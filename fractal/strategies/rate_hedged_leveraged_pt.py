"""PT looping with a Boros yield-unit overlay on the PT's mark-to-market.

A leveraged PT position is long a fixed yield: when the market's implied
APY rises the PT is marked down, the loan-to-value rises and, in the
limit, the position is liquidated. A **long** Boros yield unit on the
funding market of the same ecosystem (ETH funding for sUSDe, whose yield
is Ethena's funding) gains when the implied APR rises, so it offsets part
of that mark-to-market. The size is an empirical hedge ratio, not a
notional match:

* ``HEDGE_SIZING="beta"`` (default) — ``N_usd = HEDGE_BETA × PT value``,
  where ``HEDGE_BETA`` is the regression coefficient of the PT's
  mark-to-market on the yield unit's (both per unit of notional). With
  ``HEDGE_DURATION_SCALED`` the notional is also scaled by
  ``T_pt / T_yu`` so ``HEDGE_BETA`` reads as a rate pass-through.
* ``HEDGE_SIZING="debt"`` — ``N_usd = HEDGE_RATIO × debt value``, the
  "fix the floating cost" reading of the same trade.

Either way the notional is capped by the margin parked in the leg
(``HEDGE_MARGIN_SHARE`` of the initial balance at Boros's initial-margin
rule with a buffer), the leg is re-synced after every loop action and when
it drifts by more than ``HEDGE_REBALANCE_THRESHOLD`` (each fill pays the
taker fee), opened lazily when the Boros market lists after entry, and at
the PT's unwind either closed at the mark (``BOROS_EXIT_POLICY="close"``)
or held to its own maturity where it settles at zero cost
(``"settle"``, the default). ``RATE_HEDGE="none"`` is the plain loop.
"""
from dataclasses import dataclass

from fractal.core.base import Action, ActionToTake, BaseStrategy
from fractal.core.entities import BorosEntity
from fractal.strategies.leveraged_pt import LeveragedPTException, LeveragedPTParams, LeveragedPTStrategy, _Once

_HEDGES = ("none", "boros")
_SIZINGS = ("beta", "debt")
_EXIT_POLICIES = ("settle", "close")


@dataclass
class RateHedgedLeveragedPTParams(LeveragedPTParams):
    """:class:`LeveragedPTParams` plus the yield-unit overlay.

    RATE_HEDGE: ``"none"`` or ``"boros"``.
    HEDGE_SIZING: ``"beta"`` (hedge ratio on the PT value) or ``"debt"`` (ratio on the debt).
    HEDGE_BETA: hedge ratio of the PT's mark-to-market on the yield unit's, per unit of notional
        (``0.10`` is the pooled estimate over eight sUSDe / USDe markets, 2025-2026).
    HEDGE_DURATION_SCALED: multiply the ``beta`` notional by ``T_pt / T_yu``.
    HEDGE_RATIO: ``"debt"`` sizing — hedge notional as a multiple of the debt value.
    HEDGE_MARGIN_SHARE: share of ``INITIAL_BALANCE`` parked in the leg (not looped).
    HEDGE_REBALANCE_THRESHOLD: relative distance from the target that triggers a re-size.
    HEDGE_MARGIN_BUFFER: keep ``balance ≥ buffer × initial margin`` when sizing.
    BOROS_EXIT_POLICY: at the PT's unwind, ``"settle"`` keeps the yield unit to its own maturity,
        ``"close"`` closes it at the mark.
    """
    RATE_HEDGE: str = "none"
    HEDGE_SIZING: str = "beta"
    HEDGE_BETA: float = 0.10
    HEDGE_DURATION_SCALED: bool = False
    HEDGE_RATIO: float = 1.0
    HEDGE_MARGIN_SHARE: float = 0.10
    HEDGE_REBALANCE_THRESHOLD: float = 0.05
    HEDGE_MARGIN_BUFFER: float = 1.10
    BOROS_EXIT_POLICY: str = "settle"


class RateHedgedLeveragedPTStrategy(LeveragedPTStrategy):
    """Entities: ``PT``, ``LENDING`` and, with ``RATE_HEDGE="boros"``, ``BOROS``."""

    STRICT_OBSERVATIONS = False  # the Boros market may list after the PT window starts

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        params: RateHedgedLeveragedPTParams = self._params
        if params.RATE_HEDGE not in _HEDGES:
            raise LeveragedPTException(f"RATE_HEDGE must be one of {_HEDGES}, got {params.RATE_HEDGE!r}")
        if params.HEDGE_SIZING not in _SIZINGS:
            raise LeveragedPTException(f"HEDGE_SIZING must be one of {_SIZINGS}, got {params.HEDGE_SIZING!r}")
        if params.BOROS_EXIT_POLICY not in _EXIT_POLICIES:
            raise LeveragedPTException(
                f"BOROS_EXIT_POLICY must be one of {_EXIT_POLICIES}, got {params.BOROS_EXIT_POLICY!r}")
        if not 0.0 <= params.HEDGE_MARGIN_SHARE < 1.0 or params.HEDGE_RATIO < 0.0 or params.HEDGE_BETA < 0.0:
            raise LeveragedPTException("HEDGE_MARGIN_SHARE must be in [0, 1); HEDGE_RATIO and HEDGE_BETA >= 0")
        if params.HEDGE_REBALANCE_THRESHOLD < 0.0 or params.HEDGE_MARGIN_BUFFER < 1.0:
            raise LeveragedPTException("HEDGE_REBALANCE_THRESHOLD must be >= 0 and HEDGE_MARGIN_BUFFER >= 1")
        if params.RATE_HEDGE == "boros" and params.HEDGE_MARGIN_SHARE == 0.0:
            raise LeveragedPTException("RATE_HEDGE='boros' needs HEDGE_MARGIN_SHARE > 0")

    # ------------------------------------------------------------- setup
    def set_up(self):
        if self.hedge_kind == "boros" and not isinstance(self.get_entity("BOROS"), BorosEntity):
            raise LeveragedPTException("BOROS must be a BorosEntity")
        super().set_up()

    # ---------------------------------------------------------- readouts
    @property
    def hedge_kind(self) -> str:
        return self._params.RATE_HEDGE

    @property
    def boros(self) -> BorosEntity | None:
        return self.get_entity("BOROS") if self.hedge_kind == "boros" else None

    def hedge_balance(self) -> float:
        return self.boros.balance if self.boros is not None else 0.0

    def hedge_notional(self) -> float:
        """USD notional of the yield-unit leg (``size × underlying price``)."""
        if self.boros is None:
            return 0.0
        return self.boros.size * self.boros.global_state.underlying_price

    def pt_value(self) -> float:
        """Market value of every PT the loop holds (collateral plus any free units)."""
        pt = self.pt
        return (pt.internal_state.amount + self.lending.internal_state.collateral) * pt.current_price

    def hedge_coverage(self) -> float:
        """Hedge notional over the sizing base (PT value for ``beta``, debt for ``debt``)."""
        base = self.pt_value() if self._params.HEDGE_SIZING == "beta" else self.lending.debt_value
        return self.hedge_notional() / base if base > 0 else 0.0

    def equity(self) -> float:
        return super().equity() + self.hedge_balance()

    def _investable(self) -> float:
        share = self._params.HEDGE_MARGIN_SHARE if self.hedge_kind != "none" else 0.0
        return self._params.INITIAL_BALANCE * (1.0 - share)

    # ----------------------------------------------------------- predict
    def predict(self) -> list[ActionToTake]:
        deposited_before, exited_before = self._deposited, self._exited
        actions = super().predict()
        if self.hedge_kind == "none":
            return actions
        if not deposited_before and self._deposited:
            return self._hedge_deposit() + actions + self._hedge_resize()
        if not exited_before and self._exited:
            return actions + self._hedge_exit()
        if self._exited:
            return []
        if actions:  # the loop moved the position: follow it
            return actions + self._hedge_resize()
        if self._hedge_needs_resize():
            self._debug("Yield-unit leg drifted from its target — re-sizing")
            return self._hedge_resize()
        return []

    # ------------------------------------------------------------ sizing
    def _leg_frozen(self, boros: BorosEntity) -> bool:
        return boros.is_matured or boros.internal_state.liquidation_count > 0

    def _target_size(self, strategy: BaseStrategy) -> float:
        """Long yield units (coins) from the sizing rule, capped by the margin the leg holds."""
        boros: BorosEntity = strategy.get_entity("BOROS")
        if self._leg_frozen(boros):
            return boros.size  # a matured or liquidated leg is left alone
        params: RateHedgedLeveragedPTParams = self._params
        price = boros.global_state.underlying_price
        if params.HEDGE_SIZING == "debt":
            notional = params.HEDGE_RATIO * strategy.get_entity("LENDING").debt_value
        else:
            notional = params.HEDGE_BETA * self.pt_value()
            if params.HEDGE_DURATION_SCALED and boros.years_to_expiry > 0:
                notional *= self.pt.years_to_expiry / boros.years_to_expiry
        target = notional / price
        unit_im = boros.k_im * max(boros.years_to_expiry, boros.time_floor_years) * max(
            abs(boros.global_state.mark_rate), boros.rate_floor) * price
        if unit_im > 0:
            target = min(target, boros.balance / (unit_im * params.HEDGE_MARGIN_BUFFER))
        return max(0.0, target)

    def _hedge_needs_resize(self) -> bool:
        boros = self.boros
        if self._leg_frozen(boros):
            return False
        target = self._target_size(self)
        if target == 0 and boros.size == 0:
            return False
        return abs(boros.size - target) > self._params.HEDGE_REBALANCE_THRESHOLD * max(abs(target), 1e-12)

    # ------------------------------------------------------------ blocks
    def _hedge_deposit(self) -> list[ActionToTake]:
        share = self._params.INITIAL_BALANCE * self._params.HEDGE_MARGIN_SHARE
        return [ActionToTake("BOROS", Action("deposit", {"amount_in_notional": share}))]

    def _hedge_resize(self) -> list[ActionToTake]:
        delta = _Once(lambda s: self._target_size(s) - s.get_entity("BOROS").size)
        return [ActionToTake("BOROS", Action("open_position", {"amount_in_product": delta}))]

    def _hedge_exit(self) -> list[ActionToTake]:
        boros = self.boros
        if self._params.BOROS_EXIT_POLICY == "settle" or self._leg_frozen(boros) or boros.size == 0:
            return []  # the entity closes the unit at its maturity; settlements run until then
        return [ActionToTake("BOROS", Action("close_position", {}))]
