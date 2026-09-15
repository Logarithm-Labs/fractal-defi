"""PT looping with a floating-rate receiver leg that offsets the loan's floating cost.

A leveraged PT position earns a **fixed** yield and pays a **floating**
borrow rate. Stablecoin borrow rates and perp funding move together (both
price leverage demand), so a leg that *receives* floating funding turns
the loop's net cost from ``r_borrow`` into ``r_fixed + (r_borrow − f)``:

* ``RATE_HEDGE="boros"`` — long Boros yield units on the funding market
  of the coin whose funding drives the PT's underlying (ETH for sUSDe):
  pay the fixed implied APR, receive the venue's funding on
  ``HEDGE_RATIO × debt`` notional. Capital-light: only margin is parked.
* ``RATE_HEDGE="perp"`` — the same floating leg built as a delta-neutral
  basis position (long spot, short perp). Receiving funding on the whole
  debt would need spot capital equal to the debt, so the leg is sized by
  the capital it is given (``HEDGE_MARGIN_SHARE`` at ``PERP_TARGET_LEVERAGE``)
  and its ``hedge_coverage`` is reported.
* ``RATE_HEDGE="none"`` — the plain loop of :class:`LeveragedPTStrategy`.

The loop itself (entry, band, carry gates, unwind) is unchanged; the hedge
is re-synced to the debt after every loop action and on drift, opened
lazily when the Boros market lists after entry, and closed on unwind.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from fractal.core.base import Action, ActionToTake, BaseStrategy
from fractal.core.entities import BasePerpEntity, BaseSpotEntity, BorosEntity
from fractal.strategies.leveraged_pt import LeveragedPTException, LeveragedPTParams, LeveragedPTStrategy, _Once

_HEDGES = ("none", "boros", "perp")


@dataclass
class RateHedgedLeveragedPTParams(LeveragedPTParams):
    """:class:`LeveragedPTParams` plus the floating-rate leg.

    RATE_HEDGE: ``"none"``, ``"boros"`` or ``"perp"``.
    HEDGE_MARGIN_SHARE: share of ``INITIAL_BALANCE`` parked in the hedge leg (not looped).
    HEDGE_RATIO: hedge notional as a multiple of the loan's debt value (``1`` = full hedge).
    HEDGE_REBALANCE_THRESHOLD: relative distance from the target notional that triggers a re-size.
    HEDGE_MARGIN_BUFFER: Boros only — keep ``balance ≥ buffer × initial margin`` when sizing.
    PERP_TARGET_LEVERAGE / PERP_LEVERAGE_BAND: perp only — basis-leg leverage and the band that
        triggers a re-margin.
    """
    RATE_HEDGE: str = "none"
    HEDGE_MARGIN_SHARE: float = 0.10
    HEDGE_RATIO: float = 1.0
    HEDGE_REBALANCE_THRESHOLD: float = 0.05
    HEDGE_MARGIN_BUFFER: float = 1.10
    PERP_TARGET_LEVERAGE: float = 2.0
    PERP_LEVERAGE_BAND: Tuple[float, float] = (1.0, 4.0)


class RateHedgedLeveragedPTStrategy(LeveragedPTStrategy):
    """Entities: ``PT``, ``LENDING`` (+ ``BOROS`` | ``SPOT`` + ``PERP``)."""

    STRICT_OBSERVATIONS = False  # the Boros market may list after the PT window starts

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        params: RateHedgedLeveragedPTParams = self._params
        if params.RATE_HEDGE not in _HEDGES:
            raise LeveragedPTException(f"RATE_HEDGE must be one of {_HEDGES}, got {params.RATE_HEDGE!r}")
        if not 0.0 <= params.HEDGE_MARGIN_SHARE < 1.0 or params.HEDGE_RATIO < 0.0:
            raise LeveragedPTException("HEDGE_MARGIN_SHARE must be in [0, 1) and HEDGE_RATIO >= 0")
        if params.HEDGE_REBALANCE_THRESHOLD < 0.0 or params.HEDGE_MARGIN_BUFFER < 1.0:
            raise LeveragedPTException("HEDGE_REBALANCE_THRESHOLD must be >= 0 and HEDGE_MARGIN_BUFFER >= 1")
        lo, hi = params.PERP_LEVERAGE_BAND
        if not 0.0 < lo <= params.PERP_TARGET_LEVERAGE <= hi:
            raise LeveragedPTException("PERP_LEVERAGE_BAND must bracket PERP_TARGET_LEVERAGE")
        if params.RATE_HEDGE != "none" and params.HEDGE_MARGIN_SHARE == 0.0:
            raise LeveragedPTException("a rate hedge needs HEDGE_MARGIN_SHARE > 0")

    # ------------------------------------------------------------- setup
    def set_up(self):
        kind = self.hedge_kind
        if kind == "boros" and not isinstance(self.get_entity("BOROS"), BorosEntity):
            raise LeveragedPTException("BOROS must be a BorosEntity")
        if kind == "perp":
            if not isinstance(self.get_entity("SPOT"), BaseSpotEntity):
                raise LeveragedPTException("SPOT must be a BaseSpotEntity")
            if not isinstance(self.get_entity("PERP"), BasePerpEntity):
                raise LeveragedPTException("PERP must be a BasePerpEntity")
        super().set_up()

    # ---------------------------------------------------------- readouts
    @property
    def hedge_kind(self) -> str:
        return self._params.RATE_HEDGE

    @property
    def boros(self) -> Optional[BorosEntity]:
        return self.get_entity("BOROS") if self.hedge_kind == "boros" else None

    @property
    def spot(self) -> Optional[BaseSpotEntity]:
        return self.get_entity("SPOT") if self.hedge_kind == "perp" else None

    @property
    def perp(self) -> Optional[BasePerpEntity]:
        return self.get_entity("PERP") if self.hedge_kind == "perp" else None

    def hedge_balance(self) -> float:
        if self.hedge_kind == "boros":
            return self.boros.balance
        if self.hedge_kind == "perp":
            return self.spot.balance + self.perp.balance
        return 0.0

    def hedge_notional(self) -> float:
        """Notional on which the leg receives floating funding."""
        if self.hedge_kind == "boros":
            return self.boros.size * self.boros.global_state.underlying_price
        if self.hedge_kind == "perp":
            return -self.perp.size * self.perp.global_state.mark_price
        return 0.0

    def hedge_coverage(self) -> float:
        """``hedge_notional / debt_value`` (``0`` without debt)."""
        debt = self.lending.debt_value
        return self.hedge_notional() / debt if debt > 0 else 0.0

    def equity(self) -> float:
        return super().equity() + self.hedge_balance()

    def _investable(self) -> float:
        share = self._params.HEDGE_MARGIN_SHARE if self.hedge_kind != "none" else 0.0
        return self._params.INITIAL_BALANCE * (1.0 - share)

    # ----------------------------------------------------------- predict
    def predict(self) -> List[ActionToTake]:
        deposited_before, exited_before = self._deposited, self._exited
        actions = super().predict()
        if self.hedge_kind == "none":
            return actions
        if not deposited_before and self._deposited:
            return self._hedge_deposit() + actions + self._hedge_resize()
        if not exited_before and self._exited:
            return actions + self._hedge_close()
        if self._exited:
            return actions
        if actions:  # the loop moved the debt: follow it
            return actions + self._hedge_resize()
        if self._hedge_needs_resize():
            self._debug("Rate hedge drifted from the debt — re-sizing")
            return self._hedge_resize()
        return []

    # ------------------------------------------------------------ sizing
    def _boros_target_size(self, strategy: BaseStrategy) -> float:
        """Long YU (coins) = ``HEDGE_RATIO × debt / price``, capped by the margin the leg holds."""
        boros: BorosEntity = strategy.get_entity("BOROS")
        lending = strategy.get_entity("LENDING")
        if boros.is_matured or boros.internal_state.liquidation_count > 0:
            return boros.size  # a matured or liquidated leg is left alone
        price = boros.global_state.underlying_price
        target = self._params.HEDGE_RATIO * lending.debt_value / price
        unit_im = boros.k_im * max(boros.years_to_expiry, boros.time_floor_years) * max(
            abs(boros.global_state.mark_rate), boros.rate_floor) * price
        if unit_im > 0:
            target = min(target, boros.balance / (unit_im * self._params.HEDGE_MARGIN_BUFFER))
        return max(0.0, target)

    def _perp_target_notional(self, strategy: BaseStrategy) -> float:
        """Spot notional of the basis leg: the debt (× ratio) or what the leg's capital allows."""
        lending = strategy.get_entity("LENDING")
        spot, perp = strategy.get_entity("SPOT"), strategy.get_entity("PERP")
        lev = self._params.PERP_TARGET_LEVERAGE
        capital = spot.balance + perp.balance
        return max(0.0, min(self._params.HEDGE_RATIO * lending.debt_value, capital * lev / (1.0 + lev)))

    def _hedge_needs_resize(self) -> bool:
        threshold = self._params.HEDGE_REBALANCE_THRESHOLD
        if self.hedge_kind == "boros":
            boros = self.boros
            if boros.is_matured or boros.internal_state.liquidation_count > 0:
                return False
            target = self._boros_target_size(self)
            if target == 0 and boros.size == 0:
                return False
            return abs(boros.size - target) > threshold * max(abs(target), 1e-12)
        perp, spot = self.perp, self.spot
        if perp.balance <= 0 and perp.size == 0 and spot.internal_state.amount == 0:
            return False  # leg wiped: nothing to re-size
        target = self._perp_target_notional(self)
        current = self.hedge_notional()
        drift = abs(current - target) > threshold * max(target, 1e-12) if target > 0 else current > 0
        lo, hi = self._params.PERP_LEVERAGE_BAND
        band = perp.size != 0 and (perp.leverage > hi or perp.leverage < lo)
        return drift or band

    # ------------------------------------------------------------ blocks
    def _hedge_deposit(self) -> List[ActionToTake]:
        share = self._params.INITIAL_BALANCE * self._params.HEDGE_MARGIN_SHARE
        if self.hedge_kind == "boros":
            return [ActionToTake("BOROS", Action("deposit", {"amount_in_notional": share}))]
        lev = self._params.PERP_TARGET_LEVERAGE
        return [
            ActionToTake("SPOT", Action("deposit", {"amount_in_notional": share * lev / (1.0 + lev)})),
            ActionToTake("PERP", Action("deposit", {"amount_in_notional": share / (1.0 + lev)})),
        ]

    def _hedge_resize(self) -> List[ActionToTake]:
        if self.hedge_kind == "boros":
            delta = _Once(lambda s: self._boros_target_size(s) - s.get_entity("BOROS").size)
            return [ActionToTake("BOROS", Action("open_position", {"amount_in_product": delta}))]
        return self._perp_resize()

    def _perp_resize(self) -> List[ActionToTake]:
        """Rebuild the basis leg: flatten the perp, pool the cash on SPOT, buy/sell to the
        target coins, put ``notional / L`` back as margin, short the coins held."""
        lev = self._params.PERP_TARGET_LEVERAGE
        target_coins = _Once(lambda s: self._perp_target_notional(s) / s.get_entity("SPOT").current_price)

        def _buy(s: BaseStrategy) -> float:
            spot = s.get_entity("SPOT")
            delta = target_coins(s) - spot.internal_state.amount
            return min(max(0.0, delta) * spot.current_price, spot.internal_state.cash)

        def _sell(s: BaseStrategy) -> float:
            spot = s.get_entity("SPOT")
            return max(0.0, spot.internal_state.amount - target_coins(s))

        def _margin(s: BaseStrategy) -> float:
            spot = s.get_entity("SPOT")
            return min(spot.internal_state.cash, spot.internal_state.amount * spot.current_price / lev)

        actions = [ActionToTake("PERP", Action("close_position", {}))]
        actions.extend(self.transfer("PERP", "SPOT", _Once(lambda s: s.get_entity("PERP").balance)))
        actions.append(ActionToTake("SPOT", Action("sell", {"amount_in_product": _Once(_sell)})))
        actions.append(ActionToTake("SPOT", Action("buy", {"amount_in_notional": _Once(_buy)})))
        actions.extend(self.transfer("SPOT", "PERP", _Once(_margin)))
        actions.append(ActionToTake("PERP", Action("open_position", {"amount_in_product": _Once(
            lambda s: -s.get_entity("SPOT").internal_state.amount)})))
        return actions

    def _hedge_close(self) -> List[ActionToTake]:
        if self.hedge_kind == "boros":
            if self.boros.is_matured or self.boros.size == 0:
                return []
            return [ActionToTake("BOROS", Action("close_position", {}))]
        return [
            ActionToTake("PERP", Action("close_position", {})),
            ActionToTake("SPOT", Action("sell", {"amount_in_product": _Once(
                lambda s: s.get_entity("SPOT").internal_state.amount)})),
        ]
