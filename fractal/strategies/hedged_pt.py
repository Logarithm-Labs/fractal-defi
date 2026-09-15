"""Hedged Pendle PT carry.

Long PT of a **volatile** underlying (PT-eBTC, PT-wstETH, …) and short
the underlying on a perp so only the fixed PT yield is kept:

    carry = y_pt_fixed + funding_received − hedge_costs ± basis

The delta of ``N`` PT is ``N · pt_price_asset`` coins (it grows to ``N``
at expiry as the PT accretes to par), so the hedge is re-sized on a
drift threshold. The short perp *receives* positive funding — a
floating leg. With ``USE_BOROS`` the strategy also sells the same
notional of Boros yield units (short YU = pay floating, receive fixed),
which cancels the floating funding and locks the fixed rate quoted at
entry.

Entities: ``PT`` (:class:`PendlePTEntity`, ``asset_price`` = the coin's
notional price), ``HEDGE`` (any :class:`BasePerpEntity`; mark = the
same price), optional ``BOROS`` (:class:`BorosEntity`). Observations
may skip ``BOROS`` on bars without Boros data
(``STRICT_OBSERVATIONS = False``).

For a *stable* PT (PT-sUSDe) there is no price leg to hedge — use
:class:`LeveragedPTStrategy` with ``MAX_LOOPS=0`` instead.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams
from fractal.core.base.time import SECONDS_PER_DAY
from fractal.core.entities import BasePerpEntity, BorosEntity, PendlePTEntity

_BOROS_POLICIES = ("hold", "match_pt")


class HedgedPTException(Exception):
    """Configuration or state errors of :class:`HedgedPTStrategy`."""


class _Once:
    """Delegate evaluated once per action list (see ``leveraged_pt._Once``)."""

    def __init__(self, fn: Callable[[BaseStrategy], float]) -> None:
        self._fn = fn
        self._value: Optional[float] = None

    def __call__(self, strategy: BaseStrategy) -> float:
        if self._value is None:
            self._value = float(self._fn(strategy))
        return self._value


@dataclass
class HedgedPTParams(BaseStrategyParams):
    """Hyperparameters.

    INITIAL_BALANCE: notional deposited on the first observation.
    TARGET_HEDGE_LEVERAGE: perp leverage the hedge margin is sized for
        (``margin = |size|·mark / L``); the rest of the capital buys PT.
    HEDGE_LEVERAGE_BAND: ``(lo, hi)`` — move margin between PT and the
        hedge when the perp leverage leaves this band.
    HEDGE_REBALANCE_THRESHOLD: relative drift ``|size + N·P| / (N·P)`` that triggers a hedge re-size.
    USE_BOROS: also sell Boros yield units of the hedge's size to fix the funding.
    BOROS_MARGIN_SHARE: share of ``INITIAL_BALANCE`` parked as Boros margin when ``USE_BOROS``.
    BOROS_MATURITY_POLICY: ``"hold"`` (let the YU mature, funding floats afterwards)
        or ``"match_pt"`` (require the YU maturity to cover the PT expiry at entry).
    EXIT_BEFORE_EXPIRY_DAYS: ``0`` = hold to expiry and redeem; ``> 0`` = sell that many days before.
    MIN_DAYS_TO_MATURITY_AT_ENTRY: refuse to enter closer to expiry than this.
    """
    INITIAL_BALANCE: float
    TARGET_HEDGE_LEVERAGE: float = 2.0
    HEDGE_LEVERAGE_BAND: Tuple[float, float] = (1.0, 4.0)
    HEDGE_REBALANCE_THRESHOLD: float = 0.02
    USE_BOROS: bool = False
    BOROS_MARGIN_SHARE: float = 0.10
    BOROS_MATURITY_POLICY: str = "hold"
    EXIT_BEFORE_EXPIRY_DAYS: float = 0.0
    MIN_DAYS_TO_MATURITY_AT_ENTRY: float = 7.0


class HedgedPTStrategy(BaseStrategy[HedgedPTParams]):
    """Abstract hedged-PT carry; a venue subclass registers ``PT``, ``HEDGE`` and optionally ``BOROS``."""

    STRICT_OBSERVATIONS = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._validate_params(self._params)
        self._deposited: bool = False
        self._exited: bool = False

    # ------------------------------------------------------------- setup
    @staticmethod
    def _validate_params(params: HedgedPTParams) -> None:
        lo, hi = params.HEDGE_LEVERAGE_BAND
        if params.INITIAL_BALANCE <= 0:
            raise HedgedPTException(f"INITIAL_BALANCE must be > 0, got {params.INITIAL_BALANCE}")
        if not 0 < lo <= params.TARGET_HEDGE_LEVERAGE <= hi:
            raise HedgedPTException(
                f"HEDGE_LEVERAGE_BAND {params.HEDGE_LEVERAGE_BAND} must satisfy "
                f"0 < lo <= TARGET_HEDGE_LEVERAGE ({params.TARGET_HEDGE_LEVERAGE}) <= hi"
            )
        if params.HEDGE_REBALANCE_THRESHOLD <= 0:
            raise HedgedPTException("HEDGE_REBALANCE_THRESHOLD must be > 0")
        if not 0 <= params.BOROS_MARGIN_SHARE < 1:
            raise HedgedPTException("BOROS_MARGIN_SHARE must be in [0, 1)")
        if params.BOROS_MATURITY_POLICY not in _BOROS_POLICIES:
            raise HedgedPTException(
                f"BOROS_MATURITY_POLICY must be one of {_BOROS_POLICIES}, got {params.BOROS_MATURITY_POLICY!r}"
            )

    def set_up(self):
        """Type-check the registered entities (venue subclasses register them first)."""
        pt = self.get_entity("PT")
        if not isinstance(pt, PendlePTEntity):
            raise HedgedPTException(f"PT must be a PendlePTEntity, got {type(pt).__name__}")
        hedge = self.get_entity("HEDGE")
        if not isinstance(hedge, BasePerpEntity):
            raise HedgedPTException(f"HEDGE must be a BasePerpEntity, got {type(hedge).__name__}")
        if self._params.USE_BOROS:
            boros = self.get_entity("BOROS")
            if not isinstance(boros, BorosEntity):
                raise HedgedPTException(f"BOROS must be a BorosEntity, got {type(boros).__name__}")
        elif "BOROS" in self.get_all_available_entities():
            raise HedgedPTException("a BOROS entity is registered but USE_BOROS is False")

    # ---------------------------------------------------------- readouts
    @property
    def pt(self) -> PendlePTEntity:
        return self.get_entity("PT")

    @property
    def hedge(self) -> BasePerpEntity:
        return self.get_entity("HEDGE")

    @property
    def boros(self) -> Optional[BorosEntity]:
        return self.get_entity("BOROS") if self._params.USE_BOROS else None

    def days_to_expiry(self) -> float:
        return self.pt.seconds_to_expiry / SECONDS_PER_DAY

    def target_hedge_size(self) -> float:
        """``−N · pt_price_asset`` coins: short the PT's current delta."""
        pt = self.pt
        return -pt.internal_state.amount * pt.pt_price_asset

    def hedge_drift(self) -> float:
        """Relative distance of the hedge from its target (``0`` when both are flat)."""
        target = self.target_hedge_size()
        if target == 0:
            return abs(self.hedge.size)
        return abs(self.hedge.size - target) / abs(target)

    def equity(self) -> float:
        total = self.investable_equity()
        if self.boros is not None:
            total += self.boros.balance
        return total

    def investable_equity(self) -> float:
        """PT plus hedge margin — the capital split ``PT : margin = L : 1``."""
        return self.pt.balance + self.hedge.balance

    def boros_leg_live(self) -> bool:
        return self.boros is not None and not self.boros.is_matured

    # ----------------------------------------------------------- predict
    def predict(self) -> List[ActionToTake]:  # pylint: disable=too-many-return-statements
        params, pt, hedge = self._params, self.pt, self.hedge
        if self._deposited and not self._exited and self.equity() == 0:
            raise HedgedPTException("strategy fully wiped after the initial deposit — refusing to re-fund")
        if not self._deposited:
            if self.days_to_expiry() < params.MIN_DAYS_TO_MATURITY_AT_ENTRY:
                raise HedgedPTException(
                    f"only {self.days_to_expiry():.1f} days to expiry at entry, below "
                    f"MIN_DAYS_TO_MATURITY_AT_ENTRY={params.MIN_DAYS_TO_MATURITY_AT_ENTRY}"
                )
            if self.boros is not None and params.BOROS_MATURITY_POLICY == "match_pt":
                if self.boros.seconds_to_expiry < pt.seconds_to_expiry:
                    raise HedgedPTException("BOROS_MATURITY_POLICY='match_pt' but the YU matures before the PT")
            self._deposited = True
            self._debug("Entering the hedged PT position")
            return self._enter()
        if self._exited:
            return []
        early = params.EXIT_BEFORE_EXPIRY_DAYS > 0 and self.days_to_expiry() <= params.EXIT_BEFORE_EXPIRY_DAYS
        if pt.is_matured or early:
            self._exited = True
            self._debug("Exiting: closing the hedge and turning PT into cash")
            return self._exit(redeem=pt.is_matured)
        if hedge.balance <= 0 and hedge.size == 0 and pt.internal_state.amount > 0:
            self._debug("Hedge liquidated — re-funding it from PT")
            return self._refund_hedge()
        boros_missing = (self.boros_leg_live() and self.boros.size == 0 and hedge.size != 0
                         and self.boros.internal_state.liquidation_count == 0)
        if boros_missing:
            self._debug("Boros leg not open yet (listed after entry) — opening it against the hedge")
            return [ActionToTake("BOROS", Action("open_position", {"amount_in_product": self._boros_delta()}))]
        lo, hi = params.HEDGE_LEVERAGE_BAND
        if hedge.size != 0 and (hedge.leverage > hi or hedge.leverage < lo):
            self._debug(f"Hedge leverage {hedge.leverage:.3f} outside the band — moving margin")
            return self._rebalance_margin()
        if self.hedge_drift() > params.HEDGE_REBALANCE_THRESHOLD:
            self._debug(f"Hedge drift {self.hedge_drift():.4f} — re-sizing the hedge")
            return self._resize_hedge()
        return []

    # ------------------------------------------------------------ blocks
    def _hedge_delta(self) -> _Once:
        return _Once(lambda s: s.target_hedge_size() - s.hedge.size)

    def _boros_delta(self) -> _Once:
        """Yield units to trade so the Boros leg matches the hedge target, not the perp's delta."""
        return _Once(lambda s: s.target_hedge_size() - s.boros.size)

    def _boros_sync(self) -> List[ActionToTake]:
        if self.boros_leg_live() and self.boros.size != 0:
            return [ActionToTake("BOROS", Action("open_position", {"amount_in_product": self._boros_delta()}))]
        return []

    def _enter(self) -> List[ActionToTake]:
        params = self._params
        initial = params.INITIAL_BALANCE
        boros_share = initial * params.BOROS_MARGIN_SHARE if self.boros is not None else 0.0
        investable = initial - boros_share
        hedge_share = investable / (1.0 + params.TARGET_HEDGE_LEVERAGE)  # margin = PT value / L
        pt_share = investable - hedge_share
        delta = self._hedge_delta()
        actions = [
            ActionToTake("PT", Action("deposit", {"amount_in_notional": pt_share})),
            ActionToTake("HEDGE", Action("deposit", {"amount_in_notional": hedge_share})),
        ]
        if self.boros is not None:
            actions.append(ActionToTake("BOROS", Action("deposit", {"amount_in_notional": boros_share})))
        actions.append(ActionToTake("PT", Action("buy", {"amount_in_notional": pt_share})))
        actions.append(ActionToTake("HEDGE", Action("open_position", {"amount_in_product": delta})))
        if self.boros_leg_live():  # a YU market listed after the PT window opens later (see predict)
            actions.append(ActionToTake("BOROS", Action("open_position", {"amount_in_product": self._boros_delta()})))
        return actions

    def _resize_hedge(self) -> List[ActionToTake]:
        actions = [ActionToTake("HEDGE", Action("open_position", {"amount_in_product": self._hedge_delta()}))]
        actions.extend(self._boros_sync())
        return actions

    def _rebalance_margin(self) -> List[ActionToTake]:
        """Bring the hedge margin back to ``|size|·mark / TARGET_HEDGE_LEVERAGE``."""
        hedge, pt = self.hedge, self.pt
        # Split the investable capital ``PT : margin = L : 1`` (the entry rule):
        # sizing on the pre-move hedge would land at the band edge instead.
        target_margin = self.investable_equity() / (1.0 + self._params.TARGET_HEDGE_LEVERAGE)
        delta = target_margin - hedge.balance
        actions: List[ActionToTake] = []
        if delta > 0:
            shortfall = max(0.0, delta - pt.internal_state.cash)
            if shortfall > 0:
                units = min(shortfall / pt.current_price * 1.01, pt.internal_state.amount)
                actions.append(ActionToTake("PT", Action("sell", {"amount_in_product": units})))
            move = _Once(lambda s, d=delta: min(d, s.pt.internal_state.cash))
            actions.extend(self.transfer("PT", "HEDGE", move))
        else:
            free_margin = hedge.balance - hedge.maintenance_margin
            withdraw = min(-delta, max(0.0, free_margin))
            if withdraw <= 0:
                return []
            actions.extend(self.transfer("HEDGE", "PT", withdraw))
            actions.append(ActionToTake("PT", Action("buy", {"amount_in_notional": _Once(
                lambda s: s.pt.internal_state.cash)})))
        actions.append(ActionToTake("HEDGE", Action("open_position", {"amount_in_product": self._hedge_delta()})))
        actions.extend(self._boros_sync())
        return actions

    def _refund_hedge(self) -> List[ActionToTake]:
        """After a hedge liquidation: sell part of the PT to re-margin and re-open the short."""
        pt = self.pt
        target_margin = self.investable_equity() / (1.0 + self._params.TARGET_HEDGE_LEVERAGE)
        shortfall = max(0.0, target_margin - pt.internal_state.cash)
        units = min(shortfall / pt.current_price * 1.01, pt.internal_state.amount) if shortfall > 0 else 0.0
        move = _Once(lambda s: s.pt.internal_state.cash)
        actions = [ActionToTake("PT", Action("sell", {"amount_in_product": units}))]
        actions.extend(self.transfer("PT", "HEDGE", move))
        actions.append(ActionToTake("HEDGE", Action("open_position", {"amount_in_product": self._hedge_delta()})))
        if self.boros_leg_live():
            actions.append(ActionToTake("BOROS", Action("close_position", {})))
            actions.append(ActionToTake("BOROS", Action("open_position", {"amount_in_product": self._boros_delta()})))
        return actions

    def _exit(self, redeem: bool) -> List[ActionToTake]:
        all_pt = _Once(lambda s: s.pt.internal_state.amount)
        actions = [ActionToTake("HEDGE", Action("close_position", {}))]
        if self.boros_leg_live():
            actions.append(ActionToTake("BOROS", Action("close_position", {})))
        actions.append(ActionToTake("PT", Action("redeem" if redeem else "sell", {"amount_in_product": all_pt})))
        return actions
