"""Leveraged Pendle PT ("PT looping").

Buy PT, post it as collateral on an isolated lending market, borrow the
loan asset, buy more PT, repeat — or do it in one shot with a flash
loan. Hold to expiry (PT redeems at par), or exit early. Classic carry:

    L_n     = (1 − ℓ^(n+1)) / (1 − ℓ)          leverage after n loops at LTV ℓ
    L_inf   = 1 / (1 − ℓ)                        flash-loan "multiply"
    apy_net ≈ L · y_pt − (L − 1) · r_borrow

Entities: ``PT`` (:class:`PendlePTEntity`) and ``LENDING`` (a
:class:`BaseLendingEntity` sibling such as :class:`MorphoEntity`). The
lending entity values collateral at its **oracle** price for health;
the PT entity marks at the market price — both live on their global
states, the observation builder feeds them.

Mechanics worth knowing:

* Amounts are only known at execute time, and one amount often feeds
  two entities (the PT that was just bought is deposited *and* removed;
  the loan just borrowed is deposited into the PT entity). ``_Once``
  memoises a delegate so the second entity sees the exact number the
  first one used.
* Repaying debt uses the "flash device": ``repay(R)`` first (Morpho
  callbacks are free), then withdraw collateral, sell PT and
  ``PT.withdraw(R)`` to return the flash principal — no phantom cash.
* Rollover to the next maturity is not implemented in this version;
  the strategy unwinds at expiry (``ROLL_TO_NEXT_MATURITY=True`` raises).
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams
from fractal.core.base.time import SECONDS_PER_DAY
from fractal.core.entities import BaseLendingEntity, PendlePTEntity

_MULTIPLY_MODES = ("loop", "flash")


class LeveragedPTException(Exception):
    """Configuration or state errors of :class:`LeveragedPTStrategy`."""


class _Once:
    """Delegate that evaluates ``fn(strategy)`` on first call and replays the value.

    Built fresh inside every ``predict`` so no value leaks across steps.
    """

    def __init__(self, fn: Callable[[BaseStrategy], float]) -> None:
        self._fn = fn
        self._value: Optional[float] = None

    def __call__(self, strategy: BaseStrategy) -> float:
        if self._value is None:
            self._value = float(self._fn(strategy))
        return self._value


@dataclass
class LeveragedPTParams(BaseStrategyParams):
    """Hyperparameters.

    INITIAL_BALANCE: notional deposited on the first observation.
    TARGET_LTV / TARGET_LEVERAGE: exactly one; ``ℓ = 1 − 1/L``.
    MAX_LOOPS: manual loop count (``MULTIPLY_MODE="loop"``); blocks
        self-terminate once the increment falls below ``MIN_LOOP_INCREMENT``.
    REBALANCE_LTV_BAND: ``(lo, hi)`` — boost below ``lo``, repay to target above ``hi``.
    MIN_HEALTH_FACTOR: repay to target when the lending health factor drops below it.
    MIN_CARRY_SPREAD: deleverage fully when ``implied_apy − borrow_apy`` falls below it.
    MAX_BORROW_APY: deleverage fully when the borrow APY exceeds it.
    CARRY_GATE_LOOKBACK_BARS: the two gates use the mean borrow APY over this many
        bars (``1`` = the current bar). Morpho's rate spikes for a few hours at a
        time; without smoothing the gates whipsaw the position and pay the
        round-trip swap costs on every spike.
    EXIT_BEFORE_EXPIRY_DAYS: ``0`` = hold to expiry and redeem at par; ``> 0`` = sell into the
        market that many days before.
    MIN_DAYS_TO_MATURITY_AT_ENTRY: refuse to enter closer to expiry than this.
    ROLL_TO_NEXT_MATURITY: reserved (``True`` raises ``NotImplementedError``).
    MULTIPLY_MODE: ``"loop"`` or ``"flash"`` (one-shot ``L_inf`` entry).
    FLASH_FEE: flash-loan fee fraction (Morpho/Balancer 0, Aave 0.0005).
    MIN_LOOP_INCREMENT: dust threshold in notional / loan units.
    BAR_HOURS: bar length, used to annualise the lending entity's per-bar rate.
    """
    INITIAL_BALANCE: float
    TARGET_LTV: Optional[float] = None
    TARGET_LEVERAGE: Optional[float] = None
    MAX_LOOPS: int = 6
    REBALANCE_LTV_BAND: Tuple[float, float] = (0.70, 0.88)
    MIN_HEALTH_FACTOR: float = 1.03
    MIN_CARRY_SPREAD: float = 0.0
    MAX_BORROW_APY: float = 0.25
    CARRY_GATE_LOOKBACK_BARS: int = 1
    EXIT_BEFORE_EXPIRY_DAYS: float = 0.0
    MIN_DAYS_TO_MATURITY_AT_ENTRY: float = 7.0
    ROLL_TO_NEXT_MATURITY: bool = False
    MULTIPLY_MODE: str = "loop"
    FLASH_FEE: float = 0.0
    MIN_LOOP_INCREMENT: float = 1.0
    BAR_HOURS: float = 1.0


class LeveragedPTStrategy(BaseStrategy[LeveragedPTParams]):
    """Abstract PT-looping strategy; a venue subclass registers ``PT`` and ``LENDING``."""

    #: Extra PT sold when repaying so fees and impact still cover the flash principal.
    SELL_SLACK: float = 0.01

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._target_ltv: float = self._resolve_target_ltv(self._params)
        self._deposited: bool = False
        self._exited: bool = False
        self._borrow_apy_window: deque = deque(maxlen=max(1, int(self._params.CARRY_GATE_LOOKBACK_BARS)))

    # ------------------------------------------------------------- setup
    @staticmethod
    def _resolve_target_ltv(params: LeveragedPTParams) -> float:
        if (params.TARGET_LTV is None) == (params.TARGET_LEVERAGE is None):
            raise LeveragedPTException("set exactly one of TARGET_LTV or TARGET_LEVERAGE")
        if params.TARGET_LTV is not None:
            target = float(params.TARGET_LTV)
        else:
            if params.TARGET_LEVERAGE < 1:
                raise LeveragedPTException(f"TARGET_LEVERAGE must be >= 1, got {params.TARGET_LEVERAGE}")
            target = 1.0 - 1.0 / float(params.TARGET_LEVERAGE)
        lo, hi = params.REBALANCE_LTV_BAND
        if not 0.0 <= lo <= target <= hi < 1.0:
            raise LeveragedPTException(
                f"REBALANCE_LTV_BAND {params.REBALANCE_LTV_BAND} must satisfy "
                f"0 <= lo <= target ({target:.4f}) <= hi < 1"
            )
        if params.MULTIPLY_MODE not in _MULTIPLY_MODES:
            raise LeveragedPTException(f"MULTIPLY_MODE must be one of {_MULTIPLY_MODES}, got {params.MULTIPLY_MODE!r}")
        if params.ROLL_TO_NEXT_MATURITY:
            raise NotImplementedError(
                "rollover to the next maturity is not implemented; the strategy unwinds at expiry"
            )
        if params.MAX_LOOPS < 0 or params.FLASH_FEE < 0 or params.BAR_HOURS <= 0 or params.INITIAL_BALANCE <= 0:
            raise LeveragedPTException("MAX_LOOPS, FLASH_FEE must be >= 0; BAR_HOURS, INITIAL_BALANCE must be > 0")
        if params.CARRY_GATE_LOOKBACK_BARS < 1:
            raise LeveragedPTException("CARRY_GATE_LOOKBACK_BARS must be >= 1")
        return target

    def set_up(self):
        """Type-check the registered entities (venue subclasses register them first)."""
        pt = self.get_entity("PT")
        if not isinstance(pt, PendlePTEntity):
            raise LeveragedPTException(f"PT must be a PendlePTEntity, got {type(pt).__name__}")
        lending = self.get_entity("LENDING")
        if not isinstance(lending, BaseLendingEntity):
            raise LeveragedPTException(f"LENDING must be a BaseLendingEntity, got {type(lending).__name__}")
        readouts = ("ltv", "health_factor", "max_borrow_amount", "calculate_repay", "collateral_value", "debt_value")
        for member in readouts:
            if not hasattr(lending, member):
                raise LeveragedPTException(f"LENDING entity lacks the lending readout {member!r}")

    # ---------------------------------------------------------- readouts
    @property
    def target_ltv(self) -> float:
        return self._target_ltv

    @property
    def pt(self) -> PendlePTEntity:
        return self.get_entity("PT")

    @property
    def lending(self) -> BaseLendingEntity:
        return self.get_entity("LENDING")

    def borrow_apy(self) -> float:
        """Annualised borrow rate from the lending entity's per-bar rate."""
        bars_per_year = 24.0 * 365.0 / self._params.BAR_HOURS
        return math.expm1(self.lending.global_state.borrowing_rate * bars_per_year)

    def smoothed_borrow_apy(self) -> float:
        """Mean borrow APY over the last ``CARRY_GATE_LOOKBACK_BARS`` observed bars."""
        if not self._borrow_apy_window:
            return self.borrow_apy()
        return sum(self._borrow_apy_window) / len(self._borrow_apy_window)

    def carry_spread(self) -> float:
        """``implied_apy − smoothed_borrow_apy``."""
        return self.pt.implied_apy - self.smoothed_borrow_apy()

    def days_to_expiry(self) -> float:
        return self.pt.seconds_to_expiry / SECONDS_PER_DAY

    def equity(self) -> float:
        """Market-marked equity: PT (held + collateral) at market minus debt, plus cash."""
        pt = self.pt
        lending = self.lending
        pt_units = pt.internal_state.amount + lending.internal_state.collateral
        return pt.internal_state.cash + pt_units * pt.current_price - lending.debt_value

    # ----------------------------------------------------------- predict
    def predict(self) -> List[ActionToTake]:  # pylint: disable=too-many-return-statements
        pt, lending, params = self.pt, self.lending, self._params
        self._borrow_apy_window.append(self.borrow_apy())
        wiped = (
            lending.internal_state.collateral == 0 and lending.internal_state.borrowed == 0
            and pt.internal_state.amount == 0 and pt.internal_state.cash == 0
        )
        if self._deposited and not self._exited and wiped:
            raise LeveragedPTException(
                "position wiped (liquidated) after the initial deposit — refusing to re-fund from INITIAL_BALANCE"
            )
        if not self._deposited:
            if self.days_to_expiry() < params.MIN_DAYS_TO_MATURITY_AT_ENTRY:
                raise LeveragedPTException(
                    f"only {self.days_to_expiry():.1f} days to expiry at entry, below MIN_DAYS_TO_MATURITY_AT_ENTRY="
                    f"{params.MIN_DAYS_TO_MATURITY_AT_ENTRY}"
                )
            self._deposited = True
            self._debug("Entering the leveraged PT position")
            return self._enter()
        if self._exited:
            return []
        if pt.is_matured:
            self._debug("PT matured — unwinding at par")
            self._exited = True
            return self._unwind(redeem=True)
        if params.EXIT_BEFORE_EXPIRY_DAYS > 0 and self.days_to_expiry() <= params.EXIT_BEFORE_EXPIRY_DAYS:
            self._debug("Early exit before expiry — selling PT into the market")
            self._exited = True
            return self._unwind(redeem=False)
        if self.carry_spread() < params.MIN_CARRY_SPREAD or self.smoothed_borrow_apy() > params.MAX_BORROW_APY:
            if lending.internal_state.borrowed > 0:
                self._debug("Carry gate tripped — deleveraging to zero debt")
                return self._repay_to(0.0)
            return []
        lo, hi = params.REBALANCE_LTV_BAND
        above_band = lending.ltv > hi or lending.health_factor < params.MIN_HEALTH_FACTOR
        if lending.internal_state.borrowed > 0 and above_band:
            self._debug(f"LTV {lending.ltv:.4f} above band — repaying to target")
            return self._repay_to(self._target_ltv)
        if lending.ltv < lo:
            self._debug(f"LTV {lending.ltv:.4f} below band — boosting")
            return self._boost()
        return []

    # ----------------------------------------------------------- blocks
    def _loop_block(self) -> List[ActionToTake]:
        """One loop: buy PT with all cash → deposit it → borrow to target → cash it into PT."""
        dust = self._params.MIN_LOOP_INCREMENT
        target = self._target_ltv

        def _cash(strategy: BaseStrategy) -> float:
            cash = strategy.get_entity("PT").internal_state.cash
            return cash if cash >= dust else 0.0

        def _pt_units(strategy: BaseStrategy) -> float:
            return strategy.get_entity("PT").internal_state.amount

        def _borrow(strategy: BaseStrategy) -> float:
            lending = strategy.get_entity("LENDING")
            debt_price = lending.global_state.debt_price or 1.0
            headroom = (lending.collateral_value * target - lending.debt_value) / debt_price
            amount = max(0.0, min(headroom, lending.max_borrow_amount))
            return amount if amount >= dust else 0.0

        d_cash, d_pt, d_bor = _Once(_cash), _Once(_pt_units), _Once(_borrow)
        return [
            ActionToTake("PT", Action("buy", {"amount_in_notional": d_cash})),
            ActionToTake("LENDING", Action("deposit", {"amount_in_notional": d_pt})),
            ActionToTake("PT", Action("remove_product", {"amount": d_pt})),
            ActionToTake("LENDING", Action("borrow", {"amount_in_product": d_bor})),
            ActionToTake("PT", Action("deposit", {"amount_in_notional": d_bor})),
        ]

    def _close_block(self) -> List[ActionToTake]:
        """Half loop: deploy whatever cash is left into PT collateral without borrowing."""
        return self._loop_block()[:3]

    def _enter(self) -> List[ActionToTake]:
        initial = self._params.INITIAL_BALANCE
        if self._params.MULTIPLY_MODE == "flash":
            return self._flash_enter(initial)
        actions = [ActionToTake("PT", Action("deposit", {"amount_in_notional": initial}))]
        for _ in range(self._params.MAX_LOOPS):
            actions.extend(self._loop_block())
        actions.extend(self._close_block())
        return actions

    def _flash_enter(self, initial: float) -> List[ActionToTake]:
        """One-shot multiply: flash ``F = I·ℓ/(1−ℓ)``, buy PT with ``I + F``, borrow ``F(1+fee)``, repay the flash."""
        ltv = self._target_ltv
        flash = initial * ltv / (1.0 - ltv)
        repay_flash = flash * (1.0 + self._params.FLASH_FEE)
        d_all = _Once(lambda s: s.get_entity("PT").internal_state.cash)
        d_pt = _Once(lambda s: s.get_entity("PT").internal_state.amount)
        return [
            ActionToTake("PT", Action("deposit", {"amount_in_notional": initial})),
            ActionToTake("PT", Action("deposit", {"amount_in_notional": flash})),
            ActionToTake("PT", Action("buy", {"amount_in_notional": d_all})),
            ActionToTake("LENDING", Action("deposit", {"amount_in_notional": d_pt})),
            ActionToTake("PT", Action("remove_product", {"amount": d_pt})),
            ActionToTake("LENDING", Action("borrow", {"amount_in_product": repay_flash})),
            ActionToTake("PT", Action("deposit", {"amount_in_notional": repay_flash})),
            ActionToTake("PT", Action("withdraw", {"amount_in_notional": repay_flash})),
        ]

    def _boost(self) -> List[ActionToTake]:
        actions: List[ActionToTake] = []
        for _ in range(max(1, self._params.MAX_LOOPS)):
            actions.extend(self._loop_block())
        actions.extend(self._close_block())
        return actions

    def _repay_to(self, target_ltv: float) -> List[ActionToTake]:
        """Flash device: repay → withdraw PT → sell it → return the flash principal."""
        slack = self.SELL_SLACK

        def _repay(strategy: BaseStrategy) -> float:
            """Loan units to repay so that LTV lands on ``target_ltv`` *after* the
            collateral sold to fund the repayment has left the market:
            ``(D − R) / (C − k·R) = t`` ⇒ ``R = (D − t·C) / (1 − t·k)`` with
            ``k`` the oracle value removed per unit of debt repaid."""
            lending = strategy.get_entity("LENDING")
            pt = strategy.get_entity("PT")
            borrowed = lending.internal_state.borrowed
            if borrowed == 0:
                return 0.0
            if target_ltv == 0.0:
                return borrowed  # exact: the closed form would leave float dust behind
            debt_price = lending.global_state.debt_price or 1.0
            debt_value, coll_value = lending.debt_value, lending.collateral_value
            k = (1.0 + slack) * lending.global_state.collateral_price / pt.current_price
            denominator = 1.0 - target_ltv * k
            if denominator <= 0:
                return borrowed  # selling collateral cannot bring LTV down: deleverage fully
            repay_value = (debt_value - target_ltv * coll_value) / denominator
            return max(0.0, min(repay_value / debt_price, borrowed))

        d_repay = _Once(_repay)

        def _withdraw(strategy: BaseStrategy) -> float:
            lending = strategy.get_entity("LENDING")
            pt = strategy.get_entity("PT")
            repay_value = d_repay(strategy) * (lending.global_state.debt_price or 1.0)
            if repay_value == 0.0:
                return 0.0
            units = repay_value / pt.current_price * (1.0 + slack)
            if target_ltv == 0.0:
                units = lending.internal_state.collateral  # full deleverage frees everything
            return min(units, lending.internal_state.collateral)

        d_withdraw = _Once(_withdraw)
        return [
            ActionToTake("LENDING", Action("repay", {"amount_in_product": d_repay})),
            ActionToTake("LENDING", Action("withdraw", {"amount_in_notional": d_withdraw})),
            ActionToTake("PT", Action("inject_product", {"amount": d_withdraw})),
            ActionToTake("PT", Action("sell", {"amount_in_product": d_withdraw})),
            ActionToTake("PT", Action("withdraw", {"amount_in_notional": d_repay})),
        ]

    def _unwind(self, redeem: bool) -> List[ActionToTake]:
        """Repay all debt, free the collateral, turn every PT into cash."""
        d_debt = _Once(lambda s: s.get_entity("LENDING").internal_state.borrowed)
        d_coll = _Once(lambda s: s.get_entity("LENDING").internal_state.collateral)
        d_all_pt = _Once(lambda s: s.get_entity("PT").internal_state.amount)
        exit_action = "redeem" if redeem else "sell"
        return [
            ActionToTake("LENDING", Action("repay", {"amount_in_product": d_debt})),
            ActionToTake("LENDING", Action("withdraw", {"amount_in_notional": d_coll})),
            ActionToTake("PT", Action("inject_product", {"amount": d_coll})),
            ActionToTake("PT", Action(exit_action, {"amount_in_product": d_all_pt})),
            ActionToTake("PT", Action("withdraw", {"amount_in_notional": d_debt})),
        ]
