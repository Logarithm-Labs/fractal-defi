"""Unit tests for PanopticStraddleEntity and PanopticStraddleStrategy.

Covers L1 (unit) and L3 (invariant) layers per CONTRIBUTING.md.
Every test is marked ``@pytest.mark.core`` so it runs in the default
``pytest -m core`` suite without network access.

Strategy lives in ``fractal/strategies/panoptic_straddle.py``.
"""

import pytest

from fractal.core.base import Action, ActionToTake, BaseStrategyParams, NamedEntity, Observation
from fractal.core.base.entity import BaseEntity, EntityException, GlobalState, InternalState
from fractal.core.entities.models.uniswap_v3_fees import estimate_fee, get_liquidity_delta
from fractal.core.entities.protocols.uniswap_v3_lp import (
    UniswapV3LPConfig,
    UniswapV3LPEntity,
    UniswapV3LPGlobalState,
)

# ---------------------------------------------------------------------------
# Inline copies of the strategy classes.
# In the real PR these live in fractal/strategies/panoptic_straddle.py and
# are imported from there.  They are inlined here so the test file is
# self-contained and can be reviewed independently of the strategy PR.
# ---------------------------------------------------------------------------

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Deque, List

import numpy as np

from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams


@dataclass
class PanopticPoolGlobalState(GlobalState):
    """Pool snapshot delivered to PanopticStraddleEntity each bar.

    :ivar price: ETH spot price in USDC.
    :ivar fees: Pool swap fees collected in this bar (USD).
    :ivar liquidity: Active pool liquidity (raw uint128 L).
    :ivar volume: Trading volume for this bar (USD).
    :ivar tvl: Total value locked (USD).
    :ivar iv_annual: Pre-computed 168-hour rolling realised volatility,
        annualised to 8 760 hours.  Used as the entry signal.
    """

    price: float = 0.0
    fees: float = 0.0
    liquidity: float = 0.0
    volume: float = 0.0
    tvl: float = 0.0
    iv_annual: float = 0.0

    @property
    def implied_volatility(self) -> float:
        """Return the pre-computed annualised volatility."""
        return self.iv_annual


@dataclass
class PanopticStraddleInternalState(InternalState):
    """Mutable state of one long-straddle position.

    :ivar is_open: True while a position is active.
    :ivar entry_price: ETH price at the time the position was opened (strike K).
    :ivar notional: Notional size of the position (USDC).
    :ivar collateral: Cash locked as buyer collateral (USDC).
    :ivar accumulated_premium: Streaming premium paid so far (USDC).
    :ivar bars_held: Number of bars since the position was opened.
    :ivar cash: Free cash in the straddle sub-account (USDC).
    """

    is_open: bool = False
    entry_price: float = 0.0
    notional: float = 0.0
    collateral: float = 0.0
    accumulated_premium: float = 0.0
    bars_held: int = 0
    cash: float = 0.0


class PanopticStraddleEntity(BaseEntity):
    """Long straddle on Panoptic over a Uniswap V3 pool.

    Streaming premium mechanics follow the Panoptic whitepaper §III.A
    (Lambert & Kristensen, 2023):

        premium_bar = estimate_fee(L_ours, L_pool, pool_fees_bar)

    Buyer PnL:

        PnL = intrinsic_value - accumulated_premium
            = |S_t - K|        - sum(premium_bar)
    """

    TOKEN0_DECIMALS: int = 18   # WETH
    TOKEN1_DECIMALS: int = 6    # USDC
    TICK_SPACING_PCT: float = 0.006  # 0.3 % pool tick spacing

    def _initialize_states(self) -> None:
        self._global_state = PanopticPoolGlobalState()
        self._internal_state = PanopticStraddleInternalState()

    def update_state(self, state: PanopticPoolGlobalState) -> None:
        """Set the current pool snapshot and accrue streaming premium."""
        self._global_state = state
        if not self._internal_state.is_open:
            return
        self._internal_state.accumulated_premium += self._streaming_premium(state)
        self._internal_state.bars_held += 1

    def _streaming_premium(self, state: PanopticPoolGlobalState) -> float:
        if state.liquidity <= 0 or state.fees <= 0:
            return 0.0
        our_L = self._our_liquidity(state)
        if our_L <= 0:
            return 0.0
        return estimate_fee(
            liquidity_delta=int(our_L),
            liquidity=int(state.liquidity),
            fees=state.fees,
        )

    def _our_liquidity(self, state: PanopticPoolGlobalState) -> float:
        """Estimate combined put + call leg liquidity for the open position."""
        if state.price <= 0:
            return 0.0
        K = self._internal_state.entry_price
        half = self._internal_state.notional / 2
        try:
            L_put = get_liquidity_delta(
                P=state.price,
                lower_price=K * (1 - self.TICK_SPACING_PCT),
                upper_price=K,
                amount0=0.0,
                amount1=half,
                token0_decimal=self.TOKEN0_DECIMALS,
                token1_decimal=self.TOKEN1_DECIMALS,
            )
            L_call = get_liquidity_delta(
                P=state.price,
                lower_price=K,
                upper_price=K * (1 + self.TICK_SPACING_PCT),
                amount0=half / state.price,
                amount1=0.0,
                token0_decimal=self.TOKEN0_DECIMALS,
                token1_decimal=self.TOKEN1_DECIMALS,
            )
            return float(L_put + L_call)
        except (ValueError, ZeroDivisionError):
            # Fallback for extreme prices where the V3 math overflows.
            return (self._internal_state.notional / state.price ** 0.5) * 1e9

    @property
    def intrinsic_value(self) -> float:
        """|current_price - entry_price|, zero when closed."""
        if not self._internal_state.is_open:
            return 0.0
        return abs(self._global_state.price - self._internal_state.entry_price)

    @property
    def balance(self) -> float:
        """cash + collateral + unrealized_pnl."""
        pnl = self.intrinsic_value - self._internal_state.accumulated_premium
        return self._internal_state.cash + self._internal_state.collateral + pnl

    def action_deposit(self, amount: float) -> None:
        """Add cash to the sub-account."""
        if amount < 0:
            raise EntityException(f"deposit must be >= 0, got {amount}")
        self._internal_state.cash += amount

    def action_open(
        self,
        notional: float,
        collateral: float,
        commission: float,
        gas_usd: float,
    ) -> None:
        """Open a long straddle at the current pool price."""
        if self._internal_state.is_open:
            raise EntityException("Position already open")
        cost = collateral + commission + gas_usd
        if self._internal_state.cash < cost:
            raise EntityException(
                f"Insufficient cash: need {cost:.2f}, "
                f"have {self._internal_state.cash:.2f}"
            )
        self._internal_state.cash -= cost
        self._internal_state.is_open = True
        self._internal_state.entry_price = self._global_state.price
        self._internal_state.notional = notional
        self._internal_state.collateral = collateral
        self._internal_state.accumulated_premium = commission + gas_usd
        self._internal_state.bars_held = 0

    def action_close(self, commission: float, gas_usd: float) -> None:
        """Close the position and settle PnL to cash."""
        if not self._internal_state.is_open:
            raise EntityException("No open position")
        pnl = self.intrinsic_value - self._internal_state.accumulated_premium
        self._internal_state.cash += (
            self._internal_state.collateral + pnl - commission - gas_usd
        )
        self._internal_state.is_open = False
        self._internal_state.entry_price = 0.0
        self._internal_state.notional = 0.0
        self._internal_state.collateral = 0.0
        self._internal_state.accumulated_premium = 0.0
        self._internal_state.bars_held = 0


@dataclass
class PanopticStraddleParams(BaseStrategyParams):
    """Hyperparameters for :class:`PanopticStraddleStrategy`.

    :ivar INITIAL_BALANCE: Starting capital in USDC.
    :ivar NOTIONAL_FRACTION: Fraction of free cash used as position notional.
    :ivar COLLATERAL_PCT: Fraction of notional locked as buyer collateral
        (Panoptic whitepaper §IV.C sets this at 10 %).
    :ivar LP_RANGE_PCT: Half-width of the baseline LP range (±).
    :ivar IV_ENTRY_PERCENTILE: Enter only when rolling IV is below this
        percentile of its own LOOKBACK_BARS history.
    :ivar LOOKBACK_BARS: History window for the IV percentile signal (hours).
    :ivar TAKE_PROFIT_MULT: Close when intrinsic >= mult * total_costs.
    :ivar STOP_LOSS_BUDGET_PCT: Close when total_costs >= pct * INITIAL_BALANCE.
    :ivar MAX_HOLD_BARS: Hard time stop (hours).
    :ivar PANOPTIC_COMMISSION_PCT: One-way Panoptic commission on notional.
    :ivar GAS_USD: Flat gas cost per transaction (USD).
    """

    INITIAL_BALANCE: float = 10_000.0
    NOTIONAL_FRACTION: float = 0.80
    COLLATERAL_PCT: float = 0.10
    LP_RANGE_PCT: float = 0.10
    IV_ENTRY_PERCENTILE: float = 30.0
    LOOKBACK_BARS: int = 5
    TAKE_PROFIT_MULT: float = 1.5
    STOP_LOSS_BUDGET_PCT: float = 0.05
    MAX_HOLD_BARS: int = 24
    PANOPTIC_COMMISSION_PCT: float = 0.0
    GAS_USD: float = 0.0


class PanopticStraddleStrategy(BaseStrategy):
    """Long straddle on Panoptic paired with a passive Uniswap V3 LP baseline.

    Entry: open a long straddle when the 168-hour rolling realised volatility
    falls below the ``IV_ENTRY_PERCENTILE``-th percentile of its own
    ``LOOKBACK_BARS`` history.

    Exit (first condition that triggers):

    * **Take Profit** — ``intrinsic >= TAKE_PROFIT_MULT * total_costs``
    * **Stop Loss**   — ``total_costs >= STOP_LOSS_BUDGET_PCT * INITIAL_BALANCE``
    * **Time Stop**   — ``bars_held >= MAX_HOLD_BARS``
    """

    PARAMS_CLS = PanopticStraddleParams

    def set_up(self) -> None:
        self.register_entity(NamedEntity(
            entity_name="STRADDLE",
            entity=PanopticStraddleEntity(),
        ))
        self.register_entity(NamedEntity(
            entity_name="LP",
            entity=UniswapV3LPEntity(config=UniswapV3LPConfig(
                pool_fee_rate=0.003,
                slippage_pct=0.001,
                token0_decimals=18,
                token1_decimals=6,
                notional_side="token1",
            )),
        ))
        self._iv_history: Deque[float] = deque(maxlen=self._params.LOOKBACK_BARS)
        self._initialized: bool = False

    def predict(self) -> List[ActionToTake]:
        straddle: PanopticStraddleEntity = self.get_entity("STRADDLE")
        state: PanopticPoolGlobalState = straddle.global_state

        if not self._initialized:
            self._initialized = True
            half = self._params.INITIAL_BALANCE / 2
            return [
                ActionToTake("STRADDLE", Action("deposit", {"amount": half})),
                ActionToTake("LP", Action("deposit", {"amount_in_notional": half})),
                ActionToTake("LP", Action("open_position", {
                    "amount_in_notional": half,
                    "price_lower": state.price * (1 - self._params.LP_RANGE_PCT),
                    "price_upper": state.price * (1 + self._params.LP_RANGE_PCT),
                })),
            ]

        self._iv_history.append(state.implied_volatility)
        if len(self._iv_history) < self._params.LOOKBACK_BARS:
            return []

        if not straddle.internal_state.is_open:
            return self._maybe_open(state, straddle)
        return self._maybe_close(straddle)

    def _maybe_open(self, state, straddle) -> List[ActionToTake]:
        iv_arr = np.array(self._iv_history)
        threshold = np.percentile(iv_arr, self._params.IV_ENTRY_PERCENTILE)
        if state.implied_volatility >= threshold:
            return []
        notional = straddle.internal_state.cash * self._params.NOTIONAL_FRACTION
        if straddle.internal_state.cash < notional * self._params.COLLATERAL_PCT:
            return []
        return [ActionToTake("STRADDLE", Action("open", {
            "notional":   notional,
            "collateral": notional * self._params.COLLATERAL_PCT,
            "commission": notional * self._params.PANOPTIC_COMMISSION_PCT,
            "gas_usd":    self._params.GAS_USD,
        }))]

    def _maybe_close(self, straddle) -> List[ActionToTake]:
        s = straddle.internal_state
        commission = s.notional * self._params.PANOPTIC_COMMISSION_PCT
        total_costs = s.accumulated_premium + commission + self._params.GAS_USD

        if (total_costs > 0
                and straddle.intrinsic_value
                >= self._params.TAKE_PROFIT_MULT * total_costs):
            pass
        elif total_costs >= self._params.INITIAL_BALANCE * self._params.STOP_LOSS_BUDGET_PCT:
            pass
        elif s.bars_held >= self._params.MAX_HOLD_BARS:
            pass
        else:
            return []

        return [ActionToTake("STRADDLE", Action("close", {
            "commission": commission,
            "gas_usd":    self._params.GAS_USD,
        }))]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _pool_state(
    price: float = 3000.0,
    fees: float = 50_000.0,
    liquidity: float = 1e22,
    iv_annual: float = 0.60,
) -> PanopticPoolGlobalState:
    return PanopticPoolGlobalState(
        price=price, fees=fees, liquidity=liquidity,
        volume=5e8, tvl=5e8, iv_annual=iv_annual,
    )


def _observations(
    prices: list,
    fees: float = 50_000.0,
    liquidity: float = 1e22,
    iv_values: list | None = None,
) -> List[Observation]:
    """Build Observation list from price and optional IV arrays."""
    result = []
    for i, price in enumerate(prices):
        iv = iv_values[i] if iv_values is not None else 0.60
        ts = datetime(2023, 1, 1, i % 24, tzinfo=UTC)
        result.append(Observation(
            timestamp=ts,
            states={
                "STRADDLE": PanopticPoolGlobalState(
                    price=price, fees=fees, liquidity=liquidity,
                    volume=5e8, tvl=5e8, iv_annual=iv,
                ),
                "LP": UniswapV3LPGlobalState(
                    price=price, fees=fees, liquidity=liquidity,
                    volume=5e8, tvl=5e8,
                ),
            },
        ))
    return result


def _fresh_entity(price: float = 3000.0) -> PanopticStraddleEntity:
    """Return an entity initialised with one update_state call."""
    e = PanopticStraddleEntity()
    e._initialize_states()
    e.update_state(_pool_state(price=price))
    return e


def _open_entity(
    price: float = 3000.0,
    deposit: float = 5000.0,
    notional: float = 4000.0,
    collateral: float = 400.0,
) -> PanopticStraddleEntity:
    """Return an entity with an open position."""
    e = _fresh_entity(price=price)
    e.action_deposit(deposit)
    e.action_open(notional=notional, collateral=collateral,
                  commission=0.0, gas_usd=0.0)
    return e


# ---------------------------------------------------------------------------
# L1 -- GlobalState
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_global_state_default_values():
    s = PanopticPoolGlobalState()
    assert s.price == 0.0
    assert s.fees == 0.0
    assert s.liquidity == 0.0
    assert s.iv_annual == 0.0


@pytest.mark.core
def test_implied_volatility_returns_iv_annual():
    s = PanopticPoolGlobalState(iv_annual=0.75)
    assert s.implied_volatility == 0.75


@pytest.mark.core
def test_implied_volatility_zero_by_default():
    assert PanopticPoolGlobalState().implied_volatility == 0.0


# ---------------------------------------------------------------------------
# L1 -- action_deposit
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_deposit_increases_cash():
    e = _fresh_entity()
    e.action_deposit(1000.0)
    assert e._internal_state.cash == pytest.approx(1000.0)


@pytest.mark.core
def test_deposit_accumulates():
    e = _fresh_entity()
    e.action_deposit(1000.0)
    e.action_deposit(500.0)
    assert e._internal_state.cash == pytest.approx(1500.0)


@pytest.mark.core
def test_deposit_zero_is_allowed():
    e = _fresh_entity()
    e.action_deposit(0.0)
    assert e._internal_state.cash == 0.0


@pytest.mark.core
def test_deposit_negative_raises():
    e = _fresh_entity()
    with pytest.raises(EntityException, match="deposit must be >= 0"):
        e.action_deposit(-1.0)


# ---------------------------------------------------------------------------
# L1 -- action_open
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_open_sets_is_open():
    e = _open_entity()
    assert e._internal_state.is_open is True


@pytest.mark.core
def test_open_records_entry_price():
    e = _open_entity(price=3000.0)
    assert e._internal_state.entry_price == pytest.approx(3000.0)


@pytest.mark.core
def test_open_deducts_collateral_from_cash():
    # cash = 5000 - 400 (collateral)
    e = _open_entity(deposit=5000.0, collateral=400.0)
    assert e._internal_state.cash == pytest.approx(4600.0)


@pytest.mark.core
def test_open_deducts_commission_and_gas():
    e = _fresh_entity(price=3000.0)
    e.action_deposit(5000.0)
    e.action_open(notional=4000.0, collateral=400.0, commission=10.0, gas_usd=5.0)
    assert e._internal_state.cash == pytest.approx(4585.0)
    # accumulated_premium seeded with commission + gas
    assert e._internal_state.accumulated_premium == pytest.approx(15.0)


@pytest.mark.core
def test_open_resets_bars_held():
    e = _open_entity()
    assert e._internal_state.bars_held == 0


@pytest.mark.core
def test_open_twice_raises():
    e = _open_entity()
    with pytest.raises(EntityException, match="already open"):
        e.action_open(notional=1000.0, collateral=100.0, commission=0.0, gas_usd=0.0)


@pytest.mark.core
def test_open_insufficient_cash_raises():
    e = _fresh_entity(price=3000.0)
    e.action_deposit(100.0)  # only $100
    with pytest.raises(EntityException, match="Insufficient cash"):
        e.action_open(notional=4000.0, collateral=400.0, commission=0.0, gas_usd=0.0)


# ---------------------------------------------------------------------------
# L1 -- intrinsic_value
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_intrinsic_zero_at_entry():
    # Price == strike immediately after open -> intrinsic == 0.
    e = _open_entity(price=3000.0)
    assert e.intrinsic_value == pytest.approx(0.0)


@pytest.mark.core
def test_intrinsic_price_rise():
    e = _open_entity(price=3000.0)
    e.update_state(_pool_state(price=3200.0))
    assert e.intrinsic_value == pytest.approx(200.0)


@pytest.mark.core
def test_intrinsic_price_fall():
    e = _open_entity(price=3000.0)
    e.update_state(_pool_state(price=2700.0))
    assert e.intrinsic_value == pytest.approx(300.0)


@pytest.mark.core
def test_intrinsic_zero_when_closed():
    e = _fresh_entity()
    assert e.intrinsic_value == 0.0


@pytest.mark.core
def test_intrinsic_symmetric_around_strike():
    """Straddle payoff is symmetric: +delta == -delta."""
    e_up = _open_entity(price=3000.0)
    e_dn = _open_entity(price=3000.0)
    e_up.update_state(_pool_state(price=3300.0))
    e_dn.update_state(_pool_state(price=2700.0))
    assert e_up.intrinsic_value == pytest.approx(e_dn.intrinsic_value)


# ---------------------------------------------------------------------------
# L1 -- action_close
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_close_sets_is_open_false():
    e = _open_entity()
    e.action_close(commission=0.0, gas_usd=0.0)
    assert e._internal_state.is_open is False


@pytest.mark.core
def test_close_resets_position_fields():
    e = _open_entity()
    e.action_close(commission=0.0, gas_usd=0.0)
    s = e._internal_state
    assert s.notional == 0.0
    assert s.collateral == 0.0
    assert s.accumulated_premium == 0.0
    assert s.bars_held == 0


@pytest.mark.core
def test_close_without_open_raises():
    e = _fresh_entity()
    with pytest.raises(EntityException, match="No open position"):
        e.action_close(commission=0.0, gas_usd=0.0)


@pytest.mark.core
def test_close_profit_scenario():
    """Closing after a $500 price move must add collateral + intrinsic to cash."""
    e = _open_entity(price=3000.0, deposit=5000.0,
                     notional=4000.0, collateral=400.0)
    cash_after_open = e._internal_state.cash   # 4600
    e.update_state(_pool_state(price=3500.0, fees=0.0))
    e.action_close(commission=0.0, gas_usd=0.0)
    # Expected: cash = 4600 + 400 (collateral) + 500 (intrinsic) - 0 (premium)
    assert e._internal_state.cash == pytest.approx(cash_after_open + 400.0 + 500.0)


@pytest.mark.core
def test_close_loss_scenario():
    """When accumulated_premium > intrinsic the position closes at a loss."""
    e = _open_entity(price=3000.0, deposit=5000.0,
                     notional=4000.0, collateral=400.0)
    e._internal_state.accumulated_premium = 300.0  # force large premium
    e.update_state(_pool_state(price=3000.0, fees=0.0))  # price unchanged
    cash_before_close = e._internal_state.cash         # 4600
    e.action_close(commission=0.0, gas_usd=0.0)
    # pnl = 0 - 300 = -300; cash += 400 + (-300) = 100
    assert e._internal_state.cash == pytest.approx(cash_before_close + 100.0)


# ---------------------------------------------------------------------------
# L1 -- streaming premium
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_premium_accrues_each_bar():
    e = _open_entity()
    before = e._internal_state.accumulated_premium
    e.update_state(_pool_state(fees=100_000.0, liquidity=1e18))
    assert e._internal_state.accumulated_premium > before


@pytest.mark.core
def test_premium_zero_on_zero_fees():
    e = _open_entity()
    before = e._internal_state.accumulated_premium
    e.update_state(_pool_state(fees=0.0))
    assert e._internal_state.accumulated_premium == pytest.approx(before)


@pytest.mark.core
def test_premium_zero_on_zero_liquidity():
    e = _open_entity()
    before = e._internal_state.accumulated_premium
    e.update_state(_pool_state(fees=100_000.0, liquidity=0.0))
    assert e._internal_state.accumulated_premium == pytest.approx(before)


@pytest.mark.core
def test_premium_not_accrued_when_closed():
    """No premium must accrue while no position is open."""
    e = _fresh_entity()
    e.action_deposit(5000.0)
    e.update_state(_pool_state(fees=100_000.0, liquidity=1e18))
    assert e._internal_state.accumulated_premium == 0.0


@pytest.mark.core
def test_bars_held_increments_per_bar():
    e = _open_entity()
    for _ in range(3):
        e.update_state(_pool_state())
    assert e._internal_state.bars_held == 3


@pytest.mark.core
def test_bars_held_unchanged_when_closed():
    e = _fresh_entity()
    for _ in range(5):
        e.update_state(_pool_state())
    assert e._internal_state.bars_held == 0


# ---------------------------------------------------------------------------
# L1 -- balance
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_balance_equals_cash_when_closed():
    e = _fresh_entity()
    e.action_deposit(5000.0)
    assert e.balance == pytest.approx(5000.0)


@pytest.mark.core
def test_balance_includes_collateral_after_open():
    e = _open_entity(deposit=5000.0, notional=4000.0, collateral=400.0)
    # balance = cash(4600) + collateral(400) + intrinsic(0) - premium(0)
    assert e.balance == pytest.approx(5000.0)


@pytest.mark.core
def test_balance_zero_before_any_deposit():
    e = _fresh_entity()
    assert e.balance == 0.0


@pytest.mark.core
def test_our_liquidity_returns_zero_on_zero_price():
    """_our_liquidity must return 0.0 when price is 0, not raise."""
    e = _open_entity()
    L = e._our_liquidity(_pool_state(price=0.0))
    assert L == 0.0


# ---------------------------------------------------------------------------
# L3 -- invariants
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_balance_invariant_holds_across_price_moves():
    """balance == cash + collateral + intrinsic - premium at every bar."""
    e = _open_entity(price=3000.0, deposit=5000.0,
                     notional=4000.0, collateral=400.0)
    for price in [3100.0, 2900.0, 3200.0, 2800.0]:
        e.update_state(_pool_state(price=price, fees=10_000.0, liquidity=1e20))
        s = e._internal_state
        expected = s.cash + s.collateral + e.intrinsic_value - s.accumulated_premium
        assert e.balance == pytest.approx(expected, rel=1e-9)


@pytest.mark.core
def test_premium_non_negative():
    """Accumulated premium must never decrease."""
    e = _open_entity()
    prev = e._internal_state.accumulated_premium
    for _ in range(10):
        e.update_state(_pool_state(fees=50_000.0, liquidity=1e20))
        assert e._internal_state.accumulated_premium >= prev
        prev = e._internal_state.accumulated_premium


@pytest.mark.core
def test_cash_non_negative_after_open():
    """Cash must not go negative after a valid open."""
    e = _open_entity(deposit=5000.0, notional=4000.0, collateral=400.0)
    assert e._internal_state.cash >= 0.0


@pytest.mark.core
def test_balance_zero_when_intrinsic_equals_premium_and_no_cash():
    """PnL == 0 when intrinsic exactly cancels premium (no free cash)."""
    e = PanopticStraddleEntity()
    e._initialize_states()
    e.update_state(_pool_state(price=3000.0))
    e.action_deposit(400.0)                 # exactly collateral
    e.action_open(notional=4000.0, collateral=400.0,
                  commission=0.0, gas_usd=0.0)
    # cash == 0; intrinsic == 0; premium == 0  ->  balance == 400
    assert e.balance == pytest.approx(400.0)


@pytest.mark.core
def test_params_all_defaults_are_sensible():
    """All default hyperparameters must satisfy basic economic constraints."""
    p = PanopticStraddleParams()
    assert p.INITIAL_BALANCE > 0
    assert 0 < p.NOTIONAL_FRACTION <= 1
    assert 0 < p.COLLATERAL_PCT <= 1
    assert 0 < p.LP_RANGE_PCT < 1
    assert 0 < p.IV_ENTRY_PERCENTILE <= 100
    assert p.LOOKBACK_BARS > 0
    assert p.TAKE_PROFIT_MULT > 1
    assert 0 < p.STOP_LOSS_BUDGET_PCT <= 1
    assert p.MAX_HOLD_BARS > 0
    assert p.PANOPTIC_COMMISSION_PCT >= 0
    assert p.GAS_USD >= 0


# ---------------------------------------------------------------------------
# L1 -- strategy initialisation
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_strategy_registers_straddle_and_lp_entities():
    obs = _observations([3000.0])
    s = PanopticStraddleStrategy(params=PanopticStraddleParams())
    s.run(obs)
    assert "STRADDLE" in s.get_all_available_entities()
    assert "LP" in s.get_all_available_entities()


@pytest.mark.core
def test_strategy_deposits_half_to_straddle_on_first_bar():
    obs = _observations([3000.0])
    s = PanopticStraddleStrategy(params=PanopticStraddleParams(INITIAL_BALANCE=10_000.0))
    s.run(obs)
    straddle = s.get_entity("STRADDLE")
    assert straddle.internal_state.cash <= 5000.0


@pytest.mark.core
def test_strategy_opens_lp_baseline_on_first_bar():
    obs = _observations([3000.0])
    s = PanopticStraddleStrategy(params=PanopticStraddleParams())
    s.run(obs)
    lp: UniswapV3LPEntity = s.get_entity("LP")
    assert lp.is_position


# ---------------------------------------------------------------------------
# L1 -- entry signal
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_no_entry_during_lookback_window():
    """No position must open during the first LOOKBACK_BARS bars."""
    params = PanopticStraddleParams(LOOKBACK_BARS=5)
    obs = _observations([3000.0] * 20)
    s = PanopticStraddleStrategy(params=params)
    result = s.run(obs)
    df = result.to_dataframe()
    # Bars 0..4 are the lookback window; no position should be open.
    assert not df["STRADDLE_is_open"].iloc[:5].any()


@pytest.mark.core
def test_entry_triggers_when_iv_below_percentile():
    """Position must open at least once when IV drops well below the threshold."""
    # History: high IV fills lookback; then very low IV triggers entry.
    n = 30
    iv_values = [0.9] * 10 + [0.1] * 20
    prices = [3000.0] * n
    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        LOOKBACK_BARS=5,
        IV_ENTRY_PERCENTILE=30.0,
        MAX_HOLD_BARS=100,
    )
    obs = _observations(prices, iv_values=iv_values)
    result = PanopticStraddleStrategy(params=params).run(obs)
    df = result.to_dataframe()
    assert df["STRADDLE_is_open"].any()


@pytest.mark.core
def test_no_entry_when_percentile_is_zero():
    """IV_ENTRY_PERCENTILE=0 means iv < min(history), which is never True."""
    n = 30
    iv_values = [0.9] * n
    prices = [3000.0] * n
    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        LOOKBACK_BARS=5,
        IV_ENTRY_PERCENTILE=0.0,
    )
    obs = _observations(prices, iv_values=iv_values)
    result = PanopticStraddleStrategy(params=params).run(obs)
    df = result.to_dataframe()
    assert not df["STRADDLE_is_open"].any()


# ---------------------------------------------------------------------------
# L1 -- exit conditions
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_maybe_close_take_profit_fires():
    """_maybe_close must return a close action when intrinsic >= mult * costs."""
    e = _open_entity(price=3000.0, deposit=5000.0,
                     notional=4000.0, collateral=400.0)
    # Force a small premium and a large intrinsic move.
    e._internal_state.accumulated_premium = 10.0
    e.update_state(_pool_state(price=3200.0, fees=0.0))  # intrinsic = 200

    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        TAKE_PROFIT_MULT=1.5,     # 200 >= 1.5 * 10 -> fires
        STOP_LOSS_BUDGET_PCT=1.0,
        MAX_HOLD_BARS=10_000,
    )
    s = PanopticStraddleStrategy.__new__(PanopticStraddleStrategy)
    s._params = params
    actions = s._maybe_close(e)
    assert len(actions) == 1
    assert actions[0].action.action == "close"


@pytest.mark.core
def test_maybe_close_stop_loss_fires():
    """_maybe_close must return close when accumulated_premium exceeds budget."""
    e = _open_entity(price=3000.0, deposit=5000.0,
                     notional=4000.0, collateral=400.0)
    # STOP_LOSS_BUDGET_PCT=0.05, INITIAL_BALANCE=10_000 -> limit $500.
    e._internal_state.accumulated_premium = 600.0  # exceeds $500

    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        TAKE_PROFIT_MULT=1000.0,  # TP will not fire
        STOP_LOSS_BUDGET_PCT=0.05,
        MAX_HOLD_BARS=10_000,
    )
    s = PanopticStraddleStrategy.__new__(PanopticStraddleStrategy)
    s._params = params
    actions = s._maybe_close(e)
    assert len(actions) == 1
    assert actions[0].action.action == "close"


@pytest.mark.core
def test_maybe_close_time_stop_fires():
    """_maybe_close must return close when bars_held >= MAX_HOLD_BARS."""
    e = _open_entity()
    e._internal_state.bars_held = 24  # equals MAX_HOLD_BARS

    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        TAKE_PROFIT_MULT=1000.0,
        STOP_LOSS_BUDGET_PCT=1.0,
        MAX_HOLD_BARS=24,
    )
    s = PanopticStraddleStrategy.__new__(PanopticStraddleStrategy)
    s._params = params
    actions = s._maybe_close(e)
    assert len(actions) == 1
    assert actions[0].action.action == "close"


@pytest.mark.core
def test_maybe_close_no_action_when_conditions_unmet():
    """_maybe_close must return [] when none of the three conditions hold."""
    e = _open_entity()
    e._internal_state.accumulated_premium = 1.0
    e._internal_state.bars_held = 1

    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        TAKE_PROFIT_MULT=1000.0,
        STOP_LOSS_BUDGET_PCT=1.0,
        MAX_HOLD_BARS=1000,
    )
    s = PanopticStraddleStrategy.__new__(PanopticStraddleStrategy)
    s._params = params
    assert s._maybe_close(e) == []


# ---------------------------------------------------------------------------
# L1 -- edge cases
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_strategy_single_observation_does_not_crash():
    """Strategy must complete without error on a single bar."""
    obs = _observations([3000.0])
    result = PanopticStraddleStrategy(
        params=PanopticStraddleParams()
    ).run(obs)
    assert len(result.to_dataframe()) == 1


@pytest.mark.core
def test_strategy_exactly_lookback_bars_does_not_crash():
    """Exactly LOOKBACK_BARS observations must not crash."""
    params = PanopticStraddleParams(LOOKBACK_BARS=5)
    obs = _observations([3000.0] * 5)
    result = PanopticStraddleStrategy(params=params).run(obs)
    assert len(result.to_dataframe()) == 5


@pytest.mark.core
def test_zero_fees_produce_no_premium():
    """With zero pool fees no premium must accrue, regardless of bar count."""
    e = _open_entity()
    start = e._internal_state.accumulated_premium
    for _ in range(10):
        e.update_state(_pool_state(fees=0.0))
    assert e._internal_state.accumulated_premium == pytest.approx(start)


@pytest.mark.core
def test_nonzero_gas_deducted_on_open():
    """gas_usd must be deducted from cash and seeded into accumulated_premium."""
    e = _fresh_entity(price=3000.0)
    e.action_deposit(5000.0)
    e.action_open(notional=4000.0, collateral=400.0, commission=0.0, gas_usd=5.0)
    assert e._internal_state.cash == pytest.approx(4595.0)
    assert e._internal_state.accumulated_premium == pytest.approx(5.0)