"""Unit tests for PanopticStraddleEntity and PanopticStraddleStrategy.

Covers L1 (unit) and L3 (invariant) layers per CONTRIBUTING.md.
Every test is marked ``@pytest.mark.core`` so it runs in the default
``pytest -m core`` suite without network access.

Strategy lives in ``fractal/strategies/panoptic_straddle.py``.
"""


from datetime import UTC, datetime, timedelta
from typing import List

import pytest

from fractal.core.base import Observation
from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.uniswap_v3_lp import UniswapV3LPEntity, UniswapV3LPGlobalState
from fractal.strategies.panoptic_straddle import (
    PanopticPoolGlobalState,
    PanopticStraddleEntity,
    PanopticStraddleParams,
    PanopticStraddleStrategy,
)

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
        t0 = datetime(2023, 1, 1, tzinfo=UTC)
        ts = t0 + timedelta(hours=i)
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
    e = _open_entity(price=3000.0)   # notional=4000, entry=3000
    e.update_state(_pool_state(price=3200.0))
    contracts = 4000.0 / 3000.0
    assert e.intrinsic_value == pytest.approx(200.0 * contracts)


@pytest.mark.core
def test_intrinsic_price_fall():
    e = _open_entity(price=3000.0)
    e.update_state(_pool_state(price=2700.0))
    contracts = 4000.0 / 3000.0
    assert e.intrinsic_value == pytest.approx(300.0 * contracts)


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
    assert e._internal_state.cash > cash_after_open + 400.0


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
