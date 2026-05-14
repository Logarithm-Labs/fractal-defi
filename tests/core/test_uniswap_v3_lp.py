"""Functional tests for UniswapV3LPEntity.

Pool fee model (post-2026 refactor): see ``UniswapV3LPConfig`` docstring.
``trading_fee`` field has been replaced by ``pool_fee_rate`` + ``slippage_pct``;
fee is now applied only on the swapped portion of zap-in / zap-out.
The lower-level math helpers (``calculate_position`` etc.) have been removed
in favour of ``action_open_position`` (zap-in) and ``_open_from_pair`` (advanced).
"""
import pytest

from fractal.core.entities.protocols.uniswap_v3_lp import UniswapV3LPConfig, UniswapV3LPEntity, UniswapV3LPGlobalState


@pytest.fixture
def uniswap_lp_entity():
    return UniswapV3LPEntity(config=UniswapV3LPConfig())


@pytest.mark.core
def test_action_deposit(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    assert uniswap_lp_entity._internal_state.cash == 1000


@pytest.mark.core
def test_action_withdraw(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_withdraw(500)
    assert uniswap_lp_entity._internal_state.cash == 500


@pytest.mark.core
def test_action_open_position(uniswap_lp_entity):
    """Zap-in over [0.9, 1.1] at price=1.0 from 500 notional.

    With pool_fee_rate=0.003 (default) and slippage=0:
    * In range — split by V3 ratio (~50/50 in notional value at center).
    * Volatile portion swapped, paying fee on swap value.
    * Stable leftover (≈ stable_pre × fee) returns to cash.
    """
    uniswap_lp_entity.update_state(UniswapV3LPGlobalState(price=1.0))
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500, 0.9, 1.1)
    assert uniswap_lp_entity.is_position is True
    assert uniswap_lp_entity._internal_state.token0_amount > 0
    assert uniswap_lp_entity._internal_state.token1_amount > 0
    # cash includes the original 500 from deposit plus stable leftover
    assert uniswap_lp_entity._internal_state.cash > 500
    # Total balance is approximately deposit minus fee on swap-portion (volatile side).
    # Volatile_value_pre ≈ 237.83, fee = 237.83 × 0.003 ≈ 0.71 → balance ≈ 999.29.
    assert uniswap_lp_entity.balance == pytest.approx(999.286, rel=1e-3)


@pytest.mark.core
def test_action_close_position(uniswap_lp_entity):
    """Zap-out: stable side returns at full value, volatile swapped at fee."""
    uniswap_lp_entity.update_state(UniswapV3LPGlobalState(price=1.0))
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500, 0.9, 1.1)
    balance_in_position = uniswap_lp_entity.balance
    uniswap_lp_entity.action_close_position()
    assert uniswap_lp_entity.is_position is False
    # Round-trip: open fee on volatile_value (~237.83 × 0.003) + close fee on
    # volatile_back × p (~237.12 × 0.003). Total cost ≈ 1.42 → final ≈ 998.575.
    final_balance = uniswap_lp_entity._internal_state.cash
    assert final_balance < balance_in_position  # lost something on close
    assert final_balance == pytest.approx(998.575, rel=1e-3)


@pytest.mark.core
def test_update_state(uniswap_lp_entity):
    state = UniswapV3LPGlobalState(price=1.0)
    uniswap_lp_entity.update_state(state)
    assert uniswap_lp_entity._global_state.price == 1.0


@pytest.mark.core
def test_balance(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    assert uniswap_lp_entity.balance == 1000


@pytest.mark.core
def test_price_to_tick(uniswap_lp_entity):
    tick = uniswap_lp_entity.price_to_tick(1.0)
    assert tick >= 0


@pytest.mark.core
def test_tick_to_price(uniswap_lp_entity):
    price = uniswap_lp_entity.tick_to_price(0)
    assert price >= 0


@pytest.mark.core
def test_gas_cost_per_mint_haircut_on_open():
    """``gas_cost_per_mint`` is deducted flat from cash on open, on top of
    whatever swap-fee cost the zap-in would incur."""
    baseline = UniswapV3LPEntity(config=UniswapV3LPConfig())
    baseline.update_state(UniswapV3LPGlobalState(price=1.0))
    baseline.action_deposit(1000)
    baseline.action_open_position(500, 0.9, 1.1)

    with_gas = UniswapV3LPEntity(
        config=UniswapV3LPConfig(gas_cost_per_mint=7.5)
    )
    with_gas.update_state(UniswapV3LPGlobalState(price=1.0))
    with_gas.action_deposit(1000)
    with_gas.action_open_position(500, 0.9, 1.1)

    # Difference is exactly the gas haircut — gas doesn't touch swap math.
    assert with_gas._internal_state.cash == pytest.approx(
        baseline._internal_state.cash - 7.5
    )
    # Position composition is identical (gas does not affect mint amounts).
    assert with_gas._internal_state.liquidity == pytest.approx(
        baseline._internal_state.liquidity
    )


@pytest.mark.core
def test_gas_cost_burn_and_collect_haircut_on_close():
    """``gas_cost_per_burn + gas_cost_per_collect`` is deducted on close."""
    cfg = UniswapV3LPConfig(gas_cost_per_burn=4.0, gas_cost_per_collect=3.0)
    e = UniswapV3LPEntity(config=cfg)
    e.update_state(UniswapV3LPGlobalState(price=1.0))
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    cash_before_close = e._internal_state.cash

    e.action_close_position()

    baseline = UniswapV3LPEntity(config=UniswapV3LPConfig())
    baseline.update_state(UniswapV3LPGlobalState(price=1.0))
    baseline.action_deposit(1000)
    baseline.action_open_position(500, 0.9, 1.1)
    baseline_cash_before_close = baseline._internal_state.cash
    baseline.action_close_position()

    # Each leg's close-side proceeds match the baseline; difference is exactly 7.0.
    delta_after = baseline._internal_state.cash - e._internal_state.cash
    delta_before = baseline_cash_before_close - cash_before_close
    assert delta_after - delta_before == pytest.approx(7.0)


@pytest.mark.core
@pytest.mark.parametrize(
    "field",
    ["gas_cost_per_mint", "gas_cost_per_burn", "gas_cost_per_collect"],
)
def test_negative_gas_cost_rejected(field):
    """Each gas field must be non-negative — mirrors pool_fee_rate / slippage_pct."""
    from fractal.core.base.entity import EntityException

    cfg = UniswapV3LPConfig(**{field: -1.0})
    with pytest.raises(EntityException):
        UniswapV3LPEntity(config=cfg)
