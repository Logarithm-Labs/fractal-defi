"""Functional tests for UniswapV2LPEntity.

Pool fee model (post-2026 refactor):
* ``pool_fee_rate`` — pool's swap-fee tier; charged on the **swapped**
  portion only (half of notional in V2).
* ``slippage_pct`` — additional execution-cost handwave; default 0.
* Old ``trading_fee`` field is removed (was applied to the full deposit
  on both open AND close — incorrect modeling).
"""
import pytest

from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                            UniswapV2LPEntity,
                                                            UniswapV2LPGlobalState)


@pytest.fixture
def uniswap_lp_entity():
    config = UniswapV2LPConfig(pool_fee_rate=0.003, slippage_pct=0.0,
                                token0_decimals=6, token1_decimals=18)
    entity = UniswapV2LPEntity(config=config)
    entity.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                                fees=0, price=1000, volume=0))
    return entity


@pytest.mark.core
def test_action_deposit(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    assert uniswap_lp_entity.balance == 1000


@pytest.mark.core
def test_action_withdraw(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_withdraw(500)
    assert uniswap_lp_entity.balance == 500


@pytest.mark.core
def test_action_open_position(uniswap_lp_entity):
    """Zap-in: half stays as stable, half swaps to volatile (pays fee on the swap).

    With ``pool_fee_rate=0.003`` on a 500-notional zap-in into a pool with
    ``tvl=10_000, liquidity=10_000, price=1000``:
    * volatile_at_mint = (250 / 1000) * 0.997 = 0.24925 token1
    * stable_used (limited by volatile) = 0.24925 * 1000 = 249.25 (notional)
    * stable_leftover (returned to cash) = 250 - 249.25 = 0.75
    * cash after open = 1000 - 500 + 0.75 = 500.75
    * share = 0.24925 / 5 = 0.04985 → liquidity = 0.04985 * 10000 = 498.5
    * balance = 249.25 + 0.24925 * 1000 + 500.75 = 999.25
    Effective execution cost = 0.75 = pool_fee × half (only the swapped portion).
    """
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    assert uniswap_lp_entity.is_position is True
    assert uniswap_lp_entity._internal_state.token0_amount == pytest.approx(249.25)
    assert uniswap_lp_entity._internal_state.token1_amount == pytest.approx(0.24925)
    assert uniswap_lp_entity._internal_state.price_init == 1000
    assert uniswap_lp_entity._internal_state.liquidity == pytest.approx(498.5)
    assert uniswap_lp_entity.balance == pytest.approx(999.25)


@pytest.mark.core
def test_action_close_position(uniswap_lp_entity):
    """Zap-out: stable side returns at full value, volatile swaps back at pool fee.

    Position from zap-in: stable=249.25, volatile=0.24925, cash=500.75.
    On close: stable_back=249.25 (no swap), volatile_proceeds = 0.24925 * 1000 * 0.997 = 248.50225.
    Cash after close = 500.75 + 249.25 + 248.50225 = 998.50225.
    Round-trip cost = 1000 - 998.50225 = 1.49775 = 0.75 (open: half × fee) + 0.74775 (close: volatile_value × fee).
    """
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    uniswap_lp_entity.action_close_position()
    assert uniswap_lp_entity.is_position is False
    assert uniswap_lp_entity.balance == pytest.approx(998.50225)


@pytest.mark.core
def test_update_state(uniswap_lp_entity):
    state = UniswapV2LPGlobalState(tvl=20_000, liquidity=20_000, price=2000, fees=50, volume=0)
    uniswap_lp_entity.update_state(state)
    assert uniswap_lp_entity._global_state == state


@pytest.mark.core
def test_balance(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    assert uniswap_lp_entity.balance == 1000


@pytest.mark.core
def test_calculate_fees(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    updated_state = UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000, price=1000, fees=100, volume=0)
    uniswap_lp_entity.update_state(updated_state)
    fees_earned = uniswap_lp_entity.calculate_fees()
    assert fees_earned >= 0
    assert fees_earned <= updated_state.fees
    share = uniswap_lp_entity._internal_state.liquidity / updated_state.liquidity
    expected_fees = share * updated_state.fees
    assert fees_earned == pytest.approx(expected_fees)
