import pytest
from fractal.core.entities.protocols.uniswap_v2_lp import UniswapV2LPConfig, UniswapV2LPEntity, UniswapV2LPGlobalState


@pytest.fixture
def uniswap_lp_entity():
    config = UniswapV2LPConfig(fees_rate=0.005, token0_decimals=6, token1_decimals=18, trading_fee=0.003)
    entity = UniswapV2LPEntity(config=config)
    entity.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000, fees=0, price=1000, volume=0))
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
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    assert round(uniswap_lp_entity.balance, 6) == 998.5
    assert uniswap_lp_entity.is_position is True
    assert round(uniswap_lp_entity._internal_state.token0_amount, 6) == 249.25
    assert round(uniswap_lp_entity._internal_state.token1_amount, 6) == 0.24925
    assert uniswap_lp_entity._internal_state.price_init == 1000
    assert uniswap_lp_entity._internal_state.liquidity == 498.5


@pytest.mark.core
def test_action_close_position(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    uniswap_lp_entity.action_close_position()
    assert uniswap_lp_entity.is_position is False
    assert round(uniswap_lp_entity.balance, 6) == 997.0045


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
    assert round(fees_earned, 8) == 4.985
    share = uniswap_lp_entity._internal_state.liquidity / updated_state.liquidity
    expected_fees = share * updated_state.fees
    assert round(fees_earned, 8) == round(expected_fees, 8)
