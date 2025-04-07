import pytest

from fractal.core.entities.uniswap_v2_lp import (UniswapV2LPConfig,
                                                 UniswapV2LPEntity,
                                                 UniswapV2LPGlobalState)


@pytest.fixture
def uniswap_lp_entity():
    config = UniswapV2LPConfig(fees_rate=0.005, token0_decimals=18, token1_decimals=18, trading_fee=0.003)
    return UniswapV2LPEntity(config=config)


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
    uniswap_lp_entity.update_state(UniswapV2LPGlobalState(price=1000))
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    assert uniswap_lp_entity.balance == 998.5
    assert uniswap_lp_entity.is_position == True
    assert uniswap_lp_entity._internal_state.token0_amount == 249.25
    assert uniswap_lp_entity._internal_state.token1_amount == 0.24925
    assert uniswap_lp_entity._internal_state.price_init == 1000
    assert uniswap_lp_entity._internal_state.liquidity == 2485022.5


@pytest.mark.core
def test_action_close_position(uniswap_lp_entity):
    uniswap_lp_entity.update_state(UniswapV2LPGlobalState(price=1000))
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    uniswap_lp_entity.action_close_position()
    assert uniswap_lp_entity.balance == 995.5045
    assert uniswap_lp_entity.is_position == False


@pytest.mark.core
def test_update_state(uniswap_lp_entity):
    state = UniswapV2LPGlobalState(price=1000)
    uniswap_lp_entity.update_state(state)
    assert uniswap_lp_entity._global_state == state


@pytest.mark.core
def test_balance(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    assert uniswap_lp_entity.balance == 1000


@pytest.mark.core
def test_calculate_fees(uniswap_lp_entity):
    uniswap_lp_entity.update_state(UniswapV2LPGlobalState(price=1000, liquidity=1000000, fees=100))
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    fees = uniswap_lp_entity.calculate_fees()
    assert fees == 248.50225
