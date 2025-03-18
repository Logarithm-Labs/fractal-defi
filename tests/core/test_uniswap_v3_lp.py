import pytest

from fractal.core.entities.uniswap_v3_lp import (UniswapV3LPConfig,
                                                 UniswapV3LPEntity,
                                                 UniswapV3LPGlobalState)


@pytest.fixture
def uniswap_lp_entity():
    config = UniswapV3LPConfig()
    return UniswapV3LPEntity(config=config)


def test_action_deposit(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    assert uniswap_lp_entity._internal_state.cash == 1000


def test_action_withdraw(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_withdraw(500)
    assert uniswap_lp_entity._internal_state.cash == 500


def test_action_open_position(uniswap_lp_entity):
    uniswap_lp_entity.update_state(UniswapV3LPGlobalState(price=1.0))
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500, 0.9, 1.1)
    assert uniswap_lp_entity.is_position == True
    assert uniswap_lp_entity._internal_state.cash == 500
    assert uniswap_lp_entity._internal_state.token0_amount > 0
    assert uniswap_lp_entity._internal_state.token1_amount > 0
    assert uniswap_lp_entity.balance == 999.2866307383232


def test_action_close_position(uniswap_lp_entity):
    uniswap_lp_entity.update_state(UniswapV3LPGlobalState(price=1.0))
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500, 0.9, 1.1)
    assert uniswap_lp_entity.balance == 999.2866307383232
    uniswap_lp_entity.action_close_position()
    assert uniswap_lp_entity.is_position == False
    assert uniswap_lp_entity._internal_state.cash == 996.2887708461083


def test_update_state(uniswap_lp_entity):
    state = UniswapV3LPGlobalState(price=1.0)
    uniswap_lp_entity.update_state(state)
    assert uniswap_lp_entity._global_state.price == 1.0


def test_balance(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    assert uniswap_lp_entity.balance == 1000


def test_get_desired_token0_amount(uniswap_lp_entity):
    desired_token0_amount = uniswap_lp_entity.get_desired_token0_amount(500, 1.0, 0.9, 1.1)
    assert desired_token0_amount > 0


def test_calculate_position(uniswap_lp_entity):
    uniswap_lp_entity.calculate_position(500, 1.0, 0.9, 1.1)
    assert uniswap_lp_entity._internal_state.token0_amount > 0
    assert uniswap_lp_entity._internal_state.token1_amount > 0


def test_calculate_position_from_notional(uniswap_lp_entity):
    uniswap_lp_entity.calculate_position_from_notional(500, 1.0, 0.9, 1.1)
    assert uniswap_lp_entity._internal_state.token0_amount > 0
    assert uniswap_lp_entity._internal_state.token1_amount > 0


def test_price_to_tick(uniswap_lp_entity):
    tick = uniswap_lp_entity.price_to_tick(1.0)
    assert tick >= 0


def test_tick_to_price(uniswap_lp_entity):
    price = uniswap_lp_entity.tick_to_price(0)
    assert price >= 0
