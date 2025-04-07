import numpy as np
import pytest

from fractal.core.entities.hyperliquid import (HyperliquidEntity,
                                               HyperLiquidGlobalState,
                                               HyperLiquidPosition)


@pytest.fixture
def hyperliquid_entity():
    return HyperliquidEntity()


@pytest.mark.core
def test_action_deposit(hyperliquid_entity):
    hyperliquid_entity.action_deposit(1000)
    assert hyperliquid_entity.balance == 1000
    assert hyperliquid_entity.size == 0
    assert hyperliquid_entity.leverage == 0
    assert hyperliquid_entity.internal_state.collateral == 1000


@pytest.mark.core
def test_action_withdraw(hyperliquid_entity):
    hyperliquid_entity.action_deposit(1000)
    hyperliquid_entity.action_withdraw(500)
    assert hyperliquid_entity.balance == 500
    assert hyperliquid_entity.size == 0
    assert hyperliquid_entity.leverage == 0
    assert hyperliquid_entity.internal_state.collateral == 500


@pytest.mark.core
def test_action_withdraw_insufficient_balance(hyperliquid_entity):
    hyperliquid_entity.action_deposit(1000)
    with pytest.raises(Exception):
        hyperliquid_entity.action_withdraw(1500)
    assert hyperliquid_entity.balance == 1000  # balance should not change


@pytest.mark.core
def test_action_open_position(hyperliquid_entity):
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquid_entity.action_deposit(1000)
    hyperliquid_entity.action_open_position(0.5)
    assert hyperliquid_entity.balance == 1000 - (0.5 * 3000 * hyperliquid_entity.TRADING_FEE)


@pytest.mark.core
def test_pnl(hyperliquid_entity):
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquid_entity.action_deposit(1000)
    hyperliquid_entity.action_open_position(0.5)
    assert hyperliquid_entity.pnl == 0
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3100))
    assert hyperliquid_entity.pnl == 0.5 * 100


@pytest.mark.core
def test_balance(hyperliquid_entity):
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquid_entity.action_deposit(1000)
    hyperliquid_entity.action_open_position(0.5)
    assert hyperliquid_entity.balance == 1000 - (0.5 * 3000 * hyperliquid_entity.TRADING_FEE)
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3100))
    assert hyperliquid_entity.balance == 1000 - (0.5 * 3000 * hyperliquid_entity.TRADING_FEE) + (0.5 * (3100 - 3000))


@pytest.mark.core
def test_size(hyperliquid_entity):
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquid_entity.action_deposit(1000)
    hyperliquid_entity.action_open_position(0.1)
    assert hyperliquid_entity.size == 0.1


@pytest.mark.core
def test_leverage(hyperliquid_entity):
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquid_entity.action_deposit(1000)
    hyperliquid_entity.action_open_position(0.5)
    assert hyperliquid_entity.leverage == 0.5 * 3000 / (1000 - (0.5 * 3000 * hyperliquid_entity.TRADING_FEE))

    
@pytest.mark.core
def test_check_liquidation(hyperliquid_entity):
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquid_entity.action_deposit(1000)
    hyperliquid_entity.action_open_position(-0.7)
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=5000))
    # liquidation was triggered
    assert hyperliquid_entity.size == 0

    
@pytest.mark.core
def test_leverage_change(hyperliquid_entity):
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquid_entity.action_deposit(1000)
    hyperliquid_entity.TRADING_FEE = 0.0
    hyperliquid_entity.action_open_position(1)
    assert hyperliquid_entity.leverage == 3
    hyperliquid_entity.action_withdraw(500)
    assert hyperliquid_entity.leverage == 6
    hyperliquid_entity.action_withdraw(470)
    assert hyperliquid_entity.leverage == 100
    assert hyperliquid_entity._check_liquidation()


@pytest.mark.core
def test_state_update(hyperliquid_entity):
    state = HyperLiquidGlobalState(mark_price=10, funding_rate=0.004)
    hyperliquid_entity.update_state(state)
    assert hyperliquid_entity.global_state == state


@pytest.mark.core
def test_clearing(hyperliquid_entity):
    hyperliquid_entity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquid_entity._internal_state.positions.append(HyperLiquidPosition(amount=0.5,
                                                                           entry_price=hyperliquid_entity._global_state.mark_price,
                                                                           max_leverage=50))
    hyperliquid_entity._internal_state.collateral -= np.abs(-0.5 * hyperliquid_entity.TRADING_FEE * hyperliquid_entity._global_state.mark_price)
    hyperliquid_entity._internal_state.positions.append(HyperLiquidPosition(amount=0.5,
                                                                           entry_price=hyperliquid_entity._global_state.mark_price,
                                                                           max_leverage=50))
    hyperliquid_entity._internal_state.collateral -= np.abs(-0.5 * hyperliquid_entity.TRADING_FEE * hyperliquid_entity._global_state.mark_price)
    leverage_before_clearing = hyperliquid_entity.leverage
    balance_before_clearing = hyperliquid_entity.balance
    hyperliquid_entity._clearing()
    assert hyperliquid_entity.leverage == leverage_before_clearing
    assert hyperliquid_entity.balance == balance_before_clearing
