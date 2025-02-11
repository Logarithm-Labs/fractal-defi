import numpy as np
import pytest

from fractal.core.entities.hyperliquid import HyperliquidEntity, HyperLiquidGlobalState, HyperLiquidPosition


@pytest.fixture
def HyperliquidEntity():
    return HyperliquidEntity()


def test_action_deposit(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.action_deposit(1000)
    assert hyperliquidEntity.balance == 1000
    assert hyperliquidEntity.size == 0
    assert hyperliquidEntity.leverage == 0
    assert hyperliquidEntity.internal_state.collateral == 1000


def test_action_withdraw(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.action_deposit(1000)
    hyperliquidEntity.action_withdraw(500)
    assert hyperliquidEntity.balance == 500
    assert hyperliquidEntity.size == 0
    assert hyperliquidEntity.leverage == 0
    assert hyperliquidEntity.internal_state.collateral == 500


def test_action_withdraw_insufficient_balance(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.action_deposit(1000)
    with pytest.raises(Exception):
        hyperliquidEntity.action_withdraw(1500)
    assert hyperliquidEntity.balance == 1000                # balance should not change


def test_action_open_position(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquidEntity.action_deposit(1000)
    hyperliquidEntity.action_open_position(0.5)
    assert hyperliquidEntity.balance == 1000 - (0.5 * 3000 * hyperliquidEntity.TRADING_FEE)


def test_pnl(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquidEntity.action_deposit(1000)
    hyperliquidEntity.action_open_position(0.5)
    assert hyperliquidEntity.pnl == 0
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3100))
    assert hyperliquidEntity.pnl == 0.5 * 100


def test_balance(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquidEntity.action_deposit(1000)
    hyperliquidEntity.action_open_position(0.5)
    assert hyperliquidEntity.balance == 1000 - (0.5 * 3000 * hyperliquidEntity.TRADING_FEE)
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3100))
    assert hyperliquidEntity.balance == 1000 - (0.5 * 3000 * hyperliquidEntity.TRADING_FEE) + (0.5 * (3100 - 3000))


def test_size(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquidEntity.action_deposit(1000)
    hyperliquidEntity.action_open_position(0.5)
    assert hyperliquidEntity.size == 0.5


def test_leverage(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquidEntity.action_deposit(1000)
    hyperliquidEntity.action_open_position(0.5)
    assert hyperliquidEntity.leverage == 0.5 * 3000 / (1000 - (0.5 * 3000 * hyperliquidEntity.TRADING_FEE))


def test_check_liquidation(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquidEntity.action_deposit(1000)
    hyperliquidEntity.action_open_position(-0.7, max_leverage=50)
    assert not hyperliquidEntity._check_liquidation()
    hyperliquidEntity.action_withdraw(500)
    assert not hyperliquidEntity._check_liquidation()
    hyperliquidEntity.action_withdraw(400)
    assert not hyperliquidEntity._check_liquidation()
    hyperliquidEntity.action_withdraw(90)


def test_leverage_change(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquidEntity.action_deposit(1000)
    hyperliquidEntity.TRADING_FEE = 0.0
    hyperliquidEntity.action_open_position(1)               # 3x leverage
    assert hyperliquidEntity.leverage == 3
    hyperliquidEntity.action_withdraw(500)
    assert hyperliquidEntity.leverage == 6
    hyperliquidEntity.action_withdraw(470)
    assert hyperliquidEntity.leverage == 100
    assert hyperliquidEntity._check_liquidation()


def test_state_update(hyperliquidEntity: HyperliquidEntity):
    state = HyperLiquidGlobalState(mark_price=10, funding_rate=0.004)
    hyperliquidEntity.update_state(state)
    assert hyperliquidEntity.global_state == state


def test_clearing(hyperliquidEntity: HyperliquidEntity):
    hyperliquidEntity.update_state(HyperLiquidGlobalState(mark_price=3000))
    hyperliquidEntity._internal_state.positions.append(HyperLiquidPosition(amount=0.5,
                                                                           entry_price=hyperliquidEntity._global_state.mark_price,
                                                                           max_leverage=50))
    hyperliquidEntity._internal_state.collateral -= np.abs(-0.5 * hyperliquidEntity.TRADING_FEE * hyperliquidEntity._global_state.mark_price)
    hyperliquidEntity._internal_state.positions.append(HyperLiquidPosition(amount=0.5,
                                                                           entry_price=hyperliquidEntity._global_state.mark_price,
                                                                           max_leverage=50))
    hyperliquidEntity._internal_state.collateral -= np.abs(-0.5 * hyperliquidEntity.TRADING_FEE * hyperliquidEntity._global_state.mark_price)
    leverage_before_clearing = hyperliquidEntity.leverage
    balance_before_clearing = hyperliquidEntity.balance
    hyperliquidEntity._clearing()
    assert hyperliquidEntity.leverage == leverage_before_clearing
    assert hyperliquidEntity.balance == balance_before_clearing
