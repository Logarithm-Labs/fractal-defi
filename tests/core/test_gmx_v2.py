import numpy as np
import pytest

from fractal.core.entities import GMXV2Entity, GMXV2GlobalState, GMXV2Position


@pytest.fixture
def gmx_v2_entity():
    return GMXV2Entity()


@pytest.mark.core
def test_action_deposit(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.action_deposit(1000)
    assert gmx_v2_entity.balance == 1000
    assert gmx_v2_entity.size == 0
    assert gmx_v2_entity.leverage == 0
    assert gmx_v2_entity.internal_state.collateral == 1000


@pytest.mark.core
def test_action_withdraw(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.action_deposit(1000)
    gmx_v2_entity.action_withdraw(500)
    assert gmx_v2_entity.balance == 500
    assert gmx_v2_entity.size == 0
    assert gmx_v2_entity.leverage == 0
    assert gmx_v2_entity.internal_state.collateral == 500


@pytest.mark.core
def test_action_withdraw_insufficient_balance(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.action_deposit(1000)
    with pytest.raises(Exception):
        gmx_v2_entity.action_withdraw(1500)
    assert gmx_v2_entity.balance == 1000 # balance should not change


@pytest.mark.core
def test_action_withdraw_exceeds_max_withdrawal_limit(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3000))
    gmx_v2_entity.action_deposit(1000)
    gmx_v2_entity.action_open_position(1)
    gmx_v2_entity.update_state(GMXV2GlobalState(price=2900))
    with pytest.raises(Exception):
        gmx_v2_entity.action_withdraw(880)
    gmx_v2_entity.action_withdraw(860)
    assert gmx_v2_entity.balance == 37.0


@pytest.mark.core
def test_action_open_position(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3000))
    gmx_v2_entity.action_deposit(1000)
    gmx_v2_entity.action_open_position(0.5)
    assert gmx_v2_entity.balance == 1000 - (0.5 * 3000 * gmx_v2_entity.TRADING_FEE)


@pytest.mark.core
def test_pnl(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3000))
    gmx_v2_entity.action_deposit(1000)
    gmx_v2_entity.action_open_position(0.5)
    assert gmx_v2_entity.pnl == 0
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3100))
    assert gmx_v2_entity.pnl == 0.5 * 100


@pytest.mark.core
def test_balance(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3000))
    gmx_v2_entity.action_deposit(1000)
    gmx_v2_entity.action_open_position(0.5)
    assert gmx_v2_entity.balance == 1000 - (0.5 * 3000 * gmx_v2_entity.TRADING_FEE)
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3100))
    assert gmx_v2_entity.balance == 1000 - (0.5 * 3000 * gmx_v2_entity.TRADING_FEE) + (0.5 * (3100 - 3000))


@pytest.mark.core
def test_size(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3000))
    gmx_v2_entity.action_deposit(1000)
    gmx_v2_entity.action_open_position(0.5)
    assert gmx_v2_entity.size == 0.5


@pytest.mark.core
def test_leverage(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3000))
    gmx_v2_entity.action_deposit(1000)
    gmx_v2_entity.action_open_position(0.5)
    assert gmx_v2_entity.leverage == 0.5 * 3000 / (1000 - (0.5 * 3000 * gmx_v2_entity.TRADING_FEE))


@pytest.mark.core
def test_check_liquidation(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3000))
    gmx_v2_entity.action_deposit(1000)
    gmx_v2_entity.action_open_position(0.5)
    assert not gmx_v2_entity._check_liquidation()
    gmx_v2_entity.update_state(GMXV2GlobalState(price=1000, funding_rate_short=0.01, borrowing_rate_short=0.02))
    gmx_v2_entity.action_open_position(2)
    assert gmx_v2_entity._check_liquidation()


@pytest.mark.core
def test_leverage_change(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3000))
    gmx_v2_entity.action_deposit(1000)
    gmx_v2_entity.TRADING_FEE = 0.0
    gmx_v2_entity.action_open_position(1) # 3x leverage
    assert gmx_v2_entity.leverage == 3
    gmx_v2_entity.action_withdraw(500)
    assert gmx_v2_entity.leverage == 6
    gmx_v2_entity.action_withdraw(470)
    assert gmx_v2_entity.leverage == 100
    assert gmx_v2_entity._check_liquidation()


@pytest.mark.core
def test_state_update(gmx_v2_entity: GMXV2Entity):
    state = GMXV2GlobalState(price=10, funding_rate_short=0.001, borrowing_rate_short=0.002)
    gmx_v2_entity.update_state(state)
    assert gmx_v2_entity.global_state == state


@pytest.mark.core
def test_clearing(gmx_v2_entity: GMXV2Entity):
    gmx_v2_entity.update_state(GMXV2GlobalState(price=3000))
    gmx_v2_entity._internal_state.positions.append(GMXV2Position(amount=0.5, entry_price=gmx_v2_entity._global_state.price))
    gmx_v2_entity._internal_state.collateral -= np.abs(0.5 * gmx_v2_entity.TRADING_FEE * gmx_v2_entity._global_state.price)
    gmx_v2_entity._internal_state.positions.append(GMXV2Position(amount=0.5, entry_price=gmx_v2_entity._global_state.price))
    gmx_v2_entity._internal_state.collateral -= np.abs(0.5 * gmx_v2_entity.TRADING_FEE * gmx_v2_entity._global_state.price)
    leverage_before_clearing = gmx_v2_entity.leverage
    balance_before_clearing = gmx_v2_entity.balance
    gmx_v2_entity._clearing()
    assert gmx_v2_entity.leverage == leverage_before_clearing
    assert gmx_v2_entity.balance == balance_before_clearing
