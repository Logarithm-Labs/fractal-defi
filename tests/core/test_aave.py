import pytest

from fractal.core.entities.aave import (AaveEntity, AaveGlobalState,
                                        EntityException)


@pytest.fixture
def aave_entity():
    return AaveEntity()

def test_action_redeem(aave_entity: AaveEntity):
    aave_entity.internal_state.borrowed = 1000
    aave_entity.action_redeem(500)
    assert aave_entity.internal_state.borrowed == 500

def test_action_redeem_exceeds_borrowed_amount(aave_entity: AaveEntity):
    aave_entity.internal_state.borrowed = 1000
    with pytest.raises(EntityException):
        aave_entity.action_redeem(1500)
    assert aave_entity.internal_state.borrowed == 1000

def test_action_borrow(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.global_state.product_price = 10
    aave_entity.global_state.notional_price = 100
    aave_entity.max_ltv = 0.8
    aave_entity.action_borrow(500)
    assert aave_entity.internal_state.borrowed == 500

def test_action_borrow_exceeds_max_ltv(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.global_state.product_price = 10
    aave_entity.global_state.notional_price = 1
    aave_entity.max_ltv = 0.8
    with pytest.raises(EntityException):
        aave_entity.action_borrow(80.1)

def test_action_deposit(aave_entity: AaveEntity):
    aave_entity.action_deposit(1000)
    assert aave_entity.internal_state.collateral == 1000

def test_action_withdraw(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 500
    aave_entity.global_state.product_price = 10
    aave_entity.global_state.notional_price = 100
    aave_entity.max_ltv = 0.8
    aave_entity.action_withdraw(500)
    assert aave_entity.internal_state.collateral == 500

def test_action_withdraw_exceeds_max_ltv(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 79.9
    aave_entity.global_state.product_price = 10
    aave_entity.global_state.notional_price = 1
    aave_entity.max_ltv = 0.8
    with pytest.raises(EntityException):
        aave_entity.action_withdraw(2)

def test_balance(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 50
    aave_entity.global_state.product_price = 10
    aave_entity.global_state.notional_price = 1
    assert aave_entity.balance == 500

def test_ltv(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 50
    aave_entity.global_state.product_price = 10
    aave_entity.global_state.notional_price = 1
    assert aave_entity.ltv == 0.5

def test_check_liquidation(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 84
    aave_entity.global_state.product_price = 11
    aave_entity.global_state.notional_price = 1
    aave_entity.liq_threshold = 0.85
    aave_entity.check_liquidation()
    assert aave_entity.internal_state.collateral == 0
    assert aave_entity.internal_state.borrowed == 0

def test_update_state(aave_entity: AaveEntity):
    state = AaveGlobalState(
        notional_price=1,
        product_price=10,
        lending_rate=0.01,
        borrowing_rate=0.02
    )
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 50
    aave_entity.update_state(state)
    assert aave_entity.internal_state.collateral == 1000 * (1 + 0.01)
    assert aave_entity.internal_state.borrowed == 50 * (1 + 0.02)