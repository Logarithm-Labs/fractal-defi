"""Tests for :class:`SimpleLiquidStakingToken`."""
import pytest

from fractal.core.entities.base.liquid_staking import BaseLiquidStakingToken
from fractal.core.entities.base.spot import BaseSpotEntity
from fractal.core.entities.simple.liquid_staking import (
    SimpleLiquidStakingToken, SimpleLiquidStakingTokenException,
    SimpleLiquidStakingTokenGlobalState)


@pytest.fixture
def lst() -> SimpleLiquidStakingToken:
    e = SimpleLiquidStakingToken(trading_fee=0.0)
    e.update_state(SimpleLiquidStakingTokenGlobalState(price=1000.0, staking_rate=0.0))
    return e


# --------------------------------------------------------- typing chain
@pytest.mark.core
def test_inherits_lst_and_spot_bases():
    assert issubclass(SimpleLiquidStakingToken, BaseLiquidStakingToken)
    assert issubclass(SimpleLiquidStakingToken, BaseSpotEntity)


@pytest.mark.core
def test_can_be_instantiated():
    e = SimpleLiquidStakingToken()
    assert e.balance == 0.0
    assert e.staking_rate == 0.0


@pytest.mark.core
def test_negative_trading_fee_rejected():
    with pytest.raises(SimpleLiquidStakingTokenException):
        SimpleLiquidStakingToken(trading_fee=-0.01)


# --------------------------------------------------------- account
@pytest.mark.core
def test_deposit_increases_cash(lst):
    lst.action_deposit(1000)
    assert lst.internal_state.cash == 1000


@pytest.mark.core
def test_deposit_rejects_negative(lst):
    with pytest.raises(SimpleLiquidStakingTokenException):
        lst.action_deposit(-1)


@pytest.mark.core
def test_withdraw_basic(lst):
    lst.action_deposit(1000)
    lst.action_withdraw(400)
    assert lst.internal_state.cash == 600


@pytest.mark.core
def test_withdraw_rejects_overdraft(lst):
    lst.action_deposit(100)
    with pytest.raises(SimpleLiquidStakingTokenException):
        lst.action_withdraw(200)


# --------------------------------------------------------- buy/sell
@pytest.mark.core
def test_buy_consumes_notional_and_credits_product(lst):
    lst.action_deposit(2000)
    lst.action_buy(amount_in_notional=1000)  # 1.0 LST at price 1000, 0 fee
    assert lst.internal_state.amount == pytest.approx(1.0)
    assert lst.internal_state.cash == 1000


@pytest.mark.core
def test_buy_with_fee_reduces_received_product():
    e = SimpleLiquidStakingToken(trading_fee=0.01)
    e.update_state(SimpleLiquidStakingTokenGlobalState(price=1000, staking_rate=0))
    e.action_deposit(1000)
    e.action_buy(amount_in_notional=1000)
    # product = 1000 * (1 - 0.01) / 1000 = 0.99
    assert e.internal_state.amount == pytest.approx(0.99)


@pytest.mark.core
def test_buy_rejects_overdraft(lst):
    lst.action_deposit(100)
    with pytest.raises(SimpleLiquidStakingTokenException):
        lst.action_buy(amount_in_notional=200)


@pytest.mark.core
def test_buy_rejects_non_positive_price():
    e = SimpleLiquidStakingToken(trading_fee=0.0)
    e.update_state(SimpleLiquidStakingTokenGlobalState(price=0))
    e.action_deposit(1000)
    with pytest.raises(SimpleLiquidStakingTokenException):
        e.action_buy(amount_in_notional=100)


@pytest.mark.core
def test_sell_consumes_product_and_credits_notional(lst):
    lst.action_deposit(2000)
    lst.action_buy(amount_in_notional=1000)  # got 1 LST
    lst.action_sell(amount_in_product=0.5)
    assert lst.internal_state.amount == pytest.approx(0.5)
    assert lst.internal_state.cash == pytest.approx(1500)


@pytest.mark.core
def test_sell_rejects_more_than_held(lst):
    lst.action_deposit(1000)
    lst.action_buy(amount_in_notional=500)  # 0.5 LST
    with pytest.raises(SimpleLiquidStakingTokenException):
        lst.action_sell(amount_in_product=1.0)


# --------------------------------------------------------- rebase semantics
@pytest.mark.core
def test_update_state_rebases_amount_at_positive_rate(lst):
    lst.action_deposit(1000)
    lst.action_buy(amount_in_notional=1000)  # 1 LST
    lst.update_state(SimpleLiquidStakingTokenGlobalState(price=1000, staking_rate=0.05))
    # amount grows by 5% → 1.05
    assert lst.internal_state.amount == pytest.approx(1.05)


@pytest.mark.core
def test_update_state_rebases_amount_at_negative_rate_models_slashing(lst):
    """Negative staking_rate models slashing/downtime."""
    lst.action_deposit(1000)
    lst.action_buy(amount_in_notional=1000)
    lst.update_state(SimpleLiquidStakingTokenGlobalState(price=1000, staking_rate=-0.1))
    assert lst.internal_state.amount == pytest.approx(0.9)


@pytest.mark.core
def test_zero_rate_keeps_amount_constant(lst):
    lst.action_deposit(1000)
    lst.action_buy(amount_in_notional=1000)
    lst.update_state(SimpleLiquidStakingTokenGlobalState(price=1000, staking_rate=0.0))
    assert lst.internal_state.amount == pytest.approx(1.0)


@pytest.mark.core
def test_rebase_compounds_across_steps(lst):
    lst.action_deposit(1000)
    lst.action_buy(amount_in_notional=1000)  # 1 LST
    for _ in range(3):
        lst.update_state(
            SimpleLiquidStakingTokenGlobalState(price=1000, staking_rate=0.01)
        )
    # 1 * 1.01^3
    assert lst.internal_state.amount == pytest.approx(1.01 ** 3)


# --------------------------------------------------------- contract surface
@pytest.mark.core
def test_current_price_and_staking_rate_expose_global_state(lst):
    assert lst.current_price == 1000.0
    assert lst.staking_rate == 0.0
    lst.update_state(SimpleLiquidStakingTokenGlobalState(price=1500, staking_rate=0.02))
    assert lst.current_price == 1500.0
    assert lst.staking_rate == 0.02


@pytest.mark.core
def test_balance_marks_amount_to_price(lst):
    lst.action_deposit(2000)
    lst.action_buy(amount_in_notional=1000)  # 1 LST + 1000 cash
    assert lst.balance == pytest.approx(1 * 1000 + 1000)
    lst.update_state(SimpleLiquidStakingTokenGlobalState(price=1500, staking_rate=0.0))
    # price moved, amount unchanged → balance = 1 * 1500 + 1000
    assert lst.balance == pytest.approx(2500)


@pytest.mark.core
def test_polymorphic_lst_typing_works():
    """Strategies typing ``lst: BaseLiquidStakingToken`` see the contract."""
    lst: BaseLiquidStakingToken = SimpleLiquidStakingToken()
    assert hasattr(lst, "staking_rate")
    assert hasattr(lst, "current_price")
    assert hasattr(lst, "action_buy")
    assert hasattr(lst, "action_sell")
