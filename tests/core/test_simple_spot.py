"""Tests for :class:`SimpleSpotExchange` and the back-compat shim."""
import warnings

import pytest

from fractal.core.entities.simple.spot import (SimpleSpotExchange,
                                               SimpleSpotExchangeException,
                                               SimpleSpotExchangeGlobalState)
from fractal.core.entities.single_spot_exchange import SingleSpotExchange


@pytest.fixture
def spot() -> SimpleSpotExchange:
    e = SimpleSpotExchange(trading_fee=0.0)
    e.update_state(
        SimpleSpotExchangeGlobalState(open=1000, high=1010, low=990, close=1000)
    )
    return e


# ----------------------------------------------------------- account
@pytest.mark.core
def test_can_be_instantiated():
    e = SimpleSpotExchange()
    assert e.balance == 0.0


@pytest.mark.core
def test_negative_trading_fee_rejected():
    with pytest.raises(SimpleSpotExchangeException):
        SimpleSpotExchange(trading_fee=-0.001)


@pytest.mark.core
def test_deposit_increases_cash(spot):
    spot.action_deposit(1000)
    assert spot.internal_state.cash == 1000
    assert spot.balance == 1000


@pytest.mark.core
def test_deposit_rejects_negative(spot):
    with pytest.raises(SimpleSpotExchangeException):
        spot.action_deposit(-1)


@pytest.mark.core
def test_withdraw_basic(spot):
    spot.action_deposit(1000)
    spot.action_withdraw(400)
    assert spot.internal_state.cash == 600


@pytest.mark.core
def test_withdraw_rejects_negative(spot):
    with pytest.raises(SimpleSpotExchangeException):
        spot.action_withdraw(-1)


@pytest.mark.core
def test_withdraw_rejects_overdraft(spot):
    spot.action_deposit(100)
    with pytest.raises(SimpleSpotExchangeException):
        spot.action_withdraw(200)


# -------------------------------------------------- buy semantics (notional)
@pytest.mark.core
def test_buy_consumes_notional_and_credits_product(spot):
    spot.action_deposit(1000)
    spot.action_buy(amount_in_notional=400)
    assert spot.internal_state.cash == 600
    assert spot.internal_state.amount == pytest.approx(400 / 1000)


@pytest.mark.core
def test_buy_with_fee_reduces_received_product():
    e = SimpleSpotExchange(trading_fee=0.01)
    e.update_state(SimpleSpotExchangeGlobalState(close=1000))
    e.action_deposit(1000)
    e.action_buy(amount_in_notional=1000)
    # Cash fully spent; product = 1000 * (1 - 0.01) / 1000 = 0.99
    assert e.internal_state.cash == 0
    assert e.internal_state.amount == pytest.approx(0.99)


@pytest.mark.core
def test_buy_rejects_negative(spot):
    spot.action_deposit(1000)
    with pytest.raises(SimpleSpotExchangeException):
        spot.action_buy(amount_in_notional=-1)


@pytest.mark.core
def test_buy_rejects_overdraft(spot):
    spot.action_deposit(100)
    with pytest.raises(SimpleSpotExchangeException):
        spot.action_buy(amount_in_notional=200)


@pytest.mark.core
def test_buy_rejects_non_positive_close():
    e = SimpleSpotExchange(trading_fee=0.0)
    e.update_state(SimpleSpotExchangeGlobalState(close=0))
    e.action_deposit(1000)
    with pytest.raises(SimpleSpotExchangeException):
        e.action_buy(amount_in_notional=100)


# -------------------------------------------------- sell semantics (product)
@pytest.mark.core
def test_sell_consumes_product_and_credits_notional(spot):
    spot.action_deposit(2000)
    spot.action_buy(amount_in_notional=1000)  # bought 1.0 product
    assert spot.internal_state.amount == pytest.approx(1.0)
    spot.action_sell(amount_in_product=0.5)
    assert spot.internal_state.amount == pytest.approx(0.5)
    # No fee → 0.5 * 1000 = 500 back
    assert spot.internal_state.cash == pytest.approx(1500)


@pytest.mark.core
def test_sell_with_fee_reduces_received_notional():
    e = SimpleSpotExchange(trading_fee=0.01)
    e.update_state(SimpleSpotExchangeGlobalState(close=1000))
    e._internal_state.amount = 1.0  # seed product position directly
    e.action_sell(amount_in_product=1.0)
    # Notional received = 1 * 1000 * 0.99 = 990
    assert e.internal_state.cash == pytest.approx(990)
    assert e.internal_state.amount == 0


@pytest.mark.core
def test_sell_rejects_negative(spot):
    with pytest.raises(SimpleSpotExchangeException):
        spot.action_sell(amount_in_product=-1)


@pytest.mark.core
def test_sell_rejects_more_than_held(spot):
    spot.action_deposit(1000)
    spot.action_buy(amount_in_notional=500)  # 0.5 product
    with pytest.raises(SimpleSpotExchangeException):
        spot.action_sell(amount_in_product=1.0)


# -------------------------------------------------- balance and update
@pytest.mark.core
def test_balance_marks_to_close_price(spot):
    spot.action_deposit(2000)
    spot.action_buy(amount_in_notional=1000)
    assert spot.balance == pytest.approx(2000)  # 1.0 product * 1000 + 1000 cash
    spot.update_state(SimpleSpotExchangeGlobalState(close=1500))
    assert spot.balance == pytest.approx(1.0 * 1500 + 1000)


# -------------------------------------------------- back-compat shim
@pytest.mark.core
def test_single_spot_exchange_alias_emits_deprecation_warning_and_works():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e = SingleSpotExchange(trading_fee=0.0)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    e.update_state(SimpleSpotExchangeGlobalState(close=100))
    e.action_deposit(1000)
    e.action_buy(amount_in_notional=500)
    assert e.internal_state.amount == pytest.approx(5.0)


@pytest.mark.core
def test_single_spot_exchange_states_alias_to_new_classes():
    """Old type names still resolve, now pointing to the new classes."""
    from fractal.core.entities.single_spot_exchange import (
        SingleSpotExchangeGlobalState,
        SingleSpotExchangeInternalState,
    )
    assert SingleSpotExchangeGlobalState is SimpleSpotExchangeGlobalState
    from fractal.core.entities.simple.spot import \
        SimpleSpotExchangeInternalState as NewInternal
    assert SingleSpotExchangeInternalState is NewInternal
