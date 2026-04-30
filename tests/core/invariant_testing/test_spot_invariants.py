"""Invariant + parity tests for spot and liquid-staking entities.

Covers ``BaseSpotEntity`` family:
* :class:`UniswapV3SpotEntity` — protocol-level spot
* :class:`StakedETHEntity` — protocol-level LST (with rebasing)
* :class:`SimpleSpotExchange` — simple OHLCV spot
* :class:`SimpleLiquidStakingToken` — simple LST

All four expose ``action_buy / sell / deposit / withdraw``, ``balance``,
``current_price``, and inherit ``BaseSpotInternalState`` (``amount`` +
``cash``). LST variants additionally expose ``staking_rate``.
"""
import pytest

from fractal.core.entities.base.liquid_staking import BaseLiquidStakingToken
from fractal.core.entities.base.spot import BaseSpotEntity, BaseSpotInternalState
from fractal.core.entities.protocols.steth import (StakedETHEntity,
                                                   StakedETHEntityException,
                                                   StakedETHGlobalState)
from fractal.core.entities.protocols.uniswap_v3_spot import (UniswapV3SpotEntity,
                                                             UniswapV3SpotEntityException,
                                                             UniswapV3SpotGlobalState)
from fractal.core.entities.simple.liquid_staking import (SimpleLiquidStakingToken,
                                                         SimpleLiquidStakingTokenException,
                                                         SimpleLiquidStakingTokenGlobalState)
from fractal.core.entities.simple.spot import (SimpleSpotExchange,
                                               SimpleSpotExchangeException,
                                               SimpleSpotExchangeGlobalState)


def _univ3_spot():
    e = UniswapV3SpotEntity()
    e.update_state(UniswapV3SpotGlobalState(price=2000.0))
    return e


def _steth():
    e = StakedETHEntity()
    e.update_state(StakedETHGlobalState(price=2000.0, staking_rate=0.0))
    return e


def _simple_spot():
    e = SimpleSpotExchange()
    e.update_state(SimpleSpotExchangeGlobalState(close=2000.0, high=2010, low=1990, open=2000, volume=0))
    return e


def _simple_lst():
    e = SimpleLiquidStakingToken()
    e.update_state(SimpleLiquidStakingTokenGlobalState(price=2000.0, staking_rate=0.0))
    return e


SPOT_FACTORIES = [
    (_univ3_spot, UniswapV3SpotEntityException),
    (_steth, StakedETHEntityException),
    (_simple_spot, SimpleSpotExchangeException),
    (_simple_lst, SimpleLiquidStakingTokenException),
]

LST_FACTORIES = [_steth, _simple_lst]


@pytest.mark.core
@pytest.mark.parametrize("factory,_exc", SPOT_FACTORIES)
def test_spot_inherits_base(factory, _exc):
    """All spot entities subclass ``BaseSpotEntity`` and have ``BaseSpotInternalState``."""
    e = factory()
    assert isinstance(e, BaseSpotEntity)
    assert isinstance(e._internal_state, BaseSpotInternalState)


@pytest.mark.core
@pytest.mark.parametrize("factory", LST_FACTORIES)
def test_lst_inherits_base_liquid_staking_token(factory):
    e = factory()
    assert isinstance(e, BaseLiquidStakingToken)


@pytest.mark.core
@pytest.mark.parametrize("factory,_exc", SPOT_FACTORIES)
def test_spot_internal_state_has_amount_and_cash(factory, _exc):
    e = factory()
    assert hasattr(e._internal_state, "amount")
    assert hasattr(e._internal_state, "cash")


@pytest.mark.core
@pytest.mark.parametrize("factory,_exc", SPOT_FACTORIES)
def test_spot_exposes_required_actions(factory, _exc):
    e = factory()
    for method in ("action_buy", "action_sell", "action_deposit",
                   "action_withdraw", "update_state"):
        assert callable(getattr(e, method))


@pytest.mark.core
@pytest.mark.parametrize("factory,_exc", SPOT_FACTORIES)
def test_spot_exposes_required_properties(factory, _exc):
    e = factory()
    assert isinstance(e.balance, (int, float))
    assert isinstance(e.current_price, (int, float))


@pytest.mark.core
@pytest.mark.parametrize("factory", LST_FACTORIES)
def test_lst_exposes_staking_rate(factory):
    e = factory()
    assert isinstance(e.staking_rate, (int, float))


@pytest.mark.core
@pytest.mark.parametrize("factory,_exc", SPOT_FACTORIES)
def test_spot_initial_state_is_clean(factory, _exc):
    e = factory()
    assert e._internal_state.amount == 0
    assert e._internal_state.cash == 0
    assert e.balance == 0


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", SPOT_FACTORIES)
def test_spot_negative_buy_rejected(factory, exc):
    e = factory()
    e.action_deposit(1000)
    with pytest.raises(exc):
        e.action_buy(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", SPOT_FACTORIES)
def test_spot_negative_sell_rejected(factory, exc):
    e = factory()
    with pytest.raises(exc):
        e.action_sell(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", SPOT_FACTORIES)
def test_spot_negative_deposit_rejected(factory, exc):
    e = factory()
    with pytest.raises(exc):
        e.action_deposit(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", SPOT_FACTORIES)
def test_spot_negative_withdraw_rejected(factory, exc):
    e = factory()
    e.action_deposit(100)
    with pytest.raises(exc):
        e.action_withdraw(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", SPOT_FACTORIES)
def test_spot_buy_overdraft_rejected(factory, exc):
    e = factory()
    e.action_deposit(100)
    with pytest.raises(exc):
        e.action_buy(200)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", SPOT_FACTORIES)
def test_spot_sell_more_than_held_rejected(factory, exc):
    e = factory()
    with pytest.raises(exc):
        e.action_sell(1)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", SPOT_FACTORIES)
def test_spot_withdraw_overdraft_rejected(factory, exc):
    e = factory()
    e.action_deposit(100)
    with pytest.raises(exc):
        e.action_withdraw(200)


@pytest.mark.core
def test_univ3_spot_buy_rejects_zero_price():
    e = UniswapV3SpotEntity()
    e.action_deposit(1000)
    with pytest.raises(UniswapV3SpotEntityException, match="price must be > 0"):
        e.action_buy(100)


@pytest.mark.core
def test_steth_buy_rejects_zero_price():
    e = StakedETHEntity()
    e.action_deposit(1000)
    with pytest.raises(StakedETHEntityException, match="price must be > 0"):
        e.action_buy(100)


@pytest.mark.core
def test_simple_spot_buy_rejects_zero_close():
    e = SimpleSpotExchange()
    e.action_deposit(1000)
    with pytest.raises(SimpleSpotExchangeException, match="close price"):
        e.action_buy(100)


@pytest.mark.core
def test_simple_lst_buy_rejects_zero_price():
    e = SimpleLiquidStakingToken()
    e.action_deposit(1000)
    with pytest.raises(SimpleLiquidStakingTokenException, match="non-positive price"):
        e.action_buy(100)


@pytest.mark.core
def test_univ3_spot_sell_rejects_zero_price():
    """Symmetric guard on sell side."""
    e = UniswapV3SpotEntity()
    e._internal_state.amount = 1.0  # mock holding
    with pytest.raises(UniswapV3SpotEntityException, match="price must be > 0"):
        e.action_sell(0.5)


@pytest.mark.core
def test_steth_sell_rejects_zero_price():
    e = StakedETHEntity()
    e._internal_state.amount = 1.0
    with pytest.raises(StakedETHEntityException, match="price must be > 0"):
        e.action_sell(0.5)


@pytest.mark.core
@pytest.mark.parametrize("factory,_exc", SPOT_FACTORIES)
def test_spot_amount_and_cash_non_negative_through_lifecycle(factory, _exc):
    e = factory()
    e.action_deposit(1000)
    assert e._internal_state.amount >= 0 and e._internal_state.cash >= 0
    e.action_buy(500)
    assert e._internal_state.amount >= 0 and e._internal_state.cash >= 0
    e.action_sell(e._internal_state.amount / 2)
    assert e._internal_state.amount >= 0 and e._internal_state.cash >= 0
    e.action_withdraw(100)
    assert e._internal_state.amount >= 0 and e._internal_state.cash >= 0


@pytest.mark.core
@pytest.mark.parametrize("factory,_exc", SPOT_FACTORIES)
def test_spot_balance_equals_amount_times_price_plus_cash(factory, _exc):
    """Conservation: ``balance == amount × current_price + cash``."""
    e = factory()
    e.action_deposit(1000)
    e.action_buy(600)
    expected = e._internal_state.amount * e.current_price + e._internal_state.cash
    assert e.balance == pytest.approx(expected)


@pytest.mark.core
@pytest.mark.parametrize("factory,_exc", SPOT_FACTORIES)
def test_spot_round_trip_loses_two_trading_fees(factory, _exc):
    """Buy then sell same product → cash less than initial deposit by ~2× fee."""
    e = factory()
    e.action_deposit(1000)
    e.action_buy(1000)
    e.action_sell(e._internal_state.amount)
    # After full round-trip, all money is in cash. Fee charged on each side.
    assert e._internal_state.cash < 1000
    assert e._internal_state.cash > 980  # at most ~2% loss for default fees


@pytest.mark.core
@pytest.mark.parametrize("factory", LST_FACTORIES)
def test_lst_amount_rebases_on_positive_rate(factory):
    e = factory()
    e.action_deposit(2000)
    e.action_buy(2000)
    amount_before = e._internal_state.amount
    # Apply a 1% per-step rate
    if isinstance(e, StakedETHEntity):
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.01))
    else:
        e.update_state(SimpleLiquidStakingTokenGlobalState(price=2000, staking_rate=0.01))
    assert e._internal_state.amount == pytest.approx(amount_before * 1.01)


@pytest.mark.core
def test_lst_rate_below_minus_one_rejected_steth():
    e = StakedETHEntity()
    with pytest.raises(StakedETHEntityException, match="staking_rate must be >= -1"):
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=-2.0))


@pytest.mark.core
def test_lst_rate_below_minus_one_rejected_simple():
    e = SimpleLiquidStakingToken()
    with pytest.raises(SimpleLiquidStakingTokenException, match="staking_rate must be >= -1"):
        e.update_state(SimpleLiquidStakingTokenGlobalState(price=2000, staking_rate=-2.0))


@pytest.mark.core
def test_univ3_spot_zero_fee_round_trip_preserves_notional():
    """With trading_fee=0, buy → sell at same price → cash unchanged."""
    e = UniswapV3SpotEntity(trading_fee=0.0)
    e.update_state(UniswapV3SpotGlobalState(price=2000))
    e.action_deposit(1000)
    e.action_buy(1000)
    e.action_sell(e._internal_state.amount)
    assert e._internal_state.cash == pytest.approx(1000)


@pytest.mark.core
def test_steth_zero_fee_round_trip_preserves_notional():
    e = StakedETHEntity(trading_fee=0.0)
    e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))
    e.action_deposit(1000)
    e.action_buy(1000)
    e.action_sell(e._internal_state.amount)
    assert e._internal_state.cash == pytest.approx(1000)


@pytest.mark.core
def test_simple_spot_zero_fee_round_trip_preserves_notional():
    e = SimpleSpotExchange(trading_fee=0.0)
    e.update_state(SimpleSpotExchangeGlobalState(close=2000, high=0, low=0, open=0, volume=0))
    e.action_deposit(1000)
    e.action_buy(1000)
    e.action_sell(e._internal_state.amount)
    assert e._internal_state.cash == pytest.approx(1000)


@pytest.mark.core
def test_simple_lst_zero_fee_round_trip_preserves_notional():
    e = SimpleLiquidStakingToken(trading_fee=0.0)
    e.update_state(SimpleLiquidStakingTokenGlobalState(price=2000, staking_rate=0.0))
    e.action_deposit(1000)
    e.action_buy(1000)
    e.action_sell(e._internal_state.amount)
    assert e._internal_state.cash == pytest.approx(1000)


@pytest.mark.core
def test_lst_is_a_spot_entity():
    """LSTs are spot-tradeable; ``BaseLiquidStakingToken`` extends ``BaseSpotEntity``."""
    assert issubclass(BaseLiquidStakingToken, BaseSpotEntity)
    assert issubclass(StakedETHEntity, BaseSpotEntity)
    assert issubclass(SimpleLiquidStakingToken, BaseSpotEntity)
