"""Invariants for the cross-entity product-transfer primitives.

``BaseSpotEntity.action_inject_product`` and ``action_remove_product``
let strategies move *product* tokens (e.g. ETH) into and out of a spot
entity WITHOUT a swap or trading fee. The pair is the dual of
``action_deposit`` / ``action_withdraw`` (which move *notional* cash).

Together they let strategies compose multi-entity flows like
"borrow X collateral from Aave, hold the borrowed product in a spot
entity, then re-deposit elsewhere" without losing money to phantom
fees on cross-entity moves.

These tests pin down:
* validation parity (negative / overdraft / etc.)
* state-machine effects (only ``amount`` changes, ``cash`` untouched)
* conservation when paired with cash-side primitives across entities.
"""
import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.steth import StakedETHEntity, StakedETHGlobalState
from fractal.core.entities.protocols.uniswap_v3_spot import UniswapV3SpotEntity, UniswapV3SpotGlobalState
from fractal.core.entities.simple.liquid_staking import SimpleLiquidStakingToken, SimpleLiquidStakingTokenGlobalState
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeGlobalState


def _univ3_spot():
    e = UniswapV3SpotEntity(trading_fee=0.0)
    e.update_state(UniswapV3SpotGlobalState(price=2000.0))
    return e


def _steth():
    e = StakedETHEntity(trading_fee=0.0)
    e.update_state(StakedETHGlobalState(price=2000.0, staking_rate=0.0))
    return e


def _simple_spot():
    e = SimpleSpotExchange(trading_fee=0.0)
    e.update_state(SimpleSpotExchangeGlobalState(close=2000, high=2000, low=2000, open=2000, volume=0))
    return e


def _simple_lst():
    e = SimpleLiquidStakingToken(trading_fee=0.0)
    e.update_state(SimpleLiquidStakingTokenGlobalState(price=2000.0, staking_rate=0.0))
    return e


SPOT_FACTORIES = [_univ3_spot, _steth, _simple_spot, _simple_lst]


@pytest.mark.core
@pytest.mark.parametrize("factory", SPOT_FACTORIES)
def test_inject_product_increases_amount_only(factory):
    e = factory()
    e.action_inject_product(2.5)
    assert e._internal_state.amount == pytest.approx(2.5)
    assert e._internal_state.cash == 0  # cash NOT touched


@pytest.mark.core
@pytest.mark.parametrize("factory", SPOT_FACTORIES)
def test_remove_product_decreases_amount_only(factory):
    e = factory()
    e.action_inject_product(5.0)
    e.action_remove_product(2.0)
    assert e._internal_state.amount == pytest.approx(3.0)
    assert e._internal_state.cash == 0


@pytest.mark.core
@pytest.mark.parametrize("factory", SPOT_FACTORIES)
def test_inject_remove_round_trip_is_identity(factory):
    e = factory()
    e.action_inject_product(7.5)
    e.action_remove_product(7.5)
    assert e._internal_state.amount == 0
    assert e._internal_state.cash == 0


@pytest.mark.core
@pytest.mark.parametrize("factory", SPOT_FACTORIES)
def test_inject_product_rejects_negative(factory):
    e = factory()
    with pytest.raises(EntityException, match="inject_product amount must be >= 0"):
        e.action_inject_product(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory", SPOT_FACTORIES)
def test_remove_product_rejects_negative(factory):
    e = factory()
    with pytest.raises(EntityException, match="remove_product amount must be >= 0"):
        e.action_remove_product(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory", SPOT_FACTORIES)
def test_remove_product_rejects_overdraft(factory):
    e = factory()
    e.action_inject_product(1.0)
    with pytest.raises(EntityException, match="remove_product exceeds holding"):
        e.action_remove_product(2.0)


@pytest.mark.core
@pytest.mark.parametrize("factory", SPOT_FACTORIES)
def test_inject_zero_is_noop(factory):
    e = factory()
    e.action_inject_product(0.0)
    assert e._internal_state.amount == 0


@pytest.mark.core
@pytest.mark.parametrize("factory", SPOT_FACTORIES)
def test_remove_zero_is_noop(factory):
    e = factory()
    e.action_inject_product(1.0)
    e.action_remove_product(0.0)
    assert e._internal_state.amount == 1.0


@pytest.mark.core
def test_inject_no_fee_vs_buy_with_fee():
    """``action_inject_product`` must NOT charge a trading fee, unlike ``action_buy``."""
    no_fee = UniswapV3SpotEntity(trading_fee=0.003)  # has a fee, but inject bypasses it
    no_fee.update_state(UniswapV3SpotGlobalState(price=2000))
    no_fee.action_inject_product(0.5)
    assert no_fee._internal_state.amount == 0.5  # full amount, no fee discount

    with_fee = UniswapV3SpotEntity(trading_fee=0.003)
    with_fee.update_state(UniswapV3SpotGlobalState(price=2000))
    with_fee.action_deposit(1000)
    with_fee.action_buy(1000)
    # Buy gives less due to fee
    assert with_fee._internal_state.amount < 0.5


@pytest.mark.core
def test_remove_no_fee_vs_sell_with_fee():
    """``action_remove_product`` must NOT charge a fee or generate cash, unlike ``action_sell``."""
    no_fee = UniswapV3SpotEntity(trading_fee=0.003)
    no_fee.update_state(UniswapV3SpotGlobalState(price=2000))
    no_fee.action_inject_product(0.5)
    no_fee.action_remove_product(0.5)
    assert no_fee._internal_state.amount == 0
    assert no_fee._internal_state.cash == 0  # NO cash generated

    sell = UniswapV3SpotEntity(trading_fee=0.003)
    sell.update_state(UniswapV3SpotGlobalState(price=2000))
    sell.action_inject_product(0.5)
    sell.action_sell(0.5)
    # Sell generates cash (minus fee)
    assert sell._internal_state.amount == 0
    assert sell._internal_state.cash > 0
    assert sell._internal_state.cash < 0.5 * 2000  # less than gross due to fee


@pytest.mark.core
def test_steth_inject_remove_uses_base_exception():
    """Subclass-specific exception classes don't override the base default."""
    e = StakedETHEntity()
    with pytest.raises(EntityException, match="inject_product"):
        e.action_inject_product(-1)


@pytest.mark.core
def test_univ3_spot_inject_remove_uses_base_exception():
    e = UniswapV3SpotEntity()
    with pytest.raises(EntityException, match="remove_product"):
        e.action_remove_product(1.0)
