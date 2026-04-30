"""Tests for :class:`SimplePoolEntity`."""
import pytest

from fractal.core.entities.simple.pool import (SimplePoolEntity,
                                               SimplePoolException,
                                               SimplePoolGlobalState)


@pytest.fixture
def pool() -> SimplePoolEntity:
    e = SimplePoolEntity(trading_fee=0.0)
    e.update_state(SimplePoolGlobalState(
        price=1000.0, tvl=10_000.0, volume=5_000.0, fees=0.0, liquidity=10_000.0,
    ))
    return e


# --------------------------------------------------------- account
@pytest.mark.core
def test_can_be_instantiated():
    e = SimplePoolEntity()
    assert e.balance == 0.0
    assert not e.is_position


@pytest.mark.core
def test_negative_trading_fee_rejected():
    with pytest.raises(SimplePoolException):
        SimplePoolEntity(trading_fee=-0.01)


@pytest.mark.core
def test_deposit_increases_cash(pool):
    pool.action_deposit(2000)
    assert pool.internal_state.cash == 2000
    assert pool.balance == 2000


@pytest.mark.core
def test_deposit_rejects_negative(pool):
    with pytest.raises(SimplePoolException):
        pool.action_deposit(-1)


@pytest.mark.core
def test_withdraw_basic(pool):
    pool.action_deposit(2000)
    pool.action_withdraw(500)
    assert pool.internal_state.cash == 1500


@pytest.mark.core
def test_withdraw_rejects_overdraft(pool):
    pool.action_deposit(100)
    with pytest.raises(SimplePoolException):
        pool.action_withdraw(200)


# --------------------------------------------------------- open
@pytest.mark.core
def test_open_position_mints_proportional_lp(pool):
    pool.action_deposit(2000)
    pool.action_open_position(2000)
    # tvl = 10_000, fee = 0 → deployed = 2000, share = 2000/10000 = 0.2
    # lp_minted = 0.2 * pool_liquidity (10_000) = 2000
    assert pool.is_position
    assert pool.internal_state.liquidity == pytest.approx(2000)
    assert pool.internal_state.cash == 0
    assert pool.share == pytest.approx(0.2)


@pytest.mark.core
def test_open_position_with_fee_reduces_minted_lp():
    e = SimplePoolEntity(trading_fee=0.01)
    e.update_state(SimplePoolGlobalState(price=1000, tvl=10_000, liquidity=10_000))
    e.action_deposit(1000)
    e.action_open_position(1000)
    # deployed = 1000 * 0.99 = 990, share = 990/10000 = 0.099
    # lp_minted = 0.099 * 10_000 = 990
    assert e.internal_state.liquidity == pytest.approx(990)


@pytest.mark.core
def test_open_position_rejects_when_already_open(pool):
    pool.action_deposit(5000)
    pool.action_open_position(2000)
    with pytest.raises(SimplePoolException):
        pool.action_open_position(1000)


@pytest.mark.core
def test_open_position_rejects_overdraft(pool):
    pool.action_deposit(100)
    with pytest.raises(SimplePoolException):
        pool.action_open_position(200)


@pytest.mark.core
def test_open_position_rejects_zero_or_negative(pool):
    pool.action_deposit(1000)
    with pytest.raises(SimplePoolException):
        pool.action_open_position(0)
    with pytest.raises(SimplePoolException):
        pool.action_open_position(-1)


@pytest.mark.core
def test_open_position_rejects_dead_pool():
    e = SimplePoolEntity(trading_fee=0.0)
    e.update_state(SimplePoolGlobalState(price=1000, tvl=0, liquidity=0))
    e.action_deposit(1000)
    with pytest.raises(SimplePoolException):
        e.action_open_position(500)


# --------------------------------------------------------- close
@pytest.mark.core
def test_close_position_returns_pro_rata_value(pool):
    pool.action_deposit(2000)
    pool.action_open_position(2000)  # share = 0.2, lp = 2000
    pool.action_close_position()
    # share * tvl = 0.2 * 10_000 = 2000 (no fee)
    assert pool.internal_state.liquidity == 0
    assert pool.internal_state.cash == pytest.approx(2000)
    assert not pool.is_position


@pytest.mark.core
def test_close_position_with_fee():
    e = SimplePoolEntity(trading_fee=0.01)
    e.update_state(SimplePoolGlobalState(price=1000, tvl=10_000, liquidity=10_000))
    e.action_deposit(1000)
    e.action_open_position(1000)  # lp = 990
    e.action_close_position()
    # share = 990/10000 = 0.099; proceeds = 0.099 * 10_000 * 0.99 = 980.1
    assert e.internal_state.cash == pytest.approx(980.1)


@pytest.mark.core
def test_close_position_when_flat_is_noop(pool):
    pool.action_deposit(1000)
    pool.action_close_position()
    assert pool.internal_state.cash == 1000
    assert not pool.is_position


# --------------------------------------------------------- fee accrual
@pytest.mark.core
def test_update_state_accrues_proportional_fees():
    e = SimplePoolEntity(trading_fee=0.0)
    e.update_state(SimplePoolGlobalState(price=1000, tvl=10_000, liquidity=10_000))
    e.action_deposit(1000)
    e.action_open_position(1000)
    # share = 1000/10_000 = 0.1
    e.update_state(SimplePoolGlobalState(
        price=1000, tvl=10_000, fees=200, liquidity=10_000,
    ))
    # accrued = 0.1 * 200 = 20 → cash
    assert e.internal_state.cash == pytest.approx(20)


@pytest.mark.core
def test_update_state_no_position_no_fees(pool):
    pool.action_deposit(1000)  # no LP yet
    pool.update_state(SimplePoolGlobalState(
        price=1000, tvl=10_000, fees=200, liquidity=10_000,
    ))
    assert pool.internal_state.cash == 1000  # unchanged


# --------------------------------------------------------- balance
@pytest.mark.core
def test_balance_reflects_share_of_tvl(pool):
    pool.action_deposit(2000)
    pool.action_open_position(2000)  # share = 0.2
    # balance = cash (0) + share * tvl (10_000 * 0.2 = 2000)
    assert pool.balance == pytest.approx(2000)
    # Pool TVL grows; balance scales with share.
    pool.update_state(SimplePoolGlobalState(price=1000, tvl=15_000, liquidity=10_000))
    assert pool.balance == pytest.approx(0.2 * 15_000)
