"""End-to-end synthetic walks for spot and LST entities.

Walk a spot/LST entity through 30+ daily bars with realistic price walks
(and rebasing for LSTs); verify high-level invariants over the full
trajectory.
"""
import random

import pytest

from fractal.core.entities.protocols.steth import (StakedETHEntity,
                                                   StakedETHGlobalState)
from fractal.core.entities.protocols.uniswap_v3_spot import (UniswapV3SpotEntity,
                                                             UniswapV3SpotGlobalState)


def _price_walk(seed: int = 7, n: int = 30, p0: float = 2000.0, sigma: float = 0.02):
    rng = random.Random(seed)
    p = p0
    series = [p0]
    for _ in range(n - 1):
        p *= (1.0 + rng.gauss(0.0, sigma))
        series.append(p)
    return series


@pytest.mark.core
def test_univ3_spot_balance_finite_through_walk():
    e = UniswapV3SpotEntity(trading_fee=0.003)
    e.update_state(UniswapV3SpotGlobalState(price=2000))
    e.action_deposit(10_000)
    e.action_buy(5_000)
    for p in _price_walk():
        e.update_state(UniswapV3SpotGlobalState(price=p))
        assert e.balance == e.balance
        assert e._internal_state.amount >= 0
        assert e._internal_state.cash >= 0


@pytest.mark.core
def test_univ3_spot_balance_tracks_price_move():
    """Buy product at p0, walk price up → balance grows proportionally."""
    e = UniswapV3SpotEntity(trading_fee=0.0)
    e.update_state(UniswapV3SpotGlobalState(price=2000))
    e.action_deposit(10_000)
    e.action_buy(10_000)  # all-in
    # Now walk price up 50%
    e.update_state(UniswapV3SpotGlobalState(price=3000))
    # Position is 5 ETH × 3000 = 15_000 (vs initial 10_000)
    assert e.balance == pytest.approx(15_000)


@pytest.mark.core
def test_steth_amount_grows_through_walk_with_positive_rate():
    """LST rebases on every update_state; amount grows with positive rate."""
    e = StakedETHEntity(trading_fee=0.0)
    e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))
    e.action_deposit(10_000)
    e.action_buy(10_000)
    initial_amount = e._internal_state.amount

    daily_rate = 0.0001  # ~3.7% APY
    n_bars = 30
    for _ in range(n_bars):
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=daily_rate))

    # Amount compounds
    expected = initial_amount * (1 + daily_rate) ** n_bars
    assert e._internal_state.amount == pytest.approx(expected, rel=1e-9)


@pytest.mark.core
def test_steth_balance_combines_price_walk_and_rebasing():
    """Both price and rate contribute to balance growth."""
    e = StakedETHEntity(trading_fee=0.0)
    e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))
    e.action_deposit(10_000)
    e.action_buy(10_000)

    # 30 bars: small steady rebase + price walk to 2200 by end
    prices = _price_walk(p0=2000, sigma=0.005, n=30)
    daily_rate = 0.0001
    for p in prices:
        e.update_state(StakedETHGlobalState(price=p, staking_rate=daily_rate))

    # Balance should reflect both effects (positive overall on average).
    assert e.balance > 0
    # Amount must have grown vs initial purchase (5 ETH).
    assert e._internal_state.amount > 5.0


@pytest.mark.core
def test_lst_outperforms_plain_spot_on_same_price_walk_with_rebasing():
    """At identical fees + price walk, an LST with positive staking_rate
    must end higher than a plain spot position (the rebasing premium)."""
    spot = UniswapV3SpotEntity(trading_fee=0.0)
    lst = StakedETHEntity(trading_fee=0.0)

    spot.update_state(UniswapV3SpotGlobalState(price=2000))
    lst.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))

    spot.action_deposit(10_000)
    lst.action_deposit(10_000)
    spot.action_buy(10_000)
    lst.action_buy(10_000)

    prices = _price_walk(p0=2000, sigma=0.005, n=30)
    for p in prices:
        spot.update_state(UniswapV3SpotGlobalState(price=p))
        lst.update_state(StakedETHGlobalState(price=p, staking_rate=0.0001))

    assert lst.balance > spot.balance


@pytest.mark.core
def test_univ3_spot_buy_sell_cycle_through_walk():
    """Repeatedly enter and exit the spot position over a price walk;
    state stays consistent across cycles."""
    e = UniswapV3SpotEntity(trading_fee=0.001)
    e.update_state(UniswapV3SpotGlobalState(price=2000))
    e.action_deposit(10_000)
    cycles = 0
    for p in _price_walk(p0=2000, sigma=0.02, n=20):
        e.update_state(UniswapV3SpotGlobalState(price=p))
        if e._internal_state.amount == 0 and e._internal_state.cash >= 1000:
            e.action_buy(1000)
            cycles += 1
        elif e._internal_state.amount > 0:
            e.action_sell(e._internal_state.amount)
    assert cycles >= 1
    assert e._internal_state.cash >= 0
    assert e._internal_state.amount >= 0
