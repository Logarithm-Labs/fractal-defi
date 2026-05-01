"""Invariant and randomized property checks for ``TauResetStrategy``."""
import math
import random
from datetime import datetime, timedelta

import pytest

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState
from fractal.strategies.tau_reset_strategy import TauResetParams, TauResetStrategy


def _make_strategy(*, tau=5.0, initial=1_000_000.0,
                   token0_decimals=6, token1_decimals=18, tick_spacing=60):
    return TauResetStrategy(
        params=TauResetParams(TAU=tau, INITIAL_BALANCE=initial),
        token0_decimals=token0_decimals,
        token1_decimals=token1_decimals,
        tick_spacing=tick_spacing,
    )


def _obs(t, price, *, tvl=1_000_000.0, volume=0.0, fees=0.0, liquidity=1_000_000.0):
    return Observation(timestamp=t, states={
        'UNISWAP_V3': UniswapV3LPGlobalState(
            price=price, tvl=tvl, volume=volume, fees=fees, liquidity=liquidity,
        ),
    })


def _build_path(prices, start=datetime(2024, 1, 1), step=timedelta(hours=1)):
    return [_obs(start + i * step, p) for i, p in enumerate(prices)]


def _random_walk(rng, n=30, start=3000.0, step_pct=0.01):
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * (1 + rng.uniform(-step_pct, step_pct)))
    return out


@pytest.mark.core
def test_range_upper_lower_match_uniswap_tick_formula():
    s = _make_strategy(tau=15, tick_spacing=60)
    s.run([
        _obs(datetime(2024, 1, 1), 3000.0),
        _obs(datetime(2024, 1, 2), 4000.0),
    ])
    e = s.get_entity('UNISWAP_V3')
    expected = 1.0001 ** (15 * 60)
    assert e.internal_state.price_upper / e.internal_state.price_init == pytest.approx(expected, rel=1e-12)
    assert e.internal_state.price_init / e.internal_state.price_lower == pytest.approx(expected, rel=1e-12)


@pytest.mark.core
def test_range_init_is_geometric_mean_of_bounds():
    s = _make_strategy(tau=10, tick_spacing=60)
    s.run([
        _obs(datetime(2024, 1, 1), 2500.0),
        _obs(datetime(2024, 1, 2), 5000.0),
    ])
    e = s.get_entity('UNISWAP_V3')
    geo_mean = math.sqrt(e.internal_state.price_lower * e.internal_state.price_upper)
    assert e.internal_state.price_init == pytest.approx(geo_mean, rel=1e-12)


@pytest.mark.core
def test_range_does_not_move_when_price_stays_in_band():
    s = _make_strategy(tau=15, tick_spacing=60)
    s.run([
        _obs(datetime(2024, 1, 1, 0), 3000.0),
        _obs(datetime(2024, 1, 1, 1), 3000.0),
    ])
    e = s.get_entity('UNISWAP_V3')
    pl, pu = e.internal_state.price_lower, e.internal_state.price_upper
    cont = [_obs(datetime(2024, 1, 1, 2 + i),
                 3000.0 * (1 + 0.001 * (i % 5 - 2))) for i in range(20)]
    s.run(cont)
    assert e.internal_state.price_lower == pl
    assert e.internal_state.price_upper == pu


@pytest.mark.core
def test_reference_price_lies_inside_range_after_rebalance():
    s = _make_strategy(tau=5, tick_spacing=60)
    s.run([
        _obs(datetime(2024, 1, 1), 3000.0),
        _obs(datetime(2024, 1, 2), 3000.0),
        _obs(datetime(2024, 1, 3), 3500.0),
    ])
    e = s.get_entity('UNISWAP_V3')
    assert e.internal_state.price_lower < e.internal_state.price_init < e.internal_state.price_upper


@pytest.mark.core
def test_range_widens_with_larger_tau():
    def upper_factor(tau):
        s = _make_strategy(tau=tau, tick_spacing=60)
        s.run([
            _obs(datetime(2024, 1, 1), 3000.0),
            _obs(datetime(2024, 1, 2), 5000.0),
        ])
        e = s.get_entity('UNISWAP_V3')
        return e.internal_state.price_upper / e.internal_state.price_init

    assert upper_factor(10) == pytest.approx(upper_factor(5) ** 2, rel=1e-12)


@pytest.mark.core
def test_initial_deposit_lands_in_lp_cash_then_first_rebalance_zaps_in():
    s = _make_strategy(tau=15, tick_spacing=60, initial=100_000.0)
    s.run([_obs(datetime(2024, 1, 1), 3000.0)])
    e = s.get_entity('UNISWAP_V3')
    assert e.internal_state.cash == pytest.approx(100_000.0, rel=1e-9)
    assert e.is_position is False
    s.run([_obs(datetime(2024, 1, 2), 3000.0)])
    assert e.is_position is True
    assert e.internal_state.price_init == pytest.approx(3000.0)


@pytest.mark.core
def test_no_nan_or_inf_in_lp_state_after_run():
    s = _make_strategy(tau=10, tick_spacing=60)
    obs = [_obs(datetime(2024, 1, 1) + timedelta(hours=i),
                3000.0 * (1 + 0.005 * (i % 7 - 3))) for i in range(40)]
    s.run(obs)
    e = s.get_entity('UNISWAP_V3')
    for v in (e.internal_state.cash, e.internal_state.liquidity,
              e.internal_state.price_lower, e.internal_state.price_upper,
              e.internal_state.price_init, e.balance):
        assert math.isfinite(v)


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(20)))
def test_property_range_formula_holds_for_random_tau_and_tick_spacing(seed):
    rng = random.Random(seed)
    tau = rng.randint(1, 30)
    tick_spacing = rng.choice([10, 60, 200])
    s = _make_strategy(tau=tau, tick_spacing=tick_spacing)
    s.run(_build_path(_random_walk(rng, n=20, step_pct=0.02)))
    e = s.get_entity('UNISWAP_V3')
    if not e.is_position:
        pytest.skip("no position opened")
    expected = 1.0001 ** (tau * tick_spacing)
    assert e.internal_state.price_upper / e.internal_state.price_init == pytest.approx(expected, rel=1e-9)
    assert e.internal_state.price_init / e.internal_state.price_lower == pytest.approx(expected, rel=1e-9)


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(15)))
def test_property_no_nan_or_inf_after_random_path(seed):
    rng = random.Random(seed)
    tau = rng.randint(1, 25)
    s = _make_strategy(tau=tau, tick_spacing=rng.choice([10, 60, 200]))
    s.run(_build_path(_random_walk(rng, n=30, step_pct=0.015)))
    e = s.get_entity('UNISWAP_V3')
    for name in ('cash', 'liquidity', 'price_lower', 'price_upper',
                 'price_init', 'token0_amount', 'token1_amount'):
        assert math.isfinite(getattr(e.internal_state, name))
    assert math.isfinite(e.balance)


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(15)))
def test_property_reference_inside_range_after_rebalance(seed):
    rng = random.Random(seed)
    s = _make_strategy(tau=rng.randint(2, 20), tick_spacing=60)
    s.run(_build_path(_random_walk(rng, n=30, step_pct=0.02)))
    e = s.get_entity('UNISWAP_V3')
    if e.is_position:
        assert e.internal_state.price_lower < e.internal_state.price_init < e.internal_state.price_upper


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(10)))
def test_property_lp_cash_non_negative(seed):
    rng = random.Random(seed)
    s = _make_strategy(tau=rng.randint(3, 15), tick_spacing=60)
    s.run(_build_path(_random_walk(rng, n=25, step_pct=0.015)))
    assert s.get_entity('UNISWAP_V3').internal_state.cash >= -1e-9


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(10)))
def test_property_balance_finite_and_non_negative(seed):
    rng = random.Random(seed)
    s = _make_strategy(tau=rng.randint(3, 15), tick_spacing=60)
    s.run(_build_path(_random_walk(rng, n=30, step_pct=0.012)))
    e = s.get_entity('UNISWAP_V3')
    assert math.isfinite(e.balance)
    assert e.balance >= 0
