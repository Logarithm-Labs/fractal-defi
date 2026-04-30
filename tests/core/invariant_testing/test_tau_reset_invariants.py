"""Invariants and randomized property checks for ``TauResetStrategy``.

Layer 3 — pinned invariants on bucket geometry and rebalance behaviour:

* **Range formula** — after every rebalance, ``price_upper / price_init
  = 1.0001^(TAU·tick_spacing)`` exactly (and the symmetric lower
  bound). The ``1.0001^tick`` Uniswap V3 price-from-tick formula is
  the contract.
* **Symmetric bracket** — ``price_init`` is the geometric mean of
  ``price_lower`` and ``price_upper``.
* **No-rebalance immutability** — when the price stays inside the
  active range, ``price_lower`` / ``price_upper`` do not move.
* **In-range predicate** — ``price_lower <= reference <= price_upper``
  after a rebalance triggered by an out-of-range tick.

Layer 4 — lightweight property-based: deterministic seeds drive
randomized TAU / tick_spacing / price-walk scenarios; each scenario
re-checks the same invariants. No external Hypothesis dependency.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import List

import pytest

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState
from fractal.strategies.tau_reset_strategy import (TauResetParams,
                                                   TauResetStrategy)


# ============================================================ scaffolding


def _make_strategy(*, tau: float = 5.0, initial: float = 1_000_000.0,
                   token0_decimals: int = 6, token1_decimals: int = 18,
                   tick_spacing: int = 60) -> TauResetStrategy:
    return TauResetStrategy(
        params=TauResetParams(TAU=tau, INITIAL_BALANCE=initial),
        token0_decimals=token0_decimals,
        token1_decimals=token1_decimals,
        tick_spacing=tick_spacing,
    )


def _obs(t: datetime, price: float, *, tvl: float = 1_000_000.0,
         volume: float = 0.0, fees: float = 0.0,
         liquidity: float = 1_000_000.0) -> Observation:
    return Observation(
        timestamp=t,
        states={
            'UNISWAP_V3': UniswapV3LPGlobalState(
                price=price, tvl=tvl, volume=volume, fees=fees,
                liquidity=liquidity,
            ),
        },
    )


def _build_path(prices: List[float], start: datetime = datetime(2024, 1, 1),
                step: timedelta = timedelta(hours=1)) -> List[Observation]:
    return [_obs(start + i * step, p) for i, p in enumerate(prices)]


# ============================================================ Layer 3 — pinned invariants


@pytest.mark.core
def test_range_upper_lower_match_uniswap_tick_formula():
    """After a rebalance, ``price_upper / price_init = 1.0001^(TAU·ts)``
    and ``price_init / price_lower = 1.0001^(TAU·ts)`` — the V3
    price-from-tick formula. Pin numerically with TAU=15, ts=60."""
    s = _make_strategy(tau=15, tick_spacing=60)
    s.run([
        _obs(datetime(2024, 1, 1), 3000.0),
        # Force a rebalance: jump price way outside any plausible range.
        _obs(datetime(2024, 1, 2), 4000.0),
    ])
    e = s.get_entity('UNISWAP_V3')
    expected_factor = 1.0001 ** (15 * 60)
    assert e.internal_state.price_upper / e.internal_state.price_init == pytest.approx(
        expected_factor, rel=1e-12
    )
    assert e.internal_state.price_init / e.internal_state.price_lower == pytest.approx(
        expected_factor, rel=1e-12
    )


@pytest.mark.core
def test_range_init_is_geometric_mean_of_bounds():
    """``price_init = sqrt(price_lower · price_upper)`` — direct
    consequence of the symmetric bracket around ``reference``."""
    s = _make_strategy(tau=10, tick_spacing=60)
    s.run([
        _obs(datetime(2024, 1, 1), 2500.0),
        _obs(datetime(2024, 1, 2), 5000.0),  # huge move → rebalance
    ])
    e = s.get_entity('UNISWAP_V3')
    geo_mean = math.sqrt(e.internal_state.price_lower * e.internal_state.price_upper)
    assert e.internal_state.price_init == pytest.approx(geo_mean, rel=1e-12)


@pytest.mark.core
def test_range_does_not_move_when_price_stays_in_band():
    """Idle ticks within the active range never trigger ``_rebalance``
    — ``price_lower`` / ``price_upper`` stay frozen at the values set
    on the prior rebalance."""
    s = _make_strategy(tau=15, tick_spacing=60)
    # Initial deposit, then first rebalance.
    obs = [
        _obs(datetime(2024, 1, 1, 0), 3000.0),
        _obs(datetime(2024, 1, 1, 1), 3000.0),
    ]
    s.run(obs)
    e = s.get_entity('UNISWAP_V3')
    pl_after_first, pu_after_first = e.internal_state.price_lower, e.internal_state.price_upper
    # Continue with prices that stay safely inside the band.
    cont = [
        _obs(datetime(2024, 1, 1, 2 + i), 3000.0 * (1 + 0.001 * (i % 5 - 2)))
        for i in range(20)
    ]
    s.run(cont)
    assert e.internal_state.price_lower == pl_after_first
    assert e.internal_state.price_upper == pu_after_first


@pytest.mark.core
def test_reference_price_lies_inside_range_after_rebalance():
    """After a rebalance triggered by an out-of-range tick, the new
    reference price must sit strictly within the new range."""
    s = _make_strategy(tau=5, tick_spacing=60)
    s.run([
        _obs(datetime(2024, 1, 1), 3000.0),
        _obs(datetime(2024, 1, 2), 3000.0),  # opens position
        _obs(datetime(2024, 1, 3), 3500.0),  # outside band → rebalance
    ])
    e = s.get_entity('UNISWAP_V3')
    assert e.internal_state.price_lower < e.internal_state.price_init < e.internal_state.price_upper


@pytest.mark.core
def test_range_widens_with_larger_tau():
    """Doubling ``TAU`` widens the active-range factor by squaring it."""
    def upper_factor(tau: float) -> float:
        s = _make_strategy(tau=tau, tick_spacing=60)
        s.run([
            _obs(datetime(2024, 1, 1), 3000.0),
            _obs(datetime(2024, 1, 2), 5000.0),
        ])
        e = s.get_entity('UNISWAP_V3')
        return e.internal_state.price_upper / e.internal_state.price_init

    f5 = upper_factor(5)
    f10 = upper_factor(10)
    # Doubling TAU squares the factor (since exponent doubles).
    assert f10 == pytest.approx(f5 * f5, rel=1e-12)


@pytest.mark.core
def test_initial_deposit_lands_in_lp_cash_then_first_rebalance_zaps_in():
    """Two-step bootstrap: first observation deposits into LP cash; the
    second observation (with a fresh price) opens the first position."""
    s = _make_strategy(tau=15, tick_spacing=60, initial=100_000.0)
    s.run([_obs(datetime(2024, 1, 1), 3000.0)])
    e = s.get_entity('UNISWAP_V3')
    assert e.internal_state.cash == pytest.approx(100_000.0, rel=1e-9)
    assert e.is_position is False
    s.run([_obs(datetime(2024, 1, 2), 3000.0)])
    assert e.is_position is True
    # After zap-in, reference is the newer price (3000 again here).
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
        assert math.isfinite(v), f"non-finite {v}"


# ============================================================ Layer 4 — randomized property-based


def _random_walk(rng: random.Random, n: int = 30, start: float = 3000.0,
                 step_pct: float = 0.01) -> List[float]:
    out = [start]
    for _ in range(n - 1):
        out.append(out[-1] * (1 + rng.uniform(-step_pct, step_pct)))
    return out


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(20)))
def test_property_range_formula_holds_for_random_tau_and_tick_spacing(seed: int):
    """Across 20 seeds × random TAU / tick_spacing combos, the
    ``1.0001^(TAU·ts)`` range formula holds exactly after each
    rebalance."""
    rng = random.Random(seed)
    tau = rng.randint(1, 30)
    tick_spacing = rng.choice([10, 60, 200])
    s = _make_strategy(tau=tau, tick_spacing=tick_spacing)
    # Drive a path that triggers at least one rebalance.
    path = _random_walk(rng, n=20, step_pct=0.02)
    s.run(_build_path(path))
    e = s.get_entity('UNISWAP_V3')
    if not e.is_position:
        pytest.skip(f"seed={seed}: no position opened")
    expected_factor = 1.0001 ** (tau * tick_spacing)
    actual_upper = e.internal_state.price_upper / e.internal_state.price_init
    actual_lower = e.internal_state.price_init / e.internal_state.price_lower
    assert actual_upper == pytest.approx(expected_factor, rel=1e-9), (
        f"seed={seed} tau={tau} ts={tick_spacing}"
    )
    assert actual_lower == pytest.approx(expected_factor, rel=1e-9), (
        f"seed={seed} tau={tau} ts={tick_spacing}"
    )


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(15)))
def test_property_no_nan_or_inf_after_random_path(seed: int):
    rng = random.Random(seed)
    tau = rng.randint(1, 25)
    tick_spacing = rng.choice([10, 60, 200])
    s = _make_strategy(tau=tau, tick_spacing=tick_spacing)
    path = _random_walk(rng, n=30, step_pct=0.015)
    s.run(_build_path(path))
    e = s.get_entity('UNISWAP_V3')
    for name in ('cash', 'liquidity', 'price_lower', 'price_upper',
                 'price_init', 'token0_amount', 'token1_amount'):
        v = getattr(e.internal_state, name)
        assert math.isfinite(v), f"seed={seed} {name}={v}"
    assert math.isfinite(e.balance)


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(15)))
def test_property_reference_inside_range_after_rebalance(seed: int):
    """After whatever path, if a position is open the reference price
    used for it must be inside its range."""
    rng = random.Random(seed)
    s = _make_strategy(tau=rng.randint(2, 20), tick_spacing=60)
    path = _random_walk(rng, n=30, step_pct=0.02)
    s.run(_build_path(path))
    e = s.get_entity('UNISWAP_V3')
    if e.is_position:
        assert e.internal_state.price_lower < e.internal_state.price_init < e.internal_state.price_upper, (
            f"seed={seed} init={e.internal_state.price_init} "
            f"range=[{e.internal_state.price_lower}, {e.internal_state.price_upper}]"
        )


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(10)))
def test_property_lp_cash_non_negative(seed: int):
    """LP ``cash`` must never go negative — withdraw/zap-in are
    self-balancing in the entity, but a strategy bug could
    over-withdraw."""
    rng = random.Random(seed)
    s = _make_strategy(tau=rng.randint(3, 15), tick_spacing=60)
    path = _random_walk(rng, n=25, step_pct=0.015)
    s.run(_build_path(path))
    e = s.get_entity('UNISWAP_V3')
    assert e.internal_state.cash >= -1e-9, f"seed={seed} cash={e.internal_state.cash}"


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(10)))
def test_property_balance_finite_and_non_negative(seed: int):
    rng = random.Random(seed)
    s = _make_strategy(tau=rng.randint(3, 15), tick_spacing=60)
    path = _random_walk(rng, n=30, step_pct=0.012)
    s.run(_build_path(path))
    e = s.get_entity('UNISWAP_V3')
    assert math.isfinite(e.balance)
    assert e.balance >= 0, f"seed={seed} balance={e.balance}"
