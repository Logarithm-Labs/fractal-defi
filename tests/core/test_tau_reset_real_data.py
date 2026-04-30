"""Layer 5 smoke tests for ``TauResetStrategy`` against real CSV data.

The fixture ``examples/tau_strategy/tau_strategy_result.csv`` is a
prior strategy run output that carries the **input** columns the
strategy needs:

* ``timestamp``
* ``UNISWAP_V3_price`` — pool spot price
* ``UNISWAP_V3_tvl`` — pool TVL
* ``UNISWAP_V3_volume`` — bar volume (drives fee accrual)
* ``UNISWAP_V3_fees`` — fees collected by the pool in the bar
* ``UNISWAP_V3_liquidity`` — pool active-tick liquidity

We reconstruct ``Observation`` objects and replay the strategy through
windows of varying length. Marked ``@pytest.mark.slow`` — run with
``pytest -m slow``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List

import pytest

pd = pytest.importorskip("pandas")

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState
from fractal.strategies.tau_reset_strategy import (TauResetParams,
                                                   TauResetStrategy)


_FIXTURE = (Path(__file__).resolve().parents[2]
            / "examples" / "tau_strategy" / "tau_strategy_result.csv")


def _load_observations(*, n: int) -> List[Observation]:
    if not _FIXTURE.exists():
        pytest.skip(f"fixture missing: {_FIXTURE}")
    df = pd.read_csv(_FIXTURE).head(n)
    obs: List[Observation] = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
        obs.append(Observation(
            timestamp=ts,
            states={
                "UNISWAP_V3": UniswapV3LPGlobalState(
                    price=float(row["UNISWAP_V3_price"]),
                    tvl=float(row["UNISWAP_V3_tvl"]),
                    volume=float(row["UNISWAP_V3_volume"]),
                    fees=float(row["UNISWAP_V3_fees"]),
                    liquidity=float(row["UNISWAP_V3_liquidity"]),
                ),
            },
        ))
    return obs


def _make_strategy(*, tau: float = 15, initial: float = 1_000_000.0,
                   tick_spacing: int = 60) -> TauResetStrategy:
    return TauResetStrategy(
        params=TauResetParams(TAU=tau, INITIAL_BALANCE=initial),
        token0_decimals=6, token1_decimals=18,
        tick_spacing=tick_spacing,
    )


# ============================================================ smoke tests


@pytest.mark.slow
def test_tau_reset_runs_on_real_data_one_week():
    """**Smoke** — replay 168 hourly observations of real ETH/USDC
    pool data through the strategy. Asserts: no crash, finite final
    equity > 0, all default metrics finite."""
    obs = _load_observations(n=168)
    s = _make_strategy()
    result = s.run(obs)
    final_total = result.balances[-1]["UNISWAP_V3"]
    assert math.isfinite(final_total)
    assert final_total > 0
    metrics = result.get_default_metrics()
    metric_dict = metrics.__dict__ if hasattr(metrics, "__dict__") else dict(metrics)
    for name, value in metric_dict.items():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), f"{name} is {value}"


@pytest.mark.slow
@pytest.mark.parametrize("tau", [5, 15, 30])
def test_tau_reset_equity_within_reasonable_band_across_taus(tau: int):
    """Across ``TAU`` choices, 1-week run does not blow up. Generous
    band — concentrated LP performance varies a lot with bucket
    width but smoke is "no catastrophic drift"."""
    obs = _load_observations(n=168)
    s = _make_strategy(tau=tau)
    result = s.run(obs)
    final = result.balances[-1]["UNISWAP_V3"]
    initial = s._params.INITIAL_BALANCE
    drift = (final - initial) / initial
    assert -0.30 < drift < 0.30, (
        f"TAU={tau}: equity drifted {drift:+.1%} over 1 week"
    )


@pytest.mark.slow
def test_tau_reset_handles_long_horizon():
    """Longer horizon (700 obs ≈ 1 month) — smoke for repeated
    rebalances against real volatility."""
    obs = _load_observations(n=700)
    s = _make_strategy(tau=15)
    result = s.run(obs)
    final = sum(result.balances[-1].values())
    assert math.isfinite(final)
    assert final > 0


@pytest.mark.slow
def test_tau_reset_range_formula_holds_after_real_data_run():
    """**Lock-in** — even after a 1-month real-data run, the active
    range matches the ``1.0001^(TAU·tick_spacing)`` formula exactly."""
    obs = _load_observations(n=500)
    s = _make_strategy(tau=15, tick_spacing=60)
    s.run(obs)
    e = s.get_entity("UNISWAP_V3")
    if not e.is_position:
        pytest.skip("no position open at end of run")
    expected = 1.0001 ** (15 * 60)
    assert e.internal_state.price_upper / e.internal_state.price_init == pytest.approx(
        expected, rel=1e-9
    )
    assert e.internal_state.price_init / e.internal_state.price_lower == pytest.approx(
        expected, rel=1e-9
    )


@pytest.mark.slow
def test_tau_reset_state_invariants_hold_throughout_real_data_run():
    """End-state checks that should hold for any open position after
    replay: cash non-negative, liquidity > 0, range valid, balance
    finite and non-negative."""
    obs = _load_observations(n=500)
    s = _make_strategy(tau=15)
    s.run(obs)
    e = s.get_entity("UNISWAP_V3")
    assert e.internal_state.cash >= -1e-6
    assert math.isfinite(e.balance)
    assert e.balance >= 0
    if e.is_position:
        assert e.internal_state.price_lower < e.internal_state.price_upper
        assert e.internal_state.liquidity > 0
