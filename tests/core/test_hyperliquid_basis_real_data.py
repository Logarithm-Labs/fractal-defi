"""Layer 5 smoke tests for ``HyperliquidBasis`` against real CSV data.

The fixtures in ``examples/managed_basis_strategy/*_hyperliquid.csv``
are previously-recorded strategy run outputs that happen to also carry
the **inputs** the strategy needs as columns:

* ``timestamp``
* ``SPOT_price`` — UniswapV3 spot reference price
* ``HEDGE_mark_price`` — Hyperliquid perp mark price
* ``HEDGE_funding_rate`` — perp funding rate

We reconstruct ``Observation`` objects from these columns and replay
the strategy through a short window. The aim is *smoke* — does the
strategy survive real price/funding paths without crashes, produce
finite metrics, and finish with non-zero equity?

Marked ``@pytest.mark.slow`` so the default ``-m core`` run skips
them. Run explicitly with ``pytest -m slow``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

pd = pytest.importorskip("pandas")

from fractal.core.base import Observation
from fractal.core.entities import HyperLiquidGlobalState, UniswapV3SpotGlobalState
from fractal.strategies.basis_trading_strategy import \
    BasisTradingStrategyHyperparams
from fractal.strategies.hyperliquid_basis import HyperliquidBasis


_FIXTURES_DIR = (Path(__file__).resolve().parents[2]
                 / "examples" / "managed_basis_strategy")


@dataclass
class _HBParams(BasisTradingStrategyHyperparams):
    EXECUTION_COST: float = 0.0


# ============================================================ helpers


def _csv_path(ticker: str) -> Path:
    return _FIXTURES_DIR / f"{ticker}_hyperliquid.csv"


def _load_observations(ticker: str, *, n: int) -> List[Observation]:
    """Slice the first ``n`` rows of ``<TICKER>_hyperliquid.csv`` into
    ``Observation`` objects. Skips the test cleanly if the fixture is
    missing (e.g. on a fresh checkout without examples)."""
    path = _csv_path(ticker)
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")
    df = pd.read_csv(path).head(n)
    obs: List[Observation] = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
        obs.append(Observation(
            timestamp=ts,
            states={
                "SPOT": UniswapV3SpotGlobalState(price=float(row["SPOT_price"])),
                "HEDGE": HyperLiquidGlobalState(
                    mark_price=float(row["HEDGE_mark_price"]),
                    funding_rate=float(row["HEDGE_funding_rate"]),
                ),
            },
        ))
    return obs


def _make_strategy(*, target_lev: float = 3.0, exec_cost: float = 0.0,
                   initial: float = 1_000_000.0) -> HyperliquidBasis:
    return HyperliquidBasis(params=_HBParams(
        MIN_LEVERAGE=1.0, TARGET_LEVERAGE=target_lev,
        MAX_LEVERAGE=5.0, INITIAL_BALANCE=initial,
        EXECUTION_COST=exec_cost,
    ))


# ============================================================ smoke tests


@pytest.mark.slow
@pytest.mark.parametrize("ticker", ["BTC", "ETH", "LINK", "SOL"])
def test_hyperliquid_basis_runs_on_real_data_one_week(ticker: str):
    """**Smoke** — replay 168 hourly observations (≈ 1 week) of real
    Hyperliquid data through the strategy. Asserts:

    * strategy completes without raising,
    * final equity is finite and strictly positive,
    * default metrics (PnL, Sharpe, …) are finite.
    """
    obs = _load_observations(ticker, n=168)
    s = _make_strategy()
    result = s.run(obs)
    final_total = result.balances[-1]["HEDGE"] + result.balances[-1]["SPOT"]
    assert math.isfinite(final_total)
    assert final_total > 0, f"{ticker}: strategy fully wiped"
    metrics = result.get_default_metrics()
    metric_dict = metrics.__dict__ if hasattr(metrics, "__dict__") else dict(metrics)
    for name, value in metric_dict.items():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), f"{ticker}: {name} is {value}"


@pytest.mark.slow
@pytest.mark.parametrize("ticker", ["BTC", "ETH"])
def test_hyperliquid_basis_equity_within_reasonable_band(ticker: str):
    """Over a 1-week window with realistic execution cost (5 bps), the
    strategy should not blow up and not 10x — final equity stays
    within ±30% of initial.

    Loose bound on purpose: this is a **smoke** check (catastrophic
    drift / runaway accounting), not a benchmark.
    """
    obs = _load_observations(ticker, n=168)
    s = _make_strategy(exec_cost=0.0005, initial=1_000_000.0)
    result = s.run(obs)
    final_total = result.balances[-1]["HEDGE"] + result.balances[-1]["SPOT"]
    initial = s._params.INITIAL_BALANCE
    drift = (final_total - initial) / initial
    assert -0.30 < drift < 0.30, (
        f"{ticker}: equity drifted {drift:+.1%} over 1 week — outside band"
    )


@pytest.mark.slow
@pytest.mark.parametrize("ticker", ["BTC", "ETH", "LINK"])
def test_hyperliquid_basis_post_run_leverage_in_bounds(ticker: str):
    """After replaying real data, the hedge ends up with leverage
    inside ``[0.5·MIN, 1.5·MAX]`` — generous tolerance because the
    final tick may be mid-rebalance, but order-of-magnitude must
    match the configured trigger band.
    """
    obs = _load_observations(ticker, n=168)
    s = _make_strategy()
    s.run(obs)
    hedge = s.get_entity("HEDGE")
    assert math.isfinite(hedge.leverage)
    assert hedge.leverage >= 0.5 * s._params.MIN_LEVERAGE
    assert hedge.leverage <= 1.5 * s._params.MAX_LEVERAGE


@pytest.mark.slow
def test_hyperliquid_basis_handles_long_horizon_btc():
    """Longer horizon (1000 obs ≈ 6 weeks) on BTC — smoke for
    multi-rebalance dynamics over a meaningful sample."""
    obs = _load_observations("BTC", n=1000)
    s = _make_strategy(exec_cost=0.0005)
    result = s.run(obs)
    final_total = sum(result.balances[-1].values())
    assert math.isfinite(final_total)
    assert final_total > 0


@pytest.mark.slow
def test_hyperliquid_basis_basis_invariant_holds_on_real_data_sample():
    """Across a real-data window, the basis hedge invariant
    ``hedge.size ≈ -spot.amount`` must hold at every observation
    where the strategy is in a steady state (not mid-rebalance).

    We check the FINAL state — strict, since the strategy resolves
    rebalance fully within a single ``step``.
    """
    obs = _load_observations("ETH", n=336)  # 2 weeks
    s = _make_strategy()
    s.run(obs)
    hedge = s.get_entity("HEDGE")
    spot = s.get_entity("SPOT")
    if abs(hedge.size) > 1e-9:
        ratio = abs(hedge.size + spot.internal_state.amount) / abs(hedge.size)
        assert ratio < 0.05, (
            f"ETH: basis hedge drift ratio={ratio:.4f} after 2-week run"
        )
