"""Real-data smoke tests for ``HyperliquidBasis`` (slow)."""
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from fractal.core.base import Observation  # noqa: E402
from fractal.core.entities import HyperLiquidGlobalState, UniswapV3SpotGlobalState  # noqa: E402
from fractal.strategies.basis_trading_strategy import BasisTradingStrategyHyperparams  # noqa: E402
from fractal.strategies.hyperliquid_basis import HyperliquidBasis  # noqa: E402


_FIXTURES_DIR = (Path(__file__).resolve().parents[2]
                 / "examples" / "basis")


@dataclass
class _HBParams(BasisTradingStrategyHyperparams):
    EXECUTION_COST: float = 0.0


def _csv_path(ticker):
    return _FIXTURES_DIR / f"{ticker}_hyperliquid.csv"


def _load_observations(ticker, *, n):
    path = _csv_path(ticker)
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")
    df = pd.read_csv(path).head(n)
    obs = []
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


def _make_strategy(*, target_lev=3.0, exec_cost=0.0, initial=1_000_000.0):
    return HyperliquidBasis(params=_HBParams(
        MIN_LEVERAGE=1.0, TARGET_LEVERAGE=target_lev,
        MAX_LEVERAGE=5.0, INITIAL_BALANCE=initial,
        EXECUTION_COST=exec_cost,
    ))


@pytest.mark.slow
@pytest.mark.parametrize("ticker", ["BTC", "ETH", "LINK", "SOL"])
def test_hyperliquid_basis_runs_on_real_data_one_week(ticker):
    obs = _load_observations(ticker, n=168)
    s = _make_strategy()
    result = s.run(obs)
    final_total = result.balances[-1]["HEDGE"] + result.balances[-1]["SPOT"]
    assert math.isfinite(final_total)
    assert final_total > 0
    metrics = result.get_default_metrics()
    metric_dict = metrics.__dict__ if hasattr(metrics, "__dict__") else dict(metrics)
    for value in metric_dict.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)


@pytest.mark.slow
@pytest.mark.parametrize("ticker", ["BTC", "ETH"])
def test_hyperliquid_basis_equity_within_reasonable_band(ticker):
    obs = _load_observations(ticker, n=168)
    s = _make_strategy(exec_cost=0.0005)
    result = s.run(obs)
    final = result.balances[-1]["HEDGE"] + result.balances[-1]["SPOT"]
    drift = (final - s._params.INITIAL_BALANCE) / s._params.INITIAL_BALANCE
    assert -0.30 < drift < 0.30


@pytest.mark.slow
@pytest.mark.parametrize("ticker", ["BTC", "ETH", "LINK"])
def test_hyperliquid_basis_post_run_leverage_in_bounds(ticker):
    obs = _load_observations(ticker, n=168)
    s = _make_strategy()
    s.run(obs)
    hedge = s.get_entity("HEDGE")
    assert math.isfinite(hedge.leverage)
    assert hedge.leverage >= 0.5 * s._params.MIN_LEVERAGE
    assert hedge.leverage <= 1.5 * s._params.MAX_LEVERAGE


@pytest.mark.slow
def test_hyperliquid_basis_handles_long_horizon_btc():
    obs = _load_observations("BTC", n=1000)
    s = _make_strategy(exec_cost=0.0005)
    result = s.run(obs)
    final_total = sum(result.balances[-1].values())
    assert math.isfinite(final_total)
    assert final_total > 0


@pytest.mark.slow
def test_hyperliquid_basis_basis_invariant_holds_on_real_data_sample():
    obs = _load_observations("ETH", n=336)
    s = _make_strategy()
    s.run(obs)
    hedge = s.get_entity("HEDGE")
    spot = s.get_entity("SPOT")
    if abs(hedge.size) > 1e-9:
        ratio = abs(hedge.size + spot.internal_state.amount) / abs(hedge.size)
        assert ratio < 0.05
