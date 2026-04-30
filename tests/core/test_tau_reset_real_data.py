"""Real-data smoke tests for ``TauResetStrategy`` (slow)."""
import math
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from fractal.core.base import Observation  # noqa: E402
from fractal.core.entities import UniswapV3LPGlobalState  # noqa: E402
from fractal.strategies.tau_reset_strategy import (  # noqa: E402
    TauResetParams, TauResetStrategy,
)


_FIXTURE = (Path(__file__).resolve().parents[2]
            / "examples" / "tau_strategy" / "tau_strategy_result.csv")


def _load_observations(*, n):
    if not _FIXTURE.exists():
        pytest.skip(f"fixture missing: {_FIXTURE}")
    df = pd.read_csv(_FIXTURE).head(n)
    obs = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
        obs.append(Observation(timestamp=ts, states={
            "UNISWAP_V3": UniswapV3LPGlobalState(
                price=float(row["UNISWAP_V3_price"]),
                tvl=float(row["UNISWAP_V3_tvl"]),
                volume=float(row["UNISWAP_V3_volume"]),
                fees=float(row["UNISWAP_V3_fees"]),
                liquidity=float(row["UNISWAP_V3_liquidity"]),
            ),
        }))
    return obs


def _make_strategy(*, tau=15, initial=1_000_000.0, tick_spacing=60):
    return TauResetStrategy(
        params=TauResetParams(TAU=tau, INITIAL_BALANCE=initial),
        token0_decimals=6, token1_decimals=18,
        tick_spacing=tick_spacing,
    )


@pytest.mark.slow
def test_tau_reset_runs_on_real_data_one_week():
    obs = _load_observations(n=168)
    s = _make_strategy()
    result = s.run(obs)
    final = result.balances[-1]["UNISWAP_V3"]
    assert math.isfinite(final)
    assert final > 0
    metrics = result.get_default_metrics()
    metric_dict = metrics.__dict__ if hasattr(metrics, "__dict__") else dict(metrics)
    for value in metric_dict.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value)


@pytest.mark.slow
@pytest.mark.parametrize("tau", [5, 15, 30])
def test_tau_reset_equity_within_reasonable_band_across_taus(tau):
    obs = _load_observations(n=168)
    s = _make_strategy(tau=tau)
    result = s.run(obs)
    final = result.balances[-1]["UNISWAP_V3"]
    drift = (final - s._params.INITIAL_BALANCE) / s._params.INITIAL_BALANCE
    assert -0.30 < drift < 0.30


@pytest.mark.slow
def test_tau_reset_handles_long_horizon():
    obs = _load_observations(n=700)
    s = _make_strategy(tau=15)
    result = s.run(obs)
    final = sum(result.balances[-1].values())
    assert math.isfinite(final)
    assert final > 0


@pytest.mark.slow
def test_tau_reset_range_formula_holds_after_real_data_run():
    obs = _load_observations(n=500)
    s = _make_strategy(tau=15, tick_spacing=60)
    s.run(obs)
    e = s.get_entity("UNISWAP_V3")
    if not e.is_position:
        pytest.skip("no position open at end of run")
    expected = 1.0001 ** (15 * 60)
    assert e.internal_state.price_upper / e.internal_state.price_init == pytest.approx(expected, rel=1e-9)
    assert e.internal_state.price_init / e.internal_state.price_lower == pytest.approx(expected, rel=1e-9)


@pytest.mark.slow
def test_tau_reset_state_invariants_hold_throughout_real_data_run():
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
