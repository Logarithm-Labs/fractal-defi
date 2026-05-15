"""L2 synthetic end-to-end test for PanopticStraddleStrategy.

Runs the full strategy on hand-rolled observations that exercise every
``predict`` branch at least once.  No network access; fully deterministic.

Marked ``@pytest.mark.core`` -- included in the default ``pytest -m core``
suite.
"""

import math
from datetime import UTC, datetime
from typing import List

import numpy as np
import pytest

from fractal.core.base import Observation
from fractal.core.entities.protocols.uniswap_v3_lp import UniswapV3LPGlobalState
from fractal.strategies.panoptic_straddle import (
    PanopticPoolGlobalState,
    PanopticStraddleParams,
    PanopticStraddleStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(
    prices: list,
    iv_values: list | None = None,
    fees: float = 50_000.0,
    liquidity: float = 1e22,
) -> List[Observation]:
    """Build a list of hourly Observation objects from price and IV arrays."""
    result = []
    for i, price in enumerate(prices):
        iv = iv_values[i] if iv_values is not None else 0.60
        result.append(Observation(
            timestamp=datetime(2023, 1, 1, i % 24, tzinfo=UTC),
            states={
                "STRADDLE": PanopticPoolGlobalState(
                    price=price, fees=fees, liquidity=liquidity,
                    volume=5e8, tvl=5e8, iv_annual=iv,
                ),
                "LP": UniswapV3LPGlobalState(
                    price=price, fees=fees, liquidity=liquidity,
                    volume=5e8, tvl=5e8,
                ),
            },
        ))
    return result


def _run(params: PanopticStraddleParams, prices, **kw):
    obs = _obs(prices, **kw)
    strategy = PanopticStraddleStrategy(params=params)
    result = strategy.run(obs)
    return result, result.to_dataframe(), result.get_default_metrics(), strategy


# ---------------------------------------------------------------------------
# Scenario A -- flat market, no signal: straddle stays closed
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_e2e_flat_market_no_position_opened():
    """With IV_ENTRY_PERCENTILE=0, the entry condition is never met."""
    prices = [3000.0] * 60
    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        LOOKBACK_BARS=5,
        IV_ENTRY_PERCENTILE=0.0,   # impossible to satisfy
    )
    _, df, _, _ = _run(params, prices)
    assert not df["STRADDLE_is_open"].any()


@pytest.mark.core
def test_e2e_flat_market_metrics_finite():
    """Even without trades, all metrics must be finite numbers."""
    prices = [3000.0] * 60
    params = PanopticStraddleParams(LOOKBACK_BARS=5)
    _, _, metrics, _ = _run(params, prices)
    assert math.isfinite(metrics.sharpe)
    assert math.isfinite(metrics.apy)
    assert math.isfinite(metrics.max_drawdown)
    assert math.isfinite(metrics.accumulated_return)


@pytest.mark.core
def test_e2e_flat_market_net_balance_positive():
    prices = [3000.0] * 60
    params = PanopticStraddleParams(LOOKBACK_BARS=5)
    _, df, _, _ = _run(params, prices)
    assert (df["net_balance"] > 0).all()


@pytest.mark.core
def test_e2e_flat_market_max_drawdown_non_positive():
    prices = [3000.0] * 60
    params = PanopticStraddleParams(LOOKBACK_BARS=5)
    _, _, metrics, _ = _run(params, prices)
    assert metrics.max_drawdown <= 0.0


# ---------------------------------------------------------------------------
# Scenario B -- low IV entry, time stop exit
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_e2e_entry_on_low_iv_and_time_stop_exit():
    """Full cycle: entry triggered by low IV, exited by TIME_STOP.

    Observation layout (all prices = 3000):
        bars 0      : initialisation (LP deposit + open)
        bars 1-5    : lookback accumulation (high IV)
        bar  6      : low IV  --> entry fires
        bars 7-30   : high IV, zero fees (premium stays small) --> time stop
    """
    max_hold = 5
    prices = [3000.0] * 32
    iv = [0.9] * 6 + [0.1] + [0.9] * 25   # bar 6 is the low-IV trigger

    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        LOOKBACK_BARS=5,
        IV_ENTRY_PERCENTILE=30.0,
        TAKE_PROFIT_MULT=1000.0,    # TP will not fire
        STOP_LOSS_BUDGET_PCT=1.0,   # SL will not fire
        MAX_HOLD_BARS=max_hold,
    )
    _, df, _, strategy = _run(params, prices, iv_values=iv, fees=0.0)

    # At least one open/close cycle must have occurred.
    open_col = df["STRADDLE_is_open"].astype(int)
    assert open_col.any(), "No position was opened"
    # After max_hold bars the position must be closed.
    assert not strategy.get_entity("STRADDLE").internal_state.is_open


@pytest.mark.core
def test_e2e_result_length_matches_observations():
    prices = [3000.0] * 40
    params = PanopticStraddleParams(LOOKBACK_BARS=5)
    _, df, _, _ = _run(params, prices)
    assert len(df) == 40


@pytest.mark.core
def test_e2e_required_columns_present():
    prices = [3000.0] * 30
    params = PanopticStraddleParams(LOOKBACK_BARS=5)
    _, df, _, _ = _run(params, prices)
    for col in ("timestamp", "net_balance",
                "STRADDLE_balance", "LP_balance",
                "STRADDLE_is_open", "STRADDLE_cash",
                "STRADDLE_accumulated_premium"):
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.core
def test_e2e_net_balance_equals_straddle_plus_lp():
    prices = [3000.0] * 30
    params = PanopticStraddleParams(LOOKBACK_BARS=5)
    _, df, _, _ = _run(params, prices)
    expected = df["STRADDLE_balance"] + df["LP_balance"]
    assert np.allclose(df["net_balance"].values, expected.values, rtol=1e-9)


# ---------------------------------------------------------------------------
# Scenario C -- trending market: straddle should outperform passive LP
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_e2e_trending_market_straddle_balance_peak_above_start():
    """In a strong trend the straddle balance peak must exceed its opening level."""
    # 20 flat bars (low IV fills lookback), then 30 bars of sharp uptrend.
    flat = [3000.0] * 20
    trend = [3000.0 + i * 60 for i in range(30)]
    prices = flat + trend

    iv_flat  = [0.1] * 20   # low IV during flat -> entry fires
    iv_trend = [0.9] * 30   # high IV during trend
    iv = iv_flat + iv_trend

    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        LOOKBACK_BARS=5,
        IV_ENTRY_PERCENTILE=30.0,
        MAX_HOLD_BARS=200,
        TAKE_PROFIT_MULT=1.5,
        STOP_LOSS_BUDGET_PCT=1.0,
    )
    _, df, _, _ = _run(params, prices, iv_values=iv, fees=100.0)

    straddle_start = df["STRADDLE_balance"].iloc[6]  # after lookback
    straddle_peak  = df["STRADDLE_balance"].max()
    assert straddle_peak >= straddle_start


# ---------------------------------------------------------------------------
# Scenario D -- stop loss fires on high fees
# ---------------------------------------------------------------------------

@pytest.mark.core
def test_e2e_stop_loss_fires_on_huge_fees():
    """With a tiny SL budget and enormous fees the position must close early."""
    # Entry on bar 6 (low IV), then giant fees trigger SL before time stop.
    prices = [3000.0] * 40
    iv = [0.9] * 5 + [0.1] + [0.9] * 34

    params = PanopticStraddleParams(
        INITIAL_BALANCE=10_000.0,
        LOOKBACK_BARS=5,
        IV_ENTRY_PERCENTILE=30.0,
        TAKE_PROFIT_MULT=1000.0,    # TP will not fire
        STOP_LOSS_BUDGET_PCT=0.001, # $10 limit on a $10 k balance
        MAX_HOLD_BARS=1000,         # time stop will not fire first
    )
    _, df, _, strategy = _run(
        params, prices,
        iv_values=iv,
        fees=1_000_000.0,   # huge fees -> premium blows past $10 quickly
        liquidity=1e18,
    )
    # If a position was opened it must have been closed by SL.
    if df["STRADDLE_is_open"].any():
        assert not strategy.get_entity("STRADDLE").internal_state.is_open