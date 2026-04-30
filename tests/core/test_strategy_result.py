"""Tests for :class:`StrategyResult` and :class:`StrategyMetrics`."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from fractal.core.base import GlobalState, InternalState
from fractal.core.base.strategy.result import StrategyMetrics, StrategyResult

UTC = timezone.utc


@dataclass
class _GS(GlobalState):
    price: float = 0.0


@dataclass
class _IS(InternalState):
    amount: float = 0.0


def _build_result(prices: list[float]) -> StrategyResult:
    """Toy result: single entity ``X`` with constant amount=1, balance ≡ price."""
    n = len(prices)
    timestamps = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)]
    internal_states = [{"X": _IS(amount=1.0)} for _ in range(n)]
    global_states = [{"X": _GS(price=p)} for p in prices]
    balances = [{"X": p} for p in prices]
    return StrategyResult(
        timestamps=timestamps,
        internal_states=internal_states,
        global_states=global_states,
        balances=balances,
    )


@pytest.mark.core
def test_to_dataframe_flattens_states_and_adds_net_balance():
    r = _build_result([100.0, 110.0, 120.0])
    df = r.to_dataframe()
    assert list(df.columns) >= ["timestamp", "X_amount", "X_price", "X_balance", "net_balance"]
    assert (df["X_balance"] == df["net_balance"]).all()  # only one entity → equal
    assert df["X_price"].tolist() == [100.0, 110.0, 120.0]


@pytest.mark.core
def test_get_metrics_accumulated_return_and_apy():
    r = _build_result([100.0, 110.0, 120.0])
    df = r.to_dataframe()
    m: StrategyMetrics = r.get_metrics(df)
    assert m.accumulated_return == pytest.approx(0.2)
    # 3 hourly samples → ~2 hours; apy = 0.2 / (2/(365*24)) — large, just sanity-check sign
    assert m.apy > 0


@pytest.mark.core
def test_get_metrics_max_drawdown_negative():
    r = _build_result([100.0, 110.0, 80.0, 90.0])
    df = r.to_dataframe()
    m = r.get_metrics(df)
    # Peak 110 → 80 → drawdown ~ -27.27%
    assert m.max_drawdown == pytest.approx(-30.0 / 110.0, rel=1e-6)


@pytest.mark.core
def test_get_metrics_zero_volatility_yields_zero_sharpe():
    r = _build_result([100.0, 100.0, 100.0])
    df = r.to_dataframe()
    m = r.get_metrics(df)
    assert m.sharpe == 0


@pytest.mark.core
def test_get_default_metrics_runs_to_dataframe_internally():
    r = _build_result([100.0, 105.0, 110.0])
    m = r.get_default_metrics()
    assert m.accumulated_return == pytest.approx(0.1)


@pytest.mark.core
def test_get_metrics_with_notional_price_string_normalizes_balance():
    """Pass a column name as ``notional_price`` → metrics computed on
    balance / that column."""
    r = _build_result([100.0, 110.0, 121.0])
    df = r.to_dataframe()
    # Net balance scaled by X_price → constant 1 → zero return / drawdown
    m = r.get_metrics(df, notional_price="X_price")
    assert m.accumulated_return == pytest.approx(0.0, abs=1e-12)
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-12)


@pytest.mark.core
def test_get_metrics_with_notional_price_float_scales_balance():
    r = _build_result([100.0, 110.0])
    df = r.to_dataframe()
    m = r.get_metrics(df, notional_price=10.0)
    # Both balances divided by 10 → same return ratio
    assert m.accumulated_return == pytest.approx(0.1, rel=1e-9)


@pytest.mark.core
def test_get_metrics_rejects_invalid_notional_price():
    r = _build_result([100.0, 110.0])
    df = r.to_dataframe()
    with pytest.raises(ValueError):
        r.get_metrics(df, notional_price=[1, 2, 3])  # type: ignore[arg-type]
