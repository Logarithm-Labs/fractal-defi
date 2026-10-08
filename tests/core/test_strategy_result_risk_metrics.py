"""Hand-computed tests for the return-series risk metrics added by issue #68.

All expected values are derived from explicit plain-Python formulas here —
never by calling the code under test — so a regression in ``get_metrics``
cannot silently rewrite its own expectations.
"""
import math
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pytest

from fractal.core.base.strategy.result import StrategyMetrics, StrategyResult

UTC = timezone.utc
HOURS_PER_YEAR = 365 * 24


def _balances_from_returns(returns: List[float], initial: float = 100.0) -> List[float]:
    balances, balance = [initial], initial
    for r in returns:
        balance *= 1 + r
        balances.append(balance)
    return balances


def _result_from_balances(balances: List[float], step_hours: int = 1) -> StrategyResult:
    """Single-entity result whose ``net_balance`` path is exactly ``balances``.

    Balance is produced by a constant ``amount=1`` marked at the balance
    price, mirroring the helper in ``test_strategy_result.py``.
    """
    timestamps = [
        datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i * step_hours)
        for i in range(len(balances))
    ]
    return StrategyResult(
        timestamps=timestamps,
        internal_states=[{"X": None}] * len(balances),
        global_states=[{"X": None}] * len(balances),
        balances=[{"X": b} for b in balances],
    )


def _metrics_for(returns: List[float], step_hours: int = 1) -> StrategyMetrics:
    result = _result_from_balances(_balances_from_returns(returns), step_hours)
    return result.get_metrics(result.to_dataframe())


# ------------------------------------------------------------------ helpers
def _expected_sortino(returns: List[float], step_hours: int = 1) -> float:
    downside = [min(r, 0.0) for r in returns]
    downside_std = math.sqrt(sum(d * d for d in downside) / len(downside))
    if downside_std == 0:
        return 0.0
    # Mirror ``sharpe``'s annualization convention exactly: frequency is
    # ``len(data) / years`` — one more bar than the number of returns.
    bars = len(returns) + 1
    span_hours = len(returns) * step_hours
    frequency = bars * HOURS_PER_YEAR / span_hours
    return (sum(returns) / len(returns)) / downside_std * math.sqrt(frequency)


def _expected_var_cvar(returns: List[float]):
    values = np.sort(np.array([r for r in returns if math.isfinite(r)]))
    q05 = float(np.quantile(values, 0.05))
    tail_size = max(1, math.ceil(0.05 * values.size))
    return max(0.0, -q05), max(0.0, -float(values[:tail_size].mean()))


def _expected_omega(returns: List[float]) -> float:
    gains = sum(max(r, 0.0) for r in returns)
    losses = sum(max(-r, 0.0) for r in returns)
    return gains / losses if losses > 0 else 0.0


# -------------------------------------------------------------------- tests
@pytest.mark.core
def test_new_metrics_default_to_zero_and_construction_stays_backward_compatible():
    m = StrategyMetrics(accumulated_return=0.1, apy=0.2, sharpe=1.0, max_drawdown=-0.3)
    assert m.cagr == 0.0
    assert m.sortino == 0.0
    assert m.calmar == 0.0
    assert m.var_95 == 0.0
    assert m.cvar_95 == 0.0
    assert m.omega_ratio == 0.0
    assert m.time_in_drawdown == 0.0


@pytest.mark.core
def test_zero_metrics_includes_new_fields():
    r = _result_from_balances([100.0])
    m = r.get_metrics(r.to_dataframe())
    assert m == StrategyMetrics(
        accumulated_return=0.0, apy=0.0, sharpe=0.0, max_drawdown=0.0, cagr=0.0,
        sortino=0.0, calmar=0.0, var_95=0.0, cvar_95=0.0,
        omega_ratio=0.0, time_in_drawdown=0.0,
    )


@pytest.mark.core
def test_flat_series_yields_all_zero_risk_metrics():
    m = _metrics_for([0.0, 0.0, 0.0, 0.0])
    assert m.sortino == 0.0  # no downside deviation
    assert m.calmar == 0.0   # no drawdown
    assert m.var_95 == 0.0 and m.cvar_95 == 0.0
    assert m.omega_ratio == 0.0  # zero denominator (no losses)
    assert m.time_in_drawdown == 0.0


@pytest.mark.core
def test_all_positive_series_has_zero_downside_and_drawdown_metrics():
    m = _metrics_for([0.01, 0.02, 0.015])
    assert m.sortino == 0.0
    assert m.calmar == 0.0
    assert m.var_95 == 0.0 and m.cvar_95 == 0.0
    assert m.omega_ratio == 0.0  # no losing bars → finite 0.0 policy
    assert m.time_in_drawdown == 0.0


@pytest.mark.core
def test_loss_only_series_matches_hand_computed_values():
    returns = [-0.10, -0.05, -0.20]
    m = _metrics_for(returns)
    assert m.sortino == pytest.approx(_expected_sortino(returns))
    # Monotonic decline: every bar except the first is below the peak.
    assert m.time_in_drawdown == pytest.approx(len(returns) / (len(returns) + 1))
    assert m.max_drawdown < 0
    assert m.calmar == pytest.approx(m.apy / abs(m.max_drawdown))
    var_95, cvar_95 = _expected_var_cvar(returns)
    assert m.var_95 == pytest.approx(var_95)
    assert m.cvar_95 == pytest.approx(cvar_95)
    assert m.omega_ratio == 0.0  # no gains


@pytest.mark.core
def test_mixed_series_matches_hand_computed_values():
    returns = [0.10, -0.05, 0.03, -0.02]
    m = _metrics_for(returns)
    assert m.sortino == pytest.approx(_expected_sortino(returns))
    var_95, cvar_95 = _expected_var_cvar(returns)
    # quantile interpolation: sorted tail [-0.05, -0.02]; q05 = -0.0455
    assert m.var_95 == pytest.approx(0.0455)
    assert m.cvar_95 == pytest.approx(0.05)
    assert (var_95, cvar_95) == pytest.approx((0.0455, 0.05))
    assert m.omega_ratio == pytest.approx(_expected_omega(returns))
    assert m.omega_ratio == pytest.approx(0.13 / 0.07)
    # bars strictly below the running peak: bars 2, 3, 4 of 5
    assert m.time_in_drawdown == pytest.approx(3 / 5)
    assert m.max_drawdown == pytest.approx(-0.05)
    assert m.calmar == pytest.approx(m.apy / 0.05)


@pytest.mark.core
def test_time_in_drawdown_recounts_after_new_peak():
    # dip then full recovery to a new high: the recovered bar is not in DD
    m = _metrics_for([0.01, -0.10, 0.15])
    assert m.time_in_drawdown == pytest.approx(1 / 4)


@pytest.mark.core
def test_annualization_scales_with_timestamp_spacing():
    returns = [-0.02, 0.01, -0.03, 0.02]
    hourly = _metrics_for(returns, step_hours=1)
    daily = _metrics_for(returns, step_hours=24)
    assert daily.sortino == pytest.approx(hourly.sortino / math.sqrt(24))
    # sharpe annualization must scale identically — same convention.
    assert daily.sortino * math.sqrt(24) == pytest.approx(hourly.sortino)


@pytest.mark.core
def test_irregular_timestamps_use_real_elapsed_time_for_calmar():
    # Same returns, but the last gap spans a whole extra year.
    returns = [-0.10, 0.05, 0.06]
    balances = _balances_from_returns(returns)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start, start + timedelta(hours=1), start + timedelta(hours=2),
                  start + timedelta(days=365, hours=3)]
    result = StrategyResult(
        timestamps=timestamps,
        internal_states=[{"X": None}] * 4,
        global_states=[{"X": None}] * 4,
        balances=[{"X": b} for b in balances],
    )
    m = result.get_metrics(result.to_dataframe())
    assert m.calmar == pytest.approx(m.apy / abs(m.max_drawdown))
    # apy (linear, time-based) must be far smaller than for a 3-hour span.
    assert m.apy == pytest.approx(
        m.accumulated_return / ((timedelta(days=365, hours=3)).total_seconds() / (365 * 24 * 3600))
    )


@pytest.mark.core
def test_metrics_dict_exposes_new_fields_for_mlflow_logging():
    m = _metrics_for([0.01, -0.01])
    for key in ("sortino", "calmar", "var_95", "cvar_95", "omega_ratio", "time_in_drawdown"):
        assert key in m.__dict__  # mlflow.log_metrics(metrics.__dict__) picks these up
    assert all(isinstance(m.__dict__[key], float) and math.isfinite(m.__dict__[key]) for key in
               ("sortino", "calmar", "var_95", "cvar_95", "omega_ratio", "time_in_drawdown"))


@pytest.mark.core
def test_expected_shortfall_averages_only_the_bounded_five_percent_tail():
    """One large loss plus 22 tiny gains (ties at the quantile).

    Selecting every bar ``<= q05`` would average nearly the whole sample
    (~0.02) instead of the worst-5% tail; the expected shortfall must stay
    bounded to ``ceil(0.05 * n)`` observations.
    """
    returns = [-0.50] + [0.0001] * 22
    m = _metrics_for(returns)
    var_95, cvar_95 = _expected_var_cvar(returns)
    assert m.var_95 == pytest.approx(var_95)  # 5% quantile is a (tiny) gain -> 0.0
    assert m.cvar_95 == pytest.approx(cvar_95)
    assert m.cvar_95 == pytest.approx((0.50 - 0.0001) / 2)
    # Sanity: the tail must not be diluted by the flat majority of the sample.
    assert m.cvar_95 > 10 * abs(np.mean(returns))
    assert m.cvar_95 >= m.var_95


@pytest.mark.core
def test_zero_balance_transition_keeps_ratio_metrics_finite():
    """A balance wiped to exactly 0 and recapitalised makes ``pct_change`` emit
    ``inf``; sortino/omega_ratio must stay finite like the pre-existing sharpe."""
    result = _result_from_balances([100.0, 0.0, 50.0])
    m = result.get_metrics(result.to_dataframe())
    for key in ("sortino", "calmar", "var_95", "cvar_95", "omega_ratio", "time_in_drawdown"):
        assert math.isfinite(m.__dict__[key]), f"{key} is not finite: {m.__dict__[key]}"
    assert m.sharpe == 0.0  # unchanged pre-existing behaviour
