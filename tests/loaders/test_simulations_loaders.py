"""Tests for synthetic loaders (Monte-Carlo GBM/bootstrap + constant fundings).

These tests do **not** require network access — both loaders are pure
Python — but they live in ``tests/loaders/`` next to the real-API tests
so the suite is contiguous.
"""
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fractal.loaders import (ConstantFundingsLoader, FundingHistory,
                             MonteCarloHourPriceLoader,
                             MonteCarloPriceLoader, PriceHistory,
                             TrajectoryBundle)
from fractal.loaders.base_loader import LoaderType

UTC = timezone.utc


def _toy_price_history(n: int = 100, seed: int = 42, sigma: float = 0.01) -> PriceHistory:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0, sigma, n - 1)
    log_paths = np.concatenate([[0.0], np.cumsum(log_ret)])
    prices = 100.0 * np.exp(log_paths)
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz=UTC)
    return PriceHistory(prices=prices, time=times.values)


@pytest.mark.core
def test_monte_carlo_loader_returns_list_of_price_histories():
    history = _toy_price_history(n=200)
    loader = MonteCarloHourPriceLoader(history, trajectories_number=5, seed=42)
    out = loader.read(with_run=True)
    assert isinstance(out, list)
    assert len(out) == 5
    for traj in out:
        assert isinstance(traj, PriceHistory)
        assert len(traj) == len(history)
        assert traj.index.equals(history.index)
        assert traj["price"].dtype == "float64"


@pytest.mark.core
def test_monte_carlo_loader_is_reproducible_for_same_seed():
    history = _toy_price_history(n=120)
    a = MonteCarloHourPriceLoader(history, trajectories_number=3, seed=7).read(with_run=True)
    b = MonteCarloHourPriceLoader(history, trajectories_number=3, seed=7).read(with_run=True)
    for ta, tb in zip(a, b):
        assert np.allclose(ta["price"].values, tb["price"].values)


@pytest.mark.core
def test_monte_carlo_loader_distinct_seeds_produce_distinct_paths():
    history = _toy_price_history(n=120)
    a = MonteCarloHourPriceLoader(history, trajectories_number=1, seed=1).read(with_run=True)[0]
    b = MonteCarloHourPriceLoader(history, trajectories_number=1, seed=2).read(with_run=True)[0]
    assert not np.allclose(a["price"].values, b["price"].values)


@pytest.mark.core
def test_monte_carlo_loader_handles_empty_history():
    history = PriceHistory(prices=np.array([], dtype=float), time=np.array([], dtype="datetime64[ns]"))
    loader = MonteCarloHourPriceLoader(history, trajectories_number=2, seed=0)
    out = loader.read(with_run=True)
    assert out == []


@pytest.mark.core
def test_monte_carlo_loader_round_trip_via_disk():
    """``with_run=True`` writes a pickle; a fresh instance with the same
    inputs hits the same deterministic cache key and reads it back."""
    history = _toy_price_history(n=80)
    src = MonteCarloHourPriceLoader(history, trajectories_number=2, seed=99)
    written = src.read(with_run=True)
    # No `_file_id` hack: cache is deterministic in (history, params).
    fresh = MonteCarloHourPriceLoader(history, trajectories_number=2, seed=99)
    cached = fresh.read(with_run=False)
    for w, c in zip(written, cached):
        assert np.allclose(w["price"].values, c["price"].values)
    src.delete_dump_file()


@pytest.mark.core
def test_monte_carlo_loader_rejects_non_pickle_loader_type():
    history = _toy_price_history(n=10)
    with pytest.raises(ValueError):
        MonteCarloHourPriceLoader(history, loader_type=LoaderType.CSV)


@pytest.mark.core
def test_legacy_loader_emits_deprecation_warning():
    history = _toy_price_history(n=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MonteCarloHourPriceLoader(history, trajectories_number=1, seed=0)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


@pytest.mark.core
def test_gbm_first_value_equals_historical_first_price():
    history = _toy_price_history(n=50)
    s0 = float(history.iloc[0]["price"])
    loader = MonteCarloPriceLoader(history, trajectories_number=10, seed=1)
    out = loader.read(with_run=True)
    for traj in out:
        assert traj["price"].iloc[0] == pytest.approx(s0)


@pytest.mark.core
def test_gbm_trajectories_are_strictly_positive():
    """GBM in exp-form cannot produce non-positive prices, even with large σ."""
    history = _toy_price_history(n=200, sigma=0.1)  # absurdly volatile
    loader = MonteCarloPriceLoader(
        history, trajectories_number=20, sigma=0.5, seed=11,
    )
    out = loader.read(with_run=True)
    for traj in out:
        assert (traj["price"].values > 0).all()


@pytest.mark.core
def test_gbm_zero_sigma_keeps_price_constant():
    """σ = 0 collapses the random walk to a constant trajectory."""
    history = _toy_price_history(n=100)
    s0 = float(history.iloc[0]["price"])
    loader = MonteCarloPriceLoader(
        history, trajectories_number=3, sigma=0.0, mu=0.0, seed=0,
    )
    out = loader.read(with_run=True)
    for traj in out:
        assert np.allclose(traj["price"].values, s0)


@pytest.mark.core
def test_gbm_drift_free_median_is_close_to_s0_at_long_horizon():
    """Drift-free GBM has E[log(S_T/S_0)] = -σ²T/2, but the *median* of
    log(S_T/S_0) is 0 → median(S_T) ≈ S_0 across enough trajectories.
    """
    history = _toy_price_history(n=400, sigma=0.005)
    s0 = float(history.iloc[0]["price"])
    loader = MonteCarloPriceLoader(history, trajectories_number=2000, seed=2026)
    out = loader.read(with_run=True)
    finals = np.array([t["price"].iloc[-1] for t in out])
    # Loose: median should be within 5% of s0 (Monte-Carlo noise at n=2000).
    assert abs(np.median(finals) / s0 - 1.0) < 0.05


@pytest.mark.core
def test_gbm_drift_pushes_median_up():
    """A positive μ should drag the median above S_0 over many steps."""
    history = _toy_price_history(n=200, sigma=0.005)
    s0 = float(history.iloc[0]["price"])
    no_drift = MonteCarloPriceLoader(
        history, trajectories_number=500, mu=0.0, seed=3,
    ).read(with_run=True)
    with_drift = MonteCarloPriceLoader(
        history, trajectories_number=500, mu=0.001, seed=3,
    ).read(with_run=True)
    no_drift_finals = np.array([t["price"].iloc[-1] for t in no_drift])
    with_drift_finals = np.array([t["price"].iloc[-1] for t in with_drift])
    assert np.median(with_drift_finals) > np.median(no_drift_finals)
    assert np.median(with_drift_finals) > s0


@pytest.mark.core
def test_gbm_calibrated_sigma_matches_log_returns_std():
    history = _toy_price_history(n=500, sigma=0.02)
    expected = np.std(np.diff(np.log(history["price"].values)), ddof=1)
    loader = MonteCarloPriceLoader(history, trajectories_number=1, seed=0)
    loader.read(with_run=True)
    assert loader.calibrated_sigma == pytest.approx(expected, rel=1e-9)


@pytest.mark.core
def test_gbm_explicit_sigma_overrides_calibration():
    history = _toy_price_history(n=200, sigma=0.01)
    loader = MonteCarloPriceLoader(history, trajectories_number=1, sigma=0.123, seed=0)
    loader.read(with_run=True)
    assert loader.calibrated_sigma == 0.123


@pytest.mark.core
def test_bootstrap_mode_paths_are_strictly_positive():
    history = _toy_price_history(n=200, sigma=0.05)
    loader = MonteCarloPriceLoader(
        history, trajectories_number=10, mode="bootstrap", seed=4,
    )
    out = loader.read(with_run=True)
    for traj in out:
        assert (traj["price"].values > 0).all()


@pytest.mark.core
def test_bootstrap_mode_first_value_equals_s0():
    history = _toy_price_history(n=80, sigma=0.02)
    s0 = float(history.iloc[0]["price"])
    loader = MonteCarloPriceLoader(
        history, trajectories_number=5, mode="bootstrap", seed=4,
    )
    out = loader.read(with_run=True)
    for traj in out:
        assert traj["price"].iloc[0] == pytest.approx(s0)


@pytest.mark.core
def test_bootstrap_mode_with_constant_history_produces_constant_paths():
    """If every historical log-return is zero, bootstrap can only resample 0s."""
    n = 50
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz=UTC)
    history = PriceHistory(prices=np.full(n, 100.0), time=times.values)
    loader = MonteCarloPriceLoader(
        history, trajectories_number=3, mode="bootstrap", seed=0,
    )
    out = loader.read(with_run=True)
    for traj in out:
        assert np.allclose(traj["price"].values, 100.0)


@pytest.mark.core
def test_unknown_mode_raises():
    history = _toy_price_history(n=10)
    with pytest.raises(ValueError):
        MonteCarloPriceLoader(history, mode="garch")  # type: ignore[arg-type]


@pytest.mark.core
def test_trajectory_bundle_alias_is_list_of_price_history():
    """``TrajectoryBundle`` is the documented return type for simulators."""
    history = _toy_price_history(n=20)
    out: TrajectoryBundle = MonteCarloPriceLoader(
        history, trajectories_number=3, seed=0,
    ).read(with_run=True)
    assert isinstance(out, list)
    assert all(isinstance(t, PriceHistory) for t in out)


@pytest.mark.core
def test_cache_key_is_deterministic_for_same_inputs():
    history = _toy_price_history(n=50)
    a = MonteCarloPriceLoader(history, trajectories_number=4, mu=0.001, seed=7)
    b = MonteCarloPriceLoader(history, trajectories_number=4, mu=0.001, seed=7)
    assert a._cache_key() == b._cache_key()


@pytest.mark.core
def test_cache_key_changes_with_params():
    history = _toy_price_history(n=50)
    base = MonteCarloPriceLoader(history, trajectories_number=4, seed=7)._cache_key()
    assert MonteCarloPriceLoader(history, trajectories_number=5, seed=7)._cache_key() != base
    assert MonteCarloPriceLoader(history, trajectories_number=4, seed=8)._cache_key() != base
    assert MonteCarloPriceLoader(history, trajectories_number=4, seed=7, mu=0.01)._cache_key() != base
    assert MonteCarloPriceLoader(history, trajectories_number=4, seed=7, sigma=0.05)._cache_key() != base
    assert MonteCarloPriceLoader(history, trajectories_number=4, seed=7, mode="bootstrap")._cache_key() != base


@pytest.mark.core
def test_cache_key_changes_with_history_content():
    history_a = _toy_price_history(n=50, seed=1)
    history_b = _toy_price_history(n=50, seed=2)
    a = MonteCarloPriceLoader(history_a, trajectories_number=2, seed=0)._cache_key()
    b = MonteCarloPriceLoader(history_b, trajectories_number=2, seed=0)._cache_key()
    assert a != b


@pytest.mark.core
def test_cache_shared_across_instances():
    """Two instances built with the same inputs must read the same pickle
    without any private-field mutation."""
    history = _toy_price_history(n=40)
    src = MonteCarloPriceLoader(history, trajectories_number=2, seed=11)
    a = src.read(with_run=True)
    b = MonteCarloPriceLoader(history, trajectories_number=2, seed=11).read(with_run=False)
    for ta, tb in zip(a, b):
        assert np.allclose(ta["price"].values, tb["price"].values)
    src.delete_dump_file()


@pytest.mark.core
def test_window_slices_input_history_before_calibration():
    """``start_time``/``end_time`` slice the input history; the resulting
    trajectory length matches the windowed slice and starts at the windowed
    first price."""
    n = 200
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz=UTC)
    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    history = PriceHistory(prices=prices, time=times.values)

    start = times[50]
    end = times[150]
    expected_first = float(history.loc[history.index >= start].iloc[0]["price"])
    expected_len = int(((history.index >= start) & (history.index <= end)).sum())

    loader = MonteCarloPriceLoader(
        history, trajectories_number=3, seed=0,
        start_time=start.to_pydatetime(), end_time=end.to_pydatetime(),
    )
    out = loader.read(with_run=True)
    for traj in out:
        assert len(traj) == expected_len
        assert traj["price"].iloc[0] == pytest.approx(expected_first)
        assert traj.index.min() >= start
        assert traj.index.max() <= end


@pytest.mark.core
def test_window_outside_history_returns_empty():
    history = _toy_price_history(n=40)
    far_future_start = datetime(2099, 1, 1, tzinfo=UTC)
    far_future_end = datetime(2099, 1, 2, tzinfo=UTC)
    out = MonteCarloPriceLoader(
        history, trajectories_number=3, seed=0,
        start_time=far_future_start, end_time=far_future_end,
    ).read(with_run=True)
    assert out == []


@pytest.mark.core
def test_constant_fundings_loader_basic():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 2, 1, tzinfo=UTC)
    loader = ConstantFundingsLoader(rate=-0.0005, freq="D", start=start, end=end)
    data: FundingHistory = loader.read(with_run=True)
    assert isinstance(data, FundingHistory)
    assert (data["rate"] == -0.0005).all()
    assert data.index.tz is not None
    assert data.index.min() >= start
    assert data.index.max() <= end


@pytest.mark.core
def test_constant_fundings_loader_string_dates():
    loader = ConstantFundingsLoader(rate=0.001, freq="h", start="2024-01-01", end="2024-01-02")
    data = loader.read(with_run=True)
    assert len(data) == 25  # inclusive both endpoints with hourly freq
    assert (data["rate"] == 0.001).all()


@pytest.mark.core
def test_constant_fundings_loader_empty_when_end_before_start():
    loader = ConstantFundingsLoader(
        rate=0.0, freq="D",
        start=datetime(2024, 1, 10, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC),
    )
    data = loader.read(with_run=True)
    assert isinstance(data, FundingHistory)
    assert len(data) == 0
    assert list(data.columns) == ["rate"]


# --------------------------------------------- loader_type plumbing
@pytest.mark.core
def test_monte_carlo_loader_propagates_loader_type_to_base():
    """Regression: ``Loader.__init__(*args, loader_type=...)`` is keyword-only,
    so subclasses must pass it BY NAME. Earlier ``super().__init__(loader_type)``
    silently dropped it into ``*args`` and ``self.loader_type`` fell back to
    ``LoaderType.CSV`` — making MonteCarloHourPriceLoader.run() crash trying
    to write a list as CSV.
    """
    history = _toy_price_history(n=20)
    loader = MonteCarloPriceLoader(history, trajectories_number=2, seed=0)
    assert loader.loader_type == LoaderType.PICKLE


@pytest.mark.core
def test_loader_subclasses_keep_user_loader_type():
    """Cross-cutting check: a few loaders that historically had the same
    keyword-only-arg-passed-positionally bug. Default is ``CSV``, but the
    user's explicit choice must stick."""
    from fractal.loaders.aave import AaveV3ArbitrumLoader

    # Aave passes through loader_type. Default is CSV; we keep CSV here
    # so this stays purely offline (constructor doesn't hit the network).
    a = AaveV3ArbitrumLoader(
        asset_address="0x82af49447d8a07e3bd95bd0d56f35241523fbab1",
        loader_type=LoaderType.JSON,
    )
    assert a.loader_type == LoaderType.JSON
