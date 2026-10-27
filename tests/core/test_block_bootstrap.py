"""Synthetic tests for the moving-block bootstrap (``fractal.core.base.strategy.bootstrap``)."""
import math

import numpy as np
import pandas as pd
import pytest

from fractal.core.base.strategy import (
    BootstrapCI,
    block_bootstrap,
    block_bootstrap_indices,
    percentile_ci,
    widest_block_ci,
)

pytestmark = pytest.mark.core


def _ar1(rng: np.random.Generator, n: int, phi: float, burn_in: int = 200) -> np.ndarray:
    shocks = rng.standard_normal(n + burn_in)
    path = np.empty_like(shocks)
    path[0] = shocks[0]
    for t in range(1, path.size):
        path[t] = phi * path[t - 1] + shocks[t]
    return path[burn_in:]


# ------------------------------------------------------------------ indices
@pytest.mark.parametrize("n,block", [(10, 3), (11, 4), (30, 7), (5, 1)])
def test_indices_are_contiguous_blocks_inside_the_series(n, block):
    idx = block_bootstrap_indices(n, block, n_resamples=50, seed=1)[block]
    assert idx.shape == (50, n)
    assert idx.dtype == np.int64
    assert idx.min() >= 0
    assert idx.max() <= n - 1
    for row in idx:
        for start in range(0, n, block):
            chunk = row[start:start + block]
            # Within a block the indices are consecutive and the block fits in the series.
            np.testing.assert_array_equal(chunk, chunk[0] + np.arange(chunk.size))
            assert chunk[0] <= n - block


def test_block_starts_cover_every_admissible_position():
    n, block = 12, 4
    idx = block_bootstrap_indices(n, block, n_resamples=4000, seed=3)[block]
    starts = idx[:, ::block]
    assert set(np.unique(starts)) == set(range(n - block + 1))


def test_block_length_one_is_iid():
    idx = block_bootstrap_indices(20, 1, n_resamples=2000, seed=5)[1]
    # Consecutive picks are independent: the next index is not the previous + 1 more
    # often than chance (1/20), unlike any block length > 1.
    follow = np.mean(idx[:, 1:] == idx[:, :-1] + 1)
    assert follow < 0.08


@pytest.mark.parametrize("block", [8, 9, 50])
def test_block_length_at_or_above_n_is_clamped_to_the_whole_series(block):
    idx = block_bootstrap_indices(8, block, n_resamples=20, seed=0)[block]
    np.testing.assert_array_equal(idx, np.tile(np.arange(8), (20, 1)))


def test_single_observation():
    idx = block_bootstrap_indices(1, (1, 3), n_resamples=5, seed=0)
    for rows in idx.values():
        np.testing.assert_array_equal(rows, np.zeros((5, 1), dtype=np.int64))


def test_common_random_numbers_across_block_sets():
    alone = block_bootstrap_indices(40, 7, n_resamples=100, seed=11)[7]
    together = block_bootstrap_indices(40, (3, 7, 14), n_resamples=100, seed=11)
    np.testing.assert_array_equal(alone, together[7])
    assert list(together) == [3, 7, 14]


def test_duplicate_block_lengths_are_dropped():
    idx = block_bootstrap_indices(10, (7, 3, 7), n_resamples=4, seed=0)
    assert list(idx) == [7, 3]


@pytest.mark.parametrize("bad", [0, -2])
def test_non_positive_block_length_rejected(bad):
    with pytest.raises(ValueError, match="block lengths"):
        block_bootstrap_indices(10, bad, n_resamples=4, seed=0)


@pytest.mark.parametrize("bad", [2.5, True, "3"])
def test_non_integer_block_length_rejected(bad):
    with pytest.raises(TypeError, match="integers"):
        block_bootstrap_indices(10, (bad,), n_resamples=4, seed=0)


def test_empty_block_lengths_rejected():
    with pytest.raises(ValueError, match="empty"):
        block_bootstrap_indices(10, (), n_resamples=4, seed=0)


@pytest.mark.parametrize("kwargs", [{"n": 0}, {"n": 5, "n_resamples": 0}, {"n": 5.0}])
def test_invalid_sizes_rejected(kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        block_bootstrap_indices(block_lengths=3, **kwargs)


# ------------------------------------------------------------------ determinism
def test_same_seed_same_result_and_different_seed_differs():
    x = _ar1(np.random.default_rng(0), 120, 0.4)
    first = block_bootstrap(x, n_resamples=300, seed=42)
    second = block_bootstrap(x, n_resamples=300, seed=42)
    other = block_bootstrap(x, n_resamples=300, seed=43)
    assert (first.lower, first.upper, first.block_length) == (second.lower, second.upper, second.block_length)
    for block, draws in first.replicates.items():
        np.testing.assert_array_equal(draws, second.replicates[block])
    assert (first.lower, first.upper) != (other.lower, other.upper)


def test_generator_seed_matches_integer_seed_and_is_advanced():
    x = np.arange(30, dtype=float)
    by_int = block_bootstrap(x, n_resamples=100, seed=9)
    rng = np.random.default_rng(9)
    by_gen = block_bootstrap(x, n_resamples=100, seed=rng)
    again = block_bootstrap(x, n_resamples=100, seed=rng)
    np.testing.assert_array_equal(by_int.replicates[7], by_gen.replicates[7])
    assert not np.array_equal(by_gen.replicates[7], again.replicates[7])


# ------------------------------------------------------------------ coverage
def test_ar1_mean_coverage_close_to_nominal_and_beats_iid():
    """Coverage of the true mean (0) of a stationary AR(1) with phi = 0.5.

    The i.i.d. bootstrap ignores the autocorrelation (variance of the mean
    understated about threefold) and undercovers; the widest block CI must
    sit near the nominal 95%.
    """
    rng = np.random.default_rng(2026)
    trials, hits_widest, hits_iid = 200, 0, 0
    for trial in range(trials):
        x = _ar1(rng, 250, 0.5)
        ci = block_bootstrap(x, block_lengths=(3, 7, 14), n_resamples=400, seed=trial)
        iid = block_bootstrap(x, block_lengths=1, n_resamples=400, seed=trial)
        hits_widest += ci.contains(0.0)
        hits_iid += iid.contains(0.0)
        assert ci.width == pytest.approx(max(hi - lo for lo, hi in ci.per_block.values()))
    assert 0.88 <= hits_widest / trials <= 0.99
    assert hits_iid / trials <= 0.86
    assert hits_widest - hits_iid >= 0.08 * trials


# ------------------------------------------------------------------ statistic / data handling
def test_mean_fast_path_matches_generic_statistic():
    x = _ar1(np.random.default_rng(1), 60, 0.3)
    fast = block_bootstrap(x, n_resamples=200, seed=4)
    slow = block_bootstrap(x, statistic=lambda v: float(np.mean(v)), n_resamples=200, seed=4)
    assert fast.estimate == pytest.approx(x.mean())
    for block, draws in fast.replicates.items():
        np.testing.assert_allclose(draws, slow.replicates[block])
    assert (fast.lower, fast.upper) == pytest.approx((slow.lower, slow.upper))


def test_replicates_are_means_of_the_resampled_rows():
    x = np.arange(10, dtype=float) ** 2
    ci = block_bootstrap(x, block_lengths=3, n_resamples=50, seed=8)
    idx = block_bootstrap_indices(10, 3, n_resamples=50, seed=8)[3]
    np.testing.assert_allclose(ci.replicates[3], x[idx].mean(axis=1))


def test_dataframe_columns_are_resampled_jointly():
    rng = np.random.default_rng(6)
    a = _ar1(rng, 80, 0.5)
    frame = pd.DataFrame({"a": a, "b": 2.0 * a + 1.0})

    def gap(values: np.ndarray) -> float:
        return float(values[:, 1].mean() - 2.0 * values[:, 0].mean() - 1.0)

    ci = block_bootstrap(frame, statistic=gap, n_resamples=200, seed=1)
    for draws in ci.replicates.values():
        np.testing.assert_allclose(draws, 0.0, atol=1e-12)
    assert ci.lower == pytest.approx(0.0, abs=1e-12)
    assert ci.upper == pytest.approx(0.0, abs=1e-12)


def test_separate_calls_with_one_seed_stay_jointly_resampled():
    a = _ar1(np.random.default_rng(2), 90, 0.5)
    b = _ar1(np.random.default_rng(3), 90, 0.5)
    joint = block_bootstrap(np.column_stack([a, b]), statistic=lambda v: float(v[:, 0].mean() - v[:, 1].mean()),
                            n_resamples=200, seed=5)
    left = block_bootstrap(a, n_resamples=200, seed=5)
    right = block_bootstrap(b, n_resamples=200, seed=5)
    for block, draws in joint.replicates.items():
        np.testing.assert_allclose(draws, left.replicates[block] - right.replicates[block])


def test_series_is_sorted_by_index_and_nan_rows_dropped():
    days = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    x = pd.Series(_ar1(np.random.default_rng(4), 40, 0.5), index=days)
    shuffled = x.sample(frac=1.0, random_state=0)
    with_gap = pd.concat([x, pd.Series([np.nan], index=[days[-1] + pd.Timedelta(days=1)])])
    base = block_bootstrap(x, n_resamples=100, seed=2)
    for variant in (shuffled, with_gap):
        other = block_bootstrap(variant, n_resamples=100, seed=2)
        np.testing.assert_array_equal(base.replicates[7], other.replicates[7])


def test_dataframe_row_with_any_nan_is_dropped():
    frame = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, np.nan, 3.0, 4.0]})
    ci = block_bootstrap(frame, statistic=lambda v: float(v.shape[0]), block_lengths=1, n_resamples=10, seed=0)
    assert ci.estimate == 2.0
    np.testing.assert_array_equal(ci.replicates[1], 2.0)


def test_block_length_above_n_gives_a_zero_width_interval():
    x = np.array([1.0, 5.0, 2.0])
    ci = block_bootstrap(x, block_lengths=10, n_resamples=20, seed=0)
    assert ci.lower == ci.upper == pytest.approx(x.mean())


def test_input_errors():
    with pytest.raises(ValueError, match="explicit statistic"):
        block_bootstrap(np.ones((5, 2)), seed=0)
    with pytest.raises(ValueError, match="no complete"):
        block_bootstrap(np.array([np.nan, np.nan]), seed=0)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        block_bootstrap(np.ones((2, 2, 2)), statistic=np.sum, seed=0)
    with pytest.raises(ValueError, match="scalar"):
        block_bootstrap(np.ones((5, 2)), statistic=lambda v: v.mean(axis=0), seed=0)
    with pytest.raises(ValueError, match="level"):
        block_bootstrap(np.ones(5), level=1.0, seed=0)


# ------------------------------------------------------------------ CI helpers
def test_percentile_ci_drops_non_finite_and_handles_empty():
    draws = np.r_[np.arange(101, dtype=float), np.nan, np.inf]
    assert percentile_ci(draws, 0.9) == pytest.approx((5.0, 95.0))
    lower, upper = percentile_ci([np.nan, -np.inf], 0.9)
    assert math.isnan(lower)
    assert math.isnan(upper)


def test_widest_block_ci_picks_widest_with_first_on_ties():
    narrow = np.linspace(-1.0, 1.0, 201)
    wide = np.linspace(-3.0, 3.0, 201)
    ci = widest_block_ci({3: narrow, 7: wide, 14: wide.copy()}, level=0.9, estimate=0.25)
    assert isinstance(ci, BootstrapCI)
    assert ci.block_length == 7
    assert (ci.lower, ci.upper) == pytest.approx((-2.7, 2.7))
    assert ci.per_block[3] == pytest.approx((-0.9, 0.9))
    assert ci.estimate == 0.25
    assert ci.n_resamples == 201
    assert ci.contains(2.7)
    assert not ci.contains(2.71)


def test_widest_block_ci_prefers_an_undefined_interval():
    ci = widest_block_ci({3: np.linspace(-1.0, 1.0, 50), 7: np.full(50, np.nan)})
    assert ci.block_length == 7
    assert math.isnan(ci.lower)
    assert not ci.contains(0.0)


def test_widest_block_ci_rejects_empty_mapping():
    with pytest.raises(ValueError, match="empty"):
        widest_block_ci({})
