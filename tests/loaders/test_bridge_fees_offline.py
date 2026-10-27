"""Offline tests for the pure realised-fee helpers."""

import pytest

from fractal.loaders.bridges import comparable_input, fee_bps, fee_stats


@pytest.mark.core
def test_comparable_input_rescales_decimals_and_rejects_swaps():
    assert comparable_input(1_000_000, 999_000) == 1_000_000
    assert comparable_input(10**18, 999_000) == pytest.approx(1_000_000)
    assert comparable_input(1_000_000, 10**18) == pytest.approx(1e18)  # input in 18-decimal output units
    assert comparable_input(3_000_000, 1_000_000) is None  # 3x: a swap, not a fee
    assert comparable_input(0, 1) is None and comparable_input(1, 0) is None


@pytest.mark.core
def test_fee_bps_sign_cap_and_decimals():
    assert fee_bps(1_000_000, 999_000) == pytest.approx(10.0)
    assert fee_bps(1_000_000, 1_000_500) == pytest.approx(-5.0)  # rebate
    assert fee_bps(10**18, 999_000) == pytest.approx(10.0)  # 18 -> 6 decimals
    assert fee_bps(1_000_000, 900_000) is None  # 1000 bps exceeds the default cap
    assert fee_bps(1_000_000, 900_000, max_abs_bps=1500) == pytest.approx(1000.0)


@pytest.mark.core
def test_fee_stats_quartiles_and_volume_weighting():
    stats = fee_stats([(1.0, 100.0), (2.0, 100.0), (3.0, 100.0), (10.0, 700.0), (None, 1e9)])
    assert stats["n"] == 4
    assert stats["median_bps"] == 2.5
    assert stats["p25_bps"] == pytest.approx(1.75)
    assert stats["p75_bps"] == pytest.approx(4.75)
    assert stats["vw_mean_bps"] == pytest.approx((1 + 2 + 3) * 100 / 1000 + 10 * 0.7)
    assert fee_stats([(5.0, 0.0)])["vw_mean_bps"] is None
    assert fee_stats([]) == {"n": 0, "median_bps": None, "p25_bps": None, "p75_bps": None, "vw_mean_bps": None}
