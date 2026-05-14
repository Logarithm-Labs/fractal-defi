"""Offline tests for ``PendleMarketLoader.transform()`` — no network.

Verify the linear PT-pricing convention and the parallel-array payload
parsing against hand-crafted Pendle-style payloads.
"""
from datetime import datetime, timezone

import pytest

from fractal.loaders.pendle import PendleMarketLoader, _compute_pt_price_linear

UTC = timezone.utc

# Stub market for constructor; transform() does not consult these.
_MARKET = "0x" + "ab" * 20
_EXPIRY = 1_750_000_000  # mid-2025


def _loader_with_raw(payload: dict) -> PendleMarketLoader:
    loader = PendleMarketLoader(
        market_address=_MARKET,
        expiry_timestamp=_EXPIRY,
        start_time=datetime(2025, 1, 1, tzinfo=UTC),
        end_time=datetime(2025, 1, 2, tzinfo=UTC),
    )
    loader._raw = payload
    return loader


@pytest.mark.core
def test_compute_pt_price_linear_at_expiry_is_one():
    assert _compute_pt_price_linear(implied_yield=0.14, seconds_to_expiry=0) == 1.0
    assert _compute_pt_price_linear(implied_yield=0.14, seconds_to_expiry=-1) == 1.0


@pytest.mark.core
def test_compute_pt_price_linear_matches_formula():
    """``pt_price = 1 - r*tau``."""
    SPY = 365.25 * 24 * 3600.0
    pt = _compute_pt_price_linear(implied_yield=0.14, seconds_to_expiry=0.5 * SPY)
    assert pt == pytest.approx(1.0 - 0.14 * 0.5, abs=1e-12)


@pytest.mark.core
def test_compute_pt_price_clamps_to_unit_interval():
    """Extreme implied yields cannot push price outside [0, 1]."""
    SPY = 365.25 * 24 * 3600.0
    assert _compute_pt_price_linear(implied_yield=2.0, seconds_to_expiry=SPY) == 0.0
    assert _compute_pt_price_linear(implied_yield=-1.0, seconds_to_expiry=SPY) == 1.0


@pytest.mark.core
def test_transform_returns_expected_columns():
    SPY = 365.25 * 24 * 3600.0
    payload = {
        "timestamp": [_EXPIRY - int(0.5 * SPY), _EXPIRY - int(0.4 * SPY)],
        "impliedApy": ["0.14", "0.10"],
        "tvl": ["1000000", "1100000"],
    }
    loader = _loader_with_raw(payload)
    loader.transform()
    df = loader._data
    assert list(df.columns) == [
        "pt_price", "implied_yield", "seconds_to_expiry", "pool_liquidity",
    ]
    assert len(df) == 2
    assert df["implied_yield"].iloc[0] == pytest.approx(0.14)
    assert df["pool_liquidity"].iloc[0] == pytest.approx(1_000_000.0)


@pytest.mark.core
def test_transform_seconds_to_expiry_strictly_decreasing():
    SPY = 365.25 * 24 * 3600.0
    payload = {
        "timestamp": [
            _EXPIRY - int(0.5 * SPY),
            _EXPIRY - int(0.4 * SPY),
            _EXPIRY - int(0.3 * SPY),
        ],
        "impliedApy": ["0.10"] * 3,
        "tvl": ["1000000"] * 3,
    }
    loader = _loader_with_raw(payload)
    loader.transform()
    diffs = loader._data["seconds_to_expiry"].diff().dropna()
    assert (diffs < 0).all()


@pytest.mark.core
def test_transform_handles_empty_payload():
    payload = {"timestamp": [], "impliedApy": [], "tvl": []}
    loader = _loader_with_raw(payload)
    loader.transform()
    assert len(loader._data) == 0
    assert list(loader._data.columns) == [
        "pt_price", "implied_yield", "seconds_to_expiry", "pool_liquidity",
    ]


@pytest.mark.core
def test_transform_skips_malformed_rows():
    payload = {
        "timestamp": [1000, 2000, 3000],
        "impliedApy": ["0.10", None, "0.12"],
        "tvl": ["1000000", "1000000", "1000000"],
    }
    loader = _loader_with_raw(payload)
    loader.transform()
    # Middle row dropped (None implied yield).
    assert len(loader._data) == 2
