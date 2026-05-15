"""Offline tests for ``PendleMarketLoader.transform()`` — no network.

Verify the linear PT-pricing convention and the parallel-array payload
parsing against hand-crafted Pendle-style payloads.
"""
from datetime import datetime, timezone

import pytest

from fractal.loaders.pendle import PendleMarketLoader, _compute_pt_price_linear

UTC = timezone.utc

# Stub market for constructor; transform() does not consult these.
_STUB_MARKET = "0xa36b60a14a1a5247912584768c6e53e1a269a9f7"
_STUB_EXPIRY = 1758758400  # 25 Sep 2025
_STUB_START = datetime(2025, 7, 27, tzinfo=UTC)
_STUB_END = datetime(2025, 9, 25, tzinfo=UTC)


def _stub_loader() -> PendleMarketLoader:
    return PendleMarketLoader(
        market_address=_STUB_MARKET,
        expiry_timestamp=_STUB_EXPIRY,
        start_time=_STUB_START,
        end_time=_STUB_END,
    )


@pytest.mark.core
def test_compute_pt_price_linear_at_expiry_is_one():
    assert _compute_pt_price_linear(0.14, 0.0) == 1.0
    assert _compute_pt_price_linear(0.14, -100.0) == 1.0


@pytest.mark.core
def test_compute_pt_price_linear_matches_formula():
    secs = 90 * 24 * 3600
    tau = secs / (365.25 * 24 * 3600)
    expected = 1.0 - 0.10 * tau
    assert abs(_compute_pt_price_linear(0.10, secs) - expected) < 1e-12


@pytest.mark.core
def test_compute_pt_price_clamps_to_unit_interval():
    five_years = 5 * 365.25 * 24 * 3600
    assert _compute_pt_price_linear(1.0, five_years) == 0.0
    assert _compute_pt_price_linear(-0.10, 30 * 24 * 3600) == 1.0


@pytest.mark.core
def test_transform_returns_expected_columns():
    loader = _stub_loader()
    loader._raw = {
        "timestamp": [_STUB_EXPIRY - 86400 * i for i in range(3, 0, -1)],
        "impliedApy": [0.12, 0.13, 0.14],
        "tvl": [1_000_000.0, 1_100_000.0, 1_050_000.0],
    }
    loader.transform()
    cols = set(loader._data.columns)
    assert cols == {"pt_price", "implied_yield", "seconds_to_expiry", "pool_liquidity"}


@pytest.mark.core
def test_transform_seconds_to_expiry_strictly_decreasing():
    loader = _stub_loader()
    loader._raw = {
        "timestamp": [_STUB_EXPIRY - 86400 * i for i in range(3, 0, -1)],
        "impliedApy": [0.12, 0.13, 0.14],
        "tvl": [1_000_000.0, 1_100_000.0, 1_050_000.0],
    }
    loader.transform()
    secs = loader._data["seconds_to_expiry"].to_numpy()
    assert (secs[:-1] > secs[1:]).all()


@pytest.mark.core
def test_transform_handles_empty_payload():
    loader = _stub_loader()
    loader._raw = {}
    loader.transform()
    assert loader._data.empty


@pytest.mark.core
def test_transform_skips_malformed_rows():
    loader = _stub_loader()
    loader._raw = {
        "timestamp": [_STUB_EXPIRY - 7200, "not-a-number", _STUB_EXPIRY - 3600],
        "impliedApy": [0.10, 0.12, 0.11],
        "tvl": [1_000.0, 1_100.0, 1_050.0],
    }
    loader.transform()
    # 3 input rows, 1 malformed; 2 valid output rows.
    assert len(loader._data) == 2
