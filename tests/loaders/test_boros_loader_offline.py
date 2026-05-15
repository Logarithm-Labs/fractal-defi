"""Offline tests for ``BorosMarketLoader.transform()`` — no network.

Verify the bar-list parser, deduplication, malformed-row resilience,
window clipping and typed-struct return contract.
"""
from datetime import datetime, timezone

import pytest

from fractal.loaders.boros import BorosMarketLoader
from fractal.loaders.structs import BorosMarketHistory

UTC = timezone.utc

_START = datetime(2026, 4, 23, tzinfo=UTC)
_END = datetime(2026, 5, 14, tzinfo=UTC)


def _stub_loader() -> BorosMarketLoader:
    return BorosMarketLoader(market_id=74, start_time=_START, end_time=_END)


def _bar(ts: int, mark: float) -> dict:
    return {
        "ts": ts,
        "c": mark,
        "u": mark,
        "b7dmafr": mark,
        "b30dmafr": mark,
    }


@pytest.mark.core
def test_transform_empty_payload():
    loader = _stub_loader()
    loader._raw = {}
    loader.transform()
    assert isinstance(loader._data, BorosMarketHistory)
    assert loader._data.empty


@pytest.mark.core
def test_transform_returns_typed_struct():
    loader = _stub_loader()
    ts_inside = int(_START.timestamp()) + 3600
    loader._raw = {"results": [_bar(ts_inside, 0.03)]}
    loader.transform()
    assert isinstance(loader._data, BorosMarketHistory)
    assert list(loader._data.columns) == [
        "mark_apr",
        "observed_funding",
        "mark_apr_7d_ma",
        "mark_apr_30d_ma",
    ]
    assert len(loader._data) == 1


@pytest.mark.core
def test_transform_clips_to_requested_window():
    loader = _stub_loader()
    ts_before = int(_START.timestamp()) - 86400
    ts_inside = int(_START.timestamp()) + 86400
    ts_after = int(_END.timestamp()) + 86400
    loader._raw = {
        "results": [
            _bar(ts_before, 0.01),
            _bar(ts_inside, 0.02),
            _bar(ts_after, 0.03),
        ]
    }
    loader.transform()
    assert len(loader._data) == 1
    assert loader._data["mark_apr"].iloc[0] == pytest.approx(0.02)


@pytest.mark.core
def test_transform_deduplicates_same_timestamp():
    loader = _stub_loader()
    ts = int(_START.timestamp()) + 3600
    loader._raw = {"results": [_bar(ts, 0.02), _bar(ts, 0.99)]}
    loader.transform()
    assert len(loader._data) == 1
    # Keep first (0.02), drop second (0.99).
    assert loader._data["mark_apr"].iloc[0] == pytest.approx(0.02)


@pytest.mark.core
def test_transform_skips_malformed_rows():
    loader = _stub_loader()
    ts_ok = int(_START.timestamp()) + 3600
    loader._raw = {
        "results": [
            {"ts": "not-an-int", "c": 0.01},  # malformed ts
            _bar(ts_ok, 0.02),
        ]
    }
    loader.transform()
    assert len(loader._data) == 1
    assert loader._data["mark_apr"].iloc[0] == pytest.approx(0.02)


@pytest.mark.core
def test_cache_key_is_deterministic():
    a = _stub_loader()._cache_key()
    b = _stub_loader()._cache_key()
    assert a == b
    assert "market74" in a
