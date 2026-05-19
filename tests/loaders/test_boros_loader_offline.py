"""Offline tests for ``BorosMarketLoader`` — no network.

Cover the bar-list parser, deduplication, malformed-row resilience,
window clipping, time-frame validation, the typed-struct contract on
both cache paths, and the UTC-seconds index conversion.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from fractal.loaders.boros import BorosMarketLoader, _rebuild_from_cache
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
def test_constructor_rejects_unknown_time_frame():
    with pytest.raises(ValueError):
        BorosMarketLoader(
            market_id=74,
            start_time=_START,
            end_time=_END,
            time_frame="1m",  # not in {5m, 1h, 1d, 1w}
        )


@pytest.mark.core
def test_constructor_accepts_documented_time_frames():
    for tf in ("5m", "1h", "1d", "1w"):
        BorosMarketLoader(
            market_id=74,
            start_time=_START,
            end_time=_END,
            time_frame=tf,
        )


@pytest.mark.core
def test_constructor_rejects_reversed_window():
    with pytest.raises(ValueError):
        BorosMarketLoader(market_id=74, start_time=_END, end_time=_START)


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
def test_transform_index_is_utc_seconds_aware():
    """Epoch seconds must land in the 2026 timeline, not near 1970."""
    loader = _stub_loader()
    ts_inside = int(_START.timestamp()) + 3600
    loader._raw = {"results": [_bar(ts_inside, 0.03)]}
    loader.transform()
    assert loader._data.index[0].year == 2026
    assert loader._data.index.tz is not None


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
    assert loader._data["mark_apr"].iloc[0] == pytest.approx(0.02)


@pytest.mark.core
def test_transform_skips_malformed_rows():
    loader = _stub_loader()
    ts_ok = int(_START.timestamp()) + 3600
    loader._raw = {
        "results": [
            {"ts": "not-an-int", "c": 0.01},
            _bar(ts_ok, 0.02),
        ]
    }
    loader.transform()
    assert len(loader._data) == 1
    assert loader._data["mark_apr"].iloc[0] == pytest.approx(0.02)


@pytest.mark.core
def test_cache_key_uses_full_epoch_seconds():
    """Different start times → different cache keys (no day-level collisions)."""
    a = BorosMarketLoader(
        market_id=74,
        start_time=datetime(2026, 4, 23, 0, tzinfo=UTC),
        end_time=datetime(2026, 4, 23, 6, tzinfo=UTC),
    )
    b = BorosMarketLoader(
        market_id=74,
        start_time=datetime(2026, 4, 23, 12, tzinfo=UTC),
        end_time=datetime(2026, 4, 23, 18, tzinfo=UTC),
    )
    assert a._cache_key() != b._cache_key()
    assert "market74" in a._cache_key()


@pytest.mark.core
def test_rebuild_from_cache_restores_typed_history():
    df = pd.DataFrame(
        {
            "mark_apr": [0.03, 0.035],
            "observed_funding": [0.03, 0.035],
            "mark_apr_7d_ma": [0.029, 0.03],
            "mark_apr_30d_ma": [0.028, 0.029],
        },
        index=pd.to_datetime(
            [int(_START.timestamp()), int(_START.timestamp()) + 3600],
            unit="s",
            utc=True,
        ),
    )
    df.index.name = "time"
    rebuilt = _rebuild_from_cache(df)
    assert isinstance(rebuilt, BorosMarketHistory)
    assert rebuilt.index.tz is not None
    assert len(rebuilt) == 2


@pytest.mark.core
def test_rebuild_from_cache_handles_empty_dataframe():
    rebuilt = _rebuild_from_cache(pd.DataFrame())
    assert isinstance(rebuilt, BorosMarketHistory)
    assert rebuilt.empty
