"""Offline tests for ``PendleOHLCVLoader`` — no network.

Verify the OHLCV bar parser against both response shapes Pendle has
shipped in production (top-level list and ``{"results": [...]}``
wrapper), the constructor validations (EVM address, chain id, time
frame, window order), the deterministic cache key, the typed
``KlinesHistory`` contract on both cache paths and the UTC-seconds
index conversion.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from fractal.loaders.pendle_ohlcv import (
    PendleOHLCVLoader,
    _extract_bars,
    _rebuild_from_cache,
)
from fractal.loaders.structs import KlinesHistory

UTC = timezone.utc

_STUB_TOKEN = "0x" + "ab" * 20
_STUB_START = datetime(2025, 7, 27, tzinfo=UTC)
_STUB_END = datetime(2025, 9, 25, tzinfo=UTC)


def _stub_loader(**kw) -> PendleOHLCVLoader:
    defaults = dict(
        token_address=_STUB_TOKEN,
        start_time=_STUB_START,
        end_time=_STUB_END,
    )
    defaults.update(kw)
    return PendleOHLCVLoader(**defaults)


def _bar(t: int, c: float) -> dict:
    return {
        "time": t,
        "open": c - 0.01,
        "high": c + 0.01,
        "low": c - 0.02,
        "close": c,
        "volume": 100.0,
    }


@pytest.mark.core
def test_constructor_rejects_invalid_token_address():
    with pytest.raises(Exception):
        PendleOHLCVLoader(
            token_address="not-an-address",
            start_time=_STUB_START,
            end_time=_STUB_END,
        )


@pytest.mark.core
def test_constructor_rejects_unsupported_chain():
    with pytest.raises(ValueError):
        PendleOHLCVLoader(
            token_address=_STUB_TOKEN,
            start_time=_STUB_START,
            end_time=_STUB_END,
            chain_id=10,
        )


@pytest.mark.core
def test_constructor_rejects_unknown_time_frame():
    with pytest.raises(ValueError):
        PendleOHLCVLoader(
            token_address=_STUB_TOKEN,
            start_time=_STUB_START,
            end_time=_STUB_END,
            time_frame="2h",
        )


@pytest.mark.core
def test_constructor_accepts_all_documented_time_frames():
    for tf in ("1m", "5m", "15m", "1h", "4h", "1d", "1w"):
        PendleOHLCVLoader(
            token_address=_STUB_TOKEN,
            start_time=_STUB_START,
            end_time=_STUB_END,
            time_frame=tf,
        )


@pytest.mark.core
def test_constructor_rejects_reversed_window():
    with pytest.raises(ValueError):
        PendleOHLCVLoader(
            token_address=_STUB_TOKEN,
            start_time=_STUB_END,
            end_time=_STUB_START,
        )


@pytest.mark.core
def test_extract_bars_handles_top_level_list():
    raw = [_bar(int(_STUB_START.timestamp()), 0.95)]
    assert len(_extract_bars(raw)) == 1


@pytest.mark.core
def test_extract_bars_handles_wrapped_dict():
    raw = {"results": [_bar(int(_STUB_START.timestamp()), 0.95)]}
    assert len(_extract_bars(raw)) == 1
    raw = {"data": [_bar(int(_STUB_START.timestamp()), 0.95)]}
    assert len(_extract_bars(raw)) == 1


@pytest.mark.core
def test_extract_bars_handles_empty_payload():
    assert _extract_bars(None) == []
    assert _extract_bars({}) == []
    assert _extract_bars([]) == []


@pytest.mark.core
def test_transform_returns_klines_history():
    loader = _stub_loader()
    t0 = int(_STUB_START.timestamp())
    loader._raw = [
        _bar(t0, 0.95),
        _bar(t0 + 3600, 0.955),
        _bar(t0 + 7200, 0.96),
    ]
    loader.transform()
    assert isinstance(loader._data, KlinesHistory)
    assert list(loader._data.columns) == ["open", "high", "low", "close", "volume"]
    assert len(loader._data) == 3


@pytest.mark.core
def test_transform_index_is_utc_seconds_aware():
    """Epoch seconds must produce 2025 timestamps, not nanosecond offsets."""
    loader = _stub_loader()
    loader._raw = [_bar(int(_STUB_START.timestamp()), 0.95)]
    loader.transform()
    assert loader._data.index[0].year == 2025
    assert loader._data.index.tz is not None


@pytest.mark.core
def test_transform_empty_payload():
    loader = _stub_loader()
    loader._raw = []
    loader.transform()
    assert isinstance(loader._data, KlinesHistory)
    assert loader._data.empty


@pytest.mark.core
def test_transform_skips_malformed_rows():
    loader = _stub_loader()
    t0 = int(_STUB_START.timestamp())
    loader._raw = [
        {"time": "not-a-number", "close": 0.95},
        _bar(t0, 0.96),
    ]
    loader.transform()
    assert len(loader._data) == 1
    assert loader._data["close"].iloc[0] == pytest.approx(0.96)


@pytest.mark.core
def test_cache_key_uses_full_epoch_seconds():
    a = PendleOHLCVLoader(
        token_address=_STUB_TOKEN,
        start_time=datetime(2025, 7, 27, 0, tzinfo=UTC),
        end_time=datetime(2025, 7, 27, 6, tzinfo=UTC),
    )
    b = PendleOHLCVLoader(
        token_address=_STUB_TOKEN,
        start_time=datetime(2025, 7, 27, 12, tzinfo=UTC),
        end_time=datetime(2025, 7, 27, 18, tzinfo=UTC),
    )
    assert a._cache_key() != b._cache_key()


@pytest.mark.core
def test_rebuild_from_cache_restores_typed_history():
    df = pd.DataFrame(
        {
            "open": [0.95, 0.96],
            "high": [0.96, 0.97],
            "low": [0.94, 0.95],
            "close": [0.955, 0.965],
            "volume": [100.0, 110.0],
        },
        index=pd.to_datetime(
            [int(_STUB_START.timestamp()), int(_STUB_START.timestamp()) + 3600],
            unit="s",
            utc=True,
        ),
    )
    df.index.name = "time"
    rebuilt = _rebuild_from_cache(df)
    assert isinstance(rebuilt, KlinesHistory)
    assert rebuilt.index.tz is not None
    assert len(rebuilt) == 2


@pytest.mark.core
def test_rebuild_from_cache_handles_empty_dataframe():
    rebuilt = _rebuild_from_cache(pd.DataFrame())
    assert isinstance(rebuilt, KlinesHistory)
    assert rebuilt.empty
