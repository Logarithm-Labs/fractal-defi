"""Offline tests for the Pendle loaders: paging and params against a fake
``HttpClient``, the compounded PT price, the daily fallback for the
hourly retention cap, cache round trips, ``readState`` decoding."""
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from fractal.loaders import LoaderType, PendleMarketHistory, PendleMarketLoader, PendleOHLCVLoader
from fractal.loaders._dt import SECONDS_PER_YEAR
from fractal.loaders.pendle import PENDLE_API, PENDLE_ROUTER, get_market_info, read_market_state

MARKET = "0x47ad2cd1dd15739a7a035b9d3b7828d916fef77e"  # PT-sUSDe-26NOV2026
EXPIRY = datetime(2026, 11, 26, tzinfo=timezone.utc)
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


class _FakePendleHttp:
    """Serves synthetic rows for [start, end]; hourly rows only from ``hourly_from``."""

    def __init__(self, hourly_from: datetime, cap: int = 1000, ohlcv_csv: bool = False, ohlcv_cap: int = 1440):
        self.hourly_from = hourly_from
        self.cap = cap
        self.ohlcv_csv = ohlcv_csv
        self.ohlcv_cap = ohlcv_cap
        self.calls = []

    @staticmethod
    def _row(ts: datetime, apy: float = 0.05):
        return {"timestamp": _iso(ts), "impliedApy": apy, "underlyingApy": 0.04, "tvl": 1e6,
                "ptPrice": 0.99, "syPrice": 1.2, "totalPt": 1e6, "totalSy": 2.5e6}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url.endswith(f"/markets/{MARKET}") and "/v1/" in url:
            return {"name": "sUSDe", "expiry": _iso(EXPIRY), "pt": "1-0xPT", "yt": "1-0xYT", "sy": "1-0xSY",
                    "underlyingAsset": "1-0xUNDER", "accountingAsset": "1-0xACC"}
        start = pd.Timestamp(params["timestamp_start"]).to_pydatetime()
        end = pd.Timestamp(params["timestamp_end"]).to_pydatetime()
        if url.endswith("/historical-data"):
            step = HOUR if params["time_frame"] == "hour" else timedelta(days=1)
            first = max(start, self.hourly_from) if params["time_frame"] == "hour" else start
            rows, ts = [], first
            while ts <= end and len(rows) < self.cap:
                rows.append(self._row(ts))
                ts += step
            return {"total": len(rows), "results": rows}
        if url.endswith("/ohlcv"):
            rows, ts = [], start
            while ts <= end and len(rows) < self.ohlcv_cap:
                rows.append({"time": int(ts.timestamp()), "open": 0.98, "high": 0.99, "low": 0.97, "close": 0.985,
                             "volume": 10.0})
                ts += HOUR
            if self.ohlcv_csv:
                text = "time,open,high,low,close,volume\n" + "\n".join(
                    f"{r['time']},{r['open']},{r['high']},{r['low']},{r['close']},{r['volume']}" for r in rows)
                return {"results": text}
            return {"results": rows}
        raise AssertionError(f"unexpected url {url}")

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append((url, json))
        words = [int(1e6 * 1e18), int(2.5e6 * 1e18), 0, 0, int(57.60 * 1e18), int(EXPIRY.timestamp()),
                 int(9.8686e-4 * 1e18), 80, int(0.0480287 * 1e18)]
        return {"jsonrpc": "2.0", "id": 1, "result": "0x" + "".join(f"{w:064x}" for w in words)}


@pytest.mark.core
def test_market_info_parses_expiry_and_addresses():
    http = _FakePendleHttp(hourly_from=START)
    info = get_market_info(1, MARKET, http=http)
    assert info.expiry == EXPIRY and info.pt == "0xpt" and info.accounting_asset == "0xacc"
    assert http.calls[0][0] == f"{PENDLE_API}/v1/1/markets/{MARKET}"


@pytest.mark.core
def test_transform_derives_compounded_pt_price_and_term(monkeypatch):
    monkeypatch.setattr("fractal.loaders.pendle._time.sleep", lambda s: None)
    http = _FakePendleHttp(hourly_from=START)
    end = START + 5 * HOUR
    loader = PendleMarketLoader(MARKET, 1, START, end, expiry=EXPIRY, http=http)
    history = loader.read(with_run=True)
    assert isinstance(history, PendleMarketHistory)
    assert len(history) == 6 and history.index.name == "time" and str(history.index.tz) == "UTC"
    t = history["seconds_to_expiry"] / SECONDS_PER_YEAR
    assert np.allclose(history["pt_price_asset"], (1.05) ** (-t))
    assert history["seconds_to_expiry"].iloc[0] == (EXPIRY - START).total_seconds()
    assert history["seconds_to_expiry"].is_monotonic_decreasing
    assert np.allclose(history["pt_price_sy"], 0.99 / 1.2)
    url, params = http.calls[0]
    assert url == f"{PENDLE_API}/v3/1/markets/{MARKET}/historical-data"
    assert params["includeApyBreakdown"] == "true" and params["timestamp_start"] == _iso(START)


@pytest.mark.core
def test_paging_walks_forward_until_the_window_is_covered(monkeypatch):
    monkeypatch.setattr("fractal.loaders.pendle._time.sleep", lambda s: None)
    http = _FakePendleHttp(hourly_from=START, cap=4)
    loader = PendleMarketLoader(MARKET, 1, START, START + 9 * HOUR, expiry=EXPIRY, http=http)
    loader.extract()
    loader.transform()
    assert len(loader._data) == 10
    starts = [p["timestamp_start"] for _, p in http.calls if "historical-data" in _]
    assert starts == [_iso(START), _iso(START + 4 * HOUR), _iso(START + 8 * HOUR)]


@pytest.mark.core
def test_daily_fallback_stretches_the_prefix_and_warns(monkeypatch):
    monkeypatch.setattr("fractal.loaders.pendle._time.sleep", lambda s: None)
    hourly_from = START + timedelta(days=3)
    http = _FakePendleHttp(hourly_from=hourly_from)
    loader = PendleMarketLoader(MARKET, 1, START, hourly_from + 2 * HOUR, expiry=EXPIRY, http=http)
    with pytest.warns(UserWarning, match="stretched to hourly"):
        history = loader.read(with_run=True)
    expected_hours = int((hourly_from + 2 * HOUR - START) / HOUR) + 1
    assert len(history) == expected_hours
    assert (history.index[1:] - history.index[:-1] == pd.Timedelta(hours=1)).all()
    frames = [p["time_frame"] for u, p in http.calls if "historical-data" in u]
    assert frames == ["hour", "day"]
    strict = PendleMarketLoader(MARKET, 1, START, hourly_from + 2 * HOUR, expiry=EXPIRY, daily_fallback=False,
                                http=_FakePendleHttp(hourly_from=hourly_from))
    assert len(strict.read(with_run=True)) == 3
    assert strict._cache_key() != loader._cache_key()


@pytest.mark.core
def test_nan_in_required_columns_raises(monkeypatch):
    monkeypatch.setattr("fractal.loaders.pendle._time.sleep", lambda s: None)
    http = _FakePendleHttp(hourly_from=START)
    loader = PendleMarketLoader(MARKET, 1, START, START + HOUR, expiry=EXPIRY, http=http)
    loader.extract()
    loader._rows[0]["impliedApy"] = None
    with pytest.raises(ValueError, match="implied_apy"):
        loader.transform()


@pytest.mark.core
def test_cache_key_covers_expiry_and_frame():
    a = PendleMarketLoader(MARKET, 1, START, START + HOUR, expiry=EXPIRY, http=_FakePendleHttp(START))
    b = PendleMarketLoader(MARKET, 1, START, START + HOUR, expiry=EXPIRY + timedelta(days=30),
                           http=_FakePendleHttp(START))
    c = PendleMarketLoader(MARKET, 1, START, START + HOUR, expiry=EXPIRY, time_frame="day", http=_FakePendleHttp(START))
    assert len({a._cache_key(), b._cache_key(), c._cache_key()}) == 3
    with pytest.raises(ValueError):
        PendleMarketLoader(MARKET, 1, START + HOUR, START, expiry=EXPIRY)
    with pytest.raises(ValueError):
        PendleMarketLoader(MARKET, 1, START, START + HOUR, expiry=EXPIRY, time_frame="minute")


@pytest.mark.core
@pytest.mark.parametrize("loader_type", [LoaderType.CSV, LoaderType.JSON])
def test_cache_round_trip_returns_the_same_history(monkeypatch, tmp_path, loader_type):
    monkeypatch.setattr("fractal.loaders.pendle._time.sleep", lambda s: None)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    kwargs = dict(expiry=EXPIRY, loader_type=loader_type)
    fresh = PendleMarketLoader(MARKET, 1, START, START + 3 * HOUR, http=_FakePendleHttp(START), **kwargs)
    first = fresh.read(with_run=True)
    cached = PendleMarketLoader(MARKET, 1, START, START + 3 * HOUR, http=_FakePendleHttp(START), **kwargs)
    second = cached.read(with_run=False)
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.core
def test_read_market_state_decodes_readstate_words():
    http = _FakePendleHttp(hourly_from=START)
    state = read_market_state("http://rpc", MARKET, http=http)
    assert state.total_pt == pytest.approx(1e6) and state.total_sy == pytest.approx(2.5e6)
    assert state.scalar_root == pytest.approx(57.60) and state.ln_fee_rate_root == pytest.approx(9.8686e-4)
    assert state.last_ln_implied_rate == pytest.approx(0.0480287) and state.expiry == int(EXPIRY.timestamp())
    call = http.calls[-1][1]
    assert call["method"] == "eth_call" and call["params"][0]["to"] == MARKET
    assert call["params"][0]["data"].endswith(PENDLE_ROUTER.lower()[2:])


@pytest.mark.core
@pytest.mark.parametrize("csv", [False, True])
def test_ohlcv_parses_both_shapes_and_pages(monkeypatch, csv):
    monkeypatch.setattr("fractal.loaders.pendle._time.sleep", lambda s: None)
    http = _FakePendleHttp(hourly_from=START, ohlcv_csv=csv, ohlcv_cap=1440)
    end = START + timedelta(hours=1500)
    loader = PendleOHLCVLoader("0x" + "ab" * 20, 1, START, end, http=http)
    klines = loader.read(with_run=True)
    assert len(klines) == 1501
    assert list(klines.columns) == ["open", "high", "low", "close", "volume"]
    assert klines["close"].iloc[0] == 0.985
    ohlcv_calls = [p for u, p in http.calls if u.endswith("/ohlcv")]
    assert len(ohlcv_calls) == 2 and ohlcv_calls[0]["time_frame"] == "hour"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert loader._cache_key().startswith("1-0x")


@pytest.mark.core
def test_daily_fallback_covers_a_window_older_than_the_hourly_retention(monkeypatch):
    """An empty hourly response (window entirely older than ~60 days) must
    still produce a stretched daily history instead of an empty frame."""
    monkeypatch.setattr("fractal.loaders.pendle._time.sleep", lambda s: None)
    end = START + timedelta(days=3)
    http = _FakePendleHttp(hourly_from=end + timedelta(days=30))  # hourly retention starts after the window
    loader = PendleMarketLoader(MARKET, 1, START, end, expiry=EXPIRY, http=http)
    with pytest.warns(UserWarning, match="after the window"):
        history = loader.read(with_run=True)
    assert len(history) == 3 * 24 + 1
    assert history.index[0] == pd.Timestamp(START) and history.index[-1] == pd.Timestamp(end)
    frames = [p["time_frame"] for u, p in http.calls if "historical-data" in u]
    assert frames == ["hour", "day"]
