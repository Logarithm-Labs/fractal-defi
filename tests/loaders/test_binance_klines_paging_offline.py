"""Offline lock-in for Binance kline pagination across short pages."""
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from fractal.loaders.binance import BinanceHourPriceLoader, BinanceSpotPriceLoader

pytestmark = pytest.mark.core

HOUR_MS = 3_600_000
START = datetime(2025, 1, 1, tzinfo=timezone.utc)
START_MS = int(START.timestamp() * 1000)
N_HOURS = 2_500
LISTED_AT = 30  # first candle: the symbol lists 30 h after the requested start
OUTAGE = range(1_100, 1_105)  # a 5 h gap inside the second page
OPENS = [START_MS + i * HOUR_MS for i in range(LISTED_AT, N_HOURS) if i not in OUTAGE]


class _FakeBinanceHttp:
    """Serves synthetic hourly klines the way Binance filters them: by open time."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get(self, section: str, path: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        del section, path
        params = params or {}
        self.calls.append(params)
        opens = [t for t in OPENS if params["startTime"] <= t <= params["endTime"]]
        return [[t, "1", "1", "1", "1.5", "2"] for t in opens[: params["limit"]]]


@pytest.mark.parametrize("start_offset", [timedelta(0), timedelta(minutes=30)])
def test_get_klines_walks_past_a_late_listing_and_a_mid_window_outage(start_offset):
    http = _FakeBinanceHttp()
    start = START + start_offset
    loader = BinanceHourPriceLoader(
        "BTCUSDT", start_time=start, end_time=START + timedelta(hours=N_HOURS), http=http
    )
    rows = loader.get_klines()
    expected = [t for t in OPENS if t >= start.timestamp() * 1000]
    assert [int(r["openTime"].timestamp() * 1000) for r in rows] == expected
    assert len(http.calls) == 3  # ceil(2,500 h / 1,000 candles), no early stop


def test_read_returns_every_candle_after_short_pages(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    loader = BinanceSpotPriceLoader(
        "BTCUSDT",
        interval="1h",
        start_time=START,
        end_time=START + timedelta(hours=N_HOURS),
        http=_FakeBinanceHttp(),
    )
    history = loader.read(with_run=True)
    assert len(history) == len(OPENS)
    assert pd.DatetimeIndex(history.index).equals(pd.to_datetime(OPENS, unit="ms", utc=True))


def test_empty_range_issues_no_request():
    http = _FakeBinanceHttp()
    loader = BinanceHourPriceLoader("BTCUSDT", start_time=START, end_time=START, http=http)
    assert not loader.get_klines()
    assert not http.calls
