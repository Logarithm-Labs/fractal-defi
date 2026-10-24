"""Offline coverage for the cross-venue stablecoin mid loader."""
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from fractal.loaders import StablecoinMidLoader
from fractal.loaders._http import LoaderHttpError

pytestmark = pytest.mark.core

START = datetime(2025, 1, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=3)
START_SECONDS = int(START.timestamp())


class _FakeHttp:
    def __init__(self, payloads: dict[str, Any]):
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        del timeout
        self.calls.append((url, params or {}))
        payload = self.payloads[url]
        if isinstance(payload, Exception):
            raise payload
        return payload


def _url(venue: str) -> str:
    return {
        "binance": "https://api.binance.com/api/v3/klines",
        "kraken": "https://api.kraken.com/0/public/OHLC",
        "coinbase": "https://api.exchange.coinbase.com/products/USDT-USDC/candles",
        "bitstamp": "https://www.bitstamp.net/api/v2/ohlc/usdcusdt/",
    }[venue]


def _payload(venue: str, close: float | None) -> Any:
    if venue == "binance":
        return [] if close is None else [
            [START_SECONDS * 1000, "1", "1", "1", str(close), "2"]
        ]
    if venue == "kraken":
        rows = [] if close is None else [
            [START_SECONDS, "1", "1", "1", str(close), "1", "2", 1]
        ]
        return {"error": [], "result": {"USDCUSDT": rows, "last": START_SECONDS}}
    if venue == "coinbase":
        return [] if close is None else [
            [START_SECONDS, 1.0, 1.0, 1.0, close, 2.0]
        ]
    rows = [] if close is None else [
        {"timestamp": str(START_SECONDS), "close": str(close), "volume": "2"}
    ]
    return {"data": {"ohlc": rows}}


def _read_transformed(
    tmp_path, monkeypatch, venues: tuple[str, ...], rows: dict[str, list[tuple[int, float, float]]]
) -> pd.DataFrame:
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    loader = StablecoinMidLoader("USDC/USDT", START, END, venues=venues)
    loader._rows = rows
    loader.transform()
    loader.load()
    return loader.read()


@pytest.mark.parametrize("venue", ["binance", "kraken", "coinbase", "bitstamp"])
def test_each_venue_parses_empty_payload(venue, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    loader = StablecoinMidLoader(
        "USDC/USDT", START, END, venues=(venue,), http=_FakeHttp({_url(venue): _payload(venue, None)})
    )
    frame = loader.read(with_run=True)
    assert frame.empty
    assert list(frame.columns) == ["mid", "n_venues", venue]
    assert str(frame.index.dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize(
    ("venue", "close", "expected"),
    [
        ("binance", 1.001, 1.001),
        ("kraken", 1.002, 1.002),
        ("coinbase", 0.8, 1.25),
        ("bitstamp", 1.004, 1.004),
    ],
)
def test_each_venue_parses_single_bar_and_inverts_reciprocal(venue, close, expected, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    loader = StablecoinMidLoader(
        "USDCUSDT", START, END, venues=(venue,), http=_FakeHttp({_url(venue): _payload(venue, close)})
    )
    frame = loader.read(with_run=True)
    assert frame.index.equals(pd.DatetimeIndex([START], name="time"))
    assert frame[venue].iloc[0] == pytest.approx(expected)
    assert frame["mid"].iloc[0] == pytest.approx(expected)
    assert frame["n_venues"].iloc[0] == 1


@pytest.mark.parametrize(
    ("venues", "closes", "expected"),
    [
        (("binance",), (1.01,), 1.01),
        (("binance", "kraken"), (0.99, 1.01), 1.0),
        (("binance", "kraken", "bitstamp"), (0.99, 1.00, 1.02), 1.0),
    ],
)
def test_mid_is_median_of_one_two_or_three_venues(venues, closes, expected, tmp_path, monkeypatch):
    rows = {
        venue: [(START_SECONDS, close, 1.0)]
        for venue, close in zip(venues, closes, strict=True)
    }
    frame = _read_transformed(tmp_path, monkeypatch, venues, rows)
    assert frame["mid"].iloc[0] == pytest.approx(expected)
    assert frame["n_venues"].iloc[0] == len(venues)


def test_missing_bars_remain_sparse_and_are_never_forward_filled(tmp_path, monkeypatch):
    second = START_SECONDS + 3600
    frame = _read_transformed(
        tmp_path,
        monkeypatch,
        ("binance", "kraken"),
        {"binance": [(START_SECONDS, 1.0, 1.0)], "kraken": [(second, 1.01, 1.0)]},
    )
    assert frame.index.equals(pd.date_range(START, periods=2, freq="h", name="time"))
    assert pd.isna(frame.loc[START, "kraken"])
    assert pd.isna(frame.loc[START + timedelta(hours=1), "binance"])
    assert frame["n_venues"].tolist() == [1, 1]
    assert START + timedelta(hours=2) not in frame.index


def test_failed_venue_is_skipped_and_not_counted(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    http = _FakeHttp(
        {
            _url("binance"): LoaderHttpError("synthetic failure"),
            _url("kraken"): _payload("kraken", 1.002),
        }
    )
    frame = StablecoinMidLoader(
        "USDC/USDT", START, END, venues=("binance", "kraken"), http=http
    ).read(with_run=True)
    assert pd.isna(frame["binance"].iloc[0])
    assert frame["kraken"].iloc[0] == pytest.approx(1.002)
    assert frame["mid"].iloc[0] == pytest.approx(1.002)
    assert frame["n_venues"].iloc[0] == 1


def test_zero_volume_bar_is_absent_instead_of_carrying_a_close(tmp_path, monkeypatch):
    frame = _read_transformed(
        tmp_path,
        monkeypatch,
        ("binance",),
        {"binance": [(START_SECONDS, 1.0, 0.0)]},
    )
    assert frame.empty


def test_cache_key_is_deterministic_and_covers_all_data_inputs():
    base = StablecoinMidLoader("USDCUSDT", START, END)
    same = StablecoinMidLoader("USDC/USDT", START, END)
    assert base._cache_key() == same._cache_key()
    variants = [
        StablecoinMidLoader("USDT/USD", START, END),
        StablecoinMidLoader("USDC/USDT", START + timedelta(hours=1), END),
        StablecoinMidLoader("USDC/USDT", START, END + timedelta(hours=1)),
        StablecoinMidLoader("USDC/USDT", START, END, interval="1d"),
        StablecoinMidLoader("USDC/USDT", START, END, venues=("kraken", "binance")),
    ]
    assert all(loader._cache_key() != base._cache_key() for loader in variants)


def test_constructor_rejects_invalid_pair_interval_window_and_venues():
    with pytest.raises(ValueError, match="Unsupported stablecoin pair"):
        StablecoinMidLoader("BTC/USD", START, END)
    with pytest.raises(ValueError, match="interval"):
        StablecoinMidLoader("USDC/USDT", START, END, interval="15m")
    with pytest.raises(ValueError, match="must be after"):
        StablecoinMidLoader("USDC/USDT", END, START)
    with pytest.raises(ValueError, match="duplicates"):
        StablecoinMidLoader("USDC/USDT", START, END, venues=("kraken", "kraken"))
    with pytest.raises(ValueError, match="Unknown venue"):
        StablecoinMidLoader("USDC/USDT", START, END, venues=("other",))


@pytest.mark.integration
def test_stablecoin_mid_loader_integration_smoke(request, tmp_path, monkeypatch):
    if request.config.getoption("-m").strip() == "core":
        pytest.skip("live smoke runs only under the integration test selection")
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=2)
    frame = StablecoinMidLoader("USDC/USDT", start, end).read(with_run=True)
    assert not frame.empty
    assert list(frame.columns) == ["mid", "n_venues", "binance", "kraken", "coinbase", "bitstamp"]
    assert frame.index.tz is not None
    assert frame["n_venues"].between(1, 4).all()
    assert frame["mid"].notna().all()
