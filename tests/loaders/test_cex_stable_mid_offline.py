"""Offline coverage for the cross-venue stablecoin mid loader."""
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from fractal.loaders import StablecoinMidLoader
from fractal.loaders._http import LoaderHttpError
from fractal.loaders.base_loader import LoaderType

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
    assert str(frame.index.tz) == "UTC"
    assert frame.dtypes.astype(str).tolist() == ["float64", "int64", "float64"]


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
    with pytest.raises(ValueError, match="at least one venue"):
        StablecoinMidLoader("USDC/USDT", START, END, venues=())


class _CandleServer:
    """Synthetic hourly candles answering each venue's paging parameters."""

    def __init__(self, opens: list[int], close: float = 1.0):
        self.opens = opens
        self.close = close
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def _iso(value: str) -> int:
        return int(datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        del timeout
        params = params or {}
        self.calls.append((url, params))
        if "binance" in url:
            lo, hi = params["startTime"] // 1000, params["endTime"] // 1000
            bars = [t for t in self.opens if lo <= t <= hi][: params["limit"]]
            return [[t * 1000, "1", "1", "1", str(self.close), "5"] for t in bars]
        if "coinbase" in url:
            lo, hi = self._iso(params["start"]), self._iso(params["end"])
            bars = [t for t in self.opens if lo <= t <= hi][:300]
            return [[t, 1.0, 1.0, 1.0, self.close, 5.0] for t in reversed(bars)]
        if "bitstamp" in url:
            bars = [t for t in self.opens if t >= params["start"]][: params["limit"]]
            return {"data": {"ohlc": [{"timestamp": str(t), "close": str(self.close), "volume": "5"} for t in bars]}}
        raise KeyError(url)


@pytest.mark.parametrize("venue", ["binance", "coinbase", "bitstamp"])
def test_paging_walks_past_short_pages_and_unaligned_start(venue, tmp_path, monkeypatch):
    """An outage or listing inside one page must not end the walk early."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    opens = [START_SECONDS + i * 3600 for i in range(2_500) if not 100 <= i < 105]
    start = START + timedelta(minutes=30)
    end = START + timedelta(hours=2_500)
    server = _CandleServer(opens)
    frame = StablecoinMidLoader("USDT/USD", start, end, venues=(venue,), http=server).read(with_run=True)
    expected = pd.to_datetime([t for t in opens if t >= start.timestamp()], unit="s", utc=True)
    assert frame.index.equals(pd.DatetimeIndex(expected, name="time"))
    assert frame["n_venues"].eq(1).all()
    page = {"binance": 1_000, "coinbase": 300, "bitstamp": 1_000}[venue]
    assert len(server.calls) == -(-2_499 // page)


def test_unclosed_and_off_grid_bars_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "fractal.loaders.cex_stable_mid.utcnow", lambda: START + timedelta(hours=1, minutes=30)
    )
    frame = _read_transformed(
        tmp_path,
        monkeypatch,
        ("binance",),
        {
            "binance": [
                (START_SECONDS, 1.0, 1.0),
                (START_SECONDS + 1_800, 1.1, 1.0),
                (START_SECONDS + 3_600, 1.2, 1.0),
            ]
        },
    )
    assert frame.index.equals(pd.DatetimeIndex([START], name="time"))
    assert frame["mid"].tolist() == [1.0]


def test_daily_interval_keeps_only_midnight_bars(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    loader = StablecoinMidLoader("USDC/USDT", START, START + timedelta(days=2), interval="1d", venues=("kraken",))
    loader._rows = {"kraken": [(START_SECONDS, 1.0, 1.0), (START_SECONDS + 3_600, 1.5, 1.0)]}
    loader.transform()
    loader.load()
    frame = loader.read()
    assert frame.index.equals(pd.DatetimeIndex([START], name="time"))


@pytest.mark.parametrize(
    "payload",
    [
        {"error": ["EQuery:Unknown asset pair"], "result": {}},
        {"error": [], "result": {"USDCUSDT": [[START_SECONDS, "1"]], "last": 0}},
        {"error": [], "result": None},
    ],
)
def test_kraken_error_or_malformed_payload_skips_the_venue(payload, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    http = _FakeHttp({_url("kraken"): payload, _url("bitstamp"): _payload("bitstamp", 1.004)})
    frame = StablecoinMidLoader(
        "USDC/USDT", START, END, venues=("kraken", "bitstamp"), http=http
    ).read(with_run=True)
    assert frame["kraken"].isna().all()
    assert frame["mid"].tolist() == [pytest.approx(1.004)]


def test_binance_error_object_skips_the_venue(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    http = _FakeHttp({_url("binance"): {"code": -1121, "msg": "Invalid symbol."}})
    frame = StablecoinMidLoader("USDC/USDT", START, END, venues=("binance",), http=http).read(with_run=True)
    assert frame.empty


def test_empty_and_non_empty_reads_share_dtypes(tmp_path, monkeypatch):
    empty = _read_transformed(tmp_path, monkeypatch, ("binance", "kraken"), {})
    full = _read_transformed(
        tmp_path, monkeypatch, ("binance", "kraken"), {"binance": [(START_SECONDS, 1.0, 1.0)]}
    )
    assert empty.empty and not full.empty
    assert empty.index.dtype == full.index.dtype
    assert empty.dtypes.equals(full.dtypes)


@pytest.mark.parametrize("loader_type", ["CSV", "JSON", "PICKLE"])
def test_cached_read_matches_fresh_read(loader_type, tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    http = _FakeHttp(
        {
            _url("binance"): _payload("binance", 1.001),
            _url("kraken"): _payload("kraken", None),
            _url("coinbase"): _payload("coinbase", 0.8),
        }
    )
    kwargs = {"venues": ("binance", "kraken", "coinbase"), "loader_type": LoaderType[loader_type]}
    fresh = StablecoinMidLoader("USDC/USDT", START, END, http=http, **kwargs).read(with_run=True)
    cached = StablecoinMidLoader("USDC/USDT", START, END, http=_FakeHttp({}), **kwargs).read()
    pd.testing.assert_frame_equal(fresh, cached)
    assert fresh["kraken"].isna().all()
    assert fresh["mid"].iloc[0] == pytest.approx((1.001 + 1.25) / 2)


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
