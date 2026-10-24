"""Cross-venue CEX stablecoin close-price medians.

:class:`StablecoinMidLoader` combines public spot candles from Binance,
Kraken, Coinbase Exchange and Bitstamp. The per-bar ``mid`` is the median
of the available venue closes after reciprocal quotes are inverted. It is
a CEX last-trade proxy, not an executable on-chain price or swap quote.

No missing candle is forward-filled. Zero-volume candles are also omitted
because venues may use them to carry a previous close forward.
"""
import logging
import math
from collections.abc import Callable, Iterable
from datetime import datetime, timezone

import pandas as pd
from requests import RequestException

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient, LoaderHttpError
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.binance.binance_client import BinanceHttp
from fractal.loaders.binance.binance_prices import BinanceSpotPriceLoader

_LOG = logging.getLogger(__name__)

DEFAULT_CEX_VENUES = ("binance", "kraken", "coinbase", "bitstamp")

# Canonical pair -> venue -> (venue symbol, reciprocal quote).
_PAIR_SYMBOLS: dict[str, dict[str, tuple[str, bool]]] = {
    "USDC/USDT": {
        "binance": ("USDCUSDT", False),
        "kraken": ("USDCUSDT", False),
        "coinbase": ("USDT-USDC", True),
        "bitstamp": ("usdcusdt", False),
    },
    "USDC/USD": {
        "binance": ("USDCUSD", False),
        "kraken": ("USDCUSD", False),
        "bitstamp": ("usdcusd", False),
    },
    "USDT/USD": {
        "binance": ("USDTUSD", False),
        "kraken": ("USDTZUSD", False),
        "coinbase": ("USDT-USD", False),
        "bitstamp": ("usdtusd", False),
    },
    "PYUSD/USD": {
        "kraken": ("PYUSDUSD", False),
        "coinbase": ("PYUSD-USD", False),
        "bitstamp": ("pyusdusd", False),
    },
    "RLUSD/USD": {
        "kraken": ("RLUSDUSD", False),
        "bitstamp": ("rlusdusd", False),
    },
    "USDG/USD": {
        "kraken": ("USDGUSD", False),
        "bitstamp": ("usdgusd", False),
    },
}
_INTERVAL_SECONDS = {"1h": 3_600, "1d": 86_400}
_VENUE_ERRORS = (LoaderHttpError, RequestException, KeyError, TypeError, ValueError, StopIteration)

RawBar = tuple[int, float, float]
FetchMethod = Callable[[str], list[RawBar]]


def _canonical_pair(pair: str) -> str:
    cleaned = pair.upper().replace("-", "/").replace("_", "/")
    compact = cleaned.replace("/", "")
    for canonical in _PAIR_SYMBOLS:
        if cleaned == canonical or compact == canonical.replace("/", ""):
            return canonical
    raise ValueError(f"Unsupported stablecoin pair {pair!r}; choose one of {sorted(_PAIR_SYMBOLS)}")


def _iso_utc(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StablecoinMidLoader(Loader):
    """Median stablecoin close across public CEX spot venues.

    Args:
        pair: Stablecoin pair, with or without ``/`` (for example,
            ``"USDC/USDT"`` or ``"USDCUSDT"``).
        start_time: Inclusive start of the requested UTC window.
        end_time: Exclusive end of the requested UTC window.
        interval: Candle interval, ``"1h"`` or ``"1d"``.
        venues: Ordered subset of ``binance``, ``kraken``, ``coinbase``
            and ``bitstamp``. Unsupported pair/venue combinations are skipped.
        http: Injectable shared HTTP client; public endpoints require no keys.
        loader_type: Cache serialization format.

    ``read`` returns a UTC-indexed :class:`pandas.DataFrame` with ``mid``,
    ``n_venues`` and one nullable close column per requested venue. A failed
    venue is skipped, so ``n_venues`` counts only closes present in each bar.

    Binance and Bitstamp requests are paged at 1,000 candles and Coinbase at
    300 candles. Kraken's OHLC endpoint exposes only its most recent 720 bars,
    irrespective of the requested start time. Missing and unclosed bars are
    absent and are never forward-filled.
    """

    def __init__(
        self,
        pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h",
        venues: Iterable[str] = DEFAULT_CEX_VENUES,
        http: HttpClient | None = None,
        loader_type: LoaderType = LoaderType.CSV,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.pair = _canonical_pair(pair)
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time)
        if self.start_time is None or self.end_time is None:
            raise ValueError("start_time and end_time are required")
        if self.end_time <= self.start_time:
            raise ValueError(f"end_time {self.end_time} must be after start_time {self.start_time}")
        if interval not in _INTERVAL_SECONDS:
            raise ValueError(f"interval must be one of {sorted(_INTERVAL_SECONDS)}, got {interval!r}")
        self.interval = interval
        self.venues = tuple(venue.lower() for venue in venues)
        if len(set(self.venues)) != len(self.venues):
            raise ValueError("venues must not contain duplicates")
        unknown = sorted(set(self.venues) - set(DEFAULT_CEX_VENUES))
        if unknown:
            raise ValueError(f"Unknown venue(s) {unknown}; choose from {list(DEFAULT_CEX_VENUES)}")
        self._http = http or HttpClient()
        self._rows: dict[str, list[RawBar]] = {}

    def _cache_key(self) -> str:
        pair = self.pair.replace("/", "-")
        venues = "-".join(self.venues) or "none"
        return (
            f"{pair}-{self.interval}-{to_seconds(self.start_time)}-"
            f"{to_seconds(self.end_time)}-{venues}-v1"
        )

    def _fetch_binance(self, symbol: str) -> list[RawBar]:
        loader = BinanceSpotPriceLoader(
            ticker=symbol,
            interval=self.interval,
            start_time=self.start_time,
            end_time=self.end_time,
            http=BinanceHttp(self._http),
        )
        return [
            (int(row["openTime"].timestamp()), float(row["close"]), float(row["volume"]))
            for row in loader.get_klines()
        ]

    def _fetch_kraken(self, symbol: str) -> list[RawBar]:
        payload = self._http.get(
            "https://api.kraken.com/0/public/OHLC",
            params={
                "pair": symbol,
                "interval": _INTERVAL_SECONDS[self.interval] // 60,
                "since": to_seconds(self.start_time) - 1,
            },
        )
        if not isinstance(payload, dict):
            raise TypeError("Kraken OHLC response must be an object")
        if payload.get("error"):
            raise LoaderHttpError(f"Kraken OHLC error for {symbol}: {payload['error']}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise TypeError("Kraken OHLC result must be an object")
        result_key = next((key for key in result if key != "last"), None)
        if result_key is None:
            return []
        return [(int(row[0]), float(row[4]), float(row[6])) for row in result[result_key]]

    def _fetch_coinbase(self, symbol: str) -> list[RawBar]:
        interval_seconds = _INTERVAL_SECONDS[self.interval]
        start = to_seconds(self.start_time)
        end = to_seconds(self.end_time)
        rows: list[RawBar] = []
        cursor = start
        while cursor < end:
            stop = min(end, cursor + 300 * interval_seconds)
            payload = self._http.get(
                f"https://api.exchange.coinbase.com/products/{symbol}/candles",
                params={
                    "granularity": interval_seconds,
                    "start": _iso_utc(cursor),
                    "end": _iso_utc(stop - interval_seconds),
                },
            )
            if isinstance(payload, dict):
                raise LoaderHttpError(f"Coinbase candles error for {symbol}: {str(payload)[:160]}")
            rows.extend((int(row[0]), float(row[4]), float(row[5])) for row in payload)
            cursor = stop
        return rows

    def _fetch_bitstamp(self, symbol: str) -> list[RawBar]:
        interval_seconds = _INTERVAL_SECONDS[self.interval]
        start = to_seconds(self.start_time)
        end = to_seconds(self.end_time)
        rows: list[RawBar] = []
        cursor = start
        while cursor < end:
            payload = self._http.get(
                f"https://www.bitstamp.net/api/v2/ohlc/{symbol}/",
                params={"step": interval_seconds, "limit": 1000, "start": cursor},
            )
            if not isinstance(payload, dict):
                raise TypeError("Bitstamp OHLC response must be an object")
            if payload.get("status") == "error":
                raise LoaderHttpError(f"Bitstamp OHLC error for {symbol}: {str(payload)[:160]}")
            batch = payload.get("data", {}).get("ohlc", [])
            rows.extend(
                (int(row["timestamp"]), float(row["close"]), float(row["volume"]))
                for row in batch
            )
            cursor += 1000 * interval_seconds
        return rows

    def _fetch_venue(self, venue: str, symbol: str) -> list[RawBar]:
        fetchers: dict[str, FetchMethod] = {
            "binance": self._fetch_binance,
            "kraken": self._fetch_kraken,
            "coinbase": self._fetch_coinbase,
            "bitstamp": self._fetch_bitstamp,
        }
        return fetchers[venue](symbol)

    def extract(self) -> None:
        self._rows = {}
        symbols = _PAIR_SYMBOLS[self.pair]
        for venue in self.venues:
            if venue not in symbols:
                self._rows[venue] = []
                continue
            symbol, _ = symbols[venue]
            try:
                self._rows[venue] = self._fetch_venue(venue, symbol)
            except _VENUE_ERRORS as exc:
                _LOG.warning("Skipping %s for %s: %s", venue, self.pair, exc)
                self._rows[venue] = []

    def transform(self) -> None:
        interval_seconds = _INTERVAL_SECONDS[self.interval]
        start = to_seconds(self.start_time)
        end = to_seconds(self.end_time)
        latest_closed = min(end, to_seconds(utcnow()))
        symbols = _PAIR_SYMBOLS[self.pair]
        series: list[pd.Series] = []

        for venue in self.venues:
            inverted = symbols.get(venue, ("", False))[1]
            closes: dict[int, float] = {}
            for timestamp, close, volume in self._rows.get(venue, []):
                outside_window = (
                    timestamp % interval_seconds
                    or timestamp < start
                    or timestamp >= end
                    or timestamp + interval_seconds > latest_closed
                )
                invalid_trade = (
                    not math.isfinite(close)
                    or not math.isfinite(volume)
                    or close <= 0
                    or volume <= 0
                )
                if outside_window or invalid_trade:
                    continue
                closes[timestamp] = 1.0 / close if inverted else close
            series.append(pd.Series(closes, dtype="float64", name=venue))

        venue_frame = pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()
        if venue_frame.empty:
            self._data = pd.DataFrame(columns=["time", "mid", "n_venues", *self.venues])
            return
        venue_frame = venue_frame.reindex(columns=self.venues).dropna(how="all")
        venue_frame.insert(0, "n_venues", venue_frame.count(axis=1).astype("int64"))
        venue_frame.insert(0, "mid", venue_frame[list(self.venues)].median(axis=1))
        venue_frame.insert(0, "time", venue_frame.index.astype("int64"))
        self._data = venue_frame.reset_index(drop=True)[["time", "mid", "n_venues", *self.venues]]

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return the cached or freshly fetched UTC-indexed median series."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        columns = ["mid", "n_venues", *self.venues]
        if self._data is None or self._data.empty:
            # explicit unit: pandas 3 gives an empty index second resolution
            return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", name="time"))
        frame = self._data[columns].copy()
        frame["mid"] = pd.to_numeric(frame["mid"], errors="coerce")
        frame["n_venues"] = pd.to_numeric(frame["n_venues"], errors="raise").astype("int64")
        for venue in self.venues:
            frame[venue] = pd.to_numeric(frame[venue], errors="coerce")
        frame.index = self._utc_index("time", unit="s")
        return frame


__all__ = ["DEFAULT_CEX_VENUES", "StablecoinMidLoader"]
