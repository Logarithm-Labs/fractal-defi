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
from typing import Any

import pandas as pd
from requests import RequestException

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient, LoaderHttpError
from fractal.loaders.base_loader import Loader, LoaderType

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
_PAGE_BARS = {"binance": 1_000, "coinbase": 300, "bitstamp": 1_000}
_VENUE_ERRORS = (
    LoaderHttpError,
    RequestException,
    AttributeError,
    IndexError,
    KeyError,
    TypeError,
    ValueError,
)

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
        start_time: Inclusive start of the requested UTC window; the first
            bar is the first one opening at or after it.
        end_time: Exclusive end of the requested UTC window for bar
            *opens*: a bar opening before ``end_time`` is kept once it has
            closed, even when it closes after an unaligned ``end_time``.
        interval: Candle interval, ``"1h"`` or ``"1d"``.
        venues: Ordered subset of ``binance``, ``kraken``, ``coinbase``
            and ``bitstamp``. Unsupported pair/venue combinations are skipped.
        http: Injectable shared HTTP client; public endpoints require no keys.
        loader_type: Cache serialization format.
        include_volume: Also return one ``<venue>_volume`` column per venue,
            in units of the pair's base asset (a reciprocal venue's volume
            is converted with its close).

    ``read`` returns a UTC-indexed :class:`pandas.DataFrame` with ``mid``,
    ``n_venues`` and one nullable close column per requested venue. A failed
    venue is skipped, so ``n_venues`` counts only closes present in each bar.

    After a fetch (``run`` or ``read(with_run=True)``), :attr:`venue_report`
    maps each requested venue to its coverage: ``status`` (``"ok"``,
    ``"unsupported"`` or ``"error: <message>"``), ``symbol``, ``inverted``,
    ``expected_bars`` (bars opening in the window that have closed),
    ``present_bars``, ``missing_bars``, ``missing_ranges`` (``[start,
    end)`` epoch-second pairs) and ``zero_volume_dropped``. It is empty after
    a cache-only ``read``.

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
        include_volume: bool = False,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.pair = _canonical_pair(pair)
        self.include_volume = include_volume
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
        if not self.venues:
            raise ValueError("venues must name at least one venue")
        if len(set(self.venues)) != len(self.venues):
            raise ValueError("venues must not contain duplicates")
        unknown = sorted(set(self.venues) - set(DEFAULT_CEX_VENUES))
        if unknown:
            raise ValueError(f"Unknown venue(s) {unknown}; choose from {list(DEFAULT_CEX_VENUES)}")
        self._http = http or HttpClient()
        self._rows: dict[str, list[RawBar]] = {}
        self._errors: dict[str, str] = {}
        self.venue_report: dict[str, dict[str, Any]] = {}

    def _cache_key(self) -> str:
        pair = self.pair.replace("/", "-")
        venues = "-".join(self.venues)
        return (
            f"{pair}-{self.interval}-{to_seconds(self.start_time)}-"
            f"{to_seconds(self.end_time)}-{venues}-v2"
        )

    def _pages(self, venue: str) -> list[tuple[int, int]]:
        """Bar-aligned ``[start, stop)`` request windows of at most one venue page.

        Each window holds at most the venue's page size of candles, so a
        short page (an outage or a listing inside the window) never ends
        the walk early.
        """
        interval_seconds = _INTERVAL_SECONDS[self.interval]
        step = _PAGE_BARS[venue] * interval_seconds
        start = to_seconds(self.start_time)
        cursor = start + (-start) % interval_seconds
        end = to_seconds(self.end_time)
        windows: list[tuple[int, int]] = []
        while cursor < end:
            windows.append((cursor, min(end, cursor + step)))
            cursor += step
        return windows

    def _fetch_binance(self, symbol: str) -> list[RawBar]:
        rows: list[RawBar] = []
        for start, stop in self._pages("binance"):
            payload = self._http.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": self.interval,
                    "startTime": start * 1000,
                    "endTime": stop * 1000 - 1,
                    "limit": _PAGE_BARS["binance"],
                },
            )
            if not isinstance(payload, list):
                raise LoaderHttpError(f"Binance klines error for {symbol}: {str(payload)[:160]}")
            rows.extend((int(row[0]) // 1000, float(row[4]), float(row[5])) for row in payload)
        return rows

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
        rows: list[RawBar] = []
        for start, stop in self._pages("coinbase"):
            # ``end`` is inclusive: the last bar opens one interval before ``stop``.
            last_open = stop - 1 - (stop - 1) % interval_seconds
            payload = self._http.get(
                f"https://api.exchange.coinbase.com/products/{symbol}/candles",
                params={
                    "granularity": interval_seconds,
                    "start": _iso_utc(start),
                    "end": _iso_utc(last_open),
                },
            )
            if not isinstance(payload, list):
                raise LoaderHttpError(f"Coinbase candles error for {symbol}: {str(payload)[:160]}")
            rows.extend((int(row[0]), float(row[4]), float(row[5])) for row in payload)
        return rows

    def _fetch_bitstamp(self, symbol: str) -> list[RawBar]:
        interval_seconds = _INTERVAL_SECONDS[self.interval]
        rows: list[RawBar] = []
        for start, _ in self._pages("bitstamp"):
            # No ``end`` parameter: with one, Bitstamp anchors the page on ``end``.
            payload = self._http.get(
                f"https://www.bitstamp.net/api/v2/ohlc/{symbol}/",
                params={"step": interval_seconds, "limit": _PAGE_BARS["bitstamp"], "start": start},
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
        self._errors = {}
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
                self._errors[venue] = str(exc)[:160]

    def transform(self) -> None:
        interval_seconds = _INTERVAL_SECONDS[self.interval]
        start = to_seconds(self.start_time)
        end = to_seconds(self.end_time)
        now = to_seconds(utcnow())
        first = start + (-start) % interval_seconds
        expected = [opened for opened in range(first, end, interval_seconds) if opened + interval_seconds <= now]
        symbols = _PAIR_SYMBOLS[self.pair]
        series: list[pd.Series] = []
        self.venue_report = {}

        for venue in self.venues:
            symbol, inverted = symbols.get(venue, ("", False))
            closes: dict[int, float] = {}
            volumes: dict[int, float] = {}
            zero_volume = 0
            for timestamp, close, volume in self._rows.get(venue, []):
                outside_window = (
                    timestamp % interval_seconds
                    or timestamp < start
                    or timestamp >= end
                    or timestamp + interval_seconds > now
                )
                if outside_window or not math.isfinite(close) or close <= 0 or not math.isfinite(volume):
                    continue
                if volume <= 0:
                    # A zero-volume candle carries the previous close forward.
                    zero_volume += 1
                    continue
                closes[timestamp] = 1.0 / close if inverted else close
                volumes[timestamp] = volume * close if inverted else volume
            series.append(pd.Series(closes, dtype="float64", name=venue))
            series.append(pd.Series(volumes, dtype="float64", name=f"{venue}_volume"))
            self.venue_report[venue] = self._coverage(venue, symbol, inverted, expected, closes, zero_volume)

        volume_columns = [f"{venue}_volume" for venue in self.venues]
        cached_columns = ["time", "mid", "n_venues", *self.venues, *volume_columns]
        venue_frame = pd.concat(series, axis=1).sort_index()
        if venue_frame.empty:
            self._data = pd.DataFrame(columns=cached_columns)
            return
        venue_frame = venue_frame.dropna(how="all", subset=list(self.venues))
        venue_frame.insert(0, "n_venues", venue_frame[list(self.venues)].count(axis=1).astype("int64"))
        venue_frame.insert(0, "mid", venue_frame[list(self.venues)].median(axis=1))
        venue_frame.insert(0, "time", venue_frame.index.astype("int64"))
        self._data = venue_frame.reset_index(drop=True)[cached_columns]

    def _coverage(
        self,
        venue: str,
        symbol: str,
        inverted: bool,
        expected: list[int],
        closes: dict[int, float],
        zero_volume: int,
    ) -> dict[str, Any]:
        interval_seconds = _INTERVAL_SECONDS[self.interval]
        if not symbol:
            status = "unsupported"
        elif venue in self._errors:
            status = f"error: {self._errors[venue]}"
        else:
            status = "ok"
        ranges: list[list[int]] = []
        for opened in expected:
            if opened in closes:
                continue
            if ranges and ranges[-1][1] == opened:
                ranges[-1][1] = opened + interval_seconds
            else:
                ranges.append([opened, opened + interval_seconds])
        return {
            "status": status,
            "symbol": symbol or None,
            "inverted": inverted,
            "expected_bars": len(expected),
            "present_bars": len(closes),
            "missing_bars": len(expected) - len(closes),
            "missing_ranges": ranges,
            "zero_volume_dropped": zero_volume,
        }

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return the cached or freshly fetched UTC-indexed median series."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        volume_columns = [f"{venue}_volume" for venue in self.venues]
        columns = ["mid", "n_venues", *self.venues]
        if self._data is None:
            self._data = pd.DataFrame(columns=["time", *columns, *volume_columns])
        # Empty and non-empty reads share one path, so their dtypes (including
        # the index resolution, which differs between pandas 2 and 3) agree.
        frame = pd.DataFrame(
            {
                "mid": pd.to_numeric(self._data["mid"], errors="coerce").astype("float64"),
                "n_venues": pd.to_numeric(self._data["n_venues"], errors="raise").astype("int64"),
                **{
                    venue: pd.to_numeric(self._data[venue], errors="coerce").astype("float64")
                    for venue in self.venues
                },
            }
        )
        if self.include_volume:
            for column in volume_columns:
                frame[column] = pd.to_numeric(self._data[column], errors="coerce").astype("float64")
        frame.index = self._utc_index("time", unit="s")
        return frame


__all__ = ["DEFAULT_CEX_VENUES", "StablecoinMidLoader"]
