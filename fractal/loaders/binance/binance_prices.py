"""Binance USDT-M futures kline loaders.

The base :class:`BinancePriceLoader` paginates ``/fapi/v1/klines`` forward
in time using the ``startTime`` / ``endTime`` Binance parameters. It
returns a :class:`PriceHistory` (close prices) by default; subclasses fix
the candle interval. :class:`BinanceKlinesLoader` reuses the same data
extraction but emits a :class:`KlinesHistory` with full OHLCV.
"""
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from fractal.loaders._dt import to_ms, to_utc, utcnow
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.binance.binance_client import FUTURES_SECTION, BinanceHttp
from fractal.loaders.structs import KlinesHistory, PriceHistory

_REQUEST_SLEEP_SECONDS = 0.0  # Binance is generous; bump if 429s appear


class BinancePriceLoader(Loader):
    """Loader for Binance USDT-M futures klines.

    Returns close prices as :class:`PriceHistory`. Internal ``_data``
    DataFrame keeps the columns ``[openTime, open, high, low, close, volume]``
    so subclasses (notably :class:`BinanceKlinesLoader`) can reuse it.
    """

    _MAX_LIMIT = 1000  # Binance hard cap per /klines request
    _DEFAULT_LOOKBACK_DAYS = 365
    _KLINES_ENDPOINT = "/fapi/v1/klines"
    _INTERVAL_MS: Dict[str, int] = {
        "m": 60 * 1000,
        "h": 60 * 60 * 1000,
        "d": 24 * 60 * 60 * 1000,
        "w": 7 * 24 * 60 * 60 * 1000,
    }

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        inverse_price: bool = False,
        interval: str = "1d",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        http: Optional[BinanceHttp] = None,
    ) -> None:
        super().__init__(loader_type)
        self.ticker: str = ticker.upper()
        self.inverse_price: bool = inverse_price
        self.interval: str = interval
        self._validate_interval(interval)

        self.start_time: datetime = (
            to_utc(start_time)
            if start_time is not None
            else utcnow() - timedelta(days=self._DEFAULT_LOOKBACK_DAYS)
        )
        self.end_time: datetime = to_utc(end_time) if end_time is not None else utcnow()
        self.http = http or BinanceHttp()

    # ------------------------------------------------------------ helpers
    @classmethod
    def _validate_interval(cls, interval: str) -> None:
        if not interval or not interval[:-1].isdigit() or interval[-1] not in cls._INTERVAL_MS:
            raise ValueError(
                f"Invalid interval {interval!r}. Examples: '15m', '1h', '1d', '1w'."
            )

    def _candle_ms(self) -> int:
        unit = self.interval[-1]
        amount = int(self.interval[:-1])
        return amount * self._INTERVAL_MS[unit]

    def _cache_key(self) -> str:
        return f"{self.ticker}-{self.interval}-{to_ms(self.start_time)}-{to_ms(self.end_time)}"

    def _parse_klines(self, rows: Iterable[List[Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in rows:
            open_ms = int(item[0])
            out.append(
                {
                    "openTime": pd.Timestamp(open_ms, unit="ms", tz="UTC"),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        return out

    # ----------------------------------------------------------- pagination
    def get_klines(self) -> List[Dict[str, Any]]:
        candle_ms = self._candle_ms()
        step_ms = candle_ms * self._MAX_LIMIT
        start_ms = to_ms(self.start_time) or 0
        end_ms = to_ms(self.end_time) or int(time.time() * 1000)
        if start_ms >= end_ms:
            return []

        rows: List[Dict[str, Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            window_end = min(end_ms, cursor + step_ms - 1)
            params = {
                "symbol": self.ticker,
                "interval": self.interval,
                "limit": self._MAX_LIMIT,
                "startTime": cursor,
                "endTime": window_end,
            }
            data = self.http.get(FUTURES_SECTION, self._KLINES_ENDPOINT, params)
            if not data:
                # No data in this window — advance one step and try again.
                cursor += step_ms
                continue

            rows.extend(self._parse_klines(data))
            last_open = max(int(item[0]) for item in data)
            next_cursor = last_open + candle_ms
            if next_cursor <= cursor:
                # Should not happen, but guard against an infinite loop.
                break
            cursor = next_cursor
            if len(data) < self._MAX_LIMIT:
                # Reached the last page for this window.
                break
            if _REQUEST_SLEEP_SECONDS > 0:
                time.sleep(_REQUEST_SLEEP_SECONDS)

        # Dedup by openTime (defensive — pagination overlap should not happen).
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for r in rows:
            ts = int(r["openTime"].value)
            if ts in seen:
                continue
            seen.add(ts)
            unique.append(r)
        unique.sort(key=lambda r: r["openTime"])
        return unique

    # ----------------------------------------------------------- lifecycle
    def extract(self) -> None:
        self._data = pd.DataFrame(self.get_klines())

    def transform(self) -> None:
        cols = ["openTime", "open", "high", "low", "close", "volume"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        self._data = self._data.sort_values("openTime").reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume"]:
            self._data[c] = pd.to_numeric(self._data[c], errors="coerce")
        if self.inverse_price:
            mask = self._data["close"] != 0
            self._data.loc[mask, "close"] = 1.0 / self._data.loc[mask, "close"]
        self._data = self._data[cols]

    def load(self) -> None:
        self._load(self._cache_key())

    def read(self, with_run: bool = False) -> PriceHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return PriceHistory(prices=[], time=[])
        return PriceHistory(
            prices=self._data["close"].astype(float).values,
            time=pd.to_datetime(self._data["openTime"], utc=True).values,
        )


class BinanceDayPriceLoader(BinancePriceLoader):
    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        inverse_price: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        http: Optional[BinanceHttp] = None,
    ) -> None:
        super().__init__(
            ticker=ticker, loader_type=loader_type, inverse_price=inverse_price,
            interval="1d", start_time=start_time, end_time=end_time, http=http,
        )


class BinanceHourPriceLoader(BinancePriceLoader):
    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        inverse_price: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        http: Optional[BinanceHttp] = None,
    ) -> None:
        super().__init__(
            ticker=ticker, loader_type=loader_type, inverse_price=inverse_price,
            interval="1h", start_time=start_time, end_time=end_time, http=http,
        )


class BinanceMinutePriceLoader(BinancePriceLoader):
    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        inverse_price: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        http: Optional[BinanceHttp] = None,
    ) -> None:
        super().__init__(
            ticker=ticker, loader_type=loader_type, inverse_price=inverse_price,
            interval="1m", start_time=start_time, end_time=end_time, http=http,
        )


class BinanceKlinesLoader(BinancePriceLoader):
    """Same extraction as :class:`BinancePriceLoader` but returns OHLCV."""

    def read(self, with_run: bool = False) -> KlinesHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return KlinesHistory(time=[], open=[], high=[], low=[], close=[], volume=[])
        return KlinesHistory(
            time=pd.to_datetime(self._data["openTime"], utc=True).values,
            open=self._data["open"].astype(float).values,
            high=self._data["high"].astype(float).values,
            low=self._data["low"].astype(float).values,
            close=self._data["close"].astype(float).values,
            volume=self._data["volume"].astype(float).values,
        )
