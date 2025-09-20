from datetime import UTC, datetime, timedelta
from time import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.binance.binance_client import FUTURES_SECTION, BinanceHttp
from fractal.loaders.structs import KlinesHistory, PriceHistory


class BinancePriceLoader(Loader):
    """
    Loader for Binance kline data. Fetches forward in time for stable ordering.
    """
    _MAX_LIMIT = 1000
    _DEFAULT_START_TIME_DAYS = 365
    _KLINES_ENDPOINT = "/fapi/v1/klines"
    _INTERVAL_MS = {
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
        self.start_time: Optional[datetime] = start_time.replace(tzinfo=UTC) if start_time else datetime.now(tz=UTC) - timedelta(days=self._DEFAULT_START_TIME_DAYS)
        self.end_time: Optional[datetime] = end_time.replace(tzinfo=UTC) if end_time else datetime.now(tz=UTC)
        self.http = http or BinanceHttp()

        # Validate interval
        if not self.interval or not self.interval[:-1].isdigit() or self.interval[-1] not in self._INTERVAL_MS:
            raise ValueError("Invalid interval. Examples: '15m', '1h', '1d'.")

    @staticmethod
    def _to_ms(dt: Optional[datetime]) -> Optional[int]:
        return int(dt.timestamp() * 1000) if dt else None

    def _parse_klines(self, rows: Iterable[List[Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in rows:
            # See Binance docs: item indices 0..11
            open_time = int(item[0])
            out.append(
                {
                    "openTime": datetime.fromtimestamp(open_time / 1000, tz=UTC),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        return out

    def get_klines(self) -> List[Dict[str, Any]]:
        unit = self.interval[-1]
        amount = int(self.interval[:-1])
        candle_ms = amount * self._INTERVAL_MS[unit]
        step_ms = candle_ms * self._MAX_LIMIT  # 1000 candles per request

        start_ms = self._to_ms(self.start_time) or 0
        end_ms = self._to_ms(self.end_time) or int(time() * 1000)

        rows: List[Dict[str, Any]] = []
        cursor = start_ms

        while True:
            # Clamp request window to [cursor, end_ms]
            window_end = min(end_ms, cursor + step_ms - 1)
            if window_end <= cursor:
                break
            params = {
                "symbol": self.ticker,
                "interval": self.interval,
                "limit": self._MAX_LIMIT,
                "startTime": cursor,
                "endTime": window_end,
            }
            data = self.http.get(FUTURES_SECTION, self._KLINES_ENDPOINT, params)
            if data is None or len(data) == 0:
                cursor += step_ms
                continue

            parsed = self._parse_klines(data)
            rows.extend(parsed)

            # Advance cursor to the next candle after the last returned one
            last_open = max(int(item[0]) for item in data)
            next_cursor = last_open + candle_ms

            if next_cursor > end_ms:
                break
            
            cursor = next_cursor

            # Safety: if cursor does not move, avoid infinite loop
            if cursor <= start_ms:
                break

        # Sort & dedupe by openTime
        rows.sort(key=lambda r: r["openTime"])  # ascending
        seen_ts: set[int] = set()
        unique: List[Dict[str, Any]] = []
        for r in rows:
            ts = int(r["openTime"].timestamp() * 1000)
            if ts not in seen_ts:
                unique.append(r)
                seen_ts.add(ts)

        return unique

    # Pipeline methods
    def extract(self) -> None:
        self._data = pd.DataFrame(self.get_klines())

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=["openTime", "open", "high", "low", "close", "volume"])
            return
        self._data = self._data.sort_values("openTime").reset_index(drop=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for c in numeric_cols:
            self._data[c] = pd.to_numeric(self._data[c], errors="coerce")
        if self.inverse_price:
            # Avoid division by zero
            self._data.loc[self._data["close"] != 0, "close"] = 1.0 / self._data.loc[self._data["close"] != 0, "close"]

    def load(self) -> None:
        self._load(self.ticker)

    def read(self, with_run: bool = False) -> PriceHistory:
        if with_run:
            self.run()
        else:
            self._read(self.ticker)
        return PriceHistory(
            prices=self._data["close"].astype(float).values,
            time=pd.to_datetime(self._data["openTime"], utc=True),
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
            ticker=ticker,
            loader_type=loader_type,
            inverse_price=inverse_price,
            interval="1d",
            start_time=start_time,
            end_time=end_time,
            http=http,
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
            ticker=ticker,
            loader_type=loader_type,
            inverse_price=inverse_price,
            interval="1h",
            start_time=start_time,
            end_time=end_time,
            http=http,
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
            ticker=ticker,
            loader_type=loader_type,
            inverse_price=inverse_price,
            interval="1m",
            start_time=start_time,
            end_time=end_time,
            http=http,
        )


class BinanceKlinesLoader(BinancePriceLoader):
    """Same as BinancePriceLoader but returns OHLC in read()."""
    def read(self, with_run: bool = False) -> KlinesHistory:
        if with_run:
            self.run()
        else:
            self._read(self.ticker)
        return KlinesHistory(
            open=self._data["open"].astype(float).values,
            high=self._data["high"].astype(float).values,
            low=self._data["low"].astype(float).values,
            close=self._data["close"].astype(float).values,
            time=pd.to_datetime(self._data["openTime"], utc=True),
        )
