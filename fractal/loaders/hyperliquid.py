"""Hyperliquid public-info API loaders.

Endpoints:
  POST https://api.hyperliquid.xyz/info  with body ``{"type": ..., ...}``

We use three request types:
  - ``fundingHistory`` — paginated by ``startTime``; up to 500 rows per call.
  - ``candleSnapshot`` — bounded by ``[startTime, endTime]`` and a candle
    cap of 5000 per call; we paginate forward when the window is larger.

All loaders return one of the typed structures from ``loaders.structs``,
indexed by a UTC-aware ``DatetimeIndex``.
"""
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from fractal.loaders._dt import to_ms, to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory, KlinesHistory, PriceHistory

DEFAULT_URL = "https://api.hyperliquid.xyz/info"
_REQUEST_SLEEP_SECONDS = 0.2  # be nice to the public endpoint


class HyperliquidBaseLoader(Loader):
    """Common transport, time handling and cache-key logic."""

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        url: str = DEFAULT_URL,
    ) -> None:
        super().__init__(loader_type)
        self._ticker: str = ticker
        self._start_dt = to_utc(start_time) if start_time is not None else None
        self._end_dt = to_utc(end_time) if end_time is not None else utcnow()
        # ms epochs, kept as ints so we can use them as API params unchanged
        self._start_ms: Optional[int] = to_ms(self._start_dt)
        self._end_ms: int = to_ms(self._end_dt)
        self._url: str = url
        self._http = HttpClient()

    # -------------------------------------------------------------- helpers
    def _post(self, body: Dict[str, Any]) -> Any:
        return self._http.post(self._url, json=body)

    def _cache_key(self) -> str:
        # Include ticker + window so different windows do not collide on disk.
        # ``-`` because Hyperliquid tickers are alphanumeric only.
        start_part = self._start_ms if self._start_ms is not None else "open"
        return f"{self._ticker}-{start_part}-{self._end_ms}"

    # ------------------------------------------------------------ lifecycle
    def load(self) -> None:
        self._load(self._cache_key())

    @property
    def ticker(self) -> str:
        return self._ticker


class HyperliquidFundingRatesLoader(HyperliquidBaseLoader):
    """Funding-rate history → :class:`FundingHistory`."""

    _BATCH_LIMIT = 500  # API hard cap

    def extract(self) -> None:
        # Funding history requires startTime; if user did not provide one,
        # use a conservative default (1 year ago).
        if self._start_ms is None:
            self._start_dt = utcnow() - timedelta(days=365)
            self._start_ms = to_ms(self._start_dt)

        cursor = self._start_ms
        all_rows: List[Dict[str, Any]] = []
        seen: set = set()
        while True:
            batch = self._post(
                {
                    "type": "fundingHistory",
                    "coin": self._ticker,
                    "startTime": cursor,
                    "endTime": self._end_ms,
                }
            )
            if not batch:
                break
            new_rows = [row for row in batch if row["time"] not in seen]
            for row in new_rows:
                seen.add(row["time"])
            all_rows.extend(new_rows)

            last_time = batch[-1]["time"]
            if last_time >= self._end_ms or len(batch) < self._BATCH_LIMIT:
                break
            # Advance past last_time to avoid an infinite loop on duplicates.
            cursor = last_time + 1
            time.sleep(_REQUEST_SLEEP_SECONDS)

        self._data = pd.DataFrame(all_rows)

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=["time", "fundingRate", "coin"])
            return
        self._data["time"] = pd.to_datetime(
            self._data["time"], unit="ms", origin="unix", utc=True
        ).dt.floor("s")
        self._data["fundingRate"] = self._data["fundingRate"].astype(float)
        self._data = self._data.sort_values("time").reset_index(drop=True)
        self._data = self._data.drop_duplicates(subset=["time"])

    def read(self, with_run: bool = False) -> FundingHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return FundingHistory(rates=[], time=[])
        return FundingHistory(
            rates=self._data["fundingRate"].astype(float).values,
            time=pd.to_datetime(self._data["time"], utc=True).values,
        )


class HyperLiquidPerpsPricesLoader(HyperliquidBaseLoader):
    """Perp candle snapshots → :class:`PriceHistory` of open prices."""

    _BATCH_LIMIT = 5000  # candleSnapshot hard cap
    _INTERVAL_MS = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "12h": 12 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }

    def __init__(
        self,
        ticker: str,
        interval: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        url: str = DEFAULT_URL,
    ) -> None:
        super().__init__(ticker, loader_type, start_time, end_time, url)
        if interval not in self._INTERVAL_MS:
            raise ValueError(
                f"Unsupported interval {interval!r}. Allowed: {sorted(self._INTERVAL_MS)}"
            )
        self._interval: str = interval

    def _cache_key(self) -> str:
        start_part = self._start_ms if self._start_ms is not None else "open"
        return f"{self._ticker}-{self._interval}-{start_part}-{self._end_ms}"

    def extract(self) -> None:
        if self._start_ms is None:
            # Default to 30 days back to keep the request bounded.
            self._start_dt = utcnow() - timedelta(days=30)
            self._start_ms = to_ms(self._start_dt)

        candle_ms = self._INTERVAL_MS[self._interval]
        cursor = self._start_ms
        all_rows: List[Dict[str, Any]] = []
        seen_t: set = set()
        while cursor < self._end_ms:
            window_end = min(self._end_ms, cursor + candle_ms * self._BATCH_LIMIT)
            batch = self._post(
                {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": self._ticker,
                        "interval": self._interval,
                        "startTime": cursor,
                        "endTime": window_end,
                    },
                }
            )
            if not batch:
                break
            new_rows = [row for row in batch if row["t"] not in seen_t]
            for row in new_rows:
                seen_t.add(row["t"])
            all_rows.extend(new_rows)
            last_open = batch[-1]["t"]
            if last_open + candle_ms >= self._end_ms:
                break
            cursor = last_open + candle_ms
            time.sleep(_REQUEST_SLEEP_SECONDS)

        self._data = pd.DataFrame(all_rows)

    def transform(self) -> None:
        cols = ["open_time", "open_price", "high_price", "low_price", "close_price", "volume"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        df = self._data
        df["open_time"] = pd.to_datetime(
            df["t"], unit="ms", origin="unix", utc=True
        ).dt.floor("s")
        df["open_price"] = df["o"].astype(float)
        df["high_price"] = df["h"].astype(float)
        df["low_price"] = df["l"].astype(float)
        df["close_price"] = df["c"].astype(float)
        df["volume"] = df["v"].astype(float)
        df = df[cols].sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
        self._data = df

    def read(self, with_run: bool = False) -> PriceHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return PriceHistory(prices=[], time=[])
        return PriceHistory(
            prices=self._data["open_price"].astype(float).values,
            time=pd.to_datetime(self._data["open_time"], utc=True).values,
        )


class HyperliquidPerpsKlinesLoader(HyperLiquidPerpsPricesLoader):
    """Same extract+transform as :class:`HyperLiquidPerpsPricesLoader` but
    returns full OHLCV rows as :class:`KlinesHistory`."""

    def read(self, with_run: bool = False) -> KlinesHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return KlinesHistory(time=[], open=[], high=[], low=[], close=[], volume=[])
        return KlinesHistory(
            time=pd.to_datetime(self._data["open_time"], utc=True).values,
            open=self._data["open_price"].astype(float).values,
            high=self._data["high_price"].astype(float).values,
            low=self._data["low_price"].astype(float).values,
            close=self._data["close_price"].astype(float).values,
            volume=self._data["volume"].astype(float).values,
        )
