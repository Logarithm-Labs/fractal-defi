"""Binance USDT-M futures funding-rate loader.

Walks ``/fapi/v1/fundingRate`` forward in time in 1000-row pages until
the requested window is exhausted. Returns a :class:`FundingHistory`.
"""
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from fractal.loaders._dt import to_ms, to_utc
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.binance.binance_client import FUTURES_SECTION, BinanceHttp
from fractal.loaders.structs import FundingHistory

_REQUEST_SLEEP_SECONDS = 0.0


class BinanceFundingLoader(Loader):
    _FUNDING_ENDPOINT = "/fapi/v1/fundingRate"
    _MAX_LIMIT = 1000

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        http: Optional[BinanceHttp] = None,
    ) -> None:
        super().__init__(loader_type)
        self.ticker: str = ticker.upper()
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)
        self.http = http or BinanceHttp()

    def _cache_key(self) -> str:
        s = to_ms(self.start_time) if self.start_time is not None else "open"
        e = to_ms(self.end_time) if self.end_time is not None else "now"
        return f"{self.ticker}-{s}-{e}"

    def _fetch(self, start_ms: Optional[int], end_ms: Optional[int]) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"symbol": self.ticker, "limit": self._MAX_LIMIT}
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms
        data = self.http.get(FUTURES_SECTION, self._FUNDING_ENDPOINT, params)
        if not isinstance(data, list):
            raise ValueError(f"Unexpected response for funding rates: {data}")
        return data

    def get_funding_rates(
        self,
        ticker: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        symbol = (ticker or self.ticker).upper()
        # Allow per-call overrides while keeping defaults from constructor.
        local = BinanceFundingLoader(
            symbol,
            loader_type=self.loader_type,
            start_time=start_time or self.start_time,
            end_time=end_time or self.end_time,
            http=self.http,
        )
        rows: List[Dict[str, Any]] = []
        cursor = to_ms(local.start_time) if local.start_time is not None else None
        end_ms = to_ms(local.end_time) if local.end_time is not None else None
        while True:
            data = local._fetch(cursor, end_ms)  # pylint: disable=protected-access
            if not data:
                break
            for item in data:
                rows.append(
                    {
                        "fundingTime": pd.Timestamp(int(item["fundingTime"]), unit="ms", tz="UTC"),
                        "fundingRate": float(item["fundingRate"]),
                        "ticker": item["symbol"],
                    }
                )
            if len(data) < self._MAX_LIMIT:
                break
            last_ms = max(int(it["fundingTime"]) for it in data)
            next_cursor = last_ms + 1
            if cursor is not None and next_cursor <= cursor:
                break
            cursor = next_cursor
            if _REQUEST_SLEEP_SECONDS > 0:
                time.sleep(_REQUEST_SLEEP_SECONDS)
        return rows

    # ------------------------------------------------------------ lifecycle
    def extract(self) -> None:
        self._data = pd.DataFrame(self.get_funding_rates())

    def transform(self) -> None:
        cols = ["fundingTime", "fundingRate", "ticker"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        self._data["fundingTime"] = pd.to_datetime(self._data["fundingTime"], utc=True).dt.floor("s")
        self._data["fundingRate"] = pd.to_numeric(self._data["fundingRate"], errors="coerce").fillna(0.0)
        self._data = (
            self._data.sort_values("fundingTime")
            .drop_duplicates(subset=["fundingTime"])
            .reset_index(drop=True)
        )

    def load(self) -> None:
        self._load(self._cache_key())

    def read(self, with_run: bool = False) -> FundingHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return FundingHistory(rates=[], time=[])
        return FundingHistory(
            rates=self._data["fundingRate"].astype(float).values,
            time=pd.to_datetime(self._data["fundingTime"], utc=True).values,
        )
