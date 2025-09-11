from datetime import UTC, datetime
from time import time
from typing import Any, Dict, List, Optional

import pandas as pd

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.binance.binance_client import FUTURES_SECTION, BinanceHttp
from fractal.loaders.structs import FundingHistory


class BinanceFundingLoader(Loader):
    """
    Loader for Binance Funding Rates (USDT-margined perpetuals).

    Efficiently paginates backward in time in 1000-period (8h) windows.
    """

    _FUNDING_ENDPOINT = "/fapi/v1/fundingRate"
    _MAX_LIMIT = 1000  # Binance max limit per request

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        super().__init__(loader_type)
        self.ticker: str = ticker.upper()
        self.start_time: Optional[datetime] = (
            start_time.replace(tzinfo=UTC) if start_time else None
        )
        self.end_time: Optional[datetime] = (
            end_time.replace(tzinfo=UTC) if end_time else None
        )
        self.http = BinanceHttp()

    @staticmethod
    def _to_ms(dt: Optional[datetime]) -> Optional[int]:
        return int(dt.timestamp() * 1000) if dt else None

    def _fetch_window(self, start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
        params = {
            "symbol": self.ticker,
            "limit": self._MAX_LIMIT,
            "startTime": start_ms,
            "endTime": end_ms,
        }
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
        """Retrieve funding rate history with robust pagination.

        If *ticker*/*start_time*/*end_time* are omitted, uses instance values.
        """
        symbol = (ticker or self.ticker).upper()
        start_ms = self._to_ms(start_time or self.start_time)
        end_ms = self._to_ms(end_time or self.end_time) or int(time() * 1000)

        # Each funding period is 8 hours. One request can return up to 1000 entries.
        period_ms = 8 * 60 * 60 * 1000
        step_ms = period_ms * self._MAX_LIMIT  # cover 1000 periods per request (~333 days)

        all_rows: List[Dict[str, Any]] = []
        while True:
            query_start = max(start_ms or 0, end_ms - step_ms + 1)

            params = {
                "symbol": symbol,
                "limit": self._MAX_LIMIT,
                "startTime": query_start,
                "endTime": end_ms,
            }
            data = self.http.get(FUTURES_SECTION, self._FUNDING_ENDPOINT, params)
            if not data:
                break

            # Normalize to stable dicts
            for item in data:
                all_rows.append(
                    {
                        "fundingTime": datetime.fromtimestamp(
                            item["fundingTime"] / 1000, tz=UTC
                        ),
                        "fundingRate": float(item["fundingRate"]),
                        "ticker": item["symbol"],
                    }
                )

            # Move the window backward; stop if we've crossed the start boundary
            min_time = min(r["fundingTime"] for r in all_rows)
            min_time_ms = int(min_time.timestamp() * 1000)
            if start_ms is not None and min_time_ms <= start_ms:
                break
            end_ms = min_time_ms - 1

            # If the server returned less than the limit, we likely reached the beginning
            if len(data) < self._MAX_LIMIT:
                break

        # Sort & dedupe (just in case)
        all_rows.sort(key=lambda r: r["fundingTime"])  # ascending
        # Drop duplicates by timestamp
        seen: set[int] = set()
        unique_rows: List[Dict[str, Any]] = []
        for r in all_rows:
            ts = int(r["fundingTime"].timestamp())
            if ts not in seen:
                unique_rows.append(r)
                seen.add(ts)

        return unique_rows

    # Pipeline methods
    def extract(self) -> None:
        data = self.get_funding_rates(self.ticker, self.start_time, self.end_time)
        self._data = pd.DataFrame(data)

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=["fundingTime", "fundingRate", "ticker"])
            return
        self._data["fundingTime"] = pd.to_datetime(self._data["fundingTime"], utc=True).dt.floor("s")
        self._data["fundingRate"] = pd.to_numeric(self._data["fundingRate"], errors="coerce").fillna(0.0)
        self._data = self._data.sort_values("fundingTime").reset_index(drop=True)

    def load(self) -> None:
        self._load(self.ticker)

    def read(self, with_run: bool = False) -> FundingHistory:
        if with_run:
            self.run()
        else:
            self._read(self.ticker)
        return FundingHistory(
            rates=self._data["fundingRate"].astype(float).values,
            time=pd.to_datetime(self._data["fundingTime"], utc=True),
        )
