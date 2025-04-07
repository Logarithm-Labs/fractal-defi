import time
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory, KlinesHistory, PriceHistory

# Module-level constants for defaults
DEFAULT_START_TIME = datetime(2025, 1, 1, tzinfo=UTC)
DEFAULT_URL = 'https://api.hyperliquid.xyz/info'


class HyperliquidBaseLoader(Loader):
    """
    Base loader for Hyperliquid API that provides common functionality.
    """

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        super().__init__(loader_type)
        start_time = start_time or DEFAULT_START_TIME
        end_time = end_time or datetime.now(UTC)

        self._start_time: int = int(start_time.timestamp() * 1000)
        self._end_time: int = int(end_time.timestamp() * 1000)
        self._url: str = DEFAULT_URL
        self._ticker: str = ticker

    def _make_request(self, params: Dict[str, Any]) -> Any:
        """
        Makes a POST request to the Hyperliquid API with the given parameters.
        """
        headers = {"Content-Type": "application/json"}
        response = requests.post(self._url, headers=headers, json=params)
        if response.status_code == 200:
            return response.json()
        raise Exception(
            f"Failed to make request to {self._url}: "
            f"status code: {response.status_code} ({response.text})"
        )

    def load(self) -> None:
        """
        Trigger the load process using the inherited _load method.
        """
        self._load(self._ticker)


class HyperliquidFundingRatesLoader(HyperliquidBaseLoader):
    """
    Loader for fetching funding rate history from Hyperliquid.
    """

    def extract(self) -> None:
        """
        Extract funding rate data in batches until the end time is reached.
        """
        all_data: List[Dict[str, Any]] = self._extract_batch()
        if not all_data:
            raise Exception("No data returned from the initial request.")

        last_time: int = all_data[-1]["time"]
        while last_time < self._end_time:
            self._start_time = last_time
            batch: List[Dict[str, Any]] = self._extract_batch()
            if len(batch) <= 1:
                break
            all_data.extend(batch)
            last_time = batch[-1]["time"]
            time.sleep(1)

        self._data = pd.DataFrame(all_data)

    def _extract_batch(self) -> List[Dict[str, Any]]:
        """
        Extract a single batch of funding history data.
        """
        params = {
            "type": "fundingHistory",
            "coin": self._ticker,
            "startTime": self._start_time,
            "endTime": self._end_time,
        }
        return self._make_request(params)

    def transform(self) -> None:
        """
        Transform the raw funding history data into proper types.
        """
        self._data["time"] = (
            pd.to_datetime(self._data["time"], unit="ms", origin="unix", utc=True).dt.floor("s")
        )
        self._data["fundingRate"] = self._data["fundingRate"].astype(float)

    def read(self, with_run: bool = False) -> FundingHistory:
        """
        Read and return the funding history as a FundingHistory structure.
        """
        if with_run:
            self.run()
        else:
            self._read(self._ticker)
        return FundingHistory(
            rates=self._data["fundingRate"].values,
            time=pd.to_datetime(self._data["time"], utc=True),
        )


class HyperLiquidPerpsPricesLoader(HyperliquidBaseLoader):
    """
    Loader for fetching perpetual prices from Hyperliquid.
    """

    def __init__(
        self,
        ticker: str,
        interval: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        super().__init__(ticker, loader_type, start_time, end_time)
        self._interval: str = interval

    def extract(self) -> None:
        """
        Extract candle snapshot data for perpetual prices.
        """
        params = {
            "type": "candleSnapshot",
            "req": {
                "coin": self._ticker,
                "interval": self._interval,
                "startTime": self._start_time,
                "endTime": self._end_time,
            },
        }
        self._data = pd.DataFrame(self._make_request(params))

    def transform(self) -> None:
        """
        Transform the raw candle snapshot data.
        """
        self._data["open_time"] = (
            pd.to_datetime(self._data["t"], unit="ms", origin="unix", utc=True).dt.floor("s")
        )
        self._data["open_price"] = self._data["o"].astype(float)
        self._data["high_price"] = self._data["h"].astype(float)
        self._data["low_price"] = self._data["l"].astype(float)
        self._data["close_price"] = self._data["c"].astype(float)

    def read(self, with_run: bool = False) -> PriceHistory:
        """
        Read and return the price history as a PriceHistory structure.
        """
        if with_run:
            self.run()
        else:
            self._read(self._ticker)
        return PriceHistory(
            prices=self._data["open_price"].values,
            time=pd.to_datetime(self._data["open_time"], utc=True),
        )


class HyperliquidPerpsKlinesLoader(HyperLiquidPerpsPricesLoader):

    def read(self, with_run: bool = False) -> KlinesHistory:
        """
        Read and return the price history as a KlinesHistory structure.
        """
        if with_run:
            self.run()
        else:
            self._read(self._ticker)
        return KlinesHistory(
            open=self._data["open_price"].values,
            high=self._data["high_price"].values,
            low=self._data["low_price"].values,
            close=self._data["close_price"].values,
            time=pd.to_datetime(self._data["open_time"], utc=True),
        )
