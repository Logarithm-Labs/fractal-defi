from datetime import UTC, datetime
from time import time
from typing import Any, List, Optional

import pandas as pd
import requests

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory, KlinesHistory, PriceHistory


class BinanceFundingLoader(Loader):
    """
    Loader for Binance Funding Rates.

    Retrieves funding rate history for a given ticker symbol from the Binance API.
    """

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Initialize the BinanceFundingLoader.

        Args:
            ticker (str): Trading pair symbol, e.g. "BTCUSDT".
            loader_type (LoaderType): Loader type to use.
            start_time (datetime, optional): Start time for data retrieval.
            end_time (datetime, optional): End time for data retrieval.
        """
        super().__init__(loader_type)
        self.ticker: str = ticker
        self.start_time: Optional[datetime] = (
            start_time.replace(tzinfo=UTC) if start_time else None
        )
        self.end_time: Optional[datetime] = (
            end_time.replace(tzinfo=UTC) if end_time else None
        )
        self._url: str = "https://fapi.binance.com/fapi/v1/fundingRate"

    def get_funding_rates(
        self,
        ticker: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Any]:
        """
        Retrieve funding rate history with proper pagination.

        Args:
            ticker (str): Trading pair symbol.
            start_time (datetime, optional): Start time for data retrieval.
            end_time (datetime, optional): End time for data retrieval.

        Returns:
            List[Any]: A list of funding rate records.
        """
        funding_rates = []

        if end_time is None:
            end_time_ms = int(time() * 1000)
        else:
            end_time_ms = int(end_time.timestamp() * 1000)

        start_time_ms = int(start_time.timestamp() * 1000) if start_time else None

        # Each funding rate period is 8 hours in milliseconds.
        period_ms = 8 * 60 * 60 * 1000
        # Maximum records per call is 1000, so we cover 1000 periods in each request.
        step = period_ms * 1000

        while True:
            query_start = end_time_ms - step
            if start_time_ms is not None and query_start < start_time_ms:
                query_start = start_time_ms

            url = (
                f"{self._url}?symbol={ticker}"
                f"&limit=1000"
                f"&endTime={end_time_ms}"
                f"&startTime={query_start}"
            )
            response = requests.get(url, timeout=10).json()
            if not response:
                break

            for item in response:
                funding_rates.append({
                    "fundingTime": datetime.fromtimestamp(
                        item["fundingTime"] / 1000, tz=UTC
                    ),
                    "fundingRate": float(item["fundingRate"]),
                    "ticker": item["symbol"],
                })

            if len(response) < 1000:
                break

            new_end_time = min(item["fundingTime"] for item in response) - 1
            if start_time_ms is not None and new_end_time < start_time_ms:
                break

            end_time_ms = new_end_time

        return funding_rates

    def extract(self) -> None:
        """
        Extract funding rate data from Binance API.
        """
        data = self.get_funding_rates(
            self.ticker, start_time=self.start_time, end_time=self.end_time
        )
        self._data = pd.DataFrame(data)

    def transform(self) -> None:
        """
        Transform the raw funding rate data.

        Rounds fundingTime to the nearest second.
        """
        self._data["fundingTime"] = pd.to_datetime(
            self._data["fundingTime"], utc=True
        ).dt.floor("s")

    def load(self) -> None:
        """
        Load the data using the internal _load mechanism.
        """
        self._load(self.ticker)

    def read(self, with_run: bool = False) -> FundingHistory:
        """
        Read and return the funding history.

        Args:
            with_run (bool): Whether to run the extraction pipeline before reading.

        Returns:
            FundingHistory: The funding history data.
        """
        if with_run:
            self.run()
        else:
            self._read(self.ticker)
        return FundingHistory(
            rates=self._data["fundingRate"].astype(float).values,
            time=pd.to_datetime(self._data["fundingTime"], utc=True),
        )


class BinancePriceLoader(Loader):
    """
    Loader for Binance Price Data.

    Retrieves candlestick (kline) data for a given ticker symbol from the Binance API.
    """

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        inverse_price: bool = False,
        interval: str = "1d",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Initialize the BinancePriceLoader.

        Args:
            ticker (str): Trading pair symbol, e.g. "BTCUSDT".
            loader_type (LoaderType): Loader type to use.
            inverse_price (bool): If True, compute the inverse of the close price.
            interval (str): Interval for candlestick data (e.g., "1m", "1h", "1d").
            start_time (datetime, optional): Start time for data retrieval.
            end_time (datetime, optional): End time for data retrieval.
        """
        super().__init__(loader_type)
        self.ticker: str = ticker
        self.inverse_price: bool = inverse_price
        self.interval: str = interval
        self.start_time: Optional[datetime] = (
            start_time.replace(tzinfo=UTC) if start_time else None
        )
        self.end_time: Optional[datetime] = (
            end_time.replace(tzinfo=UTC) if end_time else None
        )
        self._url: str = "https://fapi.binance.com/fapi/v1/klines"

    def get_klines(self) -> List[Any]:
        """
        Retrieve Kline/Candlestick data from Binance API.

        Returns:
            List[Any]: A list of candlestick records.
        """
        if self.end_time is None:
            end_time_ms = int(time() * 1000)
        else:
            end_time_ms = int(self.end_time.timestamp() * 1000)

        start_time_obj = self.start_time

        interval_mapping = {
            "m": 60 * 1000,           # minutes
            "h": 60 * 60 * 1000,       # hours
            "d": 24 * 60 * 60 * 1000,   # days
            "w": 7 * 24 * 60 * 60 * 1000  # weeks
        }
        unit = self.interval[-1]
        try:
            amount = int(self.interval[:-1])
        except ValueError as err:
            raise ValueError(
                "Invalid interval format. Example valid formats: '15m', '1h', '1d'."
            ) from err
        if unit not in interval_mapping:
            raise ValueError(
                f"Interval unit '{unit}' not supported. Supported units: {list(interval_mapping.keys())}."
            )

        candle_duration_ms = amount * interval_mapping[unit]
        step_ms = 1000 * candle_duration_ms

        klines = []
        while True:
            start_ms = end_time_ms - step_ms
            url = (
                f"{self._url}?symbol={self.ticker}"
                f"&interval={self.interval}"
                f"&limit=1000"
                f"&endTime={end_time_ms}"
                f"&startTime={start_ms}"
            )
            response = requests.get(url, timeout=10).json()
            if not response:
                break

            for item in response:
                item_open_time = int(item[0])
                if start_time_obj is not None and item_open_time < int(start_time_obj.timestamp() * 1000):
                    continue
                klines.append({
                    "openTime": datetime.fromtimestamp(item_open_time / 1000, tz=UTC),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                })

            if len(response) < 1000 or (
                start_time_obj is not None
                and int(response[0][0]) < int(start_time_obj.timestamp() * 1000)
            ):
                break

            end_time_ms = int(response[0][0]) - 1000

        return klines

    def extract(self) -> None:
        """
        Extract candlestick data from Binance API.
        """
        self._data = pd.DataFrame(self.get_klines())

    def transform(self) -> None:
        """
        Transform the raw candlestick data.
        """
        self._data = self._data.sort_values("openTime")
        self._data["close"] = self._data["close"].astype(float)
        if self.inverse_price:
            self._data["close"] = 1 / self._data["close"]

    def load(self) -> None:
        """
        Load the data using the internal _load mechanism.
        """
        self._load(self.ticker)

    def read(self, with_run: bool = False) -> PriceHistory:
        """
        Read and return the price history.

        Args:
            with_run (bool): Whether to run the extraction pipeline before reading.

        Returns:
            PriceHistory: The price history data.
        """
        if with_run:
            self.run()
        else:
            self._read(self.ticker)
        return PriceHistory(
            prices=self._data["close"].astype(float).values,
            time=pd.to_datetime(self._data["openTime"], utc=True),
        )


class BinanceDayPriceLoader(BinancePriceLoader):
    """
    Loader for daily Binance price data.
    """

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        inverse_price: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Initialize the BinanceDayPriceLoader.

        Args:
            ticker (str): Trading pair symbol.
            loader_type (LoaderType): Loader type to use.
            inverse_price (bool): If True, compute the inverse of the close price.
            start_time (datetime, optional): Start time for data retrieval.
            end_time (datetime, optional): End time for data retrieval.
        """
        super().__init__(
            ticker=ticker,
            loader_type=loader_type,
            inverse_price=inverse_price,
            interval="1d",
            start_time=start_time,
            end_time=end_time,
        )


class BinanceHourPriceLoader(BinancePriceLoader):
    """
    Loader for hourly Binance price data.
    """

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        inverse_price: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Initialize the BinanceHourPriceLoader.

        Args:
            ticker (str): Trading pair symbol.
            loader_type (LoaderType): Loader type to use.
            inverse_price (bool): If True, compute the inverse of the close price.
            start_time (datetime, optional): Start time for data retrieval.
            end_time (datetime, optional): End time for data retrieval.
        """
        super().__init__(
            ticker=ticker,
            loader_type=loader_type,
            inverse_price=inverse_price,
            interval="1h",
            start_time=start_time,
            end_time=end_time,
        )


class BinanceMinutePriceLoader(BinancePriceLoader):
    """
    Loader for minute-based Binance price data.
    """

    def __init__(
        self,
        ticker: str,
        loader_type: LoaderType = LoaderType.CSV,
        inverse_price: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Initialize the BinanceMinutePriceLoader.

        Args:
            ticker (str): Trading pair symbol.
            loader_type (LoaderType): Loader type to use.
            inverse_price (bool): If True, compute the inverse of the close price.
            start_time (datetime, optional): Start time for data retrieval.
            end_time (datetime, optional): End time for data retrieval.
        """
        super().__init__(
            ticker=ticker,
            loader_type=loader_type,
            inverse_price=inverse_price,
            interval="1m",
            start_time=start_time,
            end_time=end_time,
        )


class BinanceKlinesLoader(BinancePriceLoader):
    """
    Loader for Binance Klines (candlestick) data that returns OHLC values.
    """

    def read(self, with_run: bool = False) -> KlinesHistory:
        """
        Read and return the kline history.

        Args:
            with_run (bool): Whether to run the extraction pipeline before reading.

        Returns:
            KlinesHistory: The OHLC kline history.
        """
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


class BinanceDayPriceLoader(BinancePriceLoader):

    def __init__(
            self, ticker: str, loader_type: LoaderType, inverse_price: bool = False,
            start_time: datetime = None, end_time: datetime = None):
        super().__init__(
            ticker=ticker, loader_type=loader_type, inverse_price=inverse_price, interval='1d',
            start_time=start_time, end_time=end_time
        )


class BinanceHourPriceLoader(BinancePriceLoader):

    def __init__(
            self, ticker: str, loader_type: LoaderType, inverse_price: bool = False,
            start_time: datetime = None, end_time: datetime = None):
        super().__init__(
            ticker=ticker, loader_type=loader_type, inverse_price=inverse_price, interval='1h',
            start_time=start_time, end_time=end_time
        )


class BinanceMinutePriceLoader(BinancePriceLoader):

    def __init__(
            self, ticker: str, loader_type: LoaderType, inverse_price: bool = False,
            start_time: datetime = None, end_time: datetime = None):
        super().__init__(
            ticker=ticker, loader_type=loader_type, inverse_price=inverse_price, interval='1m',
            start_time=start_time, end_time=end_time
        )
