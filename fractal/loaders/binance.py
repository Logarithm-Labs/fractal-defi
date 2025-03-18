from datetime import UTC, datetime
from time import time

import pandas as pd
import requests

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory, PriceHistory


class BinanceFundingLoader(Loader):

    def __init__(self, ticker: str, loader_type: LoaderType = LoaderType.CSV,
                 start_time: datetime = None, end_time: datetime = None):
        super().__init__(loader_type)
        self.ticker: str = ticker
        self.start_time: datetime = start_time
        self.end_time: datetime = end_time
        self._url: str = "https://fapi.binance.com/fapi/v1/fundingRate"

    def get_funding_rates(self, ticker: str, start_time: datetime = None, end_time: datetime = None) -> list:
        """
        Get funding rate history

        Args:
            ticker: str
            start_time: datetime
            end_time: datetime

        Returns:
            list: List of funding rates
        """

        funding_rates = []

        if end_time is None:
            end_time = int(time() * 1000)

        step: float = 1000 * 60 * 60 * 8 * 1000
        while True:
            response = requests.get(
                f"{self._url}?symbol={ticker}&limit=1000&endTime={end_time}&startTime={end_time - step}",
                timeout=10
            ).json()
            for item in response:
                funding_rates.append(
                    {
                        'fundingTime': datetime.utcfromtimestamp(item['fundingTime'] / 1000),
                        'fundingRate': float(item['fundingRate']),
                        'ticker': item['symbol']
                    }
                )
            if len(response) < 1000:
                break
            if start_time is not None:
                if min([int(x['fundingTime']) for x in response]) < start_time.timestamp() * 1000:
                    break
            # shift start time by 1s from the newest loaded datapoint time
            end_time = response[0]['fundingTime'] - 1000
        return funding_rates

    def extract(self):
        data = self.get_funding_rates(self.ticker, start_time=self.start_time, end_time=self.end_time)
        self._data = pd.DataFrame(data)

    def transform(self):
        self._data['date'] = pd.to_datetime(self._data['fundingTime'], unit='ms')
        self._data = self._data.sort_values('date')

    def load(self):
        self._load(self.ticker)

    def read(self, with_run: bool = False) -> FundingHistory:
        if with_run:
            self.run()
        else:
            self._read(self.ticker)
        return FundingHistory(
            rates=self._data['fundingRate'].astype(float).values,
            time=pd.to_datetime(self._data['date']).values
        )


class BinancePriceLoader(Loader):

    def __init__(
            self, ticker: str, loader_type: LoaderType,
            inverse_price: bool = False, interval: str = '1h',
            start_time: datetime = None, end_time: datetime = None):
        """
        Binance Price Loader
        Load price data from Binance API for a given ticker.

        Args:
            ticker (str): ticker to load data for. e.g. 'BTCUSDT'
            loader_type (LoaderType): Loader type to use. e.g. LoaderType.CSV
            inverse_price (bool, optional): if inverse_price returns 1/p. Defaults to False.
            interval (str, optional): interval of klines (1m, 1h, 1d). Defaults to '1h'.
        """
        super().__init__(loader_type)
        self.ticker: str = ticker
        self.inverse_price: bool = inverse_price
        self.interval: str = interval
        self.start_time: datetime = start_time.replace(tzinfo=UTC) if start_time else None
        self.end_time: datetime = end_time.replace(tzinfo=UTC) if end_time else None
        self._url: str = "https://fapi.binance.com/fapi/v1/klines"

    def get_klines(self) -> list:
        """
        Get Kline/Candlestick Data.
        Klines are uniquely identified by their open time.

        Returns:
            list: List of klines with keys 'openTime', 'open', 'high', 'low', 'close', 'volume'
        """
        # Set the time boundaries in milliseconds.
        end_time = self.end_time
        start_time = self.start_time
        if end_time is None:
            end_time_ms = int(time() * 1000)
        else:
            end_time_ms = int(end_time.timestamp() * 1000)

        # Convert self.interval (e.g. "15m", "1h", "1d") to the length of one candle in milliseconds.
        # You can extend this mapping if you have other interval types.
        interval_mapping = {
            'm': 60 * 1000,          # minutes
            'h': 60 * 60 * 1000,     # hours
            'd': 24 * 60 * 60 * 1000,  # days
            'w': 7 * 24 * 60 * 60 * 1000  # weeks
        }
        unit = self.interval[-1]
        try:
            amount = int(self.interval[:-1])
        except ValueError:
            raise ValueError("Invalid interval format. Example valid formats: '15m', '1h', '1d'.")
        if unit not in interval_mapping:
            raise ValueError(f"Interval unit '{unit}' not supported. Supported units: {list(interval_mapping.keys())}.")

        # Each candle’s duration in ms
        candle_duration_ms = amount * interval_mapping[unit]
        # Step covers 1000 candles (matching the 'limit=1000' in the API call)
        step_ms = 1000 * candle_duration_ms

        klines = []
        while True:
            start_ms = end_time_ms - step_ms
            # Build URL with parameters.
            url = (
                f"{self._url}?symbol={self.ticker}"
                f"&interval={self.interval}"
                f"&limit=1000"
                f"&endTime={end_time_ms}"
                f"&startTime={start_ms}"
            )
            response = requests.get(url, timeout=10).json()
            # If no data is returned, exit the loop.
            if not response:
                break

            # Process each returned candle.
            for item in response:
                item_open_time = int(item[0])
                # If start_time is provided and this candle is older than it, skip it.
                if start_time is not None and item_open_time < int(start_time.timestamp() * 1000):
                    continue
                klines.append({
                    'openTime': datetime.utcfromtimestamp(item_open_time / 1000),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                })

            # If fewer than 1000 candles were returned, or the oldest candle is older than our start_time,
            # we have retrieved all the desired data.
            if len(response) < 1000 or (
                start_time is not None and int(response[0][0]) < int(start_time.timestamp() * 1000)
            ):
                break

            # Update end_time_ms to just before the oldest candle in the current batch (shift by 1 second).
            end_time_ms = int(response[0][0]) - 1000

        return klines

    def extract(self):
        self._data = pd.DataFrame(self.get_klines())

    def transform(self):
        self._data['date'] = pd.to_datetime(self._data['openTime'], unit='ms', utc=True)
        self._data = self._data.sort_values('date')
        self._data['close'] = self._data['close'].astype(float)
        if self.inverse_price:
            self._data['close'] = 1 / self._data['close']

    def load(self):
        self._load(self.ticker)

    def read(self, with_run: bool = False) -> PriceHistory:
        if with_run:
            self.run()
        else:
            self._read(self.ticker)
        return PriceHistory(
            prices=self._data['close'].astype(float).values,
            time=pd.to_datetime(self._data['date']).values
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
