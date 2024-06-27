from datetime import datetime
from time import time

import pandas as pd
import requests

from fractal.loaders.loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory, PriceHistory


class BinanceDayPriceLoader(Loader):

    def __init__(self, ticker: str, loader_type: LoaderType, inverse_price: bool = False):
        super().__init__(loader_type)
        self.ticker: str = ticker
        self.inverse_price: bool = inverse_price
        self._url: str = "https://api.binance.com/api/v3/klines"

    def extract(self):
        # Load data from binance
        response = requests.get(
            f"{self._url}?symbol={self.ticker}&interval=1d&limit=1000",
            timeout=10
        )
        # Convert to pandas dataframe
        self._data = pd.DataFrame(response.json())

    def transform(self):
        self._data['date'] = pd.to_datetime(self._data[0], unit='ms')
        self._data['close'] = self._data[4].astype(float)
        if self.inverse_price:
            self._data['close'] = 1 / self._data['close']

    def load(self):
        self._load(self.ticker)

    def read(self, with_run: bool = False) -> PriceHistory:
        """
        Reads the price history data from the Binance loader.

        Args:
            with_run (bool, optional): If True, runs the loader before reading the data. Defaults to False.

        Returns:
            PriceHistory: The price history data.
        """
        if with_run:
            self.run()
        else:
            self._read(self.ticker)
        return PriceHistory(
            prices=self._data['close'].astype(float).values,
            time=pd.to_datetime(self._data['date']).values
        )


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


class BinanceHourPriceLoader(Loader):

    def __init__(self, ticker: str, loader_type: LoaderType = LoaderType.CSV,
                 start_time: datetime = None, end_time: datetime = None,
                 inverse_price: bool = False):
        super().__init__(loader_type)
        self.ticker: str = ticker
        self.start_time: datetime = start_time
        self.end_time: datetime = end_time
        self.inverse_price: bool = inverse_price
        self._url: str = "https://fapi.binance.com/fapi/v1/klines"

    def get_klines(self, ticker: str, start_time: datetime = None, end_time: datetime = None) -> list:
        """
        Get Kline/Candlestick Data
        Klines are uniquely identified by their open time.

        Args:
            ticker: str
            start_time: datetime
            end_time: datetime

        Returns:
            list: List of klines
        """
        if end_time is None:
            end_time = int(time() * 1000)

        step: float = 1000 * 60 * 60 * 1000
        klines = []
        while True:
            response = requests.get(
                f"{self._url}?symbol={ticker}&interval=1h&limit=1000&endTime={end_time}&startTime={end_time - step}",
                timeout=10
            ).json()
            for item in response:
                klines.append(
                    {
                        'openTime': datetime.utcfromtimestamp(item[0] / 1000),
                        'open': float(item[1]),
                        'high': float(item[2]),
                        'low': float(item[3]),
                        'close': float(item[4]),
                        'volume': float(item[5])
                    }
                )
            if len(response) < 1000:
                break
            if start_time is not None:
                if min([int(x[0]) for x in response]) < start_time.timestamp() * 1000:
                    break
            # shift start time by 1s from the newest loaded datapoint time
            end_time = response[0][0] - 1000
        return klines

    def extract(self):
        data = self.get_klines(self.ticker, start_time=self.start_time, end_time=self.end_time)
        self._data = pd.DataFrame(data)

    def transform(self):
        # self._data['openTime'] -= timedelta(hours=1)
        self._data['date'] = pd.to_datetime(self._data['openTime'], unit='ms')
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
