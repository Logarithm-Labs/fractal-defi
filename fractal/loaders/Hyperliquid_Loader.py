from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory, PriceHistory


class HyperliquidFundingRatesLoader(Loader):

    def __init__(self, ticker: str, loader_type: LoaderType, start_time: Optional[datetime]):
        super().__init__(loader_type)

        self.ticker: str = ticker
        self._url: str = 'https://api.hyperliquid.xyz/info'
        self.secret_key = ''
        self.public_key = ''

    def extract(self, ticker: str, start_time: datetime = None, end_time: datetime = None):
        """

        Args:
            ticker:         Hyperliquid asset alias
            start_time:     Collection start time
            end_time:       Collection end time

        Returns:
            pd.DataFrame: pandas Dataframe with collected funding rates and premiums

        """

        headers = {
            "Content-Type": "application/json"
        }

        if start_time is None:
            start_time = int(datetime.strptime("2023-01-01", "%Y-%m-%d").timestamp() * 1000)
        else:
            start_time = int(datetime.strptime(str(start_time), "%Y-%m-%d").timestamp() * 1000)

        if end_time is None:
            end_time = int(datetime.strptime("2023-06-01", "%Y-%m-%d").timestamp() * 1000)
        else:
            end_time = int(datetime.strptime(str(end_time), "%Y-%m-%d").timestamp() * 1000)

        param_dict = {'type': "fundingHistory",
                      'coin': self.ticker,
                      'startTime': start_time,
                      'endTime': end_time}

        response = requests.post(self._url, headers=headers, json=param_dict)

        if response.status_code == 200:
            self._data = pd.DataFrame(response.json())
        else:
            print(f'Failed to make request to {self._url}: status code: {response.status_code} ({response.text})')

    def transform(self):
        self._data['time'] = pd.to_datetime(self._data['time'], unit='ms', origin='unix')
        self._data['fundingRate'] = self._data['fundingRate'].astype(float)
        self._data['premium'] = self._data['premium'].astype(float)

    def load(self):
        self._load(self.ticker)

    def read(self, with_run: bool = False):
        """
        Reads the funding history data from the HyperLiquid loader.

        Args:
            with_run: with_run (bool, optional): If True, runs the loader before reading the data. Defaults to False.

        Returns:
            FundingHistory: structure with funding rate values

        """
        if with_run:
            self.run()
        else:
            self._read(self.ticker)
        return FundingHistory(
            rates=self._data['fundingRate'],
            time=pd.to_datetime(self._data['time'])
        )


class HyperLiquidPerpsPricesLoader(Loader):

    def __init__(self, ticker: str, loader_type: LoaderType):
        super().__init__(loader_type)
        self.ticker: str = ticker
        self._url: str = 'https://api.hyperliquid.xyz/info'

    def extract(self, ticker: str, interval: str, start_time: datetime = None, end_time: datetime = None):
        """

        Args:
            ticker:         Hyperliquid asset alias
            interval:       kline collection interval; Available intervals: ("1m", "5m", "15m", "1h", "1d")
            start_time:     Collection start time
            end_time:       Collection end time

        Returns:
            pd.DataFrame: pandas Dataframe with collected price klines data

        """

        headers = {
            "Content-Type": "application/json"
        }

        if start_time is None:
            start_time = int(datetime.strptime("2023-05-12", "%Y-%m-%d").timestamp() * 1000)
        else:
            start_time = int(datetime.strptime(str(start_time), "%Y-%m-%d").timestamp() * 1000)

        if end_time is None:
            end_time = int(datetime.strptime("2023-05-13", "%Y-%m-%d").timestamp() * 1000)
        else:
            end_time = int(datetime.strptime(str(end_time), "%Y-%m-%d").timestamp() * 1000)

        paramdict = {
            "type": "candleSnapshot",
            "req": {
                "coin": self.ticker,
                "interval": interval,
                "startTime": start_time,
                "endTime": end_time
            }
        }

        response_klines = requests.post(self._url, headers=headers, json=paramdict, timeout=3)

        # Check if the request was successful
        if response_klines.status_code == 200:
            self._data = pd.DataFrame(response_klines.json())
        else:
            print(f'Failed to make request to {self._url}: '
                  f'status code: {response_klines.status_code} ({response_klines.text})')

    def transform(self):
        self._data['t'] = pd.to_datetime(self._data['t'], unit='ms', origin='unix')
        self._data['T'] = pd.to_datetime(self._data['T'], unit='ms', origin='unix')
        self._data['o'] = self._data['o'].astype(float)
        self._data['c'] = self._data['c'].astype(float)
        self._data['h'] = self._data['h'].astype(float)
        self._data['l'] = self._data['l'].astype(float)

        self._data.rename(columns={"t": "open_time",
                                   "T": "close_time",
                                   "s": "ticker",
                                   "i": "interval",
                                   "o": "open_price",
                                   "c": "close_price",
                                   "h": "highest_price",
                                   "l": "lowest_price",
                                   "v": "volume"})

    def load(self):
        self._load(self.ticker)

    def read(self, with_run: bool = False) -> PriceHistory:
        """
        Reads the funding history data from the HyperLiquid loader.

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
            prices=self._data['close_price'].astype(float).values,
            time=pd.to_datetime(self._data['close_time']).values
        )
