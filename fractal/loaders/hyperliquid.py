import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory, PriceHistory


class HyperliquidBaseLoader(Loader):

    def __init__(self, ticker: str, loader_type: LoaderType = LoaderType.CSV,
                 start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        super().__init__(loader_type)
        if start_time is None:
            start_time = datetime.strptime("2024-01-01", "%Y-%m-%d")
        if end_time is None:
            end_time = datetime.now()
        self._start_time = int(start_time.timestamp() * 1000)
        self._end_time = int(end_time.timestamp() * 1000)
        self._url: str = 'https://api.hyperliquid.xyz/info'
        self._ticker: str = ticker

    def _make_request(self, param_dict):
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(self._url, headers=headers, json=param_dict)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f'Failed to make request to {self._url}: '
                            f'status code: {response.status_code} ({response.text})')

    def load(self):
        self._load(self._ticker)


class HyperliquidFundingRatesLoader(HyperliquidBaseLoader):

    def extract(self):
        end_time = self._end_time
        data = self._extract_batch()
        all_data = data
        last_time = data[-1]['time']
        while last_time < end_time:
            self._start_time = last_time
            data = self._extract_batch()
            if len(data) <= 1:
                break
            all_data += data
            last_time = data[-1]['time']
            time.sleep(1)
        self._data = pd.DataFrame(all_data)

    def _extract_batch(self):
        params = {
            'type': "fundingHistory",
            'coin': self._ticker,
            'startTime': self._start_time,
            'endTime': self._end_time
        }
        return self._make_request(params)

    def transform(self):
        self._data['time'] = pd.to_datetime(self._data['time'], unit='ms', origin='unix')
        self._data['time'] = self._data['time'].dt.floor('S')
        self._data['fundingRate'] = self._data['fundingRate'].astype(float)

    def read(self, with_run: bool = False):
        if with_run:
            self.run()
        else:
            self._read(self._ticker)
        return FundingHistory(
            rates=self._data['fundingRate'].values,
            time=pd.to_datetime(self._data['time'].values)
        )


class HyperLiquidPerpsPricesLoader(HyperliquidBaseLoader):

    def __init__(self, ticker: str, interval: str, loader_type: LoaderType = LoaderType.CSV,
                 start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        super().__init__(ticker, loader_type, start_time, end_time)
        self._interval: str = interval

    def extract(self):
        params = {
            "type": "candleSnapshot",
            "req": {
                "coin": self._ticker,
                "interval": self._interval,
                "startTime": self._start_time,
                "endTime": self._end_time
            }
        }
        self._data = pd.DataFrame(self._make_request(params))

    def transform(self):
        self._data['open_time'] = pd.to_datetime(self._data['t'], unit='ms', origin='unix')
        self._data['open_price'] = self._data['o'].astype(float)

    def read(self, with_run: bool = False) -> PriceHistory:
        if with_run:
            self.run()
        else:
            self._read(self._ticker)
        return PriceHistory(
            prices=self._data['open_price'].values,
            time=pd.to_datetime(self._data['open_time']).values
        )
