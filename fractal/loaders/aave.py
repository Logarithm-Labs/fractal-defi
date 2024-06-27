from datetime import datetime
from typing import List

import pandas as pd
import requests
from dateutil import tz

from fractal.loaders.loader import Loader, LoaderType
from fractal.loaders.structs import LendingHistory


class AaveLoader(Loader):

    def __init__(self, reserve_id: str, loader_type: LoaderType, url: str,
                 start_time: datetime = None, resolution: int = 1):
        """
        Args:
            reserve_id (str): The id of the Aave reserve you want to query.
                                For V2 markets: assetAddress + lendingPoolAddressesProvider.
                                For V3 markets: assetAddress + poolAddressesProvider + chainId.
            loader_type (LoaderType): loader type
            start_time (datetime, optional): The date for where you want to start from. Defaults to None.
            resolution (str, optional): The resolution in hours.
            For example, a resolution of 6 means return rates at every 6 hour interval. Defaults to '1'.
        """
        super().__init__(loader_type=loader_type)
        self.reserve_id: str = reserve_id.lower()
        if start_time is None:
            start_time = datetime(2022, 1, 1)
        self.start_time: datetime = start_time
        self.loader_type: LoaderType = loader_type
        self._url: str = url
        self._resolution: int = resolution

    def extract(self):
        # set tzinfo to UTC
        self.start_time = self.start_time.replace(tzinfo=tz.UTC)
        response = requests.get(f'{self._url}\
                                ?reserveId={self.reserve_id}\
                                &from={self.start_time.timestamp()}\
                                &resolutionInHours={self._resolution}',
                                timeout=10)
        self._data: List = response.json()

    def transform(self):
        for elem in self._data:
            elem['date'] = f'{elem["x"]["year"]}/{elem["x"]["month"] + 1}/\
                             {elem["x"]["date"]} {elem["x"]["hours"]}:00:00'
            elem.pop('x')

        self._data = pd.DataFrame(self._data)
        self._data['borrowing_rate'] = -self._data['variableBorrowRate_avg'].astype(float)
        self._data['lending_rate'] = -self._data['liquidityRate_avg'].astype(float)
        for col in ['borrowing_rate', 'lending_rate']:
            self._data[col] = self._data[col] / (365 * 24 / (self._resolution))
        self._data['date'] = pd.to_datetime(self._data['date'])

    def load(self):
        self._load(self.reserve_id)

    def read(self, with_run: bool = False) -> LendingHistory:
        if with_run:
            self.run()
        else:
            self._read(self.reserve_id)
        return LendingHistory(
            borrowing_rates=self._data['borrowing_rate'].astype(float).values,
            lending_rates=self._data['lending_rate'].astype(float).values,
            time=pd.to_datetime(self._data['date']).values
        )


class AaveV2EthereumLoader(AaveLoader):

    def __init__(self, asset: str, loader_type: LoaderType, start_time: datetime = None, resolution: int = 1):
        """
        Args:
            asset (str): asset address
            loader_type (LoaderType): loader type
            start_time (datetime, optional): data start time. Defaults to None.
            resolution (int, optional): interval size. Defaults to 1.
        """
        reserve_id: str = asset + '0xB53C1a33016B2DC2fF3653530bfF1848a515c8c5'
        url: str = 'https://aave-api-v2.aave.com/data/rates/history'
        super().__init__(
            reserve_id=reserve_id,
            loader_type=loader_type,
            url=url,
            start_time=start_time,
            resolution=resolution
        )


class AaveV3ArbitrumLoader(AaveLoader):

    def __init__(self, asset_address: str, loader_type: LoaderType, start_time: datetime = None, resolution: int = 1):
        """
        Args:
            asset_address: str - asset address
            loader_type (LoaderType): loader type - csv, json, sql
            start_time (datetime, optional): The date for where you want to start from.
            Defaults to None.
            resolution (str, optional): The resolution in hours.
            For example, a resolution of 6 means return rates at every 6 hour interval.
            Defaults to '1'.
        """
        reserve_id: str = asset_address + '0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb' + '42161'
        url: str = 'https://aave-api-v2.aave.com/data/rates-history'
        super().__init__(
            reserve_id=reserve_id,
            loader_type=loader_type,
            start_time=start_time,
            resolution=resolution,
            url=url
        )
