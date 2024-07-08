from datetime import datetime, timedelta

import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import PoolHistory
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_arbitrum import \
    ArbitrumUniswapV3Loader
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_ethereum import \
    EthereumUniswapV3Loader


class UniswapV3EthereumPoolDayDataLoader(EthereumUniswapV3Loader):

    def __init__(self, api_key: str, pool: str, loader_type: LoaderType) -> None:
        """
        Args:
            api_key (str): The Graph API key
            pool (str): Pool address
            loader_type (LoaderType): loader type
        """
        super().__init__(api_key=api_key, loader_type=loader_type)
        self.pool: str = pool

    def extract(self):
        query = """
        {
            poolDayDatas(
                first: 1000
                orderBy: date
                where: {pool: "%s"}
            ) {
                date
                volumeUSD
                tvlUSD
                feesUSD
                liquidity
            }
        }
        """ % self.pool.lower()
        response = self._make_request(query)
        self._data = pd.DataFrame(response['poolDayDatas'])

    def transform(self):
        self._data['date'] = self._data['date'].apply(lambda x: datetime.utcfromtimestamp(x))
        self._data['date'] = self._data['date'].dt.date + timedelta(days=1)
        self._data['volumeUSD'] = self._data['volumeUSD'].astype(float)
        self._data['tvlUSD'] = self._data['tvlUSD'].astype(float)
        self._data['feesUSD'] = self._data['feesUSD'].astype(float)
        self._data['liquidity'] = self._data['liquidity'].astype(float)

    def load(self):
        self._load(self.pool)

    def read(self, with_run: bool = False) -> PoolHistory:
        if with_run:
            self.run()
        else:
            self._read(self.pool)
        return PoolHistory(
            tvls=self._data['tvlUSD'].astype(float).values,
            volumes=self._data['volumeUSD'].astype(float).values,
            fees=self._data['feesUSD'].astype(float).values,
            liquidity=self._data['liquidity'].astype(float).values,
            time=self._data['date'].values
        )


class UniswapV3ArbitrumPoolDayDataLoader(ArbitrumUniswapV3Loader):

    def __init__(self, api_key: str, pool: str, loader_type: LoaderType) -> None:
        super().__init__(
            api_key=api_key,
            loader_type=loader_type,
        )
        self.pool: str = pool

    def extract(self):
        query = """
        {
            liquidityPoolDailySnapshots(
                first: 1000
                orderBy: timestamp
                where: {id_contains: "%s"}
                orderDirection: desc
            ) {
                dailyTotalRevenueUSD
                timestamp
                totalValueLockedUSD
                activeLiquidity
            }
        }
        """ % self.pool.lower()
        response = self._make_request(query)
        self._data = pd.DataFrame(response['liquidityPoolDailySnapshots'])

    def transform(self):
        self._data['date'] = self._data['timestamp'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x))
        self._data['date'] = self._data['date'].dt.date + timedelta(days=1)
        self._data['volumeUSD'] = 0  # mocked
        self._data['feesUSD'] = self._data['dailyTotalRevenueUSD'].astype(float)
        self._data['tvlUSD'] = self._data['totalValueLockedUSD'].astype(float)
        self._data['liquidity'] = self._data['activeLiquidity'].astype(float)

    def load(self):
        self._load(self.pool)

    def read(self, with_run: bool = False) -> PoolHistory:
        if with_run:
            self.run()
        else:
            self._read(self.pool)
        return PoolHistory(
            tvls=self._data['tvlUSD'].astype(float).values,
            volumes=self._data['volumeUSD'].astype(float).values,
            fees=self._data['feesUSD'].astype(float).values,
            liquidity=self._data['liquidity'].astype(float).values,
            time=self._data['date'].values
        )


class UniswapV3ArbitrumPoolHourDataLoader(UniswapV3ArbitrumPoolDayDataLoader):

    def __init__(self, api_key: str, pool: str, loader_type: LoaderType) -> None:
        super().__init__(api_key=api_key, pool=pool, loader_type=loader_type)

    def transform(self):
        super().transform()
        # stretch daily data to hourly and data values devided by 24
        self._data = PoolHistory(
            tvls=self._data['tvlUSD'].astype(float).values,
            volumes=self._data['volumeUSD'].astype(float).values,
            fees=self._data['feesUSD'].astype(float).values,
            liquidity=self._data['liquidity'].astype(float).values,
            time=pd.to_datetime(self._data['date']).values
        )

        self._data.index = self._data.index.to_period('D')
        self._data = self._data.resample('H').ffill()
        self._data['feesUSD'] = self._data['fees'] / 24
        self._data['volumeUSD'] = self._data['volume'] / 24
        self._data['tvlUSD'] = self._data['tvl']
        self._data.reset_index(inplace=True)
        self._data['date'] = self._data['index']


class UniswapV3EthereumPoolHourDataLoader(UniswapV3EthereumPoolDayDataLoader):

    def __init__(self, api_key: str, pool: str, loader_type: LoaderType) -> None:
        super().__init__(api_key=api_key, pool=pool, loader_type=loader_type)

    def transform(self):
        super().transform()
        # stretch daily data to hourly and data values devided by 24
        self._data = PoolHistory(
            tvls=self._data['tvlUSD'].astype(float).values,
            volumes=self._data['volumeUSD'].astype(float).values,
            fees=self._data['feesUSD'].astype(float).values,
            liquidity=self._data['liquidity'].astype(float).values,
            time=pd.to_datetime(self._data['date']).values
        )
        self._data.index = self._data.index.to_period('D')
        self._data = self._data.resample('H').ffill()
        self._data['feesUSD'] = self._data['fees'] / 24
        self._data['volumeUSD'] = self._data['volume'] / 24
        self._data['tvlUSD'] = self._data['tvl']
        self._data.reset_index(inplace=True)
        self._data['date'] = self._data['index']
