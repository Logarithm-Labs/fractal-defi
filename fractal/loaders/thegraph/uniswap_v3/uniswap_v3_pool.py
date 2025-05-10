from datetime import UTC, datetime
from typing import Optional

import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import PoolHistory
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_arbitrum import \
    ArbitrumUniswapV3Loader
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_ethereum import \
    EthereumUniswapV3Loader


class UniswapV3EthereumPoolDayDataLoader(EthereumUniswapV3Loader):

    def __init__(self, api_key: str, pool: str, loader_type: LoaderType = LoaderType.CSV) -> None:
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
                orderDirection: desc
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
        self._data['date'] = self._data['date'].astype(int).apply(lambda x: datetime.fromtimestamp(x, UTC))
        self._data['date'] = self._data['date'].dt.date
        self._data['volume'] = self._data['volumeUSD'].astype(float)
        self._data['tvl'] = self._data['tvlUSD'].astype(float)
        self._data['fees'] = self._data['feesUSD'].astype(float)
        self._data['liquidity'] = self._data['liquidity'].astype(float)

    def load(self):
        self._load(self.pool)

    def read(self, with_run: bool = False) -> PoolHistory:
        if with_run:
            self.run()
        else:
            self._read(self.pool)
        self._data['date'] = pd.to_datetime(self._data['date'])
        return PoolHistory(
            tvls=self._data['tvl'].astype(float).values,
            volumes=self._data['volume'].astype(float).values,
            fees=self._data['fees'].astype(float).values,
            liquidity=self._data['liquidity'].astype(float).values,
            time=pd.to_datetime(self._data['date'], utc=True),
            price=self._data['token0_price'].astype(float).values,
            open=self._data['open'].astype(float).values,
            high=self._data['high'].astype(float).values,
            low=self._data['low'].astype(float).values,
            close=self._data['close'].astype(float).values
        )


class UniswapV3ArbitrumPoolDayDataLoader(ArbitrumUniswapV3Loader):

    def __init__(self, api_key: str, pool: str, loader_type: LoaderType = LoaderType.CSV) -> None:
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
        self._data['date'] = pd.to_datetime(self._data['timestamp'].astype(int) * 1000, utc=True)
        self._data['date'] = self._data['date'].dt.date
        self._data['volume'] = 0  # mocked
        self._data['fees'] = self._data['dailyTotalRevenueUSD'].astype(float)
        self._data['tvl'] = self._data['totalValueLockedUSD'].astype(float)
        self._data['liquidity'] = self._data['activeLiquidity'].astype(float)

    def load(self):
        self._load(self.pool)

    def read(self, with_run: bool = False) -> PoolHistory:
        if with_run:
            self.run()
        else:
            self._read(self.pool)
        return PoolHistory(
            tvls=self._data['tvl'].astype(float).values,
            volumes=self._data['volume'].astype(float).values,
            fees=self._data['fees'].astype(float).values,
            liquidity=self._data['liquidity'].astype(float).values,
            time=pd.to_datetime(self._data['date'], utc=True),
        )


class UniswapV3ArbitrumPoolHourDataLoader(UniswapV3ArbitrumPoolDayDataLoader):

    def __init__(self, api_key: str, pool: str, loader_type: LoaderType = LoaderType.CSV) -> None:
        super().__init__(api_key=api_key, pool=pool, loader_type=loader_type)

    def transform(self):
        super().transform()
        # stretch daily data to hourly and data values devided by 24
        self._data.index = pd.to_datetime(self._data['date'])
        self._data.drop(columns=['date'], inplace=True)
        self._data.index = self._data.index.to_period('D')
        self._data = self._data.resample('H').ffill()
        self._data['fees'] /= 24
        self._data['volume'] /= 24
        self._data.reset_index(inplace=True)
        self._data['date'].dt.to_timestamp()


class UniswapV3EthereumPoolHourDataLoader(UniswapV3EthereumPoolDayDataLoader):

    def __init__(
        self,
        api_key: str,
        pool: str,
        loader_type: LoaderType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        super().__init__(api_key=api_key, pool=pool, loader_type=loader_type)
        self.start_time = start_time
        self.end_time = end_time

    def extract(self):
        current_loaded_time = self.start_time.timestamp() if self.start_time else 0
        end_time = self.end_time.timestamp() if self.end_time else datetime.now().timestamp()

        batches = []
        while current_loaded_time < end_time:
            query = """
            {
            poolHourDatas(
                first: 1000
                orderBy: periodStartUnix
                orderDirection: asc
                where: {pool: "%s", periodStartUnix_gte: %s}
            ) {
                periodStartUnix
                volumeUSD
                tvlUSD
                feesUSD
                liquidity,
                token0Price,
                open,
                high,
                low,
                close
                }
            }
            """ % (self.pool.lower(), current_loaded_time)
            response = self._make_request(query)
            batch = pd.DataFrame(response['poolHourDatas'])

            if current_loaded_time == batch['periodStartUnix'].astype(int).max():
                break

            current_loaded_time = batch['periodStartUnix'].astype(int).max()
            batches.append(batch)

        self._data = pd.concat(batches)

    def transform(self):
        self._data['date'] = self._data['periodStartUnix'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x))
        self._data['volume'] = self._data['volumeUSD'].astype(float)
        self._data['tvl'] = self._data['tvlUSD'].astype(float)
        self._data['fees'] = self._data['feesUSD'].astype(float)
        self._data['liquidity'] = self._data['liquidity'].astype(float)
        self._data['token0_price'] = self._data['token0Price'].astype(float)
        self._data['open'] = self._data['open'].astype(float)
        self._data['high'] = self._data['high'].astype(float)
        self._data['low'] = self._data['low'].astype(float)
        self._data['close'] = self._data['close'].astype(float)


class UniswapV3EthereumPoolMinuteDataLoader(UniswapV3EthereumPoolDayDataLoader):

    def __init__(self, api_key: str, pool: str, loader_type: LoaderType = LoaderType.CSV) -> None:
        super().__init__(api_key=api_key, pool=pool, loader_type=loader_type)

    def transform(self):
        super().transform()
        # stretch daily data to hourly and data values devided by 24 * 60
        self._data.index = pd.to_datetime(self._data['date'])
        self._data.drop(columns=['date'], inplace=True)
        self._data.index = self._data.index.to_period('D')
        self._data = self._data.resample('min').ffill()
        self._data['fees'] /= 24 * 60
        self._data['volume'] /= 24 * 60
        self._data.reset_index(inplace=True)
        self._data['date'] = self._data['date'].dt.to_timestamp()
