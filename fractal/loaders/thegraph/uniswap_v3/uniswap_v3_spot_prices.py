import time
from string import Template

import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import PriceHistory
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_arbitrum import \
    ArbitrumUniswapV3Loader


class UniswapV3ArbitrumPricesLoader(ArbitrumUniswapV3Loader):

    def __init__(self, api_key: str, pool: str, loader_type: LoaderType, **kwargs) -> None:
        """
        Args:
            api_key (str): The Graph API key
            pool (str): Pool address
            loader_type (LoaderType): loader type
        """
        super().__init__(api_key=api_key, loader_type=loader_type)
        self.pool: str = pool
        self.decimals: int = kwargs.get("decimals", None)
        if self.decimals is None:
            decimals0, decimals1 = self.get_pool_decimals(pool)
            self.decimals = decimals0 - decimals1

    def extract(self):
        dfs = []
        timestamp = int(time.time())
        query_template = Template("""
        {
            liquidityPoolHourlySnapshots(
                first: 1000
                where: {pool: "$pool", timestamp_lt: "$timestamp"}
                orderBy: timestamp
                orderDirection: desc
            ) {
                tick
                timestamp
            }
        }
        """)
        while True:
            query = query_template.substitute(pool=self.pool.lower(), timestamp=timestamp)
            data = self._make_request(query)
            if data is None or data["liquidityPoolHourlySnapshots"] is None or\
                len(data["liquidityPoolHourlySnapshots"]) == 0:
                break
            timestamp = data["liquidityPoolHourlySnapshots"][-1]["timestamp"]
            dfs.append(pd.DataFrame(data["liquidityPoolHourlySnapshots"])[["timestamp", "tick"]])
        self._data = pd.concat(dfs)

    def transform(self):
        self._data["time"] = pd.to_datetime(self._data["timestamp"].astype(int), unit="s")
        self._data["tick"] = self._data["tick"].astype(int)
        self._data["price"] = self._data["tick"].apply(lambda x: 1.0001**x) * 10**self.decimals
        self._data = self._data[["time", "price"]]
        self._data = self._data.sort_values("time")
        self._data = self._data.drop_duplicates("time", keep="last")
        self._data = self._data.set_index("time")
        self._data = self._data.resample("1h").ohlc()
        self._data = pd.DataFrame(self._data["price"]["close"].shift(1).ffill().dropna())
        self._data = self._data.reset_index()
        self._data.columns = ["time", "price"]

    def load(self):
        self._load(self.pool)

    def read(self, with_run: bool = False) -> PriceHistory:
        if with_run:
            self.run()
        else:
            self._load(self.pool)
        return PriceHistory(
            time=self._data["time"].values,
            prices=self._data["price"].values,
        )
