import time
from string import Template

import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import PoolHistory
from fractal.loaders.thegraph.uniswap_v2.uniswap_v2_ethereum import \
    EthereumUniswapV2Loader


class EthereumUniswapV2PoolDataLoader(EthereumUniswapV2Loader):
    """
    Loader for Uniswap V2 PoolData
    """
    def __init__(self, api_key: str, pool: str, fee_tier: float, loader_type: LoaderType) -> None:
        """
        Args:
            api_key (str): The Graph API key
            pool (str): pool address
            fee_tier (float): fee tier - it will be used to calculate fees
            loader_type (LoaderType): loader type
        """
        super().__init__(api_key=api_key, loader_type=loader_type)
        self.pool: str = pool
        self.fee_tier: float = fee_tier

    def extract(self):
        dfs = []
        timestamp = int(time.time())
        query_template = Template("""
        {
            pairHourDatas(
                orderBy: hourStartUnix
                orderDirection: desc
                where: {pair: "$pool", hourStartUnix_lt: $timestamp}
                first: 1000
            ) {
                hourStartUnix
                hourlyVolumeUSD
                totalSupply
                reserveUSD
            }
        }
        """)
        while True:
            query = query_template.substitute(pool=self.pool.lower(), timestamp=timestamp)
            data = self._make_request(query)
            if data is None or data["pairHourDatas"] is None or len(data["pairHourDatas"]) == 0:
                break
            timestamp = data["pairHourDatas"][-1]["hourStartUnix"]
            dfs.append(self._transform_batch(pd.DataFrame(data["pairHourDatas"])))
        self._data = pd.concat(dfs)

    def _transform_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        batch["time"] = pd.to_datetime(batch["hourStartUnix"].astype(int), unit="s")
        batch["volume"] = batch["hourlyVolumeUSD"].astype(float)
        batch["liquidity"] = batch["totalSupply"].astype(float)
        batch["tvl"] = batch["reserveUSD"].astype(float)
        batch = batch.dropna()
        # Remove rows with zero liquidity
        batch = batch[batch["liquidity"] != 0]
        return batch

    def transform(self):
        self._data["fees"] = self._data["volume"] * self.fee_tier

    def load(self):
        self._load(self.pool)

    def read(self, with_run: bool = False) -> PoolHistory:
        if with_run:
            self.run()
        else:
            self._read(self.pool)
        return PoolHistory(
            time=self._data["time"].values,
            tvls=self._data["tvl"].values,
            volumes=self._data["volume"].values,
            fees=self._data["fees"].values,
            liquidity=self._data["liquidity"].values,
        )
