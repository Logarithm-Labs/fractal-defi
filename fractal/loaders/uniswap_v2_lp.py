import json
import time

import numpy as np
import pandas as pd
import requests

from fractal.loaders.loader import Loader, LoaderType
from fractal.loaders.structs import PoolHistory


class UniswapV2LPLoader(Loader):

    def __init__(self, pool: str, api_key: str, fee_tier: float) -> None:
        super().__init__(LoaderType.CSV)
        self.pool: str = pool
        self.url: str = f"https://gateway-arbitrum.network.thegraph.com/api/{api_key}\
                        /subgraphs/id/EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLRFu"
        self.decimals: int = None
        self.fee_tier: float = fee_tier

    def extract(self):

        dfs = []
        timestamp = int(time.time())
        while True:
            query = """
            {
                pairHourDatas(
                    orderBy: hourStartUnix
                    orderDirection: desc
                    where: {pair: "%s", hourStartUnix_lt: %s}
                ) {
                    hourStartUnix
                    hourlyVolumeUSD
                    totalSupply
                    pair {
                    id
                    token1Price
                    token0Price
                    token1 {
                        decimals
                    }
                    token0 {
                        decimals
                    }
                    reserveUSD
                    }
                }
            }
            """ % self.pool.lower(), timestamp
            response = requests.post(self.url, json={"query": query}, timeout=10)
            if response.status_code != 200:
                time.sleep(1)
                continue
            data = json.loads(response.text)
            if not data["data"]["pairHourDatas"]:
                break
            timestamp = data["data"]["pairHourDatas"][-1]["hourStartUnix"]
            time_ = [int(x["hourStartUnix"]) for x in data["data"]["pairHourDatas"]]
            volume = [float(x["hourlyVolumeUSD"])
                      for x in data["data"]["pairHourDatas"]]
            liquidity = [float(x["totalSupply"])
                         for x in data["data"]["pairHourDatas"]]
            tvl = [float(x["pair"]["reserveUSD"])
                   for x in data["data"]["pairHourDatas"]]
            token0Price = np.array([float(x["pair"]["token0Price"])
                                    for x in data["data"]["pairHourDatas"]])
            token1Price = np.array([float(x["pair"]["token1Price"])
                                    for x in data["data"]["pairHourDatas"]])
            price = token0Price / token1Price
            dfs.append(
                pd.DataFrame(
                    {
                        "time": time_,
                        "volume": volume,
                        "liquidity": liquidity,
                        "tvl": tvl,
                        "price": price,
                    }
                )
            )
            if self.decimals is None:
                self.decimals = (
                    data["data"]["pairHourDatas"][0]["token1"]["decimals"]
                    - data["data"]["pairHourDatas"][0]["token0"]["decimals"]
                )
        self._data = pd.concat(dfs)

    def transform(self):
        self._data["time"] = pd.to_datetime(self._data["timestamp"].astype(int), unit="s")
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
            prices=self._data["price"].values,
            tvls=self._data["tvl"].values,
            volumes=self._data["volume"].values,
            fees=self._data["fees"].values,
            liquidity=self._data["liquidity"].values,
        )
