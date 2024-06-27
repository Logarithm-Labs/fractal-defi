import json
import time

import pandas as pd
import requests

from fractal.loaders.loader import Loader, LoaderType
from fractal.loaders.structs import PriceHistory


class UniswapV3PricesLoader(Loader):

    def __init__(self, pool: str) -> None:
        super().__init__(LoaderType.CSV)
        self.pool: str = pool
        self.url: str = "https://api.thegraph.com/subgraphs/name/messari/uniswap-v3-arbitrum"
        self.decimals: int = None

    def extract(self):
        dfs = []
        timestamp = int(time.time())
        while True:

            query = """
            {
            liquidityPoolHourlySnapshots(
                where: {pool: "%s", timestamp_lt: "%s"}
                orderBy: timestamp
                orderDirection: desc
            ) {
                tick
                pool {
                inputTokens {
                    decimals
                    symbol
                }
                id
                }
                timestamp
            }
            }
            """ % (
                self.pool.lower(),
                timestamp,
            )

            response = requests.post(self.url, json={"query": query}, timeout=10)
            if response.status_code != 200:
                time.sleep(1)
                continue
            data = json.loads(response.text)
            if not data["data"]["liquidityPoolHourlySnapshots"]:
                break
            timestamp = data["data"]["liquidityPoolHourlySnapshots"][-1]["timestamp"]
            dfs.append(pd.DataFrame(data["data"]["liquidityPoolHourlySnapshots"])[["timestamp", "tick"]])
            if self.decimals is None:
                self.decimals = (
                    data["data"]["liquidityPoolHourlySnapshots"][0]["pool"]["inputTokens"][0]["decimals"]
                    - data["data"]["liquidityPoolHourlySnapshots"][0]["pool"]["inputTokens"][1]["decimals"]
                )
        self._data = pd.concat(dfs)

    def transform(self):
        self._data["time"] = pd.to_datetime(self.df["timestamp"].astype(int), unit="s")
        self._data["tick"] = self._data["tick"].astype(int)
        self._data["price"] = self._data["tick"].apply(lambda x: 1.0001**x) * 10**self.decimals
        self._data = self._data[["time", "price"]]
        self._data = self._data.sort_values("time")
        self._data = self._data.drop_duplicates("time", keep="last")
        self._data = self._data.set_index("time")
        self._data = self._data.resample("1h").ohlc()
        self._data = pd.DataFrame(self.df["price"]["close"].shift(1).ffill().dropna())
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
