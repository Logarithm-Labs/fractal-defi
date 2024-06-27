import json
import time

import pandas as pd
import requests

from fractal.loaders.loader import Loader, LoaderType
from fractal.loaders.structs import RateHistory


class LidoLoader(Loader):

    def __init__(self) -> None:
        super().__init__(LoaderType.CSV)
        self.url: str = "https://api.thegraph.com/subgraphs/name/StakedETHfinance/StakedETH"

    def extract(self):
        dfs = []
        blockTime = int(time.time())
        while True:
            query = """
            {
            totalRewards(
                orderBy: blockTime
                orderDirection: desc
                where: {blockTime_lt: "%s"}
            ) {
                apr
                blockTime
            }
            }
            """ % blockTime

            response = requests.post(self.url, json={"query": query}, timeout=10)
            if response.status_code != 200:
                time.sleep(1)
                continue
            data = json.loads(response.text)
            if not data["data"]["totalRewards"]:
                break
            blockTime = data["data"]["totalRewards"][-1]["blockTime"]
            dfs.append(pd.DataFrame(data["data"]["totalRewards"]))
        self._data = pd.concat(dfs, ignore_index=True)

    def transform(self):
        self._data['blockTime'] = self._data['blockTime'].astype(int)
        self._data['apr'] = self._data['apr'].astype(float)
        self._data["blockTime"] = pd.to_datetime(self._data["blockTime"], unit="s")
        self._data = self._data.sort_values("blockTime")
        self._data = self._data.set_index("blockTime")
        self._data = self._data.resample("1h").mean().ffill()
        self._data = self._data.reset_index()
        self._data['apr'] /= (365 * 24)
        self._data = self._data.rename(columns={"blockTime": "time", "apr": "rate"})

    def load(self):
        self._load("steth")

    def read(self, with_run: bool = False) -> RateHistory:
        if with_run:
            self.run()
        else:
            self._read("steth")
        return RateHistory(
            time=self._data["time"].values,
            rates=self._data["rate"].values,
        )
