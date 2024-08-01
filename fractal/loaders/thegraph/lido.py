import time

import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import RateHistory
from fractal.loaders.thegraph.base_graph_loader import ArbitrumGraphLoader


class StETHLoader(ArbitrumGraphLoader):
    """
    StETH (Lido) Loader.
    https://thegraph.com/explorer/subgraphs/Sxx812XgeKyzQPaBpR5YZWmGV5fZuBaPdh7DFhzSwiQ?view=Query&chain=arbitrum-one

    SUBGRAPH_ID: Sxx812XgeKyzQPaBpR5YZWmGV5fZuBaPdh7DFhzSwiQ
    """
    SUBGRAPH_ID = "Sxx812XgeKyzQPaBpR5YZWmGV5fZuBaPdh7DFhzSwiQ"

    def __init__(self, api_key: str, loader_type: LoaderType) -> None:
        super().__init__(
            api_key=api_key,
            loader_type=loader_type,
            subgraph_id=self.SUBGRAPH_ID,
        )

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
            data = self._make_request(query)
            if data is None or data["totalRewards"] is None or len(data["totalRewards"]) == 0:
                break
            blockTime = data["totalRewards"][-1]["blockTime"]
            dfs.append(pd.DataFrame(data["totalRewards"]))
        self._data = pd.concat(dfs, ignore_index=True)

    def transform(self):
        self._data['blockTime'] = self._data['blockTime'].astype(int)
        self._data['apr'] = self._data['apr'].astype(float)
        self._data["blockTime"] = pd.to_datetime(self._data["blockTime"], unit="s")
        self._data = self._data.sort_values("blockTime")
        self._data = self._data.set_index("blockTime")
        self._data = self._data.resample("1h").mean().ffill()
        self._data = self._data.reset_index()
        self._data['apr'] /= (365 * 24 * 100)
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
