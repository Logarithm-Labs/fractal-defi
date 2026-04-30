"""Lido stETH staking-rate loader (TheGraph)."""
from datetime import datetime
from typing import List, Optional

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import RateHistory
from fractal.loaders.thegraph.base_graph_loader import ArbitrumGraphLoader


class StETHLoader(ArbitrumGraphLoader):
    """Hourly stETH APR series → :class:`RateHistory`.

    The returned ``rate`` column is the per-hour rate (annual APR / (365*24)).
    """

    SUBGRAPH_ID = "Sxx812XgeKyzQPaBpR5YZWmGV5fZuBaPdh7DFhzSwiQ"
    _BATCH_LIMIT = 1000

    def __init__(
        self,
        api_key: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            subgraph_id=self.SUBGRAPH_ID,
            loader_type=loader_type,
        )
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        return f"steth-{s}-{e}"

    def extract(self) -> None:
        rows: List[dict] = []
        # Walk backward from the upper bound (end_time or now); stop when we
        # reach start_time or run out of data.
        cursor = to_seconds(self.end_time) if self.end_time is not None else int(utcnow().timestamp())
        floor = to_seconds(self.start_time) if self.start_time is not None else None
        while True:
            query = (
                "{ totalRewards(first: %d, orderBy: blockTime, orderDirection: desc, "
                "where: {blockTime_lt: \"%d\"}) { apr blockTime } }"
            ) % (self._BATCH_LIMIT, cursor)
            data = self._make_request(query)
            batch = data.get("totalRewards") or []
            if not batch:
                break
            rows.extend(batch)
            last_ts = int(batch[-1]["blockTime"])
            if floor is not None and last_ts <= floor:
                break
            if len(batch) < self._BATCH_LIMIT:
                break
            cursor = last_ts
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        cols = ["time", "rate"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        df = self._data
        df["blockTime"] = df["blockTime"].astype(int)
        df["apr"] = df["apr"].astype(float)
        df["blockTime"] = pd.to_datetime(df["blockTime"], unit="s", utc=True)
        df = df.sort_values("blockTime").set_index("blockTime")
        df = df.resample("1h").mean().ffill()
        df = df.reset_index()
        df["rate"] = df["apr"] / (365 * 24 * 100)  # APR % → per-hour fraction
        df = df.rename(columns={"blockTime": "time"})[cols]
        if self.start_time is not None:
            df = df[df["time"] >= self.start_time]
        if self.end_time is not None:
            df = df[df["time"] <= self.end_time]
        self._data = df.reset_index(drop=True)

    def load(self) -> None:
        self._load(self._cache_key())

    def read(self, with_run: bool = False) -> RateHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return RateHistory(rates=[], time=[])
        return RateHistory(
            rates=self._data["rate"].astype(float).values,
            time=pd.to_datetime(self._data["time"], utc=True).values,
        )
