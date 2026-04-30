"""Uniswap V2 hourly pool-data loader (TheGraph)."""
from datetime import datetime
from typing import List, Optional

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import PoolHistory
from fractal.loaders.thegraph.uniswap_v2.uniswap_v2_ethereum import \
    EthereumUniswapV2Loader


class EthereumUniswapV2PoolDataLoader(EthereumUniswapV2Loader):
    """Hourly pool snapshots → :class:`PoolHistory`."""

    _BATCH_LIMIT = 1000

    def __init__(
        self,
        api_key: str,
        pool: str,
        fee_tier: float,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        super().__init__(api_key=api_key, loader_type=loader_type)
        self.pool: str = pool
        self.fee_tier: float = fee_tier
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        return f"{self.pool.lower()}-{s}-{e}"

    def extract(self) -> None:
        cursor = to_seconds(self.end_time) if self.end_time is not None else int(utcnow().timestamp())
        floor = to_seconds(self.start_time) if self.start_time is not None else None
        rows: List[dict] = []
        while True:
            query = (
                "{ pairHourDatas(orderBy: hourStartUnix, orderDirection: desc, "
                "where: {pair: \"%s\", hourStartUnix_lt: %d}, first: %d) "
                "{ hourStartUnix hourlyVolumeUSD totalSupply reserveUSD } }"
            ) % (self.pool.lower(), cursor, self._BATCH_LIMIT)
            data = self._make_request(query)
            batch = data.get("pairHourDatas") or []
            if not batch:
                break
            rows.extend(batch)
            last_ts = int(batch[-1]["hourStartUnix"])
            if floor is not None and last_ts <= floor:
                break
            if len(batch) < self._BATCH_LIMIT:
                break
            cursor = last_ts
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        cols = ["time", "tvl", "volume", "fees", "liquidity"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        df = self._data
        df["time"] = pd.to_datetime(df["hourStartUnix"].astype(int), unit="s", utc=True)
        df["volume"] = df["hourlyVolumeUSD"].astype(float)
        df["liquidity"] = df["totalSupply"].astype(float)
        df["tvl"] = df["reserveUSD"].astype(float)
        df["fees"] = df["volume"] * self.fee_tier
        df = df[cols].dropna()
        df = df[df["liquidity"] != 0]
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        if self.start_time is not None:
            df = df[df["time"] >= self.start_time]
        if self.end_time is not None:
            df = df[df["time"] <= self.end_time]
        self._data = df.reset_index(drop=True)

    def load(self) -> None:
        self._load(self._cache_key())

    def read(self, with_run: bool = False) -> PoolHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return PoolHistory(tvls=[], volumes=[], fees=[], liquidity=[], time=[])
        return PoolHistory(
            time=pd.to_datetime(self._data["time"], utc=True).values,
            tvls=self._data["tvl"].astype(float).values,
            volumes=self._data["volume"].astype(float).values,
            fees=self._data["fees"].astype(float).values,
            liquidity=self._data["liquidity"].astype(float).values,
        )
