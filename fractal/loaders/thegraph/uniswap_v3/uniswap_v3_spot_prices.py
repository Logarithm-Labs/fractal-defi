"""Hourly spot-price loaders derived from Uniswap V3 tick snapshots."""
from datetime import datetime
from string import Template
from typing import List, Optional

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import PriceHistory
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_arbitrum import \
    ArbitrumUniswapV3Loader
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_ethereum import \
    EthereumUniswapV3Loader


class _UniswapV3PricesBase:
    """Shared logic for the Ethereum/Arbitrum spot-price loaders.

    The two subgraphs (Messari Arbitrum vs uniswap-v3 Ethereum mainnet)
    expose different entities, so the actual GraphQL query is supplied by
    each subclass via ``_QUERY``, ``_ENTITY_NAME`` and ``_TIME_FIELD``.
    """

    _BATCH_LIMIT = 1000
    _ENTITY_NAME: str = ""
    _TIME_FIELD: str = "timestamp"
    _QUERY: Template = Template("")

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        return f"{self.pool.lower()}-prices-{s}-{e}"

    def extract(self) -> None:
        cursor = to_seconds(self.end_time) if self.end_time is not None else int(utcnow().timestamp())
        floor = to_seconds(self.start_time) if self.start_time is not None else None
        rows: List[dict] = []
        while True:
            query = self._QUERY.substitute(
                limit=self._BATCH_LIMIT,
                pool=self.pool.lower(),
                timestamp=cursor,
            )
            data = self._make_request(query)
            batch = data.get(self._ENTITY_NAME) or []
            if not batch:
                break
            # Normalize the time field name to a common ``timestamp`` column.
            for row in batch:
                row["timestamp"] = row[self._TIME_FIELD]
            rows.extend(batch)
            last_ts = int(batch[-1][self._TIME_FIELD])
            if floor is not None and last_ts <= floor:
                break
            if len(batch) < self._BATCH_LIMIT:
                break
            cursor = last_ts
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        cols = ["time", "price"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        df = self._data.copy()
        df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["tick"] = df["tick"].astype(int)
        df["price"] = df["tick"].apply(lambda x: 1.0001**x) * 10 ** self.decimals
        df = df[cols].sort_values("time").drop_duplicates("time", keep="last")
        df = df.set_index("time").resample("1h").ohlc()
        df = pd.DataFrame(df["price"]["close"].shift(1).ffill().dropna()).reset_index()
        df.columns = cols
        if self.start_time is not None:
            df = df[df["time"] >= self.start_time]
        if self.end_time is not None:
            df = df[df["time"] <= self.end_time]
        self._data = df.reset_index(drop=True)

    def load(self) -> None:
        self._load(self._cache_key())

    def read(self, with_run: bool = False) -> PriceHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return PriceHistory(prices=[], time=[])
        return PriceHistory(
            time=pd.to_datetime(self._data["time"], utc=True).values,
            prices=self._data["price"].astype(float).values,
        )


class UniswapV3ArbitrumPricesLoader(_UniswapV3PricesBase, ArbitrumUniswapV3Loader):
    """Hourly close prices for an Arbitrum UniV3 pool, derived from tick snapshots."""

    _ENTITY_NAME = "liquidityPoolHourlySnapshots"
    _TIME_FIELD = "timestamp"
    _QUERY = Template("""
        {
            liquidityPoolHourlySnapshots(
                first: $limit
                where: {pool: "$pool", timestamp_lt: "$timestamp"}
                orderBy: timestamp
                orderDirection: desc
            ) {
                tick
                timestamp
            }
        }
    """)

    def __init__(
        self,
        api_key: str,
        pool: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs,
    ) -> None:
        super().__init__(api_key=api_key, loader_type=loader_type)
        self.pool: str = pool
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)
        decimals = kwargs.get("decimals", None)
        if decimals is None:
            decimals0, decimals1 = self.get_pool_decimals(pool)
            decimals = decimals0 - decimals1
        self.decimals: float = decimals


class UniswapV3EthereumPricesLoader(_UniswapV3PricesBase, EthereumUniswapV3Loader):
    """Hourly close prices for an Ethereum mainnet UniV3 pool."""

    _ENTITY_NAME = "poolHourDatas"
    _TIME_FIELD = "periodStartUnix"
    _QUERY = Template("""
        {
            poolHourDatas(
                first: $limit
                where: {pool: "$pool", periodStartUnix_lt: $timestamp}
                orderBy: periodStartUnix
                orderDirection: desc
            ) {
                tick
                periodStartUnix
            }
        }
    """)

    def __init__(
        self,
        api_key: str,
        pool: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs,
    ) -> None:
        super().__init__(api_key=api_key, loader_type=loader_type)
        self.pool: str = pool
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)
        decimals = kwargs.get("decimals", None)
        if decimals is None:
            decimals0, decimals1 = self.get_pool_decimals(pool)
            decimals = decimals0 - decimals1
        self.decimals: float = decimals
