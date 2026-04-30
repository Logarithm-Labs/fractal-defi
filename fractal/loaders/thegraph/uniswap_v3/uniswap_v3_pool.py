"""Uniswap V3 pool-snapshot loaders (TheGraph)."""
from datetime import datetime
from typing import List, Optional

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import PoolHistory
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_arbitrum import \
    ArbitrumUniswapV3Loader
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_ethereum import \
    EthereumUniswapV3Loader


class _UniswapV3PoolBase:
    """Mixin: pagination + transform + read shared by all UniV3 pool loaders.

    The concrete subclass exposes ``self.pool``, ``self.start_time``,
    ``self.end_time`` and a ``_cache_key`` method. It also implements the
    actual GraphQL extraction (``extract``) since the schema differs
    between the Ethereum (``poolDayDatas``) and Arbitrum (Messari schema)
    subgraphs.
    """

    _BATCH_LIMIT = 1000

    def _post_transform(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        cols = ["date", "tvl", "volume", "fees", "liquidity"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        df = df.copy()
        df["date"] = pd.to_datetime(df[time_col].astype(int), unit="s", utc=True)
        df = df[cols].sort_values("date").drop_duplicates("date").reset_index(drop=True)
        if self.start_time is not None:
            df = df[df["date"] >= self.start_time]
        if self.end_time is not None:
            df = df[df["date"] <= self.end_time]
        return df.reset_index(drop=True)

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
            tvls=self._data["tvl"].astype(float).values,
            volumes=self._data["volume"].astype(float).values,
            fees=self._data["fees"].astype(float).values,
            liquidity=self._data["liquidity"].astype(float).values,
            time=pd.to_datetime(self._data["date"], utc=True).values,
        )


class UniswapV3EthereumPoolDayDataLoader(_UniswapV3PoolBase, EthereumUniswapV3Loader):
    """Daily pool data from the Ethereum mainnet subgraph (uniswap-v3 schema)."""

    def __init__(
        self,
        api_key: str,
        pool: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        super().__init__(api_key=api_key, loader_type=loader_type)
        self.pool: str = pool
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        return f"{self.pool.lower()}-day-{s}-{e}"

    def extract(self) -> None:
        cursor = to_seconds(self.end_time) if self.end_time is not None else int(utcnow().timestamp())
        floor = to_seconds(self.start_time) if self.start_time is not None else None
        rows: List[dict] = []
        while True:
            query = (
                "{ poolDayDatas(first: %d, orderBy: date, orderDirection: desc, "
                "where: {pool: \"%s\", date_lt: %d}) "
                "{ date volumeUSD tvlUSD feesUSD liquidity } }"
            ) % (self._BATCH_LIMIT, self.pool.lower(), cursor)
            data = self._make_request(query)
            batch = data.get("poolDayDatas") or []
            if not batch:
                break
            rows.extend(batch)
            last_ts = int(batch[-1]["date"])
            if floor is not None and last_ts <= floor:
                break
            if len(batch) < self._BATCH_LIMIT:
                break
            cursor = last_ts
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=["date", "tvl", "volume", "fees", "liquidity"])
            return
        df = self._data
        df["volume"] = df["volumeUSD"].astype(float)
        df["tvl"] = df["tvlUSD"].astype(float)
        df["fees"] = df["feesUSD"].astype(float)
        df["liquidity"] = df["liquidity"].astype(float)
        self._data = self._post_transform(df, time_col="date")


class UniswapV3ArbitrumPoolDayDataLoader(_UniswapV3PoolBase, ArbitrumUniswapV3Loader):
    """Daily pool snapshots from the Arbitrum Messari subgraph."""

    def __init__(
        self,
        api_key: str,
        pool: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        super().__init__(api_key=api_key, loader_type=loader_type)
        self.pool: str = pool
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        return f"{self.pool.lower()}-day-{s}-{e}"

    def extract(self) -> None:
        cursor = to_seconds(self.end_time) if self.end_time is not None else int(utcnow().timestamp())
        floor = to_seconds(self.start_time) if self.start_time is not None else None
        rows: List[dict] = []
        while True:
            query = (
                "{ liquidityPoolDailySnapshots(first: %d, orderBy: timestamp, "
                "orderDirection: desc, where: {pool: \"%s\", timestamp_lt: %d}) "
                "{ dailyTotalRevenueUSD timestamp totalValueLockedUSD activeLiquidity } }"
            ) % (self._BATCH_LIMIT, self.pool.lower(), cursor)
            data = self._make_request(query)
            batch = data.get("liquidityPoolDailySnapshots") or []
            if not batch:
                break
            rows.extend(batch)
            last_ts = int(batch[-1]["timestamp"])
            if floor is not None and last_ts <= floor:
                break
            if len(batch) < self._BATCH_LIMIT:
                break
            cursor = last_ts
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=["date", "tvl", "volume", "fees", "liquidity"])
            return
        df = self._data
        df["volume"] = 0.0  # Messari schema does not expose volume explicitly
        df["fees"] = df["dailyTotalRevenueUSD"].astype(float)
        df["tvl"] = df["totalValueLockedUSD"].astype(float)
        df["liquidity"] = df["activeLiquidity"].astype(float)
        self._data = self._post_transform(df, time_col="timestamp")


def _stretch_daily(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Fan a daily dataframe out to ``freq`` resolution by ffill, dividing
    rate-style columns (``fees``, ``volume``) by the bucket count."""
    if df.empty:
        return df
    df = df.copy()
    df = df.set_index("date").sort_index()
    if freq == "1h":
        bucket = 24
    elif freq == "1min":
        bucket = 24 * 60
    else:
        raise ValueError(f"Unsupported freq {freq!r}")
    df = df.resample(freq).ffill()
    df["fees"] = df["fees"] / bucket
    df["volume"] = df["volume"] / bucket
    df = df.reset_index()
    return df


class UniswapV3ArbitrumPoolHourDataLoader(UniswapV3ArbitrumPoolDayDataLoader):
    def transform(self) -> None:
        super().transform()
        self._data = _stretch_daily(self._data, "1h")


class UniswapV3EthereumPoolHourDataLoader(UniswapV3EthereumPoolDayDataLoader):
    def transform(self) -> None:
        super().transform()
        self._data = _stretch_daily(self._data, "1h")


class UniswapV3EthereumPoolMinuteDataLoader(UniswapV3EthereumPoolDayDataLoader):
    def transform(self) -> None:
        super().transform()
        self._data = _stretch_daily(self._data, "1min")
