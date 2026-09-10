"""Uniswap V3 pool-snapshot loaders (TheGraph)."""
import warnings
from datetime import datetime
from typing import List, Optional

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.structs import PoolHistory
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_arbitrum import ArbitrumUniswapV3Loader
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_base import BaseUniswapV3Loader
from fractal.loaders.thegraph.uniswap_v3.uniswap_v3_ethereum import EthereumUniswapV3Loader


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
        for extra in ("price", "fee_growth0", "fee_growth1"):
            if not df.empty and extra in df.columns:
                cols.append(extra)
        if df.empty:
            return pd.DataFrame(columns=cols)
        df = df.copy()
        df["date"] = pd.to_datetime(df[time_col].astype(int), unit="s", utc=True)
        df = df[cols].sort_values("date").drop_duplicates("date").reset_index(drop=True)
        if self.start_time is not None:
            df = df[df["date"] >= self.start_time]
        if self.end_time is not None:
            df = df[df["date"] <= self.end_time]
        df = df.reset_index(drop=True)
        negative_tvl = df["tvl"] < 0
        if negative_tvl.any():
            # ``tvlUSD`` is a derived subgraph field that can dip below
            # zero on accounting glitches (observed on Base V3 pools).
            # TVL plays no part in fee accrual, but downstream entities
            # reject negative snapshots — clamp and warn.
            first_bad = df.loc[negative_tvl.idxmax(), "date"]
            warnings.warn(
                f"{type(self).__name__}: negative tvlUSD for pool {self.pool} "
                f"at {first_bad} ({int(negative_tvl.sum())} bar(s)); clamped "
                f"to 0 — tvl around these bars is unreliable."
            )
            df.loc[negative_tvl, "tvl"] = 0.0
        return df

    def load(self) -> None:
        self._load(self._cache_key())

    def read(self, with_run: bool = False) -> PoolHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return PoolHistory(tvls=[], volumes=[], fees=[], liquidity=[], time=[])

        def _col(name):
            return (
                self._data[name].astype(float).values
                if name in self._data.columns
                else None
            )
        return PoolHistory(
            tvls=self._data["tvl"].astype(float).values,
            volumes=self._data["volume"].astype(float).values,
            fees=self._data["fees"].astype(float).values,
            liquidity=self._data["liquidity"].astype(float).values,
            time=pd.to_datetime(self._data["date"], utc=True).values,
            prices=_col("price"),
            fee_growth0=_col("fee_growth0"),
            fee_growth1=_col("fee_growth1"),
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
        self.pool: str = validate_evm_address(pool, field="pool")
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
        self.pool: str = validate_evm_address(pool, field="pool")
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


class _UniswapV3PoolHourBase(_UniswapV3PoolBase):
    """Native hourly pool snapshots from ``poolHourDatas`` (uniswap-v3 schema).

    Unlike the stretched-daily ``*PoolHourDataLoader`` classes above,
    this queries the true per-hour entity, derives ``price`` from the
    hourly tick, and emits ``fee_growth0/1`` — per-bar deltas of the
    cumulative ``feeGrowthGlobal{0,1}X128`` counters / ``2**128`` (the
    pool's own fee accounting: per-swap liquidity weighting built in,
    protocol fee already net). The first bar's delta is seeded by the
    pre-window row fetched during pagination, else 0.

    Window bounds are inclusive (daily loaders keep their legacy
    exclusive end). Hours with no swaps have no subgraph row — resample
    downstream if a dense grid is needed.
    """

    def __init__(
        self,
        api_key: str,
        pool: str,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        decimals: Optional[float] = None,
    ) -> None:
        """
        Args:
            api_key (str): The Graph API key
            pool (str): pool address
            loader_type (LoaderType): loader type
            start_time (datetime): window start (inclusive), optional
            end_time (datetime): window end (inclusive), optional
            decimals (float): ``decimals0 - decimals1`` of the pool tokens;
                fetched from the subgraph when omitted.
        """
        super().__init__(api_key=api_key, loader_type=loader_type)
        self.pool: str = validate_evm_address(pool, field="pool")
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)
        if decimals is None:
            decimals0, decimals1 = self.get_pool_decimals(self.pool)
            decimals = decimals0 - decimals1
        self.decimals: float = decimals

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        return f"{self.pool.lower()}-hour-{s}-{e}"

    def extract(self) -> None:
        # +1 makes the exclusive `periodStartUnix_lt` cursor include a
        # bar starting exactly at end_time (documented inclusive).
        cursor = to_seconds(self.end_time) + 1 if self.end_time is not None else int(utcnow().timestamp())
        floor = to_seconds(self.start_time) if self.start_time is not None else None
        rows: List[dict] = []
        while True:
            query = (
                "{ poolHourDatas(first: %d, orderBy: periodStartUnix, orderDirection: desc, "
                "where: {pool: \"%s\", periodStartUnix_lt: %d}) "
                "{ periodStartUnix volumeUSD tvlUSD feesUSD liquidity tick "
                "feeGrowthGlobal0X128 feeGrowthGlobal1X128 } }"
            ) % (self._BATCH_LIMIT, self.pool.lower(), cursor)
            data = self._make_request(query)
            batch = data.get("poolHourDatas") or []
            if not batch:
                break
            rows.extend(batch)
            last_ts = int(batch[-1]["periodStartUnix"])
            if floor is not None and last_ts <= floor:
                break
            if len(batch) < self._BATCH_LIMIT:
                break
            cursor = last_ts
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=[
                "date", "tvl", "volume", "fees", "liquidity", "price",
                "fee_growth0", "fee_growth1",
            ])
            return
        df = self._data
        df["volume"] = df["volumeUSD"].astype(float)
        df["tvl"] = df["tvlUSD"].astype(float)
        df["fees"] = df["feesUSD"].astype(float)
        df["liquidity"] = df["liquidity"].astype(float)
        tick = pd.to_numeric(df["tick"], errors="coerce")
        df["price"] = (1.0001 ** tick) * 10 ** self.decimals
        df["price"] = df["price"].ffill()
        # Per-bar feeGrowth deltas. Cumulative counters are uint256 that
        # only grow; Python ints keep full precision, so diff before the
        # float division by 2**128. Sort ascending first (extract walks
        # descending) — the row just before the window start, fetched by
        # pagination overshoot, seeds the first in-window delta and is
        # trimmed afterwards by ``_post_transform``.
        df = df.sort_values("periodStartUnix").reset_index(drop=True)
        for i, field in enumerate(["feeGrowthGlobal0X128", "feeGrowthGlobal1X128"]):
            growth = df[field].map(int)
            delta = (growth - growth.shift(1)).fillna(0)
            negatives = delta < 0
            if negatives.any():
                # The counter is monotonic on-chain; a dip means a
                # subgraph anomaly (reorg / re-index). The clamped bar
                # under-counts and the NEXT bar spans the dip — warn so
                # the window isn't trusted silently.
                first_bad = df.loc[negatives.idxmax(), "periodStartUnix"]
                warnings.warn(
                    f"{type(self).__name__}: non-monotonic {field} for pool "
                    f"{self.pool} at periodStartUnix={first_bad} "
                    f"({int(negatives.sum())} bar(s)); deltas clamped to 0 — "
                    f"fees around these bars are unreliable."
                )
            df[f"fee_growth{i}"] = [d / 2 ** 128 if d > 0 else 0.0 for d in delta]
        self._data = self._post_transform(df, time_col="periodStartUnix")


class UniswapV3BasePoolHourDataLoader(_UniswapV3PoolHourBase, BaseUniswapV3Loader):
    """Native hourly pool snapshots for a Uniswap V3 pool on Base."""
