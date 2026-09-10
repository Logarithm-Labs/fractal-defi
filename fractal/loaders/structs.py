"""
Typed pandas DataFrame wrappers used as the I/O contract for all loaders.

All loaders MUST return one of these structures with a UTC-aware
``DatetimeIndex`` and the documented column set. Empty periods are valid:
loaders return an instance of the correct shape with zero rows.

Simulation loaders that produce multiple synthetic trajectories return a
:data:`TrajectoryBundle` — a list of one of the structures above.
"""
from typing import List, Optional

import numpy as np
import pandas as pd


def _to_utc_index(time: np.ndarray) -> pd.DatetimeIndex:
    """Coerce an array-like of timestamps to a UTC-aware ``DatetimeIndex``."""
    idx = pd.to_datetime(time, utc=True)
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    idx.name = "time"
    return idx


class PriceHistory(pd.DataFrame):
    """Single-column price series indexed by UTC timestamps."""

    def __init__(self, prices: np.ndarray, time: np.ndarray):
        prices = np.asarray(prices, dtype=float)
        super().__init__(data={"price": prices}, index=_to_utc_index(time))


class FundingHistory(pd.DataFrame):
    """Funding-rate series indexed by UTC timestamps. Column: ``rate``."""

    def __init__(self, rates: np.ndarray, time: np.ndarray):
        rates = np.asarray(rates, dtype=float)
        super().__init__(data={"rate": rates}, index=_to_utc_index(time))


class RateHistory(pd.DataFrame):
    """Generic rate series (e.g. staking APR/hour). Column: ``rate``."""

    def __init__(self, rates: np.ndarray, time: np.ndarray):
        rates = np.asarray(rates, dtype=float)
        super().__init__(data={"rate": rates}, index=_to_utc_index(time))


class LendingHistory(pd.DataFrame):
    """Lending+borrowing rate series. Columns: ``lending_rate``, ``borrowing_rate``."""

    def __init__(
        self,
        lending_rates: np.ndarray,
        borrowing_rates: np.ndarray,
        time: np.ndarray,
    ):
        super().__init__(
            data={
                "lending_rate": np.asarray(lending_rates, dtype=float),
                "borrowing_rate": np.asarray(borrowing_rates, dtype=float),
            },
            index=_to_utc_index(time),
        )


class PoolHistory(pd.DataFrame):
    """
    AMM pool snapshots. Columns: ``tvl``, ``volume``, ``fees``, ``liquidity``.
    Optional columns when provided: ``price`` (spot price alongside
    reserves) and ``fee_growth0/1`` (per-bar ``feeGrowthGlobal{0,1}X128``
    deltas / ``2**128`` — per-token fees per unit of on-chain liquidity,
    LP-net, raw token units).
    """

    def __init__(
        self,
        tvls: np.ndarray,
        volumes: np.ndarray,
        fees: np.ndarray,
        liquidity: np.ndarray,
        time: np.ndarray,
        prices: Optional[np.ndarray] = None,
        fee_growth0: Optional[np.ndarray] = None,
        fee_growth1: Optional[np.ndarray] = None,
    ):
        data = {
            "tvl": np.asarray(tvls, dtype=float),
            "volume": np.asarray(volumes, dtype=float),
            "fees": np.asarray(fees, dtype=float),
            "liquidity": np.asarray(liquidity, dtype=float),
        }
        if prices is not None:
            data["price"] = np.asarray(prices, dtype=float)
        if fee_growth0 is not None:
            data["fee_growth0"] = np.asarray(fee_growth0, dtype=float)
        if fee_growth1 is not None:
            data["fee_growth1"] = np.asarray(fee_growth1, dtype=float)
        super().__init__(data=data, index=_to_utc_index(time))


class SwapsHistory(pd.DataFrame):
    """Per-swap event records of a V3-style pool. Columns: signed human
    ``amount0/1`` (positive = input side), ``liquidity`` (pool ``L``
    during the swap, on-chain units), post-swap ``tick``, human
    ``price``, ``block``, ``log_index``; UTC index interpolated from
    block numbers.
    """

    def __init__(
        self,
        amount0: np.ndarray,
        amount1: np.ndarray,
        liquidity: np.ndarray,
        ticks: np.ndarray,
        prices: np.ndarray,
        blocks: np.ndarray,
        log_indexes: np.ndarray,
        time: np.ndarray,
    ):
        super().__init__(
            data={
                "amount0": np.asarray(amount0, dtype=float),
                "amount1": np.asarray(amount1, dtype=float),
                "liquidity": np.asarray(liquidity, dtype=float),
                "tick": np.asarray(ticks, dtype=int),
                "price": np.asarray(prices, dtype=float),
                "block": np.asarray(blocks, dtype=int),
                "log_index": np.asarray(log_indexes, dtype=int),
            },
            index=_to_utc_index(time),
        )


# Simulation loaders fan out into multiple trajectories. We expose the
# return type as a named alias so downstream code can `isinstance`-check
# / annotate cleanly without leaking ``List[PriceHistory]`` everywhere.
TrajectoryBundle = List[PriceHistory]


class KlinesHistory(pd.DataFrame):
    """
    OHLCV klines. Columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
    ``volume`` defaults to zeros if a feed does not expose it (back-compat
    for loaders that pre-date PR #27).
    """

    def __init__(  # pylint: disable=redefined-builtin
        self,
        time: np.ndarray,
        open: np.ndarray,  # noqa: A002 - shadowing built-in is the natural OHLCV name
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: Optional[np.ndarray] = None,
    ):
        open_ = np.asarray(open, dtype=float)
        if volume is None:
            volume = np.zeros_like(open_)
        super().__init__(
            data={
                "open": open_,
                "high": np.asarray(high, dtype=float),
                "low": np.asarray(low, dtype=float),
                "close": np.asarray(close, dtype=float),
                "volume": np.asarray(volume, dtype=float),
            },
            index=_to_utc_index(time),
        )
