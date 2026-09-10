"""
Typed pandas DataFrame wrappers used as the I/O contract for all loaders.

All loaders MUST return one of these structures with a UTC-aware
``DatetimeIndex`` and the documented column set. Empty periods are valid:
loaders return an instance of the correct shape with zero rows.

Simulation loaders that produce multiple synthetic trajectories return a
:data:`TrajectoryBundle` — a list of one of the structures above.

Loaders pass ``pandas``/``numpy`` datetime arrays (or ``pd.Timestamp``
sequences) to the constructors below. Integer epochs are rejected on
purpose: convert them explicitly with ``pd.to_datetime(..., unit="s",
utc=True)`` (or ``Loader._utc_index``) so the unit is never guessed.
"""
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]
TimeLike = Union[np.ndarray, Sequence[int], Sequence[pd.Timestamp], pd.DatetimeIndex]


def _to_utc_index(time: TimeLike) -> pd.DatetimeIndex:
    """Coerce an array-like of timestamps to a UTC-aware ``DatetimeIndex`` named ``time``.

    Integer arrays are rejected: pandas would read them as nanoseconds
    and a REST loader's epoch seconds would land in 1970. Convert with
    an explicit ``unit`` before building a struct.
    """
    arr = np.asarray(time)
    if arr.size > 0 and np.issubdtype(arr.dtype, np.integer):
        raise TypeError(
            "struct time index must be datetime-like; convert integer epochs explicitly "
            "with pd.to_datetime(values, unit='s'|'ms', utc=True)"
        )
    idx = pd.to_datetime(arr, utc=True)
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
    """Lending+borrowing rate series.

    Columns:
        ``lending_rate`` — **per-bar** simple rate credited to collateral
            (entities apply ``balance *= 1 + rate``).
        ``borrowing_rate`` — **per-bar** rate charged on debt (Morpho
            loaders emit the per-bar exponent ``ln(1 + apy) · Δt / YEAR``).
        Optional, appended only when passed (never NaN-filled):
        ``utilization`` — borrowed / supplied, in ``[0, 1]``;
        ``borrow_apy`` / ``supply_apy`` — the source's annual effective
            rates, for reporting and strategy gates;
        ``rate_at_target`` — Morpho IRM ``rateAtTarget`` (annual).
    """

    def __init__(
        self,
        lending_rates: ArrayLike,
        borrowing_rates: ArrayLike,
        time: TimeLike,
        utilization: Optional[ArrayLike] = None,
        borrow_apy: Optional[ArrayLike] = None,
        supply_apy: Optional[ArrayLike] = None,
        rate_at_target: Optional[ArrayLike] = None,
    ):
        data = {
            "lending_rate": np.asarray(lending_rates, dtype=float),
            "borrowing_rate": np.asarray(borrowing_rates, dtype=float),
        }
        for name, values in (
            ("utilization", utilization), ("borrow_apy", borrow_apy),
            ("supply_apy", supply_apy), ("rate_at_target", rate_at_target),
        ):
            if values is not None:
                data[name] = np.asarray(values, dtype=float)
        super().__init__(data=data, index=_to_utc_index(time))


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


class PendleMarketHistory(pd.DataFrame):
    """Pendle PT market snapshots indexed by UTC time.

    Columns (all float):
        ``implied_apy`` — market implied APY, compounded ACT/365, decimal.
        ``pt_price_asset`` — ``(1 + implied_apy) ** (−seconds_to_expiry / YEAR)``:
            accounting asset per PT (the entity's mid price).
        ``pt_price_usd`` / ``sy_price_usd`` — the API's USD marks of PT and SY.
        ``pt_price_sy`` — ``pt_price_usd / sy_price_usd`` (PT per SY, market).
        ``underlying_apy`` — 7-day trailing underlying APY (NaN allowed).
        ``tvl`` — pool TVL in USD.
        ``total_pt`` / ``total_sy`` — AMM reserves in token units.
        ``seconds_to_expiry`` — ``max(expiry − t, 0)``.
    """

    COLUMNS = (
        "implied_apy", "pt_price_asset", "pt_price_usd", "sy_price_usd", "pt_price_sy",
        "underlying_apy", "tvl", "total_pt", "total_sy", "seconds_to_expiry",
    )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        time: TimeLike,
        implied_apy: ArrayLike,
        pt_price_asset: ArrayLike,
        pt_price_usd: ArrayLike,
        sy_price_usd: ArrayLike,
        pt_price_sy: ArrayLike,
        underlying_apy: ArrayLike,
        tvl: ArrayLike,
        total_pt: ArrayLike,
        total_sy: ArrayLike,
        seconds_to_expiry: ArrayLike,
    ):
        values = (implied_apy, pt_price_asset, pt_price_usd, sy_price_usd, pt_price_sy,
                  underlying_apy, tvl, total_pt, total_sy, seconds_to_expiry)
        data = {name: np.asarray(col, dtype=float) for name, col in zip(self.COLUMNS, values)}
        super().__init__(data=data, index=_to_utc_index(time))


class BorosMarketHistory(pd.DataFrame):
    """Pendle Boros yield-unit market bars indexed by UTC time.

    Columns (all float, rates annualised decimals):
        ``mark_apr_open/high/low/close`` — implied APR candles.
        ``volume`` — candle volume in yield units (``0`` when absent).
        ``underlying_apr`` — the venue's floating funding, annualised
            (``settlement_apr`` where a settlement landed, else the
            indicator series; NaN allowed before the market listed).
        ``settlement_apr`` — realised floating APR of the settlement on
            this bar (NaN on non-settlement bars).
        ``oi`` — open interest in yield units (NaN allowed).
        ``seconds_to_expiry`` — ``max(maturity − t, 0)``.
    """

    COLUMNS = (
        "mark_apr_open", "mark_apr_high", "mark_apr_low", "mark_apr_close", "volume",
        "underlying_apr", "settlement_apr", "oi", "seconds_to_expiry",
    )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        time: TimeLike,
        mark_apr_open: ArrayLike,
        mark_apr_high: ArrayLike,
        mark_apr_low: ArrayLike,
        mark_apr_close: ArrayLike,
        volume: ArrayLike,
        underlying_apr: ArrayLike,
        settlement_apr: ArrayLike,
        oi: ArrayLike,
        seconds_to_expiry: ArrayLike,
    ):
        values = (mark_apr_open, mark_apr_high, mark_apr_low, mark_apr_close, volume,
                  underlying_apr, settlement_apr, oi, seconds_to_expiry)
        data = {name: np.asarray(col, dtype=float) for name, col in zip(self.COLUMNS, values)}
        super().__init__(data=data, index=_to_utc_index(time))


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
