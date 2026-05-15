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
    If ``prices`` is provided it is appended as ``price`` column (used by
    pool loaders that already carry the spot price alongside reserves).
    """

    def __init__(
        self,
        tvls: np.ndarray,
        volumes: np.ndarray,
        fees: np.ndarray,
        liquidity: np.ndarray,
        time: np.ndarray,
        prices: Optional[np.ndarray] = None,
    ):
        data = {
            "tvl": np.asarray(tvls, dtype=float),
            "volume": np.asarray(volumes, dtype=float),
            "fees": np.asarray(fees, dtype=float),
            "liquidity": np.asarray(liquidity, dtype=float),
        }
        if prices is not None:
            data["price"] = np.asarray(prices, dtype=float)
        super().__init__(data=data, index=_to_utc_index(time))


# Simulation loaders fan out into multiple trajectories. We expose the
# return type as a named alias so downstream code can `isinstance`-check
# / annotate cleanly without leaking ``List[PriceHistory]`` everywhere.
TrajectoryBundle = List[PriceHistory]


class PendleMarketHistory(pd.DataFrame):
    """Pendle Principal-Token market snapshots indexed by UTC time.

    Columns:
        ``pt_price`` — mark price of 1 PT in the underlying numeraire
            (typically USDC).
        ``implied_yield`` — annualised fixed yield encoded in
            ``pt_price`` (decimal, ``0.14`` = 14% APY).
        ``seconds_to_expiry`` — seconds remaining until PT redeem
            unlocks.
        ``pool_liquidity`` — total Pendle pool liquidity in numeraire.
    """

    def __init__(
        self,
        pt_prices,
        implied_yields,
        seconds_to_expiry,
        pool_liquidity,
        time,
    ):
        data = {
            "pt_price": np.asarray(pt_prices, dtype=float),
            "implied_yield": np.asarray(implied_yields, dtype=float),
            "seconds_to_expiry": np.asarray(seconds_to_expiry, dtype=float),
            "pool_liquidity": np.asarray(pool_liquidity, dtype=float),
        }
        super().__init__(data=data, index=_to_utc_index(time))


class MorphoMarketHistory(pd.DataFrame):
    """Morpho Blue isolated-market historical state indexed by UTC time.

    Columns:
        ``borrowing_rate`` — annualised borrow APY (decimal,
            ``0.05`` = 5% APY).
        ``utilization`` — fraction of supplied debt currently
            borrowed, in ``[0, 1]``.
        ``supply_apy`` — annualised supply APY (decimal), exposed
            for completeness; PT-collateral markets pay no supplier
            yield, but other markets do.
    """

    def __init__(
        self,
        borrowing_rates,
        utilization,
        supply_apys,
        time,
    ):
        data = {
            "borrowing_rate": np.asarray(borrowing_rates, dtype=float),
            "utilization": np.asarray(utilization, dtype=float),
            "supply_apy": np.asarray(supply_apys, dtype=float),
        }
        super().__init__(data=data, index=_to_utc_index(time))


class BorosMarketHistory(pd.DataFrame):
    """Pendle Boros funding-rate-forward market bars indexed by UTC time.

    Columns:
        ``mark_apr`` — close mark APR (annualised decimal). Long-Boros
            holders receive this as funding per unit of time.
        ``observed_funding`` — close observed funding rate
            (annualised decimal).
        ``mark_apr_7d_ma`` — 7-day moving average of observed funding.
        ``mark_apr_30d_ma`` — 30-day moving average of observed funding.
    """

    def __init__(
        self,
        mark_apr,
        observed_funding,
        mark_apr_7d_ma,
        mark_apr_30d_ma,
        time,
    ):
        data = {
            "mark_apr": np.asarray(mark_apr, dtype=float),
            "observed_funding": np.asarray(observed_funding, dtype=float),
            "mark_apr_7d_ma": np.asarray(mark_apr_7d_ma, dtype=float),
            "mark_apr_30d_ma": np.asarray(mark_apr_30d_ma, dtype=float),
        }
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
