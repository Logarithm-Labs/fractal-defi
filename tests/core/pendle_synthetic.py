"""Synthetic Pendle PT + Morpho observation builder shared by the PT strategy tests.

Builds a daily (or hourly) path to a fixed expiry with a constant or
user-supplied implied-APY series, the Morpho oracle on a linear
discount, the PT market price on Pendle's compounded convention and a
constant borrow APY — the minimum a ``LeveragedPTStrategy`` needs.
"""
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional, Sequence, Union

from fractal.core.base import Observation
from fractal.core.base.time import SECONDS_PER_DAY, SECONDS_PER_YEAR
from fractal.core.entities import MorphoGlobalState, PendlePTGlobalState
from fractal.core.entities.models.morpho_math import per_bar_borrow_rate
from fractal.core.entities.models.pendle_math import linear_discount_oracle_price, pt_price_from_apy

EXPIRY = datetime(2026, 12, 1, tzinfo=timezone.utc)


def synthetic_observations(
    days: int = 90,
    *,
    bar_hours: float = 24.0,
    implied_apy: Union[float, Sequence[float], Callable[[int], float]] = 0.10,
    borrow_apy: Union[float, Sequence[float], Callable[[int], float]] = 0.0,
    base_discount: float = 0.06,
    expiry: datetime = EXPIRY,
    past_expiry_bars: int = 1,
    asset_price: float = 1.0,
    sy_exchange_rate: float = 1.0,
    pool_reserves: Optional[float] = 1e9,
) -> List[Observation]:
    """``days`` of bars ending ``past_expiry_bars`` bars after ``expiry``."""
    bars = int(days * 24 / bar_hours) + past_expiry_bars
    start = expiry - timedelta(days=days)
    step = timedelta(hours=bar_hours)
    observations: List[Observation] = []
    for i in range(bars + 1):
        ts = start + i * step
        seconds = (expiry - ts).total_seconds()
        apy = _at(implied_apy, i)
        borrow = _at(borrow_apy, i)
        price = pt_price_from_apy(apy, max(seconds, 0.0) / SECONDS_PER_YEAR) * asset_price
        oracle = linear_discount_oracle_price(base_discount, seconds) * asset_price
        pt_state = PendlePTGlobalState(
            seconds_to_expiry=seconds, implied_apy=apy, asset_price=asset_price,
            sy_exchange_rate=sy_exchange_rate,
            total_pt=pool_reserves or 0.0, total_sy=pool_reserves or 0.0,
            scalar_root=50.0 if pool_reserves else 0.0, ln_fee_rate_root=0.0,
        )
        lending_state = MorphoGlobalState(
            collateral_price=oracle, debt_price=1.0, lending_rate=0.0,
            borrowing_rate=per_bar_borrow_rate(borrow, bar_hours * 3600),
            collateral_market_price=price,
        )
        observations.append(Observation(timestamp=ts, states={"PT": pt_state, "LENDING": lending_state}))
    return observations


def _at(series, index: int) -> float:
    if callable(series):
        return float(series(index))
    if isinstance(series, (int, float)):
        return float(series)
    return float(series[min(index, len(series) - 1)])


def days_to_seconds(days: float) -> float:
    return days * SECONDS_PER_DAY
