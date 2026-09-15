"""Datetime helpers shared by all loaders.

The contract: every loader normalizes ``start_time``/``end_time`` to
UTC-aware ``datetime`` objects on construction. Conversions to API
representations (ms epoch, s epoch) live here so we don't duplicate
``int(dt.timestamp() * 1000)`` boilerplate.
"""
from datetime import datetime, timezone
from typing import Iterable, Optional

import pandas as pd

from fractal.core.base.time import SECONDS_PER_YEAR

__all__ = [
    "SECONDS_PER_YEAR", "to_utc", "to_ms", "to_seconds", "utcnow",
    "annualise_funding", "require_no_nan",
]


def to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Return ``dt`` as a UTC-aware datetime. Naive datetimes are assumed UTC.

    Returns ``None`` unchanged so call sites can keep optional semantics.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_ms(dt: Optional[datetime]) -> Optional[int]:
    """UTC-aware datetime → millisecond epoch (or ``None`` if input is None)."""
    dt = to_utc(dt)
    if dt is None:
        return None
    return int(dt.timestamp() * 1000)


def to_seconds(dt: Optional[datetime]) -> Optional[int]:
    """UTC-aware datetime → second epoch (or ``None`` if input is None)."""
    dt = to_utc(dt)
    if dt is None:
        return None
    return int(dt.timestamp())


def utcnow() -> datetime:
    """Convenience: ``datetime.now`` in UTC."""
    return datetime.now(tz=timezone.utc)


def annualise_funding(rate: float, period_seconds: float) -> float:
    """Per-period funding rate → annualised simple rate: ``rate * YEAR / period``.

    Matches how Boros reports ``settlementApr`` for a venue's funding
    (Binance 8h ⇒ ``rate * 1095``, Hyperliquid 1h ⇒ ``rate * 8760``).
    """
    if period_seconds <= 0:
        raise ValueError(f"period_seconds must be > 0, got {period_seconds}")
    return rate * SECONDS_PER_YEAR / period_seconds


def require_no_nan(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise ``ValueError`` if any of ``columns`` holds NaN.

    Loaders call this at the end of ``transform`` for every column a
    backtest relies on: a missing rate must surface, never be zero-filled
    (see the ``AaveV3RatesLoader`` precedent).
    """
    bad = {c: int(df[c].isna().sum()) for c in columns if c in df.columns and df[c].isna().any()}
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"required column(s) missing from loader output: {missing}")
    if bad:
        raise ValueError(
            f"NaN in required column(s) {bad}; the source feed has gaps in "
            f"this window — narrow the window or fix the feed instead of filling zeros"
        )
