"""Datetime helpers shared by all loaders.

The contract: every loader normalizes ``start_time``/``end_time`` to
UTC-aware ``datetime`` objects on construction. Conversions to API
representations (ms epoch, s epoch) live here so we don't duplicate
``int(dt.timestamp() * 1000)`` boilerplate.
"""
from datetime import datetime, timezone
from typing import Optional


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
