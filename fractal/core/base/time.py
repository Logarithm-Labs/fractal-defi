"""Time constants shared by entities, models and loaders.

One home for the day-count convention so entities, pure-model modules
and loaders cannot drift apart. Every on-chain protocol modelled here
(Pendle ``lnImpliedRate * t``, Morpho ``exp(rate * 31_536_000)``,
Boros ``PaymentLib``) and the library's own loader convention
(``365 * 24`` bars per year) use a **365-day year** (ACT/365).
"""

SECONDS_PER_DAY: float = 24.0 * 3600.0
SECONDS_PER_YEAR: float = 365.0 * SECONDS_PER_DAY  # 31_536_000, ACT/365

__all__ = ["SECONDS_PER_DAY", "SECONDS_PER_YEAR"]
