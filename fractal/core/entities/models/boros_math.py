"""Pure Pendle Boros maths: fixed-vs-floating yield-unit accounting.

A Boros position of signed size ``N`` (yield units, in coins of the
underlying: long = pay fixed / receive floating) entered at fixed rate
``r_fix`` settles every venue funding period ``Δ`` and is marked to
maturity between settlements. All rates are annualised decimals, time
is ACT/365 (``SECONDS_PER_YEAR``); every amount below is in **coins**
of the underlying — the entity converts to its margin asset.

Formulas match ``boros-core`` (whitepaper §2.2–2.3, §3.2; the
``PaymentLib`` port in Demeter is the same arithmetic in WAD ints):

    settlement_i = N·f_i − N·r_fix·Δ/YEAR − |N|·settleFee·Δ/YEAR
    unrealised   = N·(mark − r_fix)·TTM
    IM / MM      = k·|N|·max(TTM, tThresh/YEAR)·max(|mark|, RateFloor)
    open fee     = |ΔN|·takerFee·TTM

Sign convention follows the library's perps: ``funding_rate > 0`` means
perp longs pay, so a **long** yield unit *receives* ``N·f``.
"""
from fractal.core.base.time import SECONDS_PER_YEAR

LIQUIDATION_BASE_INCENTIVE = 0.25   # Boros liqSettings.base (live BTC/ETH markets)
LIQUIDATION_INCENTIVE_SLOPE = 0.5   # Boros liqSettings.slope
LIQUIDATION_PROTOCOL_FEE_RATE = 0.0005  # protocol cut on the liquidated size, per year of TTM

__all__ = [
    "LIQUIDATION_BASE_INCENTIVE",
    "LIQUIDATION_INCENTIVE_SLOPE",
    "LIQUIDATION_PROTOCOL_FEE_RATE",
    "settlement_pnl_coin",
    "fixed_leg_coin",
    "mark_to_maturity_coin",
    "open_fee_coin",
    "margin_coin",
    "liquidation_penalty_fraction",
]


def fixed_leg_coin(size: float, fixed_rate: float, seconds: float) -> float:
    """Fixed-leg cash flow over ``seconds``: ``size · fixed_rate · seconds / YEAR`` (paid by the long)."""
    if seconds < 0.0:
        raise ValueError(f"seconds must be >= 0, got {seconds}")
    return size * fixed_rate * seconds / SECONDS_PER_YEAR


def settlement_pnl_coin(
    size: float,
    funding_rate: float,
    entry_rate: float,
    period_seconds: float,
    settle_fee_rate: float,
) -> float:
    """One settlement: floating received minus fixed paid minus the settlement fee.

    Args:
        size: signed yield units (coins); ``> 0`` long (pay fixed, receive floating).
        funding_rate: the venue's **raw per-period** funding for this period
            (e.g. Binance's 8-hour rate), positive = perp longs pay.
        entry_rate: the position's fixed rate (annualised).
        period_seconds: the venue's funding interval Δ.
        settle_fee_rate: annualised settlement fee (``0.001`` live).
    """
    if period_seconds < 0.0 or settle_fee_rate < 0.0:
        raise ValueError("period_seconds and settle_fee_rate must be >= 0")
    floating = size * funding_rate
    fixed = fixed_leg_coin(size, entry_rate, period_seconds)
    fee = abs(size) * settle_fee_rate * period_seconds / SECONDS_PER_YEAR
    return floating - fixed - fee


def mark_to_maturity_coin(size: float, mark_rate: float, entry_rate: float, years: float) -> float:
    """Unrealised value ``size · (mark − entry) · TTM``; ``0`` at/after maturity."""
    return size * (mark_rate - entry_rate) * max(years, 0.0)


def open_fee_coin(delta_size: float, taker_fee_rate: float, years: float) -> float:
    """Taker fee on a fill: ``|ΔN| · takerFee · TTM`` (maker fee is zero on Boros)."""
    if taker_fee_rate < 0.0:
        raise ValueError(f"taker_fee_rate must be >= 0, got {taker_fee_rate}")
    return abs(delta_size) * taker_fee_rate * max(years, 0.0)


def margin_coin(
    size: float,
    mark_rate: float,
    years: float,
    k: float,
    rate_floor: float,
    time_floor_years: float,
) -> float:
    """``k · |N| · max(TTM, time floor) · max(|mark|, rate floor)`` — IM with ``k = kIM``, MM with ``k = kMM``."""
    if k < 0.0 or rate_floor < 0.0 or time_floor_years < 0.0:
        raise ValueError("k, rate_floor and time_floor_years must be >= 0")
    return k * abs(size) * max(years, time_floor_years) * max(abs(mark_rate), rate_floor)


def liquidation_penalty_fraction(health_ratio: float) -> float:
    """Liquidator incentive as a fraction of the maintenance margin: ``clip(min(base + slope·(1 − HR), HR), 0, 1)``."""
    incentive = min(LIQUIDATION_BASE_INCENTIVE + LIQUIDATION_INCENTIVE_SLOPE * (1.0 - health_ratio), health_ratio)
    return min(1.0, max(0.0, incentive))
