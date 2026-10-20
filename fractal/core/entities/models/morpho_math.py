"""Pure Morpho Blue maths: interest accrual, liquidation incentive, IRM.

Sources: ``morpho-blue/src/Morpho.sol`` (``_accrueInterest``,
``_isHealthy``, ``liquidate``), ``libraries/MathLib.sol``
(``wTaylorCompounded``), ``libraries/ConstantsLib.sol`` and
``morpho-blue-irm/AdaptiveCurveIrm.sol``. Rates are decimals, time is
seconds or ACT/365 years.
"""
import math

import numpy as np

from fractal.core.base.time import SECONDS_PER_YEAR

LIQUIDATION_CURSOR = 0.3                 # ConstantsLib.LIQUIDATION_CURSOR
MAX_LIQUIDATION_INCENTIVE_FACTOR = 1.15  # ConstantsLib.MAX_LIQUIDATION_INCENTIVE_FACTOR
TARGET_UTILIZATION = 0.9                 # AdaptiveCurveIrm
CURVE_STEEPNESS = 4.0                    # AdaptiveCurveIrm
ADJUSTMENT_SPEED = 50.0                  # AdaptiveCurveIrm ConstantsLib.ADJUSTMENT_SPEED, per year
MIN_RATE_AT_TARGET = 0.001               # ConstantsLib.MIN_RATE_AT_TARGET: 0.1% / year (continuous)
MAX_RATE_AT_TARGET = 2.0                 # ConstantsLib.MAX_RATE_AT_TARGET: 200% / year (continuous)
WAD = 1e18                               # MathLib.WAD — fixed-point scale of lltv, fee, rates

__all__ = [
    "ADJUSTMENT_SPEED",
    "CURVE_STEEPNESS",
    "LIQUIDATION_CURSOR",
    "MAX_LIQUIDATION_INCENTIVE_FACTOR",
    "MAX_RATE_AT_TARGET",
    "MIN_RATE_AT_TARGET",
    "TARGET_UTILIZATION",
    "WAD",
    "adaptive_curve_borrow_apr",
    "adaptive_curve_error",
    "borrow_apy_from_per_second_rate",
    "from_wad",
    "liquidation_incentive_factor",
    "max_borrow",
    "per_bar_borrow_rate",
    "rate_at_target_drift",
    "supply_apy_curve",
    "supply_apy_from_curve",
    "taylor_compounded",
    "utilization_for_supply_apy",
]


def from_wad(value: float | int | str) -> float:
    """A WAD-scaled on-chain integer (``lltv``, ``fee``, per-second rate) as a float fraction."""
    return float(value) / WAD


def taylor_compounded(x: float) -> float:
    """``MathLib.wTaylorCompounded``: ``x + x²/2 + x³/6`` ≈ ``e^x − 1`` for small ``x``.

    Morpho accrues ``totalBorrowAssets * wTaylorCompounded(borrowRate * elapsed)``
    on every interaction; entities apply it per bar.
    """
    if x < -1.0:
        raise ValueError(f"per-bar rate must be >= -1, got {x}")
    return x + x * x / 2.0 + x * x * x / 6.0


def per_bar_borrow_rate(borrow_apy: float, seconds: float) -> float:
    """Per-bar exponent ``x = r_sec * seconds`` from the API's effective ``borrowApy``.

    Morpho reports ``borrowApy = exp(r_sec * YEAR) − 1``, so
    ``r_sec = ln(1 + borrowApy) / YEAR``. Feed the result to
    :func:`taylor_compounded` (or ``expm1``) for the per-bar growth.
    """
    if borrow_apy <= -1.0:
        raise ValueError(f"borrow_apy must be > -1, got {borrow_apy}")
    if seconds < 0.0:
        raise ValueError(f"seconds must be >= 0, got {seconds}")
    return math.log1p(borrow_apy) * seconds / SECONDS_PER_YEAR


def borrow_apy_from_per_second_rate(rate_per_second: float) -> float:
    """``exp(r_sec * YEAR) − 1`` — Morpho's APY convention for a per-second WAD rate."""
    return math.expm1(rate_per_second * SECONDS_PER_YEAR)


def liquidation_incentive_factor(lltv: float) -> float:
    """``LIF = min(1.15, 1 / (1 − 0.3 · (1 − LLTV)))``: 0.86 → 1.0438, 0.915 → 1.0262."""
    if not 0.0 < lltv <= 1.0:
        raise ValueError(f"lltv must be in (0, 1], got {lltv}")
    return min(MAX_LIQUIDATION_INCENTIVE_FACTOR, 1.0 / (1.0 - LIQUIDATION_CURSOR * (1.0 - lltv)))


def max_borrow(collateral: float, oracle_price: float, lltv: float) -> float:
    """``collateral * oraclePrice * LLTV`` in loan-asset units; ``0`` for non-positive inputs."""
    if collateral <= 0.0 or oracle_price <= 0.0:
        return 0.0
    return collateral * oracle_price * lltv


def adaptive_curve_borrow_apr(utilization: float, rate_at_target: float) -> float:
    """Instantaneous borrow rate of the AdaptiveCurve IRM for a given ``rateAtTarget``.

    ``err = (u − 0.9) / (0.1 if u > 0.9 else 0.9)``;
    ``coeff = 1 − 1/4`` below target, ``4 − 1`` above;
    ``rate = rateAtTarget * (1 + coeff * err)`` — ``rT/4`` at ``u = 0``,
    ``rT`` at target, ``4·rT`` at full utilisation. Both inputs and the
    result share one unit (per-second or annual). The drift of
    ``rateAtTarget`` itself (``exp(50/yr · err · elapsed)``) is not
    modelled here.
    """
    if not 0.0 <= utilization <= 1.0:
        raise ValueError(f"utilization must be in [0, 1], got {utilization}")
    if rate_at_target < 0.0:
        raise ValueError(f"rate_at_target must be >= 0, got {rate_at_target}")
    if utilization > TARGET_UTILIZATION:
        err = (utilization - TARGET_UTILIZATION) / (1.0 - TARGET_UTILIZATION)
        coeff = CURVE_STEEPNESS - 1.0
    else:
        err = (utilization - TARGET_UTILIZATION) / TARGET_UTILIZATION
        coeff = 1.0 - 1.0 / CURVE_STEEPNESS
    return rate_at_target * (1.0 + coeff * err)


def adaptive_curve_error(utilization: float) -> float:
    """Normalised utilisation error ``err ∈ [−1, 1]`` the AdaptiveCurve IRM steers on.

    ``(u − target) / (1 − target)`` above target, ``(u − target) / target`` below.
    """
    if not 0.0 <= utilization <= 1.0:
        raise ValueError(f"utilization must be in [0, 1], got {utilization}")
    if utilization > TARGET_UTILIZATION:
        return (utilization - TARGET_UTILIZATION) / (1.0 - TARGET_UTILIZATION)
    return (utilization - TARGET_UTILIZATION) / TARGET_UTILIZATION


def rate_at_target_drift(utilization: float, elapsed_years: float) -> float:
    """Multiplicative drift of ``rateAtTarget`` over ``elapsed_years`` at a constant ``utilization``.

    ``AdaptiveCurveIrm``: ``endRateAtTarget = startRateAtTarget · exp(ADJUSTMENT_SPEED · err · elapsed)``
    (the on-chain code bounds the result to ``[MIN_RATE_AT_TARGET, MAX_RATE_AT_TARGET]``; callers
    clamp after multiplying). At ``u = 1`` the rate doubles in ``ln 2 / 50`` years ≈ 5.06 days.
    """
    if elapsed_years < 0.0:
        raise ValueError(f"elapsed_years must be >= 0, got {elapsed_years}")
    return math.exp(ADJUSTMENT_SPEED * adaptive_curve_error(utilization) * elapsed_years)


def supply_apy_curve(utilization, rate_at_target: float, fee: float = 0.0):
    """Vectorised :func:`supply_apy_from_curve` over an array of utilizations for one rate and fee."""
    utilization = np.asarray(utilization, dtype=float)
    if np.any(utilization < 0.0) or np.any(utilization > 1.0):
        raise ValueError("utilization values must be in [0, 1]")
    if rate_at_target < 0.0:
        raise ValueError(f"rate_at_target must be >= 0, got {rate_at_target}")
    if not 0.0 <= fee < 1.0:
        raise ValueError(f"fee must be in [0, 1), got {fee}")
    err = np.where(utilization > TARGET_UTILIZATION,
                   (utilization - TARGET_UTILIZATION) / (1.0 - TARGET_UTILIZATION),
                   (utilization - TARGET_UTILIZATION) / TARGET_UTILIZATION)
    coeff = np.where(utilization > TARGET_UTILIZATION, CURVE_STEEPNESS - 1.0, 1.0 - 1.0 / CURVE_STEEPNESS)
    return np.expm1(rate_at_target * (1.0 + coeff * err)) * utilization * (1.0 - fee)


def supply_apy_from_curve(utilization: float, rate_at_target: float, fee: float = 0.0) -> float:
    """Annual supplier APY at ``utilization`` for a continuous annual ``rate_at_target``.

    ``borrow_apy = exp(r(u)) − 1`` with ``r`` the IRM curve, then Morpho's
    ``supplyApy = borrowApy · u · (1 − fee)``.
    """
    if not 0.0 <= fee < 1.0:
        raise ValueError(f"fee must be in [0, 1), got {fee}")
    borrow_rate = adaptive_curve_borrow_apr(utilization, rate_at_target)
    return math.expm1(borrow_rate) * utilization * (1.0 - fee)


def utilization_for_supply_apy(target_apy: float, rate_at_target: float, fee: float = 0.0,
                               tolerance: float = 1e-12) -> float | None:
    """Utilisation at which the curve pays suppliers exactly ``target_apy``; ``None`` if unreachable.

    ``supply_apy_from_curve`` is strictly increasing in ``u`` (both factors are),
    so a bisection on ``[0, 1]`` is exact. Returns ``None`` when even ``u = 1``
    pays less than ``target_apy``.
    """
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be finite and > 0, got {tolerance}")
    if target_apy <= 0.0:
        return 0.0
    if supply_apy_from_curve(1.0, rate_at_target, fee) < target_apy:
        return None
    low, high = 0.0, 1.0
    for _ in range(64):
        if high - low <= tolerance:
            break
        mid = 0.5 * (low + high)
        if mid in (low, high):
            break
        if supply_apy_from_curve(mid, rate_at_target, fee) < target_apy:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
