"""Pure Morpho Blue maths: interest accrual, liquidation incentive, IRM.

Sources: ``morpho-blue/src/Morpho.sol`` (``_accrueInterest``,
``_isHealthy``, ``liquidate``), ``libraries/MathLib.sol``
(``wTaylorCompounded``), ``libraries/ConstantsLib.sol`` and
``morpho-blue-irm/AdaptiveCurveIrm.sol``. Rates are decimals, time is
seconds or ACT/365 years.
"""
import math

from fractal.core.base.time import SECONDS_PER_YEAR

LIQUIDATION_CURSOR = 0.3                 # ConstantsLib.LIQUIDATION_CURSOR
MAX_LIQUIDATION_INCENTIVE_FACTOR = 1.15  # ConstantsLib.MAX_LIQUIDATION_INCENTIVE_FACTOR
TARGET_UTILIZATION = 0.9                 # AdaptiveCurveIrm
CURVE_STEEPNESS = 4.0                    # AdaptiveCurveIrm

__all__ = [
    "LIQUIDATION_CURSOR",
    "MAX_LIQUIDATION_INCENTIVE_FACTOR",
    "TARGET_UTILIZATION",
    "CURVE_STEEPNESS",
    "taylor_compounded",
    "per_bar_borrow_rate",
    "borrow_apy_from_per_second_rate",
    "liquidation_incentive_factor",
    "max_borrow",
    "adaptive_curve_borrow_apr",
]


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
