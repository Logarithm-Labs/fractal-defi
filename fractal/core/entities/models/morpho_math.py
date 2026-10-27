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
    "adaptive_curve_borrow_rate_array",
    "adaptive_curve_error",
    "adaptive_curve_error_array",
    "borrow_apy_from_per_second_rate",
    "from_wad",
    "horizon_average_supply_apy",
    "horizon_supply_capacity",
    "liquidation_incentive_factor",
    "max_borrow",
    "per_bar_borrow_rate",
    "rate_at_target_drift",
    "static_supply_capacity",
    "supply_apy_curve",
    "supply_apy_curve_array",
    "supply_apy_from_curve",
    "taylor_compounded",
    "utilization_after_deposit",
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


def utilization_after_deposit(supply, borrow, deposit, alpha=0.0, beta=0.0):
    """Post-deposit utilization under two-sided borrower and supplier response.

    ``u'(P) = (borrow + alpha * P) / (supply + (1 - beta) * P)``.
    ``alpha`` in ``[0, 1]`` is the deposited share borrowed and ``beta <= 1`` is
    the share offset by other suppliers withdrawing; a negative ``beta`` is
    crowding-in (other suppliers add to the deposit). Inputs broadcast as NumPy
    arrays; scalar inputs return a float. The result is clamped to ``[0, 1]``
    and is ``0`` when effective supply is zero.
    """
    supply_values, borrow_values, deposit_values, alpha_values, beta_values = np.broadcast_arrays(
        np.asarray(supply, dtype=float),
        np.asarray(borrow, dtype=float),
        np.asarray(deposit, dtype=float),
        np.asarray(alpha, dtype=float),
        np.asarray(beta, dtype=float),
    )
    values = (supply_values, borrow_values, deposit_values, alpha_values, beta_values)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("supply, borrow, deposit, alpha and beta must be finite")
    if np.any(supply_values < 0.0) or np.any(borrow_values < 0.0) or np.any(deposit_values < 0.0):
        raise ValueError("supply, borrow and deposit must be >= 0")
    if np.any(alpha_values < 0.0) or np.any(alpha_values > 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if np.any(beta_values > 1.0):
        raise ValueError("beta must be <= 1")
    total_supply = supply_values + (1.0 - beta_values) * deposit_values
    total_borrow = borrow_values + alpha_values * deposit_values
    utilization = np.divide(
        total_borrow,
        total_supply,
        out=np.zeros_like(total_borrow),
        where=total_supply > 0.0,
    )
    utilization = np.clip(utilization, 0.0, 1.0)
    return float(utilization) if utilization.ndim == 0 else utilization


def static_supply_capacity(target_apy: float, supply: float, borrow: float, rate_at_target: float,
                           fee: float = 0.0, alpha: float = 0.0, beta: float = 0.0) -> float:
    """Largest deposit whose instantaneous supplier APY remains at least ``target_apy``.

    The AdaptiveCurve supplier-rate curve is inverted to a target utilization,
    then ``u'(P) = u*`` is solved in closed form:
    ``P = (borrow - u* * supply) / (u* * (1 - beta) - alpha)``. Returns ``0``
    when the undeployed market is already below target (even when a response
    with ``alpha / (1 - beta) > u*`` would lift a large deposit above it) or
    the target exceeds the curve maximum. Returns ``inf`` when utilization
    cannot fall below the target, ``u* * (1 - beta) <= alpha``, while the
    current rate clears it. ``alpha = beta = 0`` reduces exactly to the static
    ``borrow / (supply + deposit)`` model. ``rate_at_target`` is the
    continuous annual rate; divide ``target_apy`` by any observed-to-curve APY
    scale before calling.
    """
    if math.isnan(target_apy) or target_apy < 0.0:
        raise ValueError(f"target_apy must be >= 0, got {target_apy}")
    if not math.isfinite(rate_at_target) or rate_at_target < 0.0:
        raise ValueError(f"rate_at_target must be finite and >= 0, got {rate_at_target}")
    current_utilization = utilization_after_deposit(supply, borrow, 0.0, alpha, beta)
    current_apy = supply_apy_from_curve(current_utilization, rate_at_target, fee)
    if target_apy <= 0.0:
        return float("inf")
    if current_apy < target_apy:
        return 0.0
    target_utilization = utilization_for_supply_apy(target_apy, rate_at_target, fee)
    if target_utilization is None:
        return 0.0
    denominator = target_utilization * (1.0 - beta) - alpha
    if denominator <= 0.0:
        return float("inf")
    capacity = (borrow - target_utilization * supply) / denominator
    return max(capacity, 0.0)


def _horizon_path_arrays(first, second, elapsed_years_per_step):
    """Broadcast two paths and elapsed step lengths to flat float arrays."""
    first_values, second_values, elapsed_values = np.broadcast_arrays(
        np.asarray(first, dtype=float),
        np.asarray(second, dtype=float),
        np.asarray(elapsed_years_per_step, dtype=float),
    )
    if first_values.size == 0:
        raise ValueError("horizon paths must not be empty")
    if not all(np.all(np.isfinite(value)) for value in (first_values, second_values, elapsed_values)):
        raise ValueError("horizon paths and elapsed_years_per_step must be finite")
    if np.any(elapsed_values < 0.0) or float(np.sum(elapsed_values)) <= 0.0:
        raise ValueError("elapsed_years_per_step must be >= 0 with a positive total")
    return first_values.ravel(), second_values.ravel(), elapsed_values.ravel()


def horizon_average_supply_apy(utilization_path, rate_at_target: float, elapsed_years_per_step,
                               fee: float = 0.0, drift_reference_utilization=None) -> float:
    """Annualized supplier APY along a utilization path with AdaptiveCurve drift.

    Each step first drifts ``rate_at_target`` for that step and clamps it to
    ``[MIN_RATE_AT_TARGET, MAX_RATE_AT_TARGET]``, matching the controller-before-
    accrual order. Step APYs compound over their ACT/365 durations; the return
    value is the constant annual APY with the same horizon growth. A scalar
    utilization broadcasts across an array of step durations.

    By default the controller steers on the absolute error ``err(u)``. A
    ``drift_reference_utilization`` (scalar or path) makes it steer on
    ``err(u) - err(reference)`` instead: the pre-deposit utilization isolates
    the drift caused by a deposit, and the path itself freezes
    ``rate_at_target``.
    """
    reference = utilization_path if drift_reference_utilization is None else drift_reference_utilization
    utilization_values, reference_values, elapsed_values = _horizon_path_arrays(
        utilization_path, reference, elapsed_years_per_step
    )
    for values in (utilization_values, reference_values):
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("utilization values must be in [0, 1]")
    if not math.isfinite(rate_at_target) or rate_at_target < 0.0:
        raise ValueError(f"rate_at_target must be finite and >= 0, got {rate_at_target}")
    if not 0.0 <= fee < 1.0:
        raise ValueError(f"fee must be in [0, 1), got {fee}")
    current_rate = min(max(rate_at_target, MIN_RATE_AT_TARGET), MAX_RATE_AT_TARGET)
    log_growth = 0.0
    for utilization, reference_utilization, elapsed_years in zip(
        utilization_values, reference_values, elapsed_values, strict=True
    ):
        error = adaptive_curve_error(float(utilization))
        if drift_reference_utilization is not None:
            error -= adaptive_curve_error(float(reference_utilization))
        try:
            drift = math.exp(ADJUSTMENT_SPEED * error * float(elapsed_years))
        except OverflowError:
            drift = float("inf")
        current_rate = min(max(current_rate * drift, MIN_RATE_AT_TARGET), MAX_RATE_AT_TARGET)
        step_apy = supply_apy_from_curve(float(utilization), current_rate, fee)
        log_growth += float(elapsed_years) * math.log1p(step_apy)
    return math.expm1(log_growth / float(np.sum(elapsed_values)))


def horizon_supply_capacity(target_apy: float, supply: float, borrow: float, rate_at_target: float,
                            elapsed_years_per_step, fee: float = 0.0, alpha=0.0, beta=0.0,
                            tolerance: float = 1e-6, drift_basis: str = "absolute") -> float:
    """Largest deposit whose horizon-average supplier APY remains at least ``target_apy``.

    ``alpha`` and ``beta`` may be scalars for a constant post-deposit
    utilization response or arrays describing a response path. They broadcast
    against ``elapsed_years_per_step``. The capacity is found by bisection,
    using ``tolerance`` as a relative deposit-size tolerance. Returns ``0``
    when the zero-deposit horizon rate misses the target and ``inf`` when the
    infinite-deposit response still clears it. The bisection assumes the
    horizon APY does not rise again with the deposit once it has fallen below
    the target; a response path that raises utilization with the deposit in
    some steps and lowers it in others can break that.

    ``drift_basis`` selects the controller error: ``"absolute"`` steers on
    ``err(u)``, ``"deposit"`` on ``err(u) - err(borrow / supply)`` (only the
    deposit's own effect drifts the rate) and ``"frozen"`` keeps
    ``rate_at_target`` fixed. See :func:`horizon_average_supply_apy`.
    """
    if math.isnan(target_apy) or target_apy < 0.0:
        raise ValueError(f"target_apy must be >= 0, got {target_apy}")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"tolerance must be finite and > 0, got {tolerance}")
    if drift_basis not in ("absolute", "deposit", "frozen"):
        raise ValueError(f"drift_basis must be 'absolute', 'deposit' or 'frozen', got {drift_basis!r}")
    alpha_values, beta_values, elapsed_values = _horizon_path_arrays(alpha, beta, elapsed_years_per_step)
    if np.any(alpha_values < 0.0) or np.any(alpha_values > 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if np.any(beta_values > 1.0):
        raise ValueError("beta must be <= 1")

    pre_deposit_utilization = utilization_after_deposit(supply, borrow, 0.0)

    def path_average_apy(utilization_path) -> float:
        reference = {"absolute": None, "deposit": pre_deposit_utilization, "frozen": utilization_path}[drift_basis]
        return horizon_average_supply_apy(utilization_path, rate_at_target, elapsed_values, fee, reference)

    def average_apy(deposit: float) -> float:
        return path_average_apy(utilization_after_deposit(supply, borrow, deposit, alpha_values, beta_values))

    current_apy = average_apy(0.0)
    if target_apy <= 0.0:
        return float("inf")
    if current_apy < target_apy:
        return 0.0
    current_utilization = utilization_after_deposit(supply, borrow, 0.0, alpha_values, beta_values)
    retained_supply = 1.0 - beta_values
    limit_utilization = np.divide(
        alpha_values,
        retained_supply,
        out=np.zeros_like(alpha_values),
        where=retained_supply > 0.0,
    )
    limit_utilization = np.where(
        retained_supply > 0.0,
        limit_utilization,
        np.where(alpha_values > 0.0, 1.0, current_utilization),
    )
    limit_utilization = np.clip(limit_utilization, 0.0, 1.0)
    if path_average_apy(limit_utilization) >= target_apy:
        return float("inf")

    low, high = 0.0, max(supply, borrow, 1.0)
    for _ in range(256):
        if average_apy(high) < target_apy:
            break
        high *= 2.0
    else:
        return float("inf")
    for _ in range(256):
        if high - low <= tolerance * max(high, 1.0):
            break
        mid = 0.5 * (low + high)
        if average_apy(mid) >= target_apy:
            low = mid
        else:
            high = mid
    return low


def adaptive_curve_error_array(utilization):
    """Vectorised :func:`adaptive_curve_error` over an array of utilizations in ``[0, 1]``."""
    utilization = np.asarray(utilization, dtype=float)
    if np.any(~np.isfinite(utilization)) or np.any(utilization < 0.0) or np.any(utilization > 1.0):
        raise ValueError("utilization values must be in [0, 1]")
    return np.where(utilization > TARGET_UTILIZATION,
                    (utilization - TARGET_UTILIZATION) / (1.0 - TARGET_UTILIZATION),
                    (utilization - TARGET_UTILIZATION) / TARGET_UTILIZATION)


def adaptive_curve_borrow_rate_array(utilization, rate_at_target):
    """Vectorised :func:`adaptive_curve_borrow_apr` with an array ``rate_at_target``.

    Both inputs broadcast; the result shares the unit of ``rate_at_target``
    (continuous annual in this module).
    """
    rate_at_target = np.asarray(rate_at_target, dtype=float)
    if np.any(~np.isfinite(rate_at_target)) or np.any(rate_at_target < 0.0):
        raise ValueError("rate_at_target must be finite and >= 0")
    error = adaptive_curve_error_array(utilization)
    coeff = np.where(error > 0.0, CURVE_STEEPNESS - 1.0, 1.0 - 1.0 / CURVE_STEEPNESS)
    return rate_at_target * (1.0 + coeff * error)


def supply_apy_curve_array(utilization, rate_at_target, fee=0.0):
    """Vectorised :func:`supply_apy_from_curve` with array ``rate_at_target`` and ``fee``.

    ``expm1(r(u)) * u * (1 - fee)``; all inputs broadcast. Use it when every
    market or path step has its own controller rate.
    """
    fee = np.asarray(fee, dtype=float)
    if np.any(~np.isfinite(fee)) or np.any(fee < 0.0) or np.any(fee >= 1.0):
        raise ValueError("fee must be in [0, 1)")
    utilization = np.asarray(utilization, dtype=float)
    return np.expm1(adaptive_curve_borrow_rate_array(utilization, rate_at_target)) * utilization * (1.0 - fee)
