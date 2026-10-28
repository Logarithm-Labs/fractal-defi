"""Forward simulation of Morpho Blue markets under the AdaptiveCurve IRM.

:func:`simulate_market` steps a batch of independent markets through a
sequence of steps (hourly, daily or uneven ACT/365 durations). Each step it
applies optional borrower and supplier flow rules, borrow interest, an
optional borrow cap and our own deposit path, then updates utilization, drifts
``rateAtTarget`` and prices the borrow and supply APYs. Rates are continuous
annual decimals, as in :mod:`fractal.core.entities.models.morpho_math`.

Flow rules are callables ``rule(state) -> per-step relative change`` (an
array broadcastable to the number of markets). :func:`piecewise_utilization_supply_flow`
and :func:`utilization_reset_supply_flow` build supplier rules;
:func:`fit_knot_means` estimates knot values from observed
``(utilization, ΔS/S)`` samples. The module ships no fitted values: callers
supply their own knots and values.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from fractal.core.entities.models.morpho_math import (
    ADJUSTMENT_SPEED,
    MAX_RATE_AT_TARGET,
    MIN_RATE_AT_TARGET,
    adaptive_curve_borrow_rate_array,
    adaptive_curve_error_array,
)

__all__ = [
    "DRIFT_TIMINGS",
    "FlowRule",
    "KnotMeans",
    "MarketPaths",
    "fit_knot_means",
    "piecewise_utilization_supply_flow",
    "simulate_market",
    "utilization_reset_supply_flow",
]

FlowRule = Callable[[Mapping[str, np.ndarray]], np.ndarray]
DRIFT_TIMINGS = ("same_step", "lagged")


@dataclass(frozen=True)
class MarketPaths:
    """Paths of a forward simulation.

    Market arrays have shape ``(markets, steps + 1)``; column ``0`` is the
    origin (after the initial deposit) and column ``k`` the state at the end
    of step ``k``. ``supply`` includes our deposit. ``elapsed_years`` holds the
    ``steps`` step durations.
    """

    supply: np.ndarray
    borrow: np.ndarray
    deposit: np.ndarray
    utilization: np.ndarray
    rate_at_target: np.ndarray
    borrow_apy: np.ndarray
    supply_apy: np.ndarray
    elapsed_years: np.ndarray

    def horizon_average_supply_apy(self, compounded: bool = True) -> np.ndarray:
        """Per-market horizon average of the supply APYs of steps ``1..steps``.

        ``compounded=True`` (default): the constant APY with the same growth,
        step APYs compounding over their ACT/365 durations, as in
        :func:`~fractal.core.entities.models.morpho_math.horizon_average_supply_apy`.
        ``compounded=False``: the duration-weighted arithmetic mean of the step
        APYs. The arithmetic mean is never below the compounded one (Jensen);
        the gap grows with the APY level and its variation over the horizon and
        can reach tens of basis points when the controller moves the rate a lot.
        """
        weights = self.elapsed_years / float(self.elapsed_years.sum())
        if not compounded:
            return self.supply_apy[:, 1:] @ weights
        return np.expm1(np.log1p(self.supply_apy[:, 1:]) @ weights)


def _market_values(value, markets: int, name: str) -> np.ndarray:
    values = np.broadcast_to(np.asarray(value, dtype=float), (markets,)).copy()
    if np.any(np.isnan(values)):
        raise ValueError(f"{name} must not be NaN")
    return values


def _path_values(value, markets: int, columns: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 1:
        values = values[:, None] if 1 < markets == values.size else values[None, :]
    try:
        values = np.broadcast_to(values, (markets, columns)).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must be a scalar, per-market, (steps + 1,) or (markets, steps + 1)") from exc
    if np.any(np.isnan(values)):
        raise ValueError(f"{name} must not be NaN")
    return values


def _ratio(borrow: np.ndarray, supply: np.ndarray) -> np.ndarray:
    return np.divide(borrow, supply, out=np.zeros_like(borrow), where=supply > 0.0)


def _flow(rule: FlowRule | None, state: Mapping[str, np.ndarray], markets: int, name: str) -> np.ndarray:
    if rule is None:
        return np.zeros(markets)
    values = np.broadcast_to(np.asarray(rule(state), dtype=float), (markets,))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} returned non-finite values")
    return values


def simulate_market(supply, borrow, rate_at_target, elapsed_years_per_step, *, fee=0.0, deposit=0.0,
                    supply_flow: FlowRule | None = None, borrow_flow: FlowRule | None = None,
                    borrow_cap=None, accrue_borrow: bool = True, accrue_supply: bool = True,
                    drift_timing: str = "same_step", initial_utilization=None,
                    apy_scale=1.0, observed_supply_apy=None,
                    undefined_anchor: str = "clamp") -> MarketPaths:
    """Simulate markets forward under the AdaptiveCurve IRM.

    Args:
        supply, borrow: market totals before our deposit, one value or one per market.
        rate_at_target: continuous annual ``rateAtTarget`` at the origin; clamped to
            ``[MIN_RATE_AT_TARGET, MAX_RATE_AT_TARGET]``.
        elapsed_years_per_step: ACT/365 duration of each step (for example
            ``np.full(720, 1 / 8760)`` for 30 hourly steps).
        fee: market fee in ``[0, 1)``.
        deposit: our position. A scalar or per-market value is held for the whole
            horizon; a ``(steps + 1,)`` or ``(markets, steps + 1)`` path sets the
            position at each column, and changes are added to or removed from
            supply. A 1-D array of length ``markets > 1`` is read as per-market.
        supply_flow, borrow_flow: rules ``state -> relative change per step`` of
            supply and borrow. ``state`` holds the pre-step arrays ``step``,
            ``elapsed_years``, ``utilization``, ``rate_at_target``, ``borrow_rate``
            (continuous), ``supply_apy``, ``supply``, ``borrow`` and ``deposit``.
        borrow_cap: optional cap on borrow, per market or as a path like ``deposit``.
        accrue_borrow: accrue borrow interest at the pre-step borrow rate.
        accrue_supply: credit the same interest to supply, as Morpho does. Pass
            ``False`` when ``supply_flow`` already models total supply changes,
            interest included (for example a rule fitted on observed ΔS/S).
        drift_timing: ``"same_step"`` drifts ``rateAtTarget`` over step ``k`` with
            ``err(u_k)`` (the convention of ``horizon_average_supply_apy``);
            ``"lagged"`` uses ``err(u_{k-1})``. Both price step ``k`` at ``u_k``.
        initial_utilization: optional observed utilization at the origin. When
            given, ``u_k = u_0 + B_k / S_k - B_0 / S_0`` (increment convention,
            anchored to the observed level) instead of ``B_k / S_k``.
        apy_scale: per-market multiplier of the supply APY, for example the
            ratio of an observed APY to the curve APY at the origin.
        observed_supply_apy: per-market observed supply APY at the origin, used
            only by ``undefined_anchor="hold_observed"``.
        undefined_anchor: what to do where the curve cannot be anchored, i.e.
            ``rate_at_target <= 0`` or the origin curve APY (unclamped rate,
            ``u_0``, fee) is ``0``. ``"clamp"`` (default) prices those markets
            like any other, at the rate clamped to ``MIN_RATE_AT_TARGET``.
            ``"hold_observed"`` keeps their supply APY at
            ``observed_supply_apy`` for the whole path (the convention of the
            research engine this simulator was checked against); utilization,
            rate and borrow APY paths still evolve.

    Supply never falls below ``max(borrow, deposit)``: other suppliers cannot
    withdraw borrowed liquidity or our deposit. Utilization is clamped to
    ``[0, 1]`` and is ``0`` on an empty market.
    """
    if drift_timing not in DRIFT_TIMINGS:
        raise ValueError(f"drift_timing must be one of {DRIFT_TIMINGS}, got {drift_timing!r}")
    if undefined_anchor not in ("clamp", "hold_observed"):
        raise ValueError(f"undefined_anchor must be 'clamp' or 'hold_observed', got {undefined_anchor!r}")
    if undefined_anchor == "hold_observed" and observed_supply_apy is None:
        raise ValueError("undefined_anchor='hold_observed' needs observed_supply_apy")
    elapsed = np.asarray(elapsed_years_per_step, dtype=float).ravel()
    if elapsed.size == 0 or not np.all(np.isfinite(elapsed)) or np.any(elapsed < 0.0) or elapsed.sum() <= 0.0:
        raise ValueError("elapsed_years_per_step must be non-empty, finite, >= 0 with a positive total")
    steps = elapsed.size
    per_market = [supply, borrow, rate_at_target, fee, apy_scale]
    per_market += [value for value in (initial_utilization, observed_supply_apy) if value is not None]
    if any(np.ndim(value) > 1 for value in per_market):
        raise ValueError("per-market inputs must be scalars or 1-D arrays")
    shapes = [np.shape(value) for value in per_market]
    shapes += [np.shape(path)[:1] for path in (deposit, borrow_cap) if path is not None and np.ndim(path) == 2]
    markets = int(np.broadcast_shapes((1,), *shapes)[0])
    s0 = _market_values(supply, markets, "supply")
    b0 = _market_values(borrow, markets, "borrow")
    r0 = _market_values(rate_at_target, markets, "rate_at_target")
    fees = _market_values(fee, markets, "fee")
    scale = _market_values(apy_scale, markets, "apy_scale")
    deposits = _path_values(deposit, markets, steps + 1, "deposit")
    caps = None if borrow_cap is None else _path_values(borrow_cap, markets, steps + 1, "borrow_cap")
    finite = (s0, b0, r0, fees, scale, deposits)
    if not all(np.all(np.isfinite(v)) for v in finite):
        raise ValueError("supply, borrow, rate_at_target, fee, apy_scale and deposit must be finite")
    if np.any(s0 < 0.0) or np.any(b0 < 0.0) or np.any(deposits < 0.0) or np.any(r0 < 0.0) or np.any(scale < 0.0):
        raise ValueError("supply, borrow, deposit, rate_at_target and apy_scale must be >= 0")
    if np.any(fees < 0.0) or np.any(fees >= 1.0):
        raise ValueError("fee must be in [0, 1)")
    if caps is not None and np.any(caps < 0.0):
        raise ValueError("borrow_cap must be >= 0")

    shape = (markets, steps + 1)
    supply_path, borrow_path = np.empty(shape), np.empty(shape)
    utilization, rate_path = np.empty(shape), np.empty(shape)
    borrow_apy, supply_apy = np.empty(shape), np.empty(shape)

    supply_path[:, 0] = np.maximum(s0 + deposits[:, 0], np.maximum(b0, deposits[:, 0]))
    borrow_path[:, 0] = b0
    ratio0 = _ratio(b0, supply_path[:, 0])
    if initial_utilization is None:
        u0 = np.clip(ratio0, 0.0, 1.0)
    else:
        u0 = _market_values(initial_utilization, markets, "initial_utilization")
        if np.any(u0 < 0.0) or np.any(u0 > 1.0):
            raise ValueError("initial_utilization must be in [0, 1]")
    utilization[:, 0] = u0
    rate_path[:, 0] = np.clip(r0, MIN_RATE_AT_TARGET, MAX_RATE_AT_TARGET)
    held = np.zeros(markets, dtype=bool)
    observed = np.zeros(markets)
    if undefined_anchor == "hold_observed":
        observed = _market_values(observed_supply_apy, markets, "observed_supply_apy")
        if not np.all(np.isfinite(observed)):
            raise ValueError("observed_supply_apy must be finite")
        origin_curve = np.expm1(adaptive_curve_borrow_rate_array(u0, r0)) * u0 * (1.0 - fees)
        held = (r0 <= 0.0) | (origin_curve <= 0.0)

    def price(column: int) -> None:
        rate = adaptive_curve_borrow_rate_array(utilization[:, column], rate_path[:, column])
        borrow_apy[:, column] = np.expm1(rate)
        supply_apy[:, column] = borrow_apy[:, column] * utilization[:, column] * (1.0 - fees) * scale
        if held.any():
            supply_apy[held, column] = observed[held]

    price(0)
    for k in range(1, steps + 1):
        dt = elapsed[k - 1]
        s_prev, b_prev, u_prev = supply_path[:, k - 1], borrow_path[:, k - 1], utilization[:, k - 1]
        borrow_rate = adaptive_curve_borrow_rate_array(u_prev, rate_path[:, k - 1])
        state = {"step": np.full(markets, k), "elapsed_years": np.full(markets, dt), "utilization": u_prev,
                 "rate_at_target": rate_path[:, k - 1], "borrow_rate": borrow_rate,
                 "supply_apy": supply_apy[:, k - 1], "supply": s_prev, "borrow": b_prev,
                 "deposit": deposits[:, k - 1]}
        f_borrow = _flow(borrow_flow, state, markets, "borrow_flow")
        f_supply = _flow(supply_flow, state, markets, "supply_flow")
        interest = b_prev * np.expm1(borrow_rate * dt) if accrue_borrow else np.zeros(markets)
        b_new = np.maximum(b_prev + interest + f_borrow * b_prev, 0.0)
        if caps is not None:
            b_new = np.minimum(b_new, caps[:, k])
        s_new = s_prev * (1.0 + f_supply) + deposits[:, k] - deposits[:, k - 1]
        if accrue_supply:
            s_new = s_new + interest
        s_new = np.maximum(s_new, np.maximum(b_new, deposits[:, k]))
        supply_path[:, k], borrow_path[:, k] = s_new, b_new
        ratio = _ratio(b_new, s_new)
        utilization[:, k] = np.clip(ratio if initial_utilization is None else u0 + ratio - ratio0, 0.0, 1.0)
        error = adaptive_curve_error_array(utilization[:, k] if drift_timing == "same_step" else u_prev)
        with np.errstate(over="ignore"):
            drift = np.exp(ADJUSTMENT_SPEED * error * dt)
        rate_path[:, k] = np.clip(rate_path[:, k - 1] * drift, MIN_RATE_AT_TARGET, MAX_RATE_AT_TARGET)
        price(k)
    return MarketPaths(supply_path, borrow_path, deposits, utilization, rate_path, borrow_apy, supply_apy, elapsed)


def piecewise_utilization_supply_flow(knots, values) -> FlowRule:
    """Supplier rule ``f_S(u)``: linear between ``(knot, value)`` pairs, flat outside the knots.

    ``values`` are relative supply changes per simulation step at each knot
    utilization; they must match the step length used in :func:`simulate_market`.
    """
    knots = np.asarray(knots, dtype=float)
    values = np.asarray(values, dtype=float)
    if knots.ndim != 1 or knots.size == 0 or knots.shape != values.shape:
        raise ValueError("knots and values must be non-empty 1-D arrays of the same length")
    if not (np.all(np.isfinite(knots)) and np.all(np.isfinite(values))):
        raise ValueError("knots and values must be finite")
    if np.any(knots < 0.0) or np.any(knots > 1.0) or np.any(np.diff(knots) <= 0.0):
        raise ValueError("knots must be strictly increasing in [0, 1]")
    if np.any(values <= -1.0):
        raise ValueError("values must be > -1")

    def rule(state: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.interp(state["utilization"], knots, values)

    return rule


def utilization_reset_supply_flow(target_utilization: float) -> FlowRule:
    """Parameter-free supplier rule: supply moves each step to ``borrow / target_utilization``.

    The change is computed from the pre-step state, so utilization returns to
    the target up to the step's borrow flow and interest.
    """
    if not 0.0 < target_utilization <= 1.0:
        raise ValueError(f"target_utilization must be in (0, 1], got {target_utilization}")

    def rule(state: Mapping[str, np.ndarray]) -> np.ndarray:
        supply = np.asarray(state["supply"], dtype=float)
        borrow = np.asarray(state["borrow"], dtype=float)
        target_supply = borrow / target_utilization
        return np.divide(target_supply - supply, supply, out=np.zeros_like(supply), where=supply > 0.0)

    return rule


@dataclass(frozen=True)
class KnotMeans:
    """Mean flow per knot cell: ``values[i]`` is ``NaN`` when ``counts[i] == 0``."""

    knots: np.ndarray
    values: np.ndarray
    counts: np.ndarray


def fit_knot_means(utilization, flow, knots, winsor_quantile: float = 0.0) -> KnotMeans:
    """Knot values for :func:`piecewise_utilization_supply_flow` from observed samples.

    Each finite ``(utilization, flow)`` pair is assigned to its nearest knot
    (cells split at knot midpoints, the end cells are open) and each cell
    value is the mean flow. ``flow`` is the per-step relative supply change
    ``ΔS / S``. With ``winsor_quantile > 0`` flows are first clipped to the
    ``[q, 1 - q]`` sample quantiles.
    """
    knots = np.asarray(knots, dtype=float)
    if knots.ndim != 1 or knots.size == 0 or not np.all(np.isfinite(knots)) or np.any(np.diff(knots) <= 0.0):
        raise ValueError("knots must be a non-empty, finite, strictly increasing 1-D array")
    if not 0.0 <= winsor_quantile < 0.5:
        raise ValueError(f"winsor_quantile must be in [0, 0.5), got {winsor_quantile}")
    utilization = np.asarray(utilization, dtype=float).ravel()
    flow = np.asarray(flow, dtype=float).ravel()
    if utilization.shape != flow.shape:
        raise ValueError("utilization and flow must have the same length")
    keep = np.isfinite(utilization) & np.isfinite(flow)
    utilization, flow = utilization[keep], flow[keep]
    if winsor_quantile > 0.0 and flow.size:
        low, high = np.quantile(flow, [winsor_quantile, 1.0 - winsor_quantile])
        flow = np.clip(flow, low, high)
    cells = np.searchsorted((knots[:-1] + knots[1:]) / 2.0, utilization, side="right")
    counts = np.bincount(cells, minlength=knots.size)
    sums = np.bincount(cells, weights=flow, minlength=knots.size)
    values = np.divide(sums, counts, out=np.full(knots.size, np.nan), where=counts > 0)
    return KnotMeans(knots.copy(), values, counts)
