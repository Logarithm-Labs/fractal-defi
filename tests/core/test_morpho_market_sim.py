"""Synthetic unit tests for the AdaptiveCurve market forward simulator and supplier rules."""
import math

import numpy as np
import pytest

from fractal.core.entities.models.morpho_market_sim import (
    fit_knot_means,
    piecewise_utilization_supply_flow,
    simulate_market,
    utilization_reset_supply_flow,
)
from fractal.core.entities.models.morpho_math import (
    MAX_RATE_AT_TARGET,
    MIN_RATE_AT_TARGET,
    adaptive_curve_borrow_apr,
    adaptive_curve_borrow_rate_array,
    adaptive_curve_error,
    adaptive_curve_error_array,
    horizon_average_supply_apy,
    supply_apy_curve_array,
    supply_apy_from_curve,
)

pytestmark = pytest.mark.core

HOUR = 1.0 / 8760.0


def _reference_loop(supply, borrow, rate, fee, elapsed, *, supply_rule=None, borrow_rule=None, deposit=0.0,
                    lagged=False, accrue=True, accrue_supply=True, cap=math.inf, u_obs=None, scale=1.0):
    """Independent scalar loop of one market (pre-step flows and accrual, drift, then pricing)."""
    s, b = supply + deposit, borrow
    ratio0 = b / s if s > 0 else 0.0
    u = ratio0 if u_obs is None else u_obs
    r = min(max(rate, MIN_RATE_AT_TARGET), MAX_RATE_AT_TARGET)
    apy = math.expm1(adaptive_curve_borrow_apr(u, r)) * u * (1 - fee) * scale
    out = [apy]
    for dt in elapsed:
        rb = adaptive_curve_borrow_apr(u, r)
        state = {"utilization": np.array([u]), "supply": np.array([s]), "borrow": np.array([b])}
        f_s = float(supply_rule(state)[0]) if supply_rule else 0.0
        f_b = float(borrow_rule(state)[0]) if borrow_rule else 0.0
        interest = b * math.expm1(rb * dt) if accrue else 0.0
        b_new = min(max(b + interest + f_b * b, 0.0), cap)
        s_new = s * (1 + f_s) + (interest if accrue_supply else 0.0)
        s = max(s_new, b_new, deposit)
        b = b_new
        ratio = b / s if s > 0 else 0.0
        u_new = min(max(ratio if u_obs is None else u_obs + ratio - ratio0, 0.0), 1.0)
        err_u = adaptive_curve_error(u if lagged else u_new)
        r = min(max(r * math.exp(50.0 * err_u * dt), MIN_RATE_AT_TARGET), MAX_RATE_AT_TARGET)
        u = u_new
        apy = math.expm1(adaptive_curve_borrow_apr(u, r)) * u * (1 - fee) * scale
        out.append(apy)
    return np.array(out)


def test_array_curve_helpers_match_scalar_forms():
    u = np.linspace(0.0, 1.0, 41)
    rates = np.linspace(0.0, 0.3, 41)
    fees = np.linspace(0.0, 0.25, 41)
    assert np.array_equal(adaptive_curve_error_array(u), [adaptive_curve_error(x) for x in u])
    assert np.allclose(adaptive_curve_borrow_rate_array(u, rates),
                       [adaptive_curve_borrow_apr(x, r) for x, r in zip(u, rates, strict=True)], rtol=1e-15, atol=0)
    expected = [supply_apy_from_curve(x, r, f) for x, r, f in zip(u, rates, fees, strict=True)]
    assert np.allclose(supply_apy_curve_array(u, rates, fees), expected, rtol=1e-15, atol=0)
    with pytest.raises(ValueError):
        adaptive_curve_error_array([1.1])
    with pytest.raises(ValueError):
        supply_apy_curve_array(0.5, 0.05, 1.0)


@pytest.mark.parametrize("drift_timing", ["same_step", "lagged"])
def test_frozen_utilization_matches_the_horizon_helper(drift_timing):
    elapsed = np.full(30 * 24, HOUR)
    paths = simulate_market([100.0, 100.0, 100.0], [97.0, 80.0, 90.0], [0.05, 0.08, 0.02], elapsed,
                            fee=[0.1, 0.0, 0.05], deposit=[0.0, 10.0, 5.0], accrue_borrow=False,
                            drift_timing=drift_timing)
    for i in range(3):
        u = paths.utilization[i, 0]
        assert np.all(paths.utilization[i] == u)
        expected = horizon_average_supply_apy(u, paths.rate_at_target[i, 0], elapsed, [0.1, 0.0, 0.05][i])
        assert paths.horizon_average_supply_apy()[i] == pytest.approx(expected, rel=1e-12)


def test_same_step_drift_matches_the_horizon_helper_on_a_deposit_path():
    elapsed = np.full(240, HOUR)
    deposit = np.linspace(0.0, 40.0, 241)
    paths = simulate_market(100.0, 96.0, 0.06, elapsed, fee=0.1, deposit=deposit, accrue_borrow=False)
    u_path = paths.utilization[0, 1:]
    assert u_path[-1] == pytest.approx(96.0 / 140.0)
    expected = horizon_average_supply_apy(u_path, 0.06, elapsed, 0.1)
    assert paths.horizon_average_supply_apy()[0] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("lagged", [False, True])
def test_simulator_matches_an_independent_loop_with_flows_accrual_cap_and_anchor(lagged):
    elapsed = np.r_[np.full(48, HOUR), np.full(5, 24 * HOUR)]
    supply_rule = piecewise_utilization_supply_flow([0.8, 0.9, 0.95], [0.004, 0.0, -0.003])

    def borrow_rule(state):
        return 0.002 * (0.92 - state["utilization"])

    kwargs = {"supply_rule": supply_rule, "borrow_rule": borrow_rule, "deposit": 15.0, "lagged": lagged,
              "accrue_supply": False, "cap": 101.0, "u_obs": 0.9, "scale": 1.1}
    expected = _reference_loop(110.0, 99.0, 0.07, 0.1, elapsed, **kwargs)
    paths = simulate_market(110.0, 99.0, 0.07, elapsed, fee=0.1, deposit=15.0, supply_flow=supply_rule,
                            borrow_flow=borrow_rule, borrow_cap=101.0, accrue_supply=False,
                            drift_timing="lagged" if lagged else "same_step", initial_utilization=0.9,
                            apy_scale=1.1)
    assert np.allclose(paths.supply_apy[0], expected, rtol=1e-12, atol=0)
    assert np.all(paths.borrow[0] <= 101.0 + 1e-12)


def test_drift_timings_differ_only_when_utilization_moves():
    elapsed = np.full(72, HOUR)
    rule = utilization_reset_supply_flow(0.88)
    same = simulate_market(100.0, 97.0, 0.05, elapsed, supply_flow=rule, accrue_borrow=False)
    lagged = simulate_market(100.0, 97.0, 0.05, elapsed, supply_flow=rule, accrue_borrow=False,
                            drift_timing="lagged")
    assert np.array_equal(same.utilization, lagged.utilization)
    assert lagged.rate_at_target[0, 1] > same.rate_at_target[0, 1]  # step 1 drifts on u_0 = 0.97 vs u_1 ≈ 0.88


def test_morpho_accrual_conserves_liquidity_and_floors_supply():
    elapsed = np.full(24 * 10, HOUR)
    paths = simulate_market(100.0, 90.0, 0.1, elapsed, deposit=20.0)
    liquidity = paths.supply - paths.borrow
    assert np.allclose(liquidity, liquidity[:, :1], rtol=0, atol=1e-9)
    assert paths.borrow[0, -1] > 90.0

    def drain(state):
        return np.full_like(state["supply"], -0.5)

    drained = simulate_market(100.0, 60.0, 0.05, np.full(10, HOUR), deposit=25.0, supply_flow=drain)
    assert np.all(drained.supply >= np.maximum(drained.borrow, 25.0) - 1e-12)
    assert drained.utilization[0, -1] == pytest.approx(1.0)


def test_rate_at_target_is_clamped_and_long_steps_do_not_overflow():
    paths = simulate_market(100.0, 100.0, 5.0, np.array([10.0, 10.0]), accrue_borrow=False)
    assert np.all(paths.rate_at_target == MAX_RATE_AT_TARGET)
    low = simulate_market(100.0, 10.0, 0.05, np.array([10.0]), accrue_borrow=False)
    assert low.rate_at_target[0, -1] == MIN_RATE_AT_TARGET


def test_reset_rule_returns_utilization_to_target_without_borrow_moves():
    paths = simulate_market([100.0, 200.0], [97.0, 150.0], 0.05, np.full(5, HOUR), accrue_borrow=False,
                            supply_flow=utilization_reset_supply_flow(0.9))
    assert np.allclose(paths.utilization[:, 1:], 0.9, rtol=0, atol=1e-12)


def test_piecewise_rule_is_flat_outside_the_knots():
    rule = piecewise_utilization_supply_flow([0.5, 0.9], [0.01, -0.01])
    got = rule({"utilization": np.array([0.0, 0.5, 0.7, 0.9, 1.0])})
    assert np.allclose(got, [0.01, 0.01, 0.0, -0.01, -0.01])
    with pytest.raises(ValueError, match="increasing"):
        piecewise_utilization_supply_flow([0.9, 0.5], [0.0, 0.0])
    with pytest.raises(ValueError, match="same length"):
        piecewise_utilization_supply_flow([0.5, 0.9], [0.0])


def test_fit_knot_means_assigns_nearest_knot_and_winsorises():
    knots = [0.8, 0.9, 1.0]
    u = np.array([0.7, 0.84, 0.86, 0.94, 0.96, np.nan, 0.9])
    flow = np.array([1.0, 3.0, 5.0, 7.0, 100.0, 9.0, np.nan])
    fit = fit_knot_means(u, flow, knots)
    assert fit.counts.tolist() == [2, 2, 1]
    assert np.allclose(fit.values, [2.0, 6.0, 100.0])
    empty = fit_knot_means([0.1], [1.0], [0.1, 0.5])
    assert empty.values[0] == 1.0 and math.isnan(empty.values[1])
    clipped = fit_knot_means(np.full(101, 0.8), np.r_[np.zeros(100), 1000.0], knots, winsor_quantile=0.01)
    assert clipped.values[0] == pytest.approx(0.0)


def test_rule_fitted_from_simulated_flows_recovers_the_rule():
    knots = np.array([0.80, 0.88, 0.95])
    rule = piecewise_utilization_supply_flow(knots, [0.003, 0.0, -0.004])
    u = np.repeat(knots, 5)
    fitted = fit_knot_means(u, rule({"utilization": u}), knots)
    assert np.allclose(fitted.values, [0.003, 0.0, -0.004])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"elapsed_years_per_step": []}, "elapsed"),
        ({"elapsed_years_per_step": [HOUR, -HOUR]}, "elapsed"),
        ({"drift_timing": "next"}, "drift_timing"),
        ({"fee": 1.0}, "fee"),
        ({"supply": -1.0}, ">= 0"),
        ({"deposit": np.ones((2, 4))}, "deposit"),
        ({"initial_utilization": 1.5}, "initial_utilization"),
        ({"borrow_cap": -1.0}, "borrow_cap"),
        ({"supply_flow": lambda state: np.full(1, np.nan)}, "supply_flow"),
    ],
)
def test_simulator_rejects_invalid_inputs(kwargs, match):
    call = {"supply": 100.0, "borrow": 90.0, "rate_at_target": 0.05, "elapsed_years_per_step": [HOUR, HOUR]}
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        simulate_market(**call)
