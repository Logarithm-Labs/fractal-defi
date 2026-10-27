"""Synthetic unit tests for standalone Morpho supply-capacity helpers."""
import math

import numpy as np
import pytest

from fractal.core.entities.models.morpho_math import (
    MAX_RATE_AT_TARGET,
    MIN_RATE_AT_TARGET,
    TARGET_UTILIZATION,
    horizon_average_supply_apy,
    horizon_supply_capacity,
    monotone_response,
    rate_at_target_drift,
    rate_at_target_transition_years,
    static_supply_capacity,
    supply_apy_from_curve,
    utilization_after_deposit,
    utilization_for_supply_apy,
)

pytestmark = pytest.mark.core


def _numeric_static_capacity(target_apy, supply, borrow, rate_at_target, fee, alpha, beta):
    def earns_enough(deposit):
        utilization = utilization_after_deposit(supply, borrow, deposit, alpha, beta)
        return supply_apy_from_curve(utilization, rate_at_target, fee) >= target_apy

    low, high = 0.0, max(supply, 1.0)
    while earns_enough(high):
        high *= 2.0
    for _ in range(100):
        mid = 0.5 * (low + high)
        if earns_enough(mid):
            low = mid
        else:
            high = mid
    return low


def test_zero_response_matches_plain_post_deposit_utilization_exactly():
    supply, borrow, deposit = 125.0, 90.0, 37.0
    assert utilization_after_deposit(supply, borrow, deposit) == borrow / (supply + deposit)


def test_static_capacity_zero_response_matches_existing_curve_inverse():
    supply, borrow, rate_at_target, fee = 125.0, 112.5, 0.08, 0.07
    target_apy = supply_apy_from_curve(0.75, rate_at_target, fee)
    target_utilization = utilization_for_supply_apy(target_apy, rate_at_target, fee)
    assert target_utilization is not None
    expected = max(borrow / target_utilization - supply, 0.0)
    assert static_supply_capacity(target_apy, supply, borrow, rate_at_target, fee) == pytest.approx(
        expected, rel=0.0, abs=1e-12
    )


def test_static_capacity_closed_form_matches_numeric_bisection():
    args = {
        "supply": 100.0,
        "borrow": 90.0,
        "rate_at_target": 0.06,
        "fee": 0.1,
        "alpha": 0.2,
        "beta": 0.1,
    }
    target_apy = supply_apy_from_curve(0.75, args["rate_at_target"], args["fee"])
    expected = _numeric_static_capacity(target_apy=target_apy, **args)
    assert static_supply_capacity(target_apy=target_apy, **args) == pytest.approx(expected, rel=1e-10)


def test_static_capacity_is_monotone_in_target_alpha_and_beta():
    supply, borrow, rate_at_target = 100.0, 90.0, 0.08
    low_target = supply_apy_from_curve(0.7, rate_at_target)
    high_target = supply_apy_from_curve(0.8, rate_at_target)
    assert static_supply_capacity(low_target, supply, borrow, rate_at_target) >= static_supply_capacity(
        high_target, supply, borrow, rate_at_target
    )
    by_alpha = [
        static_supply_capacity(high_target, supply, borrow, rate_at_target, alpha=alpha)
        for alpha in (0.0, 0.1, 0.2)
    ]
    by_beta = [
        static_supply_capacity(high_target, supply, borrow, rate_at_target, beta=beta)
        for beta in (0.0, 0.1, 0.2)
    ]
    assert by_alpha == sorted(by_alpha)
    assert by_beta == sorted(by_beta)


def test_static_capacity_handles_zero_supply_unreachable_target_and_full_offset():
    rate_at_target = 0.05
    above_current = supply_apy_from_curve(0.91, rate_at_target)
    above_max = supply_apy_from_curve(1.0, rate_at_target) + 1e-6
    assert static_supply_capacity(0.01, 0.0, 0.0, rate_at_target) == 0.0
    assert math.isinf(static_supply_capacity(0.0, 0.0, 0.0, rate_at_target))
    assert static_supply_capacity(above_current, 100.0, 90.0, rate_at_target) == 0.0
    assert static_supply_capacity(above_max, 100.0, 90.0, rate_at_target) == 0.0
    target = supply_apy_from_curve(0.8, rate_at_target)
    assert math.isinf(static_supply_capacity(target, 100.0, 90.0, rate_at_target, alpha=0.1, beta=1.0))
    assert math.isinf(static_supply_capacity(target, 100.0, 90.0, rate_at_target, alpha=0.4, beta=0.7))


def test_horizon_average_equals_instant_apy_when_controller_drift_is_zero():
    rate_at_target, fee = 0.07, 0.08
    elapsed = np.full(48, 1.0 / (365.0 * 24.0))
    average = horizon_average_supply_apy(TARGET_UTILIZATION, rate_at_target, elapsed, fee)
    instant = supply_apy_from_curve(TARGET_UTILIZATION, rate_at_target, fee)
    assert average == pytest.approx(instant, rel=0.0, abs=1e-15)


@pytest.mark.parametrize(
    "utilization,bound",
    [(1.0, MAX_RATE_AT_TARGET), (0.5, MIN_RATE_AT_TARGET)],
)
def test_horizon_average_clamps_controller_rate_over_long_steps(utilization, bound):
    average = horizon_average_supply_apy(utilization, 0.05, 100.0)
    assert average == pytest.approx(supply_apy_from_curve(utilization, bound))


def test_horizon_capacity_accepts_a_two_sided_response_path_and_is_monotone_in_target():
    elapsed = np.full(72, 1.0 / (365.0 * 24.0))
    alpha = np.linspace(0.0, 0.15, elapsed.size)
    beta = np.linspace(0.0, 0.1, elapsed.size)
    low_target = supply_apy_from_curve(0.72, 0.06)
    high_target = supply_apy_from_curve(0.78, 0.06)
    low = horizon_supply_capacity(
        low_target, 100.0, 88.0, 0.06, elapsed, alpha=alpha, beta=beta, tolerance=1e-10
    )
    high = horizon_supply_capacity(
        high_target, 100.0, 88.0, 0.06, elapsed, alpha=alpha, beta=beta, tolerance=1e-10
    )
    higher_alpha = horizon_supply_capacity(
        high_target, 100.0, 88.0, 0.06, elapsed, alpha=alpha + 0.05, beta=beta, tolerance=1e-10
    )
    higher_beta = horizon_supply_capacity(
        high_target, 100.0, 88.0, 0.06, elapsed, alpha=alpha, beta=beta + 0.05, tolerance=1e-10
    )
    assert low >= high >= 0.0
    assert higher_alpha >= high
    assert higher_beta >= high


@pytest.mark.parametrize(
    "supply,borrow,target_utilization,comparison",
    [
        (100.0, 95.0, 0.94, "above"),
        (100.0, 80.0, 0.75, "below"),
    ],
)
def test_horizon_capacity_moves_with_controller_drift_direction(supply, borrow, target_utilization, comparison):
    rate_at_target = 0.05
    target_apy = supply_apy_from_curve(target_utilization, rate_at_target)
    static = static_supply_capacity(target_apy, supply, borrow, rate_at_target)
    elapsed = np.full(7 * 24, 1.0 / (365.0 * 24.0))
    horizon = horizon_supply_capacity(
        target_apy, supply, borrow, rate_at_target, elapsed, tolerance=1e-10
    )
    if comparison == "above":
        assert horizon >= static
    else:
        assert horizon <= static


def _reference_horizon_apy(utilization_path, rate_at_target, elapsed, fee, reference_path=None):
    """Independent loop: log-space drift clamp, drift before accrual, geometric annualization."""
    def error(u):
        return (u - 0.9) / 0.1 if u > 0.9 else (u - 0.9) / 0.9

    low_log = math.log(MIN_RATE_AT_TARGET / rate_at_target)
    high_log = math.log(MAX_RATE_AT_TARGET / rate_at_target)
    log_drift, log_growth = 0.0, 0.0
    for k, (u, dt) in enumerate(zip(utilization_path, elapsed, strict=True)):
        offset = 0.0 if reference_path is None else error(reference_path[k])
        log_drift = min(max(log_drift + 50.0 * (error(u) - offset) * dt, low_log), high_log)
        log_growth += dt * math.log1p(supply_apy_from_curve(u, rate_at_target * math.exp(log_drift), fee))
    return math.expm1(log_growth / sum(elapsed))


@pytest.mark.parametrize("use_reference", [False, True])
def test_horizon_average_matches_independent_loop_on_a_varying_path(use_reference):
    path = np.linspace(0.97, 0.80, 240)
    elapsed = np.where(np.arange(240) % 3 == 0, 2.0, 1.0) / (365.0 * 24.0)
    reference = np.full(240, 0.93) if use_reference else None
    got = horizon_average_supply_apy(path, 0.06, elapsed, 0.1, drift_reference_utilization=reference)
    expected = _reference_horizon_apy(path, 0.06, elapsed, 0.1, reference)
    assert got == pytest.approx(expected, rel=1e-12)


def test_drift_reference_at_the_path_itself_freezes_rate_at_target():
    elapsed = np.full(24 * 30, 1.0 / (365.0 * 24.0))
    u = 0.97
    frozen = horizon_average_supply_apy(u, 0.05, elapsed, 0.05, drift_reference_utilization=u)
    drifting = horizon_average_supply_apy(u, 0.05, elapsed, 0.05)
    assert frozen == pytest.approx(supply_apy_from_curve(u, 0.05, 0.05), rel=1e-12)
    assert drifting > frozen * 1.5


def test_horizon_capacity_is_the_boundary_of_the_horizon_apy():
    elapsed = np.full(7 * 24, 1.0 / (365.0 * 24.0))
    supply, borrow, rate_at_target = 100.0, 93.0, 0.05
    target = supply_apy_from_curve(0.91, rate_at_target)
    capacity = horizon_supply_capacity(target, supply, borrow, rate_at_target, elapsed, tolerance=1e-12)
    assert 0.0 < capacity < math.inf

    def horizon(deposit):
        return horizon_average_supply_apy(borrow / (supply + deposit), rate_at_target, elapsed)

    assert horizon(capacity) >= target
    assert horizon(capacity * (1.0 + 1e-9)) < target


def test_horizon_drift_bases_bracket_the_static_capacity_above_target():
    elapsed = np.full(14 * 24, 1.0 / (365.0 * 24.0))
    supply, borrow, rate_at_target = 100.0, 97.0, 0.05
    target = supply_apy_from_curve(0.93, rate_at_target)
    static = static_supply_capacity(target, supply, borrow, rate_at_target, alpha=0.1, beta=0.05)
    by_basis = {
        basis: horizon_supply_capacity(
            target, supply, borrow, rate_at_target, elapsed, alpha=0.1, beta=0.05, tolerance=1e-12,
            drift_basis=basis,
        )
        for basis in ("absolute", "deposit", "frozen")
    }
    assert by_basis["frozen"] == pytest.approx(static, rel=1e-9)
    assert by_basis["deposit"] < static < by_basis["absolute"]


def test_horizon_deposit_basis_earns_the_instant_apy_at_zero_deposit():
    elapsed = np.full(48, 1.0 / (365.0 * 24.0))
    supply, borrow, rate_at_target = 100.0, 80.0, 0.05
    instant = supply_apy_from_curve(0.8, rate_at_target)
    # just below the instant APY: absolute drift (u < target, rate falls) misses it, the deposit basis clears it
    target = instant * (1.0 - 1e-3)
    assert horizon_supply_capacity(target, supply, borrow, rate_at_target, elapsed) == 0.0
    assert horizon_supply_capacity(target, supply, borrow, rate_at_target, elapsed, drift_basis="deposit") > 0.0


def test_static_capacity_accepts_crowding_in_beta():
    args = {"supply": 100.0, "borrow": 90.0, "rate_at_target": 0.06, "fee": 0.1, "alpha": 0.1, "beta": -0.5}
    target_apy = supply_apy_from_curve(0.8, args["rate_at_target"], args["fee"])
    capacity = static_supply_capacity(target_apy=target_apy, **args)
    assert capacity == pytest.approx(_numeric_static_capacity(target_apy=target_apy, **args), rel=1e-10)
    assert capacity < static_supply_capacity(target_apy, 100.0, 90.0, 0.06, 0.1, alpha=0.1)
    utilization = utilization_after_deposit(100.0, 90.0, capacity, 0.1, -0.5)
    assert supply_apy_from_curve(utilization, 0.06, 0.1) == pytest.approx(target_apy, rel=1e-9)


def test_capacity_is_zero_below_target_even_when_the_response_lifts_utilization():
    supply, borrow, rate_at_target, alpha, beta = 100.0, 80.0, 0.05, 0.9, 0.1
    target = supply_apy_from_curve(0.85, rate_at_target)
    large = utilization_after_deposit(supply, borrow, 1e6, alpha, beta)
    assert supply_apy_from_curve(large, rate_at_target) > target  # documented: still reported as 0
    assert static_supply_capacity(target, supply, borrow, rate_at_target, alpha=alpha, beta=beta) == 0.0
    elapsed = np.full(24, 1.0 / (365.0 * 24.0))
    assert horizon_supply_capacity(
        target, supply, borrow, rate_at_target, elapsed, alpha=alpha, beta=beta, drift_basis="frozen"
    ) == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"elapsed_years_per_step": np.array([])}, "must not be empty"),
        ({"elapsed_years_per_step": np.array([1.0, -0.5])}, ">= 0"),
        ({"elapsed_years_per_step": np.zeros(3)}, "positive total"),
        ({"alpha": 1.1}, "alpha"),
        ({"beta": 1.1}, "beta"),
        ({"alpha": math.nan}, "finite"),
        ({"drift_basis": "relative"}, "drift_basis"),
        ({"tolerance": 0.0}, "tolerance"),
    ],
)
def test_horizon_capacity_rejects_invalid_inputs(kwargs, match):
    call = {"target_apy": 0.03, "supply": 100.0, "borrow": 90.0, "rate_at_target": 0.05,
            "elapsed_years_per_step": np.full(3, 1.0 / 8760.0)}
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        horizon_supply_capacity(**call)


def test_utilization_helpers_reject_out_of_range_inputs():
    with pytest.raises(ValueError, match="beta"):
        utilization_after_deposit(100.0, 90.0, 10.0, beta=1.5)
    with pytest.raises(ValueError, match=">= 0"):
        utilization_after_deposit(-1.0, 90.0, 10.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        horizon_average_supply_apy(0.9, 0.05, 1.0, drift_reference_utilization=1.2)
    assert utilization_after_deposit(0.0, 0.0, 10.0, beta=1.0) == 0.0
    assert utilization_after_deposit(100.0, 90.0, 10.0, alpha=1.0, beta=1.0) == 1.0


def test_monotone_response_matches_the_sign_of_the_utilization_slope():
    supply, borrow = 100.0, 80.0
    u = borrow / supply
    cases = [(0.0, 0.0), (0.8, 0.0), (0.81, 0.0), (0.5, 0.3), (0.6, 0.3), (0.0, -0.5), (0.1, 1.0)]
    for alpha, beta in cases:
        rising = utilization_after_deposit(supply, borrow, 1e3, alpha, beta) > u + 1e-12
        assert monotone_response(u, alpha, beta) is (not rising)
    path_alpha = np.array([0.0, 0.2, 0.7])
    assert monotone_response(u, path_alpha, 0.1)
    assert not monotone_response(u, np.r_[path_alpha, 0.75], 0.1)
    with pytest.raises(ValueError):
        monotone_response(1.2)
    with pytest.raises(ValueError):
        monotone_response(0.5, alpha=1.5)


def test_monotone_response_makes_horizon_apy_non_increasing_in_the_deposit():
    elapsed = np.full(72, 1.0 / (365.0 * 24.0))
    alpha, beta = np.linspace(0.0, 0.5, 72), np.linspace(0.0, 0.3, 72)
    assert monotone_response(0.9, alpha, beta)
    deposits = np.geomspace(1e-2, 1e4, 60)
    apys = [horizon_average_supply_apy(utilization_after_deposit(100.0, 90.0, p, alpha, beta), 0.05, elapsed)
            for p in deposits]
    assert np.all(np.diff(apys) <= 1e-15)


def test_rate_at_target_transition_years():
    assert rate_at_target_transition_years(1.0) == pytest.approx(1.0 / 50.0)
    assert rate_at_target_transition_years(-0.5) == pytest.approx(1.0 / 25.0)
    assert math.isinf(rate_at_target_transition_years(0.0))
    years = rate_at_target_transition_years(0.4)
    assert math.log(rate_at_target_drift(0.9 + 0.4 * 0.1, years)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        rate_at_target_transition_years(math.nan)
