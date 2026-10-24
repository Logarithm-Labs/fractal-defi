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
