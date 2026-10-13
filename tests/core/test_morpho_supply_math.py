"""Synthetic unit tests for shared Morpho supply and controller calculations."""
import math

import pytest

from fractal.core.entities.models.morpho_math import (
    adaptive_curve_error,
    from_wad,
    rate_at_target_drift,
    supply_apy_from_curve,
    utilization_for_supply_apy,
)

pytestmark = pytest.mark.core


@pytest.mark.parametrize("utilization,error", [(0.0, -1.0), (0.9, 0.0), (1.0, 1.0)])
def test_controller_error_anchors(utilization, error):
    """The controller error is normalized separately on either side of the target."""
    assert adaptive_curve_error(utilization) == pytest.approx(error)


def test_controller_drift_and_invalid_inputs():
    """At target there is no drift; above target the rate increases exponentially."""
    assert rate_at_target_drift(0.9, 1.0) == pytest.approx(1.0)
    assert rate_at_target_drift(1.0, math.log(2.0) / 50.0) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="utilization"):
        adaptive_curve_error(1.01)
    with pytest.raises(ValueError, match="elapsed_years"):
        rate_at_target_drift(0.9, -1.0)


@pytest.mark.parametrize("utilization", [0.1, 0.8, 0.9, 0.99])
def test_supply_curve_inverse_and_fee_units(utilization):
    """Curve inversion recovers utilization on both sides of the kink."""
    rate, fee = 0.05, 0.1
    target = supply_apy_from_curve(utilization, rate, fee)
    assert target == pytest.approx(supply_apy_from_curve(utilization, rate) * (1.0 - fee))
    assert utilization_for_supply_apy(target, rate, fee) == pytest.approx(utilization, abs=1e-10)


def test_supply_curve_unreachable_zero_target_and_wad():
    """Unreachable targets are explicit; WAD input is always scaled, including strings."""
    assert utilization_for_supply_apy(1.0, 0.01) is None
    assert utilization_for_supply_apy(0.0, 0.01) == 0.0
    assert from_wad("915000000000000000") == pytest.approx(0.915)
    assert from_wad(1) == pytest.approx(1e-18, abs=1e-30)
    with pytest.raises(ValueError, match="fee"):
        supply_apy_from_curve(0.9, 0.05, fee=1.0)
