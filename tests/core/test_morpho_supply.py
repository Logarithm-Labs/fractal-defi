"""Synthetic supply-side invariants: liquidity gates, accrual, costs and own-position impact."""
import copy
import math
import random

import numpy as np
import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.models.morpho_math import MIN_RATE_AT_TARGET, adaptive_curve_borrow_apr
from fractal.core.entities.protocols.morpho_supply import (
    MorphoSupplyEntity,
    MorphoSupplyException,
    MorphoSupplyGlobalState,
)

pytestmark = pytest.mark.core

SUPPLY_USD = 16_000_000.0
BORROW_USD = 15_000_000.0
LIQUIDITY_USD = SUPPLY_USD - BORROW_USD
HISTORICAL_UTILIZATION = BORROW_USD / SUPPLY_USD
RATE_AT_TARGET_APY = 0.07


def _curve_apy(utilization: float, rate_at_target_apy: float = RATE_AT_TARGET_APY, fee: float = 0.0) -> float:
    """Supply APY the IRM curve implies, converted rate -> APY (as the validator does)."""
    borrow_apy = math.expm1(adaptive_curve_borrow_apr(utilization, math.log1p(rate_at_target_apy)))
    return borrow_apy * utilization * (1.0 - fee)


def _state(**overrides) -> MorphoSupplyGlobalState:
    base = {
        "utilization": HISTORICAL_UTILIZATION,
        "supply_rate": 0.0,
        "supply_apy": 0.08,
        "borrow_apy": 0.09,
        "rate_at_target": RATE_AT_TARGET_APY,
        "fee": 0.0,
        "supply_assets_usd": SUPPLY_USD,
        "borrow_assets_usd": BORROW_USD,
        "liquidity_assets_usd": LIQUIDITY_USD,
    }
    base.update(overrides)
    return MorphoSupplyGlobalState(**base)


def _entity(**kwargs) -> MorphoSupplyEntity:
    entity = MorphoSupplyEntity(**kwargs)
    entity.update_state(_state())
    return entity


def test_own_impact_accrual_is_anchored_to_the_observed_rate():
    """E1: at P = 0 the credited APY is exactly the observed one; dilution still bites."""
    state = _state(utilization=0.99, supply_apy=0.30, rate_at_target=0.5,
                   borrow_assets_usd=15_840_000.0, liquidity_assets_usd=160_000.0)
    entity = MorphoSupplyEntity(own_impact=True)
    entity.update_state(state)
    assert entity.projected_supply_apy(0.0, state) == pytest.approx(0.30)     # the observation, not the curve
    diluted = entity.projected_supply_apy(1_000_000.0, state)
    assert 0.0 <= diluted < 0.30
    assert entity.projected_supply_apy(4_000_000.0, state) < diluted          # self-dilution still bites


def test_capacity_at_inverts_the_anchored_projection_on_random_states():
    """E2: after anchoring, the capacity inversion round-trips on random synthetic states."""
    rng = random.Random(20261019)
    for _ in range(200):
        supply = rng.uniform(1e6, 200e6)
        utilization = rng.uniform(0.05, 0.99)
        state = _state(
            utilization=utilization,
            supply_assets_usd=supply,
            borrow_assets_usd=supply * utilization,
            liquidity_assets_usd=supply * (1.0 - utilization),
            rate_at_target=rng.uniform(0.01, 0.5),
            supply_apy=rng.uniform(0.001, 0.5),     # independent of the curve: the observation is the level
            fee=rng.uniform(0.0, 0.1),
        )
        entity = MorphoSupplyEntity(own_impact=True, absorption=rng.uniform(0.0, utilization))
        entity.update_state(state)
        position = rng.uniform(1.0, 0.2 * supply)
        apy = entity.projected_supply_apy(position, state)
        assert apy > 0.0
        assert entity.capacity_at(apy, state) == pytest.approx(position, rel=1e-6)


def test_drift_is_clamped_to_the_rate_bounds_after_a_long_hold():
    """E3: 120 days of own-position drift cannot wind R* past its on-chain floor."""
    entity = MorphoSupplyEntity(own_impact=True, irm_feedback=True)
    held = _state(utilization=0.9, borrow_assets_usd=14_400_000.0, liquidity_assets_usd=1_600_000.0)
    entity.update_state(held)
    entity.action_deposit(16_000_000.0)                     # P = S: utilization halves under our supply
    for _ in range(120 * 24):                                # 120 days of hourly bars
        entity.update_state(held)
    assert entity.curve_rate_at_target() >= MIN_RATE_AT_TARGET
    entity.action_withdraw(entity.balance)                   # exit: the drift source is gone
    doubled = _state(utilization=0.9, rate_at_target=0.14, borrow_assets_usd=14_400_000.0,
                     liquidity_assets_usd=1_600_000.0)
    entity.update_state(doubled)
    assert entity.curve_rate_at_target() > MIN_RATE_AT_TARGET    # rose with the market, not locked at the floor


def test_liquidity_capacity_branches():
    """E4: the entry-gate capacity across modes and multiples."""
    exogenous = _entity(own_impact=False)
    assert exogenous.liquidity_capacity(0.0) == float("inf")
    assert exogenous.liquidity_capacity(1.0) == pytest.approx(LIQUIDITY_USD)
    absorbing = _entity(own_impact=True, absorption=1.0)
    unabsorbed = 1.0 - HISTORICAL_UTILIZATION
    assert absorbing.liquidity_capacity(0.5 * unabsorbed) == float("inf")          # multiple <= unabsorbed
    assert absorbing.liquidity_capacity(0.5) == pytest.approx(LIQUIDITY_USD / (0.5 - unabsorbed))
    with pytest.raises(MorphoSupplyException, match="liquidity multiple"):
        absorbing.liquidity_capacity(-1.0)


def test_numpy_scalar_fields_are_accepted_by_update_state():
    """E5: an ``np.int64`` market size is a real number, not a rejected type."""
    entity = MorphoSupplyEntity()
    entity.update_state(_state(supply_assets_usd=np.int64(16_000_000)))
    assert entity.global_state.supply_assets_usd == 16_000_000


def test_take_curve_is_the_capacity_solution_capped_by_cap():
    """The vectorised take curve and ``capacity_at`` invert the same curve."""
    entity = _entity(own_impact=True)
    state = _state()
    lambdas = np.linspace(0.0, 0.3, 61)
    cap = 2_000_000.0
    take = entity.take_curve(lambdas, state, cap)
    assert np.all(take >= 0.0) and np.all(take <= cap * (1.0 + 1e-9))
    assert np.all(np.diff(take) <= 1e-6)                     # non-increasing in the required rate
    for lam, position in zip(lambdas, take, strict=True):
        if 1.0 < position < cap - 1.0:                       # interior: the same solution as the scalar inverse
            assert entity.projected_supply_apy(float(position), state) == pytest.approx(lam, rel=2e-2)
    assert take[0] == pytest.approx(min(cap, entity.capacity_at(0.0, state)), rel=1e-3)
    assert entity.take_curve(np.array([0.5]), state, cap)[0] == 0.0    # unreachable even at u = 1


def test_take_curve_absorption_and_exogenous_branches():
    """Full absorption takes the cap up to the historical rate; exogenous is all-or-nothing."""
    absorbing = _entity(own_impact=True, absorption=1.0)
    state = _state()
    lambdas = np.linspace(0.0, 0.2, 21)
    take = absorbing.take_curve(lambdas, state, 1e6)
    at_hist = state.supply_apy                         # the observed level anchors the historical utilization
    assert np.all(take[lambdas <= at_hist] == 1e6)     # u* below alpha for every reachable rate: take the cap
    assert np.all(take[lambdas > at_hist] == 0.0)      # supplying can never raise utilization above u_hist
    exogenous = _entity(own_impact=False)
    take = exogenous.take_curve(lambdas, state, 1e6)
    assert np.all(take[lambdas <= 0.08] == 1e6) and np.all(take[lambdas > 0.08] == 0.0)  # headline: all-or-nothing


# ------------------------------------------------------------------- basics
def test_initial_state_is_empty_and_supply_only():
    entity = MorphoSupplyEntity()
    assert entity.balance == 0.0
    assert entity.internal_state.supplied == 0.0
    assert sorted(entity.get_available_actions()) == ["deposit", "withdraw"]  # never borrows


def test_balance_is_the_supplied_notional():
    entity = _entity(own_impact=False)
    entity.action_deposit(1_000.0)
    assert entity.balance == pytest.approx(1_000.0)


def test_position_state_is_internal_not_global():
    """No collateral/debt fields: this entity is deliberately not a BaseLendingEntity."""
    assert not hasattr(_entity(), "ltv")
    assert not hasattr(_entity(), "borrowed")


# --------------------------------------------------------------------- gate
def test_withdraw_above_withdrawable_raises():
    entity = _entity(own_impact=False)
    entity.action_deposit(2_000_000.0)
    assert entity.withdrawable == pytest.approx(LIQUIDITY_USD)
    with pytest.raises(MorphoSupplyException, match="exceeds withdrawable"):
        entity.action_withdraw(2_000_000.0)
    assert entity.balance == pytest.approx(2_000_000.0)  # unchanged by the failed call


def test_gate_is_closed_at_full_utilization():
    entity = MorphoSupplyEntity(own_impact=False)
    entity.update_state(_state(utilization=1.0, liquidity_assets_usd=0.0))
    entity.action_deposit(50_000.0)
    assert entity.withdrawable == 0.0
    with pytest.raises(MorphoSupplyException, match="exceeds withdrawable"):
        entity.action_withdraw(1.0)
    assert entity.balance == pytest.approx(50_000.0)


def test_withdraw_above_position_raises_even_when_market_is_liquid():
    entity = _entity(own_impact=False)
    entity.action_deposit(1_000.0)
    with pytest.raises(MorphoSupplyException, match="exceeds position"):
        entity.action_withdraw(1_001.0)


def test_round_trip_withdraws_exactly_the_deposit_when_the_gate_is_open():
    entity = _entity(own_impact=False)
    entity.action_deposit(500_000.0)
    entity.action_withdraw(500_000.0)
    assert entity.balance == pytest.approx(0.0)


def test_negative_amounts_are_rejected():
    entity = _entity(own_impact=False)
    with pytest.raises(MorphoSupplyException, match="deposit amount must be >= 0"):
        entity.action_deposit(-1.0)
    with pytest.raises(MorphoSupplyException, match="withdraw amount must be >= 0"):
        entity.action_withdraw(-1.0)


# ------------------------------------------------------------------ accrual
def test_update_state_accrues_the_per_bar_supply_rate():
    entity = _entity(own_impact=False)
    entity.action_deposit(1_000.0)
    entity.update_state(_state(supply_rate=0.01))
    assert entity.balance == pytest.approx(1_010.0)


def test_exogenous_mode_reproduces_the_historical_rate_and_gate():
    """``own_impact=False`` must not invent a rate: it replays the loader's."""
    entity = _entity(own_impact=False)
    entity.action_deposit(1_000.0)
    entity.update_state(_state(supply_rate=0.002))
    assert entity.effective_supply_rate == pytest.approx(0.002)
    assert entity.withdrawable == pytest.approx(LIQUIDITY_USD)
    assert entity.effective_utilization == pytest.approx(HISTORICAL_UTILIZATION)


# --------------------------------------------------------------- own impact
def test_own_impact_preserves_borrowing_and_adds_supply():
    """Adding supply with fixed borrowing reduces utilization and increases liquidity."""
    entity = _entity(own_impact=True, absorption=0.0)
    entity.action_deposit(1_000_000.0)
    assert entity.effective_utilization == pytest.approx(BORROW_USD / (SUPPLY_USD + 1_000_000.0))
    assert entity.withdrawable == pytest.approx(LIQUIDITY_USD + 1_000_000.0)


def test_own_impact_self_dilutes_the_earned_rate():
    """The position lowers the rate it earns through self-dilution."""
    entity = _entity(own_impact=True, absorption=0.0)
    entity.action_deposit(1_000_000.0)
    observed = entity.global_state.supply_apy
    assert 0.0 < entity.effective_supply_apy < observed
    expected = observed * _curve_apy(entity.effective_utilization) / _curve_apy(HISTORICAL_UTILIZATION)
    assert entity.effective_supply_apy == pytest.approx(expected, rel=1e-5)
    # The synthetic example spans the utilization target: dilution is material.
    assert observed - entity.effective_supply_apy > 0.02


def test_absorption_of_u_preserves_utilization():
    entity = _entity(own_impact=True, absorption=HISTORICAL_UTILIZATION)
    entity.action_deposit(1_000_000.0)
    assert entity.effective_utilization == pytest.approx(HISTORICAL_UTILIZATION, abs=5e-5)


def test_absorption_is_capped_at_utilization_so_size_never_raises_the_rate():
    """``absorption = 1`` means "no dilution", never "our deposit pushes u above history"."""
    full, at_u = _entity(own_impact=True, absorption=1.0), _entity(own_impact=True, absorption=HISTORICAL_UTILIZATION)
    for entity in (full, at_u):
        entity.action_deposit(50_000_000.0)                # several times the market
    assert full.effective_utilization == pytest.approx(HISTORICAL_UTILIZATION, abs=1e-9)
    assert full.effective_supply_apy == pytest.approx(at_u.effective_supply_apy, rel=1e-9)
    # withdrawable grows only by the un-borrowed share of our supply
    assert full.withdrawable == pytest.approx(LIQUIDITY_USD + (1 - HISTORICAL_UTILIZATION) * 50_000_000.0, rel=1e-9)
    # and a bigger position never earns more than the undiluted curve rate
    assert full.projected_supply_apy(1e9) <= full.projected_supply_apy(0.0) * (1 + 1e-9)


def test_larger_positions_dilute_more():
    small, large = _entity(own_impact=True), _entity(own_impact=True)
    small.action_deposit(0.01 * SUPPLY_USD)
    large.action_deposit(0.5 * SUPPLY_USD)
    assert large.effective_supply_apy < small.effective_supply_apy


# -------------------------------------------------------------------- costs
def test_costs_are_charged_on_the_moved_notional_and_recorded():
    recorded = []
    entity = MorphoSupplyEntity(own_impact=False, cost_bps_per_leg=10.0)
    entity.attach_execution_recorder(lambda action, notional, fee: recorded.append((action, notional, fee)))
    entity.update_state(_state())
    entity.action_deposit(100_000.0)
    assert entity.balance == pytest.approx(100_000.0 * (1 - 0.001))
    entity.action_withdraw(50_000.0)
    assert recorded == [("deposit", 100_000.0, 100.0), ("withdraw", 50_000.0, 50.0)]


def test_zero_cost_still_records_the_leg():
    recorded = []
    entity = MorphoSupplyEntity(own_impact=False)
    entity.attach_execution_recorder(lambda action, notional, fee: recorded.append((action, notional, fee)))
    entity.update_state(_state())
    entity.action_deposit(1_000.0)
    assert recorded == [("deposit", 1_000.0, 0.0)]


# ------------------------------------------------------------- declarations
def test_update_state_does_not_mutate_the_passed_state():
    entity = _entity()
    state = _state(supply_rate=0.003)
    snapshot = copy.deepcopy(state)
    entity.update_state(state)
    assert state == snapshot and state == _state(supply_rate=0.003)


def test_invalid_arguments_and_states_are_rejected():
    with pytest.raises(MorphoSupplyException, match="absorption"):
        MorphoSupplyEntity(absorption=1.5)
    with pytest.raises(MorphoSupplyException, match="cost_bps_per_leg"):
        MorphoSupplyEntity(cost_bps_per_leg=-1.0)
    with pytest.raises(MorphoSupplyException, match="bars_per_year"):
        MorphoSupplyEntity(bars_per_year=0.0)
    entity = MorphoSupplyEntity(own_impact=False)
    with pytest.raises(MorphoSupplyException, match="fee must be in"):
        entity.update_state(_state(fee=0.5))
    with pytest.raises(MorphoSupplyException, match="rate_at_target"):
        entity.update_state(_state(rate_at_target=-0.01))
    with pytest.raises(MorphoSupplyException, match="supply_rate must be >= -1"):
        entity.update_state(_state(supply_rate=-1.5))


@pytest.mark.parametrize(
    "kwargs", [
        {"cost_bps_per_leg": float("nan")},
        {"cost_bps_per_leg": float("inf")},
        {"gas_usd_per_leg": float("nan")},
        {"gas_usd_per_leg": float("inf")},
        {"bars_per_year": float("nan")},
        {"bars_per_year": float("inf")},
    ],
)
def test_non_finite_cost_and_frequency_configuration_is_rejected(kwargs):
    with pytest.raises(MorphoSupplyException, match="finite"):
        MorphoSupplyEntity(**kwargs)


@pytest.mark.parametrize("action", ["deposit", "withdraw"])
@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_action_amount_is_rejected_without_mutation(action, amount):
    recorded = []
    entity = _entity(own_impact=False)
    entity.attach_execution_recorder(lambda *args: recorded.append(args))
    entity.action_deposit(100.0)
    recorded.clear()
    before = entity.balance
    with pytest.raises(MorphoSupplyException, match="finite"):
        getattr(entity, f"action_{action}")(amount)
    assert entity.balance == before
    assert not recorded


def test_entity_exception_is_an_entity_exception():
    assert issubclass(MorphoSupplyException, EntityException)


def test_bars_per_year_matches_the_hourly_loader_convention():
    """A 5% supply APY must round-trip through the per-bar rate at 8760 bars/year."""
    entity = MorphoSupplyEntity(own_impact=False, bars_per_year=8760.0)
    state = _state(supply_rate=math.log1p(0.05) / 8760.0)
    assert math.expm1(math.log1p(entity._effective_supply_rate(state, 0.0)) * 8760.0) == pytest.approx(0.05, rel=1e-5)


# ------------------------------------------------------- review fixes (PR #116)
@pytest.mark.parametrize("field", ["supply_assets_usd", "borrow_assets_usd", "liquidity_assets_usd",
                                   "supply_rate", "rate_at_target", "utilization", "fee"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_state_is_rejected_not_silently_accepted(field, bad):
    """A NaN size used to read as an open gate (``x > nan`` is False). Now it raises."""
    for own_impact in (False, True):
        entity = _entity(own_impact=own_impact)
        with pytest.raises(MorphoSupplyException, match=f"{field} must be finite"):
            entity.update_state(_state(**{field: bad}))


def test_nan_sizes_cannot_open_the_gate():
    """Regression for the review finding: exogenous mode, NaN liquidity, full withdraw was accepted."""
    entity = _entity(own_impact=False)
    entity.action_deposit(1_000_000.0)
    with pytest.raises(MorphoSupplyException, match="must be finite"):
        entity.update_state(_state(utilization=0.99, liquidity_assets_usd=float("nan"),
                                   supply_assets_usd=float("nan"), borrow_assets_usd=float("nan")))
    # the state did not change, so the gate still reflects the last finite snapshot
    assert entity.withdrawable == pytest.approx(LIQUIDITY_USD)


def test_withdraw_fee_leaves_the_position_so_nav_drop_equals_recorded_fees():
    """``amount`` is what the destination receives; ``amount + fee`` leaves the position."""
    recorded = []
    entity = MorphoSupplyEntity(own_impact=False, cost_bps_per_leg=10.0)
    entity.attach_execution_recorder(lambda action, notional, fee: recorded.append(fee))
    entity.update_state(_state())
    entity.action_deposit(1_000_000.0)
    received = entity.max_withdrawable_notional
    entity.action_withdraw(received)
    assert entity.balance == pytest.approx(0.0, abs=1e-6)
    nav_drop = 1_000_000.0 - received
    assert nav_drop == pytest.approx(sum(recorded), rel=1e-9)
    assert received == pytest.approx(1_000_000.0 * (1 - 0.001) / 1.001)


def test_max_withdrawable_notional_is_the_largest_accepted_amount():
    entity = _entity(own_impact=False, cost_bps_per_leg=25.0)
    entity.action_deposit(2_000_000.0)                     # more than the 1.15M of liquidity
    cap = entity.max_withdrawable_notional
    assert cap == pytest.approx(LIQUIDITY_USD / 1.0025)
    with pytest.raises(MorphoSupplyException, match="exceeds withdrawable"):
        entity.action_withdraw(cap * 1.001)
    entity.action_withdraw(cap)                           # exactly the cap passes
    assert entity.balance == pytest.approx(2_000_000.0 * (1 - 0.0025) - LIQUIDITY_USD, rel=1e-9)


def test_withdraw_plus_fee_must_fit_inside_the_position():
    entity = _entity(own_impact=False, cost_bps_per_leg=10.0)
    entity.action_deposit(1_000.0)                         # supplied = 999
    with pytest.raises(MorphoSupplyException, match="exceeds position"):
        entity.action_withdraw(999.0)                      # 999 + 0.999 > 999


def test_projected_supply_apy_is_the_whole_position_figure():
    entity = _entity(own_impact=True, absorption=0.0)
    assert entity.projected_supply_apy(0.0) == pytest.approx(entity.global_state.supply_apy)
    entity.action_deposit(1_000_000.0)
    assert entity.projected_supply_apy(1_000_000.0) == pytest.approx(entity.effective_supply_apy, rel=1e-9)
    # monotone: more of our capital, lower rate for all of it
    sizes = [0.0, 1e5, 1e6, 1e7]
    projected = [entity.projected_supply_apy(size) for size in sizes]
    assert projected == sorted(projected, reverse=True)
    with pytest.raises(MorphoSupplyException, match="position_notional"):
        entity.projected_supply_apy(-1.0)


def test_projected_supply_apy_is_the_headline_without_own_impact():
    entity = _entity(own_impact=False)
    assert entity.projected_supply_apy(1e9) == pytest.approx(0.08)


def test_utilization_outside_unit_interval_is_rejected():
    entity = _entity()
    with pytest.raises(MorphoSupplyException, match="utilization must be in"):
        entity.update_state(_state(utilization=1.2))


# ------------------------------------------------------------ sizing helpers
def test_capacity_at_inverts_projected_supply_apy():
    """``projected_supply_apy(capacity_at(y)) == y`` for every absorption below u*."""
    for absorption in (0.0, 0.3):
        entity = _entity(own_impact=True, absorption=absorption)
        for target in (0.03, 0.05, 0.07):
            size = entity.capacity_at(target)
            assert size > 0.0 and math.isfinite(size)
            assert entity.projected_supply_apy(size) == pytest.approx(target, rel=1e-6)
            # a bigger position earns less than the target, a smaller one more
            assert entity.projected_supply_apy(size * 1.01) < target < entity.projected_supply_apy(size * 0.99)


def test_capacity_at_edge_cases():
    entity = _entity(own_impact=True, absorption=0.0)
    undiluted = entity.projected_supply_apy(0.0)
    assert entity.capacity_at(undiluted * 1.01) == 0.0          # market already pays less than the target
    assert entity.capacity_at(10.0) == 0.0                      # unreachable even at u = 1
    assert entity.capacity_at(0.0) == float("inf")              # any size clears a zero target
    # absorption at or above u*: utilization never falls to u*, capacity is unbounded
    saturated = _entity(own_impact=True, absorption=1.0)
    assert saturated.capacity_at(undiluted * 0.5) == float("inf")
    # exogenous mode: headline decides, all or nothing
    exo = _entity(own_impact=False)
    assert exo.capacity_at(0.05) == float("inf") and exo.capacity_at(0.10) == 0.0
    with pytest.raises(MorphoSupplyException, match="target_apy"):
        entity.capacity_at(-0.01)


def test_projected_apy_and_capacity_accept_an_explicit_snapshot():
    entity = _entity(own_impact=True)
    other = _state(supply_assets_usd=2 * SUPPLY_USD, borrow_assets_usd=2 * BORROW_USD,
                   liquidity_assets_usd=2 * LIQUIDITY_USD)
    # same utilization, twice the depth: a given position dilutes half as much
    assert entity.projected_supply_apy(1e6, other) > entity.projected_supply_apy(1e6)
    assert entity.capacity_at(0.05, other) == pytest.approx(2 * entity.capacity_at(0.05), rel=1e-9)
    assert entity.global_state == _state()                      # evaluating on a snapshot does not apply it


# -------------------------------------------------------------- irm feedback
def test_irm_feedback_requires_own_impact_and_starts_at_zero_drift():
    with pytest.raises(MorphoSupplyException, match="irm_feedback requires own_impact"):
        MorphoSupplyEntity(own_impact=False, irm_feedback=True)
    entity = _entity(own_impact=True, irm_feedback=True)
    assert entity.internal_state.rstar_log_drift == 0.0
    assert entity.effective_rate_at_target == pytest.approx(RATE_AT_TARGET_APY)


def test_irm_feedback_halves_rate_at_target_in_about_ten_days_at_u_045_vs_target():
    """The controller differential halves R* after ln(2) / 25 years at utilization 0.45."""
    entity = _entity(own_impact=True, absorption=0.0, irm_feedback=True)
    supply, borrow = 1_000_000.0, 900_000.0                     # historical u = 0.9 (err = 0)
    state = _state(utilization=0.9, supply_assets_usd=supply, borrow_assets_usd=borrow,
                   liquidity_assets_usd=supply - borrow)
    entity.update_state(state)
    entity.action_deposit(1_000_000.0)                          # u' = 0.9M / 2.0M = 0.45
    bars = round(10.1 * 24)
    for _ in range(bars):
        entity.update_state(_state(**{**state.__dict__}))
    assert entity.effective_utilization == pytest.approx(0.45, rel=1e-3)
    assert entity.effective_rate_at_target == pytest.approx(RATE_AT_TARGET_APY / 2, rel=0.03)
    # The earned APY is the drifted curve at u' anchored to the observed level on the historical curve:
    # the historical R* stays in the denominator, so the drift is not cancelled.
    expected = (
        state.supply_apy
        * _curve_apy(entity.effective_utilization, entity.effective_rate_at_target)
        / _curve_apy(state.utilization, RATE_AT_TARGET_APY)
    )
    assert entity.effective_supply_apy == pytest.approx(expected, rel=1e-5)


def test_irm_feedback_lowers_earned_yield_and_sizing_consistently():
    """Halving R* through feedback must lower the accrued APY, the projection, capacity and the take curve."""
    supply, borrow = 1_000_000.0, 900_000.0
    state = _state(utilization=0.9, supply_assets_usd=supply, borrow_assets_usd=borrow,
                   liquidity_assets_usd=supply - borrow)
    runs = {}
    for feedback in (False, True):
        entity = _entity(own_impact=True, absorption=0.0, irm_feedback=feedback)
        entity.update_state(state)
        entity.action_deposit(1_000_000.0)
        start = entity.balance
        for _ in range(round(10.1 * 24)):
            entity.update_state(_state(**{**state.__dict__}))
        runs[feedback] = (entity, entity.balance - start)
    base, drifted = runs[False][0], runs[True][0]
    assert drifted.effective_rate_at_target == pytest.approx(RATE_AT_TARGET_APY / 2, rel=0.03)
    assert drifted.effective_supply_apy < 0.6 * base.effective_supply_apy          # roughly halved, not unchanged
    assert runs[True][1] < runs[False][1]                                           # less interest accrued
    size = 1_000_000.0
    assert drifted.projected_supply_apy(size) < base.projected_supply_apy(size)
    target = drifted.projected_supply_apy(size)
    assert drifted.capacity_at(target) == pytest.approx(size, rel=1e-6)            # inversion stays exact
    assert float(drifted.take_curve([target], drifted.global_state, cap=1e12)[0]) == pytest.approx(size, rel=1e-3)
    assert drifted.capacity_at(target) < base.capacity_at(target)


def test_irm_feedback_has_no_drift_without_a_position_and_is_clamped():
    entity = _entity(own_impact=True, irm_feedback=True)
    for _ in range(100):
        entity.update_state(_state())
    assert entity.internal_state.rstar_log_drift == 0.0
    # a huge position held for a long time cannot push R* below the protocol floor
    entity.action_deposit(1e9)
    for _ in range(24 * 120):
        entity.update_state(_state())
    assert math.log1p(entity.effective_rate_at_target) == pytest.approx(0.001, rel=1e-6)


# ------------------------------------------------------------ fixed gas per leg
def test_fixed_gas_is_charged_per_leg_and_recorded():
    recorded = []
    entity = MorphoSupplyEntity(own_impact=False, gas_usd_per_leg=5.0, cost_bps_per_leg=1.0)
    entity.attach_execution_recorder(lambda action, notional, fee: recorded.append((action, notional, fee)))
    entity.update_state(_state())
    entity.action_deposit(10_000.0)
    assert entity.balance == pytest.approx(10_000.0 - 5.0 - 1.0)
    out = entity.max_withdrawable_notional
    assert out == pytest.approx((entity.balance - 5.0) / 1.0001)
    entity.action_withdraw(out)
    assert entity.balance == pytest.approx(0.0, abs=1e-6)
    assert [r[2] for r in recorded] == [pytest.approx(6.0), pytest.approx(5.0 + out * 1e-4)]
    assert 10_000.0 - out == pytest.approx(sum(r[2] for r in recorded), rel=1e-9)   # NAV drop == fees


def test_gas_makes_tiny_legs_impossible_and_empty_legs_free():
    recorded = []
    entity = MorphoSupplyEntity(own_impact=False, gas_usd_per_leg=5.0)
    entity.attach_execution_recorder(lambda action, notional, fee: recorded.append(fee))
    entity.update_state(_state())
    with pytest.raises(MorphoSupplyException, match="does not cover its own cost"):
        entity.action_deposit(3.0)
    entity.action_deposit(0.0)
    entity.action_withdraw(0.0)
    assert not recorded and entity.balance == 0.0
    entity.action_deposit(8.0)                                  # 8 − 5 gas = 3 supplied
    assert entity.max_withdrawable_notional == 0.0              # 3 left does not cover the 5 gas
    with pytest.raises(MorphoSupplyException, match="gas_usd_per_leg"):
        MorphoSupplyEntity(gas_usd_per_leg=-1.0)
    assert entity.leg_cost_model == (5.0, 0.0)


# ----------------------------------------------------------------- rewards
def test_rewards_are_off_by_default_and_ignored_then():
    entity = _entity(own_impact=True)
    base = entity.projected_supply_apy(1e6)
    entity.update_state(_state(reward_apy=0.05))
    assert entity.projected_supply_apy(1e6) == pytest.approx(base)


def test_rewards_dilute_as_supply_share():
    """Campaign pays a fixed amount to the market: our reward APY is reward · S/(S+P)."""
    with_rw = MorphoSupplyEntity(own_impact=True, use_rewards=True)
    without = MorphoSupplyEntity(own_impact=True)
    for e in (with_rw, without):
        e.update_state(_state(reward_apy=0.04))
    for size in (0.0, 1e6, SUPPLY_USD):
        extra = with_rw.projected_supply_apy(size) - without.projected_supply_apy(size)
        assert extra == pytest.approx(0.04 * SUPPLY_USD / (SUPPLY_USD + size), rel=1e-6)


def test_reward_weight_haircuts_and_exogenous_mode_adds_undiluted():
    half = MorphoSupplyEntity(own_impact=True, use_rewards=True, reward_weight=0.5)
    full = MorphoSupplyEntity(own_impact=True, use_rewards=True)
    for e in (half, full):
        e.update_state(_state(reward_apy=0.04))
    base = _entity(own_impact=True).projected_supply_apy(1e6)
    expected = 0.5 * (full.projected_supply_apy(1e6) - base)
    assert half.projected_supply_apy(1e6) - base == pytest.approx(expected, rel=1e-6)
    exo = MorphoSupplyEntity(own_impact=False, use_rewards=True)
    exo.update_state(_state(reward_apy=0.03))
    assert exo.projected_supply_apy(1e9) == pytest.approx(0.08 + 0.03)
    with pytest.raises(MorphoSupplyException, match="reward_weight"):
        MorphoSupplyEntity(use_rewards=True, reward_weight=1.5)
    with pytest.raises(MorphoSupplyException, match="reward_apy"):
        exo.update_state(_state(reward_apy=-0.01))


def test_exogenous_rewards_accrue_the_same_apy_that_is_projected():
    base_apy, reward_apy = 0.08, 0.03
    entity = MorphoSupplyEntity(own_impact=False, use_rewards=True)
    snapshot = _state(
        supply_rate=math.log1p(base_apy) / 8760.0,
        supply_apy=base_apy,
        reward_apy=reward_apy,
    )
    entity.update_state(snapshot)
    entity.action_deposit(1_000.0)
    initial = entity.balance
    for _ in range(8760):
        entity.update_state(snapshot)
    assert entity.balance / initial - 1.0 == pytest.approx(
        entity.projected_supply_apy(entity.balance), rel=1e-5,
    )


def test_rewards_accrue_and_capacity_inverts_with_rewards():
    entity = MorphoSupplyEntity(own_impact=True, use_rewards=True)
    entity.update_state(_state(reward_apy=0.03))
    for target in (0.05, 0.08, 0.10):
        size = entity.capacity_at(target)
        assert 0.0 < size < float("inf")
        assert entity.projected_supply_apy(size) == pytest.approx(target, rel=1e-4)
    # with absorption keeping u fixed, rewards still dilute: capacity becomes finite for a target above the base
    absorbed = MorphoSupplyEntity(own_impact=True, absorption=1.0, use_rewards=True)
    absorbed.update_state(_state(reward_apy=0.03))
    base_rate = absorbed.projected_supply_apy(1e12)                 # rewards diluted away
    assert math.isfinite(absorbed.capacity_at(base_rate + 0.01))
    # accrual: one bar earns base + diluted reward
    entity.action_deposit(1e6)
    before = entity.balance
    entity.update_state(_state(reward_apy=0.03))
    assert entity.balance / before - 1 == pytest.approx(math.log1p(entity.effective_supply_apy) / 8760, rel=1e-6)


@pytest.mark.parametrize("absorption", [0.0, 0.5])
def test_market_fee_cancels_out_of_the_anchored_rate(absorption):
    """The observed supply APY is already net of the fee, and (1 - fee) scales both curves of the anchor ratio."""
    state = _state(utilization=0.85, supply_assets_usd=10e6, borrow_assets_usd=8.5e6, liquidity_assets_usd=1.5e6)
    results = []
    for fee in (0.0, 0.25):
        entity = _entity(own_impact=True, absorption=absorption)
        snap = MorphoSupplyGlobalState(**{**state.__dict__, "fee": fee})
        entity.update_state(snap)
        results.append((entity.projected_supply_apy(2e6, snap), entity.capacity_at(0.05, snap),
                        float(entity.take_curve([0.05], snap, cap=1e12)[0])))
    for without_fee, with_fee in zip(*results, strict=True):
        assert with_fee == pytest.approx(without_fee, rel=1e-9)
