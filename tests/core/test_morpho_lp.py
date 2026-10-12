"""Supply-side invariants for :class:`MorphoLpEntity`.

The LP entity is the only lending entity in the repo whose *exit* can fail, so
these tests pin the gate (it can never be violated), the accrual, the two
replay modes and the RQ4 arithmetic that CONTEXT §2.3 quotes ($1.0M into a
$15.83M supply / $14.68M borrow market).
"""
import copy
import math

import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.models.morpho_math import adaptive_curve_borrow_apr
from fractal.core.entities.protocols.morpho_lp import (
    MorphoLpEntity,
    MorphoLpException,
    MorphoLpGlobalState,
)

pytestmark = pytest.mark.core

SUPPLY_USD = 15_830_000.0
BORROW_USD = 14_680_000.0
LIQUIDITY_USD = SUPPLY_USD - BORROW_USD           # 1.15M
HISTORICAL_UTILIZATION = BORROW_USD / SUPPLY_USD  # 0.9273
RATE_AT_TARGET_APY = 0.07


def _curve_apy(utilization: float, rate_at_target_apy: float = RATE_AT_TARGET_APY, fee: float = 0.0) -> float:
    """Supply APY the IRM curve implies, converted rate -> APY (as the validator does)."""
    borrow_apy = math.expm1(adaptive_curve_borrow_apr(utilization, math.log1p(rate_at_target_apy)))
    return borrow_apy * utilization * (1.0 - fee)


def _state(**overrides) -> MorphoLpGlobalState:
    base = {
        "utilization": HISTORICAL_UTILIZATION,
        "supply_rate": 0.0,
        "supply_apy": 0.0821,
        "borrow_apy": 0.0893,
        "rate_at_target": RATE_AT_TARGET_APY,
        "fee": 0.0,
        "supply_assets_usd": SUPPLY_USD,
        "borrow_assets_usd": BORROW_USD,
        "liquidity_assets_usd": LIQUIDITY_USD,
    }
    base.update(overrides)
    return MorphoLpGlobalState(**base)


def _entity(**kwargs) -> MorphoLpEntity:
    entity = MorphoLpEntity(**kwargs)
    entity.update_state(_state())
    return entity


# ------------------------------------------------------------------- basics
def test_initial_state_is_empty_and_supply_only():
    entity = MorphoLpEntity()
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
    with pytest.raises(MorphoLpException, match="exceeds withdrawable"):
        entity.action_withdraw(2_000_000.0)
    assert entity.balance == pytest.approx(2_000_000.0)  # unchanged by the failed call


def test_gate_is_closed_at_full_utilization():
    entity = MorphoLpEntity(own_impact=False)
    entity.update_state(_state(utilization=1.0, liquidity_assets_usd=0.0))
    entity.action_deposit(50_000.0)
    assert entity.withdrawable == 0.0
    with pytest.raises(MorphoLpException, match="exceeds withdrawable"):
        entity.action_withdraw(1.0)
    assert entity.balance == pytest.approx(50_000.0)


def test_withdraw_above_position_raises_even_when_market_is_liquid():
    entity = _entity(own_impact=False)
    entity.action_deposit(1_000.0)
    with pytest.raises(MorphoLpException, match="exceeds position"):
        entity.action_withdraw(1_001.0)


def test_round_trip_withdraws_exactly_the_deposit_when_the_gate_is_open():
    entity = _entity(own_impact=False)
    entity.action_deposit(500_000.0)
    entity.action_withdraw(500_000.0)
    assert entity.balance == pytest.approx(0.0)


def test_negative_amounts_are_rejected():
    entity = _entity(own_impact=False)
    with pytest.raises(MorphoLpException, match="deposit amount must be >= 0"):
        entity.action_deposit(-1.0)
    with pytest.raises(MorphoLpException, match="withdraw amount must be >= 0"):
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
def test_own_impact_arithmetic_matches_context_section_2_3():
    """$1.0M into $15.83M/$14.68M: u 0.927 -> 0.872, gate = liquidity + P."""
    entity = _entity(own_impact=True, absorption=0.0)
    entity.action_deposit(1_000_000.0)
    assert entity.effective_utilization == pytest.approx(0.872, abs=5e-4)
    assert entity.withdrawable == pytest.approx(LIQUIDITY_USD + 1_000_000.0)


def test_own_impact_self_dilutes_the_earned_rate():
    """The position lowers the rate it earns — the RQ4 effect, modelled not assumed."""
    entity = _entity(own_impact=True, absorption=0.0)
    entity.action_deposit(1_000_000.0)
    undistorted = _curve_apy(HISTORICAL_UTILIZATION)
    assert entity.effective_supply_apy < undistorted  # CONTEXT 2.3: our own size cuts our rate
    assert entity.effective_supply_apy > 0.0
    assert entity.effective_supply_apy == pytest.approx(_curve_apy(entity.effective_utilization), rel=1e-5)
    # the dilution is material, not a rounding artefact: 8.2% -> ~4.9% at this R*
    assert undistorted - entity.effective_supply_apy > 0.02


def test_absorption_of_u_preserves_utilization():
    entity = _entity(own_impact=True, absorption=HISTORICAL_UTILIZATION)
    entity.action_deposit(1_000_000.0)
    assert entity.effective_utilization == pytest.approx(HISTORICAL_UTILIZATION, abs=5e-5)


def test_full_absorption_does_not_make_our_own_supply_withdrawable():
    entity = _entity(own_impact=True, absorption=1.0)
    entity.action_deposit(1_000_000.0)
    assert entity.withdrawable == pytest.approx(LIQUIDITY_USD)


def test_larger_positions_dilute_more():
    small, large = _entity(own_impact=True), _entity(own_impact=True)
    small.action_deposit(0.01 * SUPPLY_USD)
    large.action_deposit(0.5 * SUPPLY_USD)
    assert large.effective_supply_apy < small.effective_supply_apy


# -------------------------------------------------------------------- costs
def test_costs_are_charged_on_the_moved_notional_and_recorded():
    recorded = []
    entity = MorphoLpEntity(own_impact=False, cost_bps_per_leg=10.0)
    entity.attach_execution_recorder(lambda action, notional, fee: recorded.append((action, notional, fee)))
    entity.update_state(_state())
    entity.action_deposit(100_000.0)
    assert entity.balance == pytest.approx(100_000.0 * (1 - 0.001))
    entity.action_withdraw(50_000.0)
    assert recorded == [("deposit", 100_000.0, 100.0), ("withdraw", 50_000.0, 50.0)]


def test_zero_cost_still_records_the_leg():
    recorded = []
    entity = MorphoLpEntity(own_impact=False)
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
    with pytest.raises(MorphoLpException, match="absorption"):
        MorphoLpEntity(absorption=1.5)
    with pytest.raises(MorphoLpException, match="cost_bps_per_leg"):
        MorphoLpEntity(cost_bps_per_leg=-1.0)
    with pytest.raises(MorphoLpException, match="bars_per_year"):
        MorphoLpEntity(bars_per_year=0.0)
    entity = MorphoLpEntity(own_impact=False)
    with pytest.raises(MorphoLpException, match="fee must be in"):
        entity.update_state(_state(fee=0.5))
    with pytest.raises(MorphoLpException, match="rate_at_target"):
        entity.update_state(_state(rate_at_target=-0.01))
    with pytest.raises(MorphoLpException, match="supply_rate must be >= -1"):
        entity.update_state(_state(supply_rate=-1.5))


def test_entity_exception_is_an_entity_exception():
    assert issubclass(MorphoLpException, EntityException)


def test_bars_per_year_matches_the_hourly_loader_convention():
    """A 5% supply APY must round-trip through the per-bar rate at 8760 bars/year."""
    entity = MorphoLpEntity(own_impact=False, bars_per_year=8760.0)
    state = _state(supply_rate=math.log1p(0.05) / 8760.0)
    assert math.expm1(math.log1p(entity._effective_supply_rate(state)) * 8760.0) == pytest.approx(0.05, rel=1e-5)
