"""L1 tests for :class:`BorosEntity` and :mod:`boros_math`: settlement
cash flows at the venue cadence, mark-to-maturity, floors in margin,
netting, maturity, liquidation, coin margining."""
import math

import pytest

from fractal.core.base import Action
from fractal.core.base.time import SECONDS_PER_DAY, SECONDS_PER_YEAR
from fractal.core.entities import BasePerpEntity
from fractal.core.entities.models.boros_math import (
    fixed_leg_coin,
    liquidation_penalty_fraction,
    margin_coin,
    mark_to_maturity_coin,
    open_fee_coin,
    settlement_pnl_coin,
)
from fractal.core.entities.protocols.boros import BorosEntity, BorosException, BorosGlobalState

EIGHT_HOURS = 8 * 3600
DAYS_30 = 30 * SECONDS_PER_DAY


def state(**overrides) -> BorosGlobalState:
    base = dict(seconds_to_expiry=DAYS_30, mark_rate=0.05, funding_rate=0.0, funding_period_seconds=0.0,
                underlying_price=1.0)
    base.update(overrides)
    return BorosGlobalState(**base)


def make(collateral: float = 1_000.0, **kwargs) -> BorosEntity:
    entity = BorosEntity(taker_fee_rate=0.0, settle_fee_rate=0.0, **kwargs)
    entity.update_state(state())
    entity.action_deposit(collateral)
    return entity


# ------------------------------------------------------------ boros_math
@pytest.mark.core
def test_settlement_formula_and_sign():
    """Long receives N·f, pays N·r·Δ/Y and |N|·fee·Δ/Y; short mirrors the legs."""
    long_flow = settlement_pnl_coin(10.0, 0.0001, 0.05, EIGHT_HOURS, 0.001)
    expected = 10 * 0.0001 - 10 * 0.05 * EIGHT_HOURS / SECONDS_PER_YEAR - 10 * 0.001 * EIGHT_HOURS / SECONDS_PER_YEAR
    assert long_flow == pytest.approx(expected)
    short_flow = settlement_pnl_coin(-10.0, 0.0001, 0.05, EIGHT_HOURS, 0.001)
    assert short_flow == pytest.approx(-10 * 0.0001 + 10 * 0.05 * EIGHT_HOURS / SECONDS_PER_YEAR
                                       - 10 * 0.001 * EIGHT_HOURS / SECONDS_PER_YEAR)
    assert fixed_leg_coin(10.0, 0.05, SECONDS_PER_YEAR) == pytest.approx(0.5)


@pytest.mark.core
def test_per_period_fixed_leg_sums_to_the_upfront_fixed_cost():
    """Boros charges the fixed leg upfront (N·r·T_entry); paying it per period sums to the same."""
    size, rate, term = 10.0, 0.05, 90 * SECONDS_PER_DAY
    periods = int(term / EIGHT_HOURS)
    per_period = sum(-settlement_pnl_coin(size, 0.0, rate, EIGHT_HOURS, 0.0) for _ in range(periods))
    assert per_period == pytest.approx(fixed_leg_coin(size, rate, term), rel=1e-12)


@pytest.mark.core
def test_mtm_margin_fee_and_penalty_shapes():
    assert mark_to_maturity_coin(10.0, 0.06, 0.05, 0.5) == pytest.approx(10 * 0.01 * 0.5)
    assert mark_to_maturity_coin(10.0, 0.06, 0.05, 0.0) == 0.0
    assert open_fee_coin(-10.0, 0.0005, 0.5) == pytest.approx(10 * 0.0005 * 0.5)
    # floors: tiny TTM → time floor; tiny mark → rate floor
    assert margin_coin(10.0, 0.05, 1 / 365, 0.645, 0.06, 10 / 365) == pytest.approx(0.645 * 10 * (10 / 365) * 0.06)
    assert margin_coin(10.0, 0.20, 0.5, 0.645, 0.06, 10 / 365) == pytest.approx(0.645 * 10 * 0.5 * 0.20)
    assert liquidation_penalty_fraction(1.0) == pytest.approx(0.25)
    assert liquidation_penalty_fraction(0.5) == pytest.approx(0.5)
    assert liquidation_penalty_fraction(0.1) == pytest.approx(0.1)
    assert liquidation_penalty_fraction(-1.0) == 0.0


# ---------------------------------------------------------------- config
@pytest.mark.core
@pytest.mark.parametrize("kwargs", [
    {"max_leverage": 0.0}, {"mm_to_im_ratio": 0.0}, {"mm_to_im_ratio": 1.5}, {"rate_floor": -0.1},
    {"time_threshold_seconds": -1.0}, {"taker_fee_rate": -1e-4}, {"settle_fee_rate": -1e-4}, {"slippage_rate": -1e-4},
])
def test_config_validation(kwargs):
    with pytest.raises(BorosException):
        BorosEntity(**kwargs)


@pytest.mark.core
def test_is_a_perp_with_a_term():
    entity = BorosEntity()
    assert isinstance(entity, BasePerpEntity)
    assert entity.is_matured
    assert entity.k_im == pytest.approx(1 / 1.55)
    assert entity.k_mm == pytest.approx(0.5 / 1.55)
    assert entity.trading_fee == 0.0005
    for action in ("deposit", "withdraw", "open_position", "close_position"):
        assert action in entity.get_available_actions()
    entity.action_deposit(10.0)
    with pytest.raises(BorosException, match="after expiry"):
        entity.action_open_position(1.0)


# -------------------------------------------------------------- lifecycle
@pytest.mark.core
def test_settlement_only_on_period_bars_and_long_receives_positive_funding():
    entity = make()
    entity.action_open_position(10.0)  # long 10 YU at 5 % fixed
    entity.update_state(state(funding_rate=0.0001))  # no period → nothing settles
    assert entity.internal_state.collateral == 1_000.0
    entity.update_state(state(funding_rate=0.0001, funding_period_seconds=EIGHT_HOURS))
    expected = settlement_pnl_coin(10.0, 0.0001, 0.05, EIGHT_HOURS, 0.0)
    assert entity.internal_state.collateral == pytest.approx(1_000.0 + expected)
    assert entity.internal_state.realized_settlements == pytest.approx(expected)
    assert expected == pytest.approx(10 * 0.0001 - 10 * 0.05 * EIGHT_HOURS / SECONDS_PER_YEAR)


@pytest.mark.core
def test_long_yu_offsets_a_short_perp_funding_leg():
    """Short perp pays −size·mark·f (i.e. receives when f > 0); long YU receives N·f: net floating = 0."""
    entity = make()
    entity.action_open_position(10.0)
    entity.update_state(state(funding_rate=-0.0002, funding_period_seconds=EIGHT_HOURS, mark_rate=0.05))
    fixed_only = settlement_pnl_coin(10.0, 0.0, 0.0, EIGHT_HOURS, 0.0)
    floating = settlement_pnl_coin(10.0, -0.0002, 0.0, EIGHT_HOURS, 0.0) - fixed_only
    assert floating == pytest.approx(10 * -0.0002)  # long YU pays when funding is negative


@pytest.mark.core
def test_mark_to_maturity_and_netting():
    entity = make()
    entity.action_open_position(10.0)
    assert entity.entry_rate == 0.05
    entity.update_state(state(mark_rate=0.07))
    years = DAYS_30 / SECONDS_PER_YEAR
    assert entity.pnl == pytest.approx(10 * 0.02 * years)
    assert entity.balance == pytest.approx(1_000.0 + 10 * 0.02 * years)
    entity.action_open_position(10.0)  # average up
    assert entity.entry_rate == pytest.approx(0.06)
    entity.action_open_position(-5.0)  # partial close realises 5·(0.07−0.06)·T
    assert entity.size == 15.0
    assert entity.internal_state.collateral == pytest.approx(1_000.0 + 5 * 0.01 * years)
    entity.action_open_position(-25.0)  # flip to short 10 at the fill rate
    assert entity.size == -10.0 and entity.entry_rate == 0.07
    entity.action_close_position()
    assert entity.size == 0.0 and entity.entry_rate == 0.0 and entity.pnl == 0.0


@pytest.mark.core
def test_open_fee_and_slippage_scale_with_ttm():
    near = BorosEntity(taker_fee_rate=0.0005, settle_fee_rate=0.0, slippage_rate=0.001)
    near.update_state(state(seconds_to_expiry=10 * SECONDS_PER_DAY))
    near.action_deposit(1_000.0)
    near.action_open_position(100.0)
    far = BorosEntity(taker_fee_rate=0.0005, settle_fee_rate=0.0, slippage_rate=0.001)
    far.update_state(state(seconds_to_expiry=300 * SECONDS_PER_DAY))
    far.action_deposit(1_000.0)
    far.action_open_position(100.0)
    fee_near = 1_000.0 - near.internal_state.collateral
    fee_far = 1_000.0 - far.internal_state.collateral
    assert fee_far / fee_near == pytest.approx(30, rel=1e-9)
    assert near.entry_rate == pytest.approx(0.05 + 0.001)  # long pays the spread


@pytest.mark.core
def test_position_expires_worthless_at_maturity():
    entity = make()
    entity.action_open_position(10.0)
    entity.update_state(state(mark_rate=0.08, seconds_to_expiry=SECONDS_PER_DAY))
    assert entity.pnl > 0
    entity.update_state(state(mark_rate=0.08, seconds_to_expiry=0.0, funding_rate=0.0003,
                              funding_period_seconds=EIGHT_HOURS))
    last_flow = settlement_pnl_coin(10.0, 0.0003, 0.05, EIGHT_HOURS, 0.0)
    assert entity.size == 0.0
    assert entity.pnl == 0.0
    assert entity.balance == pytest.approx(1_000.0 + last_flow)
    with pytest.raises(BorosException, match="after expiry"):
        entity.action_open_position(1.0)


@pytest.mark.core
def test_initial_margin_gate_rolls_back():
    entity = make(collateral=1.0)
    size_ok = 1.0 / (entity.k_im * max(DAYS_30 / SECONDS_PER_YEAR, entity.time_floor_years) * 0.06)
    with pytest.raises(BorosException, match="initial margin"):
        entity.action_open_position(size_ok * 1.01)
    assert entity.size == 0.0 and entity.internal_state.collateral == 1.0
    entity.action_open_position(size_ok * 0.99)
    assert entity.size > 0
    with pytest.raises(BorosException, match="initial margin"):
        entity.action_withdraw(0.5)


@pytest.mark.core
def test_settlement_before_liquidation_can_save_the_position():
    """A favourable settlement lands before the HR check on the same bar."""
    # C = 0.7 vs IM 0.636 at open (rate floor 6 %, T = 30 d): thin but allowed.
    entity = make(collateral=0.7, mm_to_im_ratio=1.0)
    entity.action_open_position(200.0)
    entity.update_state(state(mark_rate=0.03, seconds_to_expiry=20 * SECONDS_PER_DAY))
    assert entity.size == 200.0 and entity.health_ratio > 1
    # without settlement this bar would liquidate (mark falls further)
    probe = make(collateral=0.7, mm_to_im_ratio=1.0)
    probe.action_open_position(200.0)
    probe.update_state(state(mark_rate=0.03, seconds_to_expiry=20 * SECONDS_PER_DAY))
    probe.update_state(state(mark_rate=0.00, seconds_to_expiry=19 * SECONDS_PER_DAY))
    assert probe.size == 0.0 and probe.internal_state.liquidation_count == 1
    # with a big positive funding settlement the same bar keeps it alive
    entity.update_state(state(mark_rate=0.00, seconds_to_expiry=19 * SECONDS_PER_DAY,
                              funding_rate=0.01, funding_period_seconds=EIGHT_HOURS))
    assert entity.size == 200.0 and entity.internal_state.liquidation_count == 0


@pytest.mark.core
def test_liquidation_closes_at_mark_and_charges_the_penalty():
    entity = make(collateral=1.0, mm_to_im_ratio=1.0)
    entity.action_open_position(200.0)
    entity.update_state(state(mark_rate=0.00, seconds_to_expiry=DAYS_30))
    post_mtm_balance = 1.0 + 200 * (0.0 - 0.05) * DAYS_30 / SECONDS_PER_YEAR  # ≈ 0.178
    assert entity.size == 0.0 and entity.entry_rate == 0.0
    assert entity.internal_state.liquidation_count == 1
    assert 0.0 <= entity.internal_state.collateral < post_mtm_balance  # closed at mark, penalty taken
    assert entity.pnl == 0.0


@pytest.mark.core
def test_coin_margined_balance_tracks_the_underlying_price():
    entity = BorosEntity(coin_margined=True, taker_fee_rate=0.0, settle_fee_rate=0.0)
    entity.update_state(state(underlying_price=2_000.0))
    entity.action_deposit(4_000.0)  # 2 ETH of margin
    assert entity.internal_state.collateral == pytest.approx(2.0)
    entity.action_open_position(1.0)
    entity.update_state(state(underlying_price=3_000.0, mark_rate=0.06))
    years = DAYS_30 / SECONDS_PER_YEAR
    assert entity.balance == pytest.approx((2.0 + 1.0 * 0.01 * years) * 3_000.0)
    assert entity.initial_margin == pytest.approx(entity.k_im * 1.0 * years * 0.06 * 3_000.0)


@pytest.mark.core
def test_update_state_validation_and_dispatch():
    entity = make()
    for bad in (state(underlying_price=0.0), state(funding_period_seconds=-1.0), state(mark_rate=math.nan),
                state(seconds_to_expiry=math.inf)):
        with pytest.raises(BorosException):
            entity.update_state(bad)
    entity.execute(Action("open_position", {"amount_in_product": -3.0}))
    assert entity.size == -3.0
    entity.execute(Action("close_position", {}))
    assert entity.size == 0.0
    with pytest.raises(BorosException, match="increased"):
        entity.update_state(state(seconds_to_expiry=DAYS_30 + 1.0))
