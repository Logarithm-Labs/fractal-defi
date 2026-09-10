"""L1 tests for :class:`PendlePTEntity`: pricing in the accounting asset,
swap costs through both impact models, redemption semantics and the
fixed-term guards."""
import math

import pytest

from fractal.core.base import Action
from fractal.core.base.time import SECONDS_PER_DAY
from fractal.core.entities.models.pendle_math import pt_price_from_apy
from fractal.core.entities.protocols.pendle_pt import (
    PendlePTConfig,
    PendlePTEntity,
    PendlePTException,
    PendlePTGlobalState,
)

DAYS_90 = 90 * SECONDS_PER_DAY


def make_state(**overrides) -> PendlePTGlobalState:
    base = dict(
        seconds_to_expiry=DAYS_90, implied_apy=0.10, asset_price=1.0, sy_exchange_rate=1.0,
        total_pt=1_000_000.0, total_sy=3_000_000.0, scalar_root=50.0, ln_fee_rate_root=5e-4,
    )
    base.update(overrides)
    return PendlePTGlobalState(**base)


def make_entity(impact_model: str = "amm", **cfg) -> PendlePTEntity:
    entity = PendlePTEntity(PendlePTConfig(impact_model=impact_model, **cfg))
    entity.update_state(make_state())
    return entity


@pytest.mark.core
@pytest.mark.parametrize("cfg", [
    {"impact_model": "linear"},
    {"fee_ln_rate": -1e-4},
    {"impact_ln_rate_per_share": -1.0},
    {"max_pool_share": 0.0},
    {"max_pool_share": 1.5},
])
def test_config_validation(cfg):
    with pytest.raises(PendlePTException):
        PendlePTEntity(PendlePTConfig(**cfg))


@pytest.mark.core
def test_default_entity_is_matured_and_cannot_trade():
    entity = PendlePTEntity()
    entity.action_deposit(10.0)
    assert entity.is_matured
    with pytest.raises(PendlePTException, match="after expiry"):
        entity.action_buy(10.0)


@pytest.mark.core
def test_price_is_derived_from_implied_apy_in_asset_terms():
    entity = make_entity()
    expected = pt_price_from_apy(0.10, DAYS_90 / (365 * SECONDS_PER_DAY))
    assert entity.pt_price_asset == pytest.approx(expected)
    assert entity.current_price == pytest.approx(expected)
    entity.update_state(make_state(asset_price=2.0))
    assert entity.current_price == pytest.approx(2.0 * expected)


@pytest.mark.core
@pytest.mark.parametrize("impact_model", ["amm", "rate_spread"])
def test_buy_then_sell_round_trip_loses_only_to_costs(impact_model):
    entity = make_entity(impact_model)
    entity.action_deposit(10_000.0)
    entity.action_buy(10_000.0)
    assert entity.internal_state.cash == 0.0
    pt = entity.internal_state.amount
    mid = entity.pt_price_asset
    assert 0 < pt < 10_000.0 / mid  # paid more than mid: fee + impact
    entity.action_sell(pt)
    assert entity.internal_state.amount == 0.0
    assert 9_900.0 < entity.internal_state.cash < 10_000.0


@pytest.mark.core
def test_zero_cost_rate_spread_buys_at_mid():
    entity = make_entity("rate_spread", fee_ln_rate=0.0, impact_ln_rate_per_share=0.0)
    entity.action_deposit(1_000.0)
    entity.action_buy(1_000.0)
    assert entity.internal_state.amount == pytest.approx(1_000.0 / entity.pt_price_asset)
    assert entity.balance == pytest.approx(1_000.0)


@pytest.mark.core
def test_amm_model_needs_pool_state():
    entity = PendlePTEntity(PendlePTConfig(impact_model="amm"))
    entity.update_state(make_state(total_pt=0.0, total_sy=0.0, scalar_root=0.0))
    entity.action_deposit(100.0)
    with pytest.raises(PendlePTException, match="rate_spread"):
        entity.action_buy(100.0)


@pytest.mark.core
def test_buy_sell_validation():
    entity = make_entity()
    entity.action_deposit(100.0)
    with pytest.raises(PendlePTException):
        entity.action_buy(-1.0)
    with pytest.raises(PendlePTException):
        entity.action_buy(101.0)
    with pytest.raises(PendlePTException):
        entity.action_sell(-1.0)
    with pytest.raises(PendlePTException):
        entity.action_sell(1.0)
    entity.action_buy(0.0)
    entity.action_sell(0.0)
    assert entity.internal_state.cash == 100.0


@pytest.mark.core
def test_actions_dispatch_through_execute_with_spot_argument_names():
    entity = make_entity("rate_spread")
    entity.execute(Action("deposit", {"amount_in_notional": 500.0}))
    entity.execute(Action("buy", {"amount_in_notional": 500.0}))
    held = entity.internal_state.amount
    entity.execute(Action("remove_product", {"amount": held}))
    entity.execute(Action("inject_product", {"amount": held}))
    entity.execute(Action("sell", {"amount_in_product": held}))
    assert entity.internal_state.amount == 0.0
    assert "redeem" in entity.get_available_actions()


@pytest.mark.core
def test_balance_is_continuous_across_expiry_and_pt_earns_nothing_after():
    entity = make_entity("rate_spread", fee_ln_rate=0.0, impact_ln_rate_per_share=0.0)
    entity.action_deposit(1_000.0)
    entity.action_buy(1_000.0)
    pt = entity.internal_state.amount
    entity.update_state(make_state(seconds_to_expiry=1.0))
    just_before = entity.balance
    entity.update_state(make_state(seconds_to_expiry=0.0))
    assert entity.is_matured
    assert entity.balance == pytest.approx(pt)  # par
    assert entity.balance == pytest.approx(just_before, rel=1e-6)
    entity.update_state(make_state(seconds_to_expiry=-30 * SECONDS_PER_DAY, implied_apy=0.5))
    assert entity.balance == pytest.approx(pt)  # no accrual, implied APY irrelevant
    with pytest.raises(PendlePTException, match="after expiry"):
        entity.action_buy(0.0)
    with pytest.raises(PendlePTException, match="after expiry"):
        entity.action_sell(0.0)


@pytest.mark.core
def test_redeem_requires_expiry_and_pays_par_times_asset_price():
    entity = make_entity()
    entity.action_inject_product(100.0)
    with pytest.raises(PendlePTException, match="after expiry"):
        entity.action_redeem(100.0)
    entity.update_state(make_state(seconds_to_expiry=0.0, asset_price=0.97))
    with pytest.raises(PendlePTException):
        entity.action_redeem(100.1)
    entity.action_redeem(100.0)
    assert entity.internal_state.amount == 0.0
    assert entity.internal_state.cash == pytest.approx(97.0)


@pytest.mark.core
def test_redeem_applies_py_index_haircut_when_sy_rate_falls():
    """Pendle ratchets pyIndex to the highest exchange rate seen; a later
    drop is borne by PT holders (syIndex / pyIndex)."""
    entity = make_entity()
    entity.action_inject_product(100.0)
    entity.update_state(make_state(sy_exchange_rate=1.25))
    entity.update_state(make_state(sy_exchange_rate=1.20, seconds_to_expiry=0.0))
    assert entity.redeem_haircut == pytest.approx(1.20 / 1.25)
    assert entity.balance == pytest.approx(100.0 * 1.20 / 1.25)
    entity.action_redeem(100.0)
    assert entity.internal_state.cash == pytest.approx(96.0)
    # a rising rate never inflates the payout above par
    other = make_entity()
    other.action_inject_product(1.0)
    other.update_state(make_state(sy_exchange_rate=1.30, seconds_to_expiry=0.0))
    assert other.redeem_haircut == 1.0


@pytest.mark.core
def test_update_state_validates_and_does_not_mutate_input():
    entity = make_entity()
    for bad in (
        make_state(implied_apy=-1.0), make_state(asset_price=0.0), make_state(sy_exchange_rate=-1.0),
        make_state(total_pt=-1.0), make_state(scalar_root=-1.0), make_state(ln_fee_rate_root=-1e-4),
        make_state(seconds_to_expiry=math.nan),
    ):
        with pytest.raises(PendlePTException):
            entity.update_state(bad)
    state = make_state(seconds_to_expiry=0.0, implied_apy=0.3)
    entity.update_state(state)
    assert state.seconds_to_expiry == 0.0 and state.implied_apy == 0.3
    with pytest.raises(PendlePTException, match="increased"):
        entity.update_state(make_state(seconds_to_expiry=DAYS_90))


@pytest.mark.core
def test_two_instances_do_not_share_state():
    a, b = make_entity("rate_spread"), make_entity("rate_spread")
    a.action_deposit(10.0)
    assert b.internal_state.cash == 0.0
    assert a.internal_state is not b.internal_state
