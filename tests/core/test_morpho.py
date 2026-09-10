"""L1 tests for :class:`MorphoEntity` and :mod:`morpho_math`: LLTV health,
Taylor accrual, LIF liquidation that closes the debt and leaves the
residual collateral, oracle-vs-market pricing."""
import math

import pytest

from fractal.core.base import Action
from fractal.core.base.time import SECONDS_PER_YEAR
from fractal.core.entities.models.morpho_math import (
    adaptive_curve_borrow_apr,
    borrow_apy_from_per_second_rate,
    liquidation_incentive_factor,
    max_borrow,
    per_bar_borrow_rate,
    taylor_compounded,
)
from fractal.core.entities.protocols.morpho import MorphoEntity, MorphoException, MorphoGlobalState


def make(lltv: float = 0.86, **kwargs) -> MorphoEntity:
    entity = MorphoEntity(lltv=lltv, **kwargs)
    entity.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0))
    return entity


# ------------------------------------------------------------- morpho_math
@pytest.mark.core
def test_lif_matches_protocol_table():
    assert liquidation_incentive_factor(0.86) == pytest.approx(1.0438, abs=5e-4)
    assert liquidation_incentive_factor(0.915) == pytest.approx(1.0262, abs=5e-4)
    assert liquidation_incentive_factor(0.625) == pytest.approx(1.1268, abs=5e-4)
    assert liquidation_incentive_factor(0.385) == 1.15  # capped
    with pytest.raises(ValueError):
        liquidation_incentive_factor(0.0)


@pytest.mark.core
def test_taylor_accrual_matches_exp_at_hourly_steps():
    x = per_bar_borrow_rate(0.09, 3600)
    assert x == pytest.approx(math.log1p(0.09) * 3600 / SECONDS_PER_YEAR)
    assert taylor_compounded(x) == pytest.approx(math.expm1(x), abs=1e-15)
    hours = int(SECONDS_PER_YEAR // 3600)
    growth = (1 + taylor_compounded(x)) ** hours
    assert growth == pytest.approx(1.09, rel=1e-6)  # a year of hourly bars reproduces the APY
    assert borrow_apy_from_per_second_rate(math.log1p(0.09) / SECONDS_PER_YEAR) == pytest.approx(0.09)
    with pytest.raises(ValueError):
        taylor_compounded(-1.5)


@pytest.mark.core
def test_adaptive_curve_shape():
    assert adaptive_curve_borrow_apr(0.0, 0.04) == pytest.approx(0.01)
    assert adaptive_curve_borrow_apr(0.9, 0.04) == pytest.approx(0.04)
    assert adaptive_curve_borrow_apr(1.0, 0.04) == pytest.approx(0.16)
    assert max_borrow(1000.0, 0.97, 0.915) == pytest.approx(1000 * 0.97 * 0.915)
    assert max_borrow(0.0, 1.0, 0.9) == 0.0


# ---------------------------------------------------------------- config
@pytest.mark.core
@pytest.mark.parametrize("kwargs,pattern", [
    ({"lltv": 0.0}, "lltv must be in"),
    ({"lltv": 1.5}, "lltv must be in"),
    ({"lltv": 0.86, "max_ltv": 0.9}, "liq_thr.*must be >="),
    ({"lltv": 0.86, "max_ltv": 0.0}, "max_ltv must be in"),
    ({"lltv": 0.86, "liquidation_incentive_factor": 0.99}, "liquidation_incentive_factor"),
])
def test_config_validation(kwargs, pattern):
    with pytest.raises(MorphoException, match=pattern):
        MorphoEntity(**kwargs)


@pytest.mark.core
def test_defaults_follow_the_protocol():
    entity = MorphoEntity(lltv=0.915)
    assert entity.max_ltv == 0.915 and entity.liq_thr == 0.915
    assert entity.lif == pytest.approx(liquidation_incentive_factor(0.915))
    assert entity.collateral_is_volatile is True
    cushioned = MorphoEntity(lltv=0.915, max_ltv=0.8)
    assert cushioned.max_ltv == 0.8


# --------------------------------------------------------------- actions
@pytest.mark.core
def test_borrow_up_to_lltv_and_health():
    entity = make(0.86)
    entity.action_deposit(1000.0)
    entity.action_borrow(860.0)  # exactly at LLTV is allowed
    assert entity.ltv == pytest.approx(0.86)
    assert entity.health_factor == pytest.approx(1.0)
    with pytest.raises(MorphoException, match="borrow would push LTV"):
        entity.action_borrow(1.0)
    assert entity.internal_state.borrowed == 860.0
    assert entity.max_borrow_amount == pytest.approx(0.0)


@pytest.mark.core
def test_actions_use_lending_argument_names_through_execute():
    entity = make()
    entity.execute(Action("deposit", {"amount_in_notional": 1000.0}))
    entity.execute(Action("borrow", {"amount_in_product": 500.0}))
    entity.execute(Action("repay", {"amount_in_product": 200.0}))
    entity.execute(Action("withdraw", {"amount_in_notional": 100.0}))
    assert entity.internal_state.borrowed == 300.0
    assert entity.internal_state.collateral == 900.0
    assert entity.max_borrow_amount == pytest.approx(900 * 0.86 - 300)
    assert entity.calculate_repay(0.1) == pytest.approx(900 * (300 / 900 - 0.1))


@pytest.mark.core
def test_liquidation_closes_debt_and_leaves_residual_collateral():
    """Oracle price drops 10 %: LTV 0.889 > 0.86 → whole debt closed at LIF, rest stays."""
    entity = make(0.86)
    entity.action_deposit(1000.0)
    entity.action_borrow(800.0)
    entity.update_state(MorphoGlobalState(collateral_price=0.9, debt_price=1.0))
    seized = 800.0 * liquidation_incentive_factor(0.86) / 0.9
    assert entity.internal_state.borrowed == 0.0
    assert entity.internal_state.collateral == pytest.approx(1000.0 - seized)
    assert entity.internal_state.liquidation_count == 1
    assert entity.balance == pytest.approx((1000.0 - seized) * 0.9)
    # position stays usable — no latch
    entity.action_borrow(10.0)
    assert entity.internal_state.borrowed == 10.0


@pytest.mark.core
def test_liquidation_is_strictly_above_lltv():
    entity = make(0.86)
    entity.action_deposit(1000.0)
    entity.action_borrow(860.0)
    entity.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0))
    assert entity.internal_state.liquidation_count == 0
    entity.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0, borrowing_rate=1e-6))
    assert entity.internal_state.liquidation_count == 1


@pytest.mark.core
def test_bad_debt_zeroes_both_legs():
    entity = make(0.86)
    entity.action_deposit(1000.0)
    entity.action_borrow(800.0)
    entity.update_state(MorphoGlobalState(collateral_price=0.5, debt_price=1.0))
    assert entity.internal_state.collateral == 0.0
    assert entity.internal_state.borrowed == 0.0
    assert entity.balance == 0.0


@pytest.mark.core
def test_market_price_affects_balance_but_not_health():
    entity = make(0.86)
    entity.action_deposit(1000.0)
    entity.action_borrow(500.0)
    entity.update_state(MorphoGlobalState(collateral_price=0.97, debt_price=1.0, collateral_market_price=0.99))
    assert entity.ltv == pytest.approx(500 / 970)
    assert entity.collateral_value == pytest.approx(970.0)
    assert entity.collateral_market_value == pytest.approx(990.0)
    assert entity.balance == pytest.approx(990.0 - 500.0)
    entity.update_state(MorphoGlobalState(collateral_price=0.97, debt_price=1.0))
    assert entity.balance == pytest.approx(970.0 - 500.0)


@pytest.mark.core
def test_accrual_uses_taylor_and_rejects_bad_rates():
    entity = make()
    entity.action_deposit(1000.0)
    entity.action_borrow(500.0)
    x = per_bar_borrow_rate(0.09, 3600)
    entity.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0, borrowing_rate=x))
    assert entity.internal_state.borrowed == pytest.approx(500.0 * (1 + taylor_compounded(x)))
    assert entity.internal_state.collateral == 1000.0  # PT collateral earns nothing
    with pytest.raises(MorphoException, match="borrowing_rate must be >= -1"):
        entity.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0, borrowing_rate=-2))
    with pytest.raises(MorphoException, match="lending_rate must be >= -1"):
        entity.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0, lending_rate=-2))
    with pytest.raises(MorphoException, match="collateral_market_price"):
        entity.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0, collateral_market_price=-1))


@pytest.mark.core
def test_update_state_does_not_mutate_input_and_instances_are_independent():
    state = MorphoGlobalState(collateral_price=1.0, debt_price=1.0, borrowing_rate=0.01)
    a, b = MorphoEntity(), MorphoEntity()
    a.update_state(state)
    a.action_deposit(10.0)
    assert state.borrowing_rate == 0.01
    assert b.internal_state.collateral == 0.0


@pytest.mark.core
def test_liquidation_price_and_insolvency_branches():
    entity = make(0.86)
    assert math.isnan(entity.liquidation_price)
    entity.action_deposit(1000.0)
    entity.action_borrow(430.0)
    assert entity.liquidation_price == pytest.approx(430.0 / (1000.0 * 0.86))
    entity._internal_state.collateral = 0.0
    assert entity.ltv == float("inf")
    assert entity.health_factor == 0.0
    with pytest.raises(MorphoException, match="non-finite"):
        entity.calculate_repay(0.1)
