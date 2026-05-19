"""Offline tests for :class:`MorphoEntity`.

Cover the LLTV-bounded mutations, the borrow-rate accrual at hourly
granularity, the latching liquidation flag, derived health-factor /
LTV properties (including the insolvent inf branch) and the config
validation on construction.
"""
import math

import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.morpho import (
    MorphoConfig,
    MorphoEntity,
    MorphoGlobalState,
)

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def _make_market(lltv: float = 0.86) -> MorphoEntity:
    return MorphoEntity(MorphoConfig(lltv=lltv))


@pytest.mark.core
def test_initial_ltv_is_zero_without_collateral_or_debt():
    e = _make_market()
    e.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0, timestamp_seconds=0.0))
    assert e.ltv == 0.0
    assert e.health_factor == float("inf")


@pytest.mark.core
def test_deposit_then_borrow_to_80_percent_ltv():
    e = _make_market(lltv=0.86)
    e.update_state(MorphoGlobalState(collateral_price=1.0, timestamp_seconds=0.0))
    e.action_deposit(1000.0)
    e.action_borrow(800.0)
    assert math.isclose(e.ltv, 0.80, rel_tol=1e-9)
    assert math.isclose(e.health_factor, 0.86 / 0.80, rel_tol=1e-9)


@pytest.mark.core
def test_borrow_rejects_above_lltv():
    e = _make_market(lltv=0.86)
    e.update_state(MorphoGlobalState(collateral_price=1.0, timestamp_seconds=0.0))
    e.action_deposit(1000.0)
    with pytest.raises(EntityException):
        e.action_borrow(900.0)
    assert e._internal_state.debt == 0.0


@pytest.mark.core
def test_withdraw_rejects_if_pushes_above_lltv():
    e = _make_market(lltv=0.86)
    e.update_state(MorphoGlobalState(collateral_price=1.0, timestamp_seconds=0.0))
    e.action_deposit(1000.0)
    e.action_borrow(800.0)
    with pytest.raises(EntityException):
        e.action_withdraw(100.0)
    assert e._internal_state.collateral == 1000.0


@pytest.mark.core
def test_borrow_rate_accrues_over_one_hour():
    e = _make_market()
    e.update_state(MorphoGlobalState(collateral_price=1.0, borrowing_rate=0.10, timestamp_seconds=0.0))
    e.action_deposit(1000.0)
    e.action_borrow(500.0)
    e.update_state(
        MorphoGlobalState(
            collateral_price=1.0,
            borrowing_rate=0.10,
            timestamp_seconds=3600.0,
        )
    )
    expected = 500.0 * (1.0 + 0.10 / (365.25 * 24))
    assert math.isclose(e._internal_state.debt, expected, rel_tol=1e-9)


@pytest.mark.core
def test_collateral_price_drop_latches_liquidation_flag():
    e = _make_market(lltv=0.86)
    e.update_state(MorphoGlobalState(collateral_price=1.0, timestamp_seconds=0.0))
    e.action_deposit(1000.0)
    e.action_borrow(800.0)
    e.update_state(MorphoGlobalState(collateral_price=0.8, timestamp_seconds=3600.0))
    assert e.is_liquidated
    with pytest.raises(EntityException):
        e.action_deposit(100.0)


@pytest.mark.core
def test_repay_more_than_outstanding_rejected():
    e = _make_market()
    e.update_state(MorphoGlobalState(collateral_price=1.0, timestamp_seconds=0.0))
    e.action_deposit(1000.0)
    e.action_borrow(500.0)
    with pytest.raises(EntityException):
        e.action_repay(600.0)
    assert e._internal_state.debt == 500.0


@pytest.mark.core
def test_balance_is_collateral_minus_debt():
    e = _make_market()
    e.update_state(MorphoGlobalState(collateral_price=2.0, timestamp_seconds=0.0))
    e.action_deposit(100.0)
    e.action_borrow(50.0)
    assert math.isclose(e.balance, 150.0, rel_tol=1e-9)


@pytest.mark.core
def test_ltv_is_inf_when_insolvent_no_collateral_with_debt():
    """Collateral price → 0 with outstanding debt → ltv = +inf, latches liq."""
    e = _make_market(lltv=0.86)
    e.update_state(MorphoGlobalState(collateral_price=1.0, timestamp_seconds=0.0))
    e.action_deposit(1000.0)
    e.action_borrow(500.0)
    # Total collateral wipe: positive debt, zero collateral_value → ltv inf.
    e.update_state(MorphoGlobalState(collateral_price=0.0, timestamp_seconds=3600.0))
    assert e.ltv == float("inf")
    assert e.is_liquidated
    # Latched: any further action must raise.
    with pytest.raises(EntityException):
        e.action_borrow(1.0)


@pytest.mark.core
def test_config_rejects_lltv_above_one():
    with pytest.raises(EntityException):
        MorphoEntity(MorphoConfig(lltv=1.5))


@pytest.mark.core
def test_config_rejects_lltv_at_zero():
    with pytest.raises(EntityException):
        MorphoEntity(MorphoConfig(lltv=0.0))


@pytest.mark.core
def test_config_rejects_negative_lltv():
    with pytest.raises(EntityException):
        MorphoEntity(MorphoConfig(lltv=-0.1))


@pytest.mark.core
def test_config_rejects_negative_liquidation_penalty():
    with pytest.raises(EntityException):
        MorphoEntity(MorphoConfig(lltv=0.86, liquidation_penalty=-0.01))
