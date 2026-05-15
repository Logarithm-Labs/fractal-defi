"""Offline tests for :class:`FundingHedgeEntity`.

Cover the linear PnL accrual, the long/short sign flip, and the
invariant that withdrawing notional does not unwind already-realised
funding payments.
"""
import math

import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.funding_hedge import (
    FundingHedgeConfig,
    FundingHedgeEntity,
    FundingHedgeGlobalState,
)


SECONDS_PER_YEAR = 365.25 * 24 * 3600


@pytest.mark.core
def test_first_update_seeds_timestamp_without_accrual():
    e = FundingHedgeEntity()
    e.action_deposit(10_000.0)
    e.update_state(FundingHedgeGlobalState(funding_rate=0.10, timestamp_seconds=0.0))
    assert e._internal_state.accrued_pnl == 0.0
    assert e._internal_state.last_timestamp == 0.0


@pytest.mark.core
def test_one_hour_long_accrual_matches_closed_form():
    e = FundingHedgeEntity()
    e.action_deposit(10_000.0)
    e.update_state(FundingHedgeGlobalState(funding_rate=0.10, timestamp_seconds=0.0))
    e.update_state(FundingHedgeGlobalState(funding_rate=0.10, timestamp_seconds=3600.0))
    expected = 10_000.0 * 0.10 * (3600.0 / SECONDS_PER_YEAR)
    assert math.isclose(e._internal_state.accrued_pnl, expected, rel_tol=1e-9)


@pytest.mark.core
def test_short_direction_flips_sign():
    e = FundingHedgeEntity(FundingHedgeConfig(direction="short"))
    e.action_deposit(10_000.0)
    e.update_state(FundingHedgeGlobalState(funding_rate=0.10, timestamp_seconds=0.0))
    e.update_state(FundingHedgeGlobalState(funding_rate=0.10, timestamp_seconds=3600.0))
    expected = -10_000.0 * 0.10 * (3600.0 / SECONDS_PER_YEAR)
    assert math.isclose(e._internal_state.accrued_pnl, expected, rel_tol=1e-9)


@pytest.mark.core
def test_negative_funding_rate_loses_money():
    e = FundingHedgeEntity()
    e.action_deposit(10_000.0)
    e.update_state(FundingHedgeGlobalState(funding_rate=-0.05, timestamp_seconds=0.0))
    e.update_state(FundingHedgeGlobalState(funding_rate=-0.05, timestamp_seconds=3600.0))
    assert e._internal_state.accrued_pnl < 0.0


@pytest.mark.core
def test_withdraw_keeps_realised_pnl():
    e = FundingHedgeEntity()
    e.action_deposit(10_000.0)
    e.update_state(FundingHedgeGlobalState(funding_rate=0.10, timestamp_seconds=0.0))
    e.update_state(FundingHedgeGlobalState(funding_rate=0.10, timestamp_seconds=3600.0))
    pnl_before = e._internal_state.accrued_pnl
    e.action_withdraw(10_000.0)
    assert e._internal_state.notional == 0.0
    assert e._internal_state.accrued_pnl == pnl_before


@pytest.mark.core
def test_action_deposit_rejects_negative():
    e = FundingHedgeEntity()
    with pytest.raises(EntityException):
        e.action_deposit(-1.0)


@pytest.mark.core
def test_action_withdraw_rejects_oversize():
    e = FundingHedgeEntity()
    e.action_deposit(100.0)
    with pytest.raises(EntityException):
        e.action_withdraw(500.0)


@pytest.mark.core
def test_balance_equals_accrued_pnl():
    e = FundingHedgeEntity()
    e._internal_state.accrued_pnl = 123.45
    assert e.balance == 123.45
