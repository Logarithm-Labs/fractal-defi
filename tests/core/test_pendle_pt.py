"""Offline tests for :class:`PendlePTEntity`.

Cover the swap fee + slippage math, the redeem invariants, the
balance / mark-to-market identity (pre- and post-expiry, with and
without underlying depeg) and the depeg-aware redeem path.
"""
import math

import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.pendle_pt import (
    PendlePTConfig,
    PendlePTEntity,
    PendlePTGlobalState,
    compute_pt_price,
)


def _open_entity(pt_price: float = 0.95, pool: float = 1_000_000.0) -> PendlePTEntity:
    e = PendlePTEntity(PendlePTConfig(amm_fee_rate=0.001, slippage_factor=0.5))
    e.update_state(
        PendlePTGlobalState(
            pt_price=pt_price,
            implied_yield=0.10,
            seconds_to_expiry=30 * 24 * 3600,
            pool_liquidity=pool,
        )
    )
    return e


@pytest.mark.core
def test_compute_pt_price_at_expiry_is_unity():
    assert compute_pt_price(0.14, 0.0) == 1.0
    assert compute_pt_price(0.14, -100.0) == 1.0


@pytest.mark.core
def test_compute_pt_price_linear_matches_formula():
    secs = 90 * 24 * 3600
    tau = secs / (365.25 * 24 * 3600)
    expected = 1.0 - 0.10 * tau
    assert math.isclose(compute_pt_price(0.10, secs, "linear"), expected, rel_tol=1e-9)


@pytest.mark.core
def test_compute_pt_price_clamped_to_unit_interval():
    five_years = 5 * 365.25 * 24 * 3600
    assert compute_pt_price(1.0, five_years, "linear") == 0.0
    assert compute_pt_price(-0.10, 30 * 24 * 3600, "linear") == 1.0


@pytest.mark.core
def test_compute_pt_price_unknown_mode_raises():
    with pytest.raises(EntityException):
        compute_pt_price(0.10, 1000.0, "log-curve")


@pytest.mark.core
def test_buy_pt_deducts_cash_and_adds_face():
    e = _open_entity()
    e.action_deposit(1000.0)
    e.action_buy_pt(1000.0)
    assert e._internal_state.cash == 0.0
    assert 0.0 < e._internal_state.pt_face_amount < 1000.0 / 0.95


@pytest.mark.core
def test_buy_pt_rejects_oversize_cash_draw():
    e = _open_entity()
    e.action_deposit(100.0)
    with pytest.raises(EntityException):
        e.action_buy_pt(1000.0)


@pytest.mark.core
def test_buy_pt_rejects_negative_amount():
    e = _open_entity()
    e.action_deposit(100.0)
    with pytest.raises(EntityException):
        e.action_buy_pt(-10.0)


@pytest.mark.core
def test_sell_pt_round_trip_loses_to_fees():
    e = _open_entity()
    e.action_deposit(1000.0)
    e.action_buy_pt(1000.0)
    face = e._internal_state.pt_face_amount
    e.action_sell_pt(face)
    assert e._internal_state.pt_face_amount == 0.0
    assert e._internal_state.cash < 1000.0


@pytest.mark.core
def test_redeem_requires_expired():
    e = _open_entity()
    e._internal_state.pt_face_amount = 100.0
    with pytest.raises(EntityException):
        e.action_redeem(100.0)


@pytest.mark.core
def test_redeem_at_expiry_pays_face_in_usdc():
    e = _open_entity()
    e._internal_state.pt_face_amount = 100.0
    e.update_state(
        PendlePTGlobalState(
            pt_price=0.5,  # caller value is overridden by update_state
            implied_yield=0.0,
            seconds_to_expiry=0.0,
            pool_liquidity=1_000_000.0,
            sy_price_in_usdc=1.0,
        )
    )
    e.action_redeem(100.0)
    assert e._internal_state.pt_face_amount == 0.0
    assert math.isclose(e._internal_state.cash, 100.0, rel_tol=1e-9)


@pytest.mark.core
def test_redeem_realises_underlying_depeg():
    e = _open_entity()
    e._internal_state.pt_face_amount = 100.0
    e.update_state(
        PendlePTGlobalState(
            pt_price=1.0,
            implied_yield=0.0,
            seconds_to_expiry=0.0,
            pool_liquidity=1_000_000.0,
            sy_price_in_usdc=0.97,
        )
    )
    e.action_redeem(100.0)
    assert math.isclose(e._internal_state.cash, 97.0, rel_tol=1e-9)


@pytest.mark.core
def test_balance_pre_expiry_uses_pt_price_only():
    e = _open_entity(pt_price=0.95)
    e._internal_state.cash = 200.0
    e._internal_state.pt_face_amount = 100.0
    # Pre-expiry: sy_price_in_usdc is irrelevant.
    assert math.isclose(e.balance, 200.0 + 100.0 * 0.95, rel_tol=1e-9)


@pytest.mark.core
def test_balance_post_expiry_marks_face_at_sy_price():
    """Post-expiry NAV must reflect underlying depeg even before redeem."""
    e = _open_entity()
    e._internal_state.cash = 50.0
    e._internal_state.pt_face_amount = 100.0
    # Snap to expired with a 4% underlying depeg.
    e.update_state(
        PendlePTGlobalState(
            pt_price=1.0,
            implied_yield=0.0,
            seconds_to_expiry=0.0,
            pool_liquidity=1_000_000.0,
            sy_price_in_usdc=0.96,
        )
    )
    # Expected: cash + face * sy_price_in_usdc = 50 + 100*0.96 = 146.0,
    # not 50 + 100*1.0 = 150.0 (the pre-fix overstatement).
    assert math.isclose(e.balance, 146.0, rel_tol=1e-9)
