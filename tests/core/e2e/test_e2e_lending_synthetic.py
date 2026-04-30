"""End-to-end synthetic walks for lending entities (Aave / SimpleLending).

Walk a lending position through 30+ daily bars with realistic interest
accrual and price moves; verify high-level invariants hold over the full
trajectory: balance finite, debt monotone-modulo-rate, liquidation
handled correctly, deposit/borrow/repay/withdraw cycle maintainable.
"""
from __future__ import annotations

import random
from typing import List

import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.aave import (AaveEntity, AaveGlobalState)
from fractal.core.entities.simple.lending import (SimpleLendingEntity,
                                                  SimpleLendingException,
                                                  SimpleLendingGlobalState)


# ============================================================ data
def _bars(seed: int = 17, n: int = 30, p_n: float = 1.0, p_p: float = 1.0,
          lend_apy: float = 0.02, borrow_apy: float = 0.05, sigma: float = 0.0):
    """Synthetic lending bars: prices walk, rates per-day from APY."""
    rng = random.Random(seed)
    bars = []
    pn, pp = p_n, p_p
    daily_lend = (1 + lend_apy) ** (1 / 365) - 1
    daily_borrow = (1 + borrow_apy) ** (1 / 365) - 1
    for _ in range(n):
        if sigma > 0:
            pn *= (1.0 + rng.gauss(0.0, sigma * 0.3))
            pp *= (1.0 + rng.gauss(0.0, sigma))
        bars.append({
            "collateral_price": pn,
            "debt_price": pp,
            "lending_rate": daily_lend,
            "borrowing_rate": daily_borrow,
        })
    return bars


# ============================================================ Aave walks
@pytest.mark.core
def test_aave_balance_finite_through_walk():
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(10_000)
    e.action_borrow(5_000)
    bars = _bars(sigma=0.005)
    for bar in bars:
        e.update_state(AaveGlobalState(**bar))
        assert e.balance == e.balance, "NaN balance"
        assert e._internal_state.collateral >= 0
        assert e._internal_state.borrowed >= 0


@pytest.mark.core
def test_aave_collateral_grows_with_lending_rate_no_price_move():
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(10_000)
    bars = _bars(lend_apy=0.05, borrow_apy=0.05, sigma=0)
    initial_coll = e._internal_state.collateral
    for bar in bars:
        e.update_state(AaveGlobalState(**bar))
    # 30 days at ~5% APY → ~0.4% growth
    assert e._internal_state.collateral > initial_coll
    expected_growth = (1 + 0.05) ** (30 / 365)
    assert e._internal_state.collateral == pytest.approx(
        initial_coll * expected_growth, rel=1e-3
    )


@pytest.mark.core
def test_aave_liquidation_triggered_by_price_move():
    """Walk price up enough that LTV crosses liq_thr."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(10_000)
    e.action_borrow(7_000)  # LTV = 0.7 (below liq_thr = 0.85)

    # Sharp debt_price spike to push LTV above 0.85
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.5))
    # New LTV ~= 7000*1.5 / (10000*1.0) = 1.05 → liquidated
    assert e._internal_state.collateral == 0
    assert e._internal_state.borrowed == 0


@pytest.mark.core
def test_aave_repay_to_target_ltv():
    """Use ``calculate_repay`` to hit a target LTV exactly."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(10_000)
    e.action_borrow(7_000)  # LTV = 0.7
    target = 0.4
    repay = e.calculate_repay(target)
    e.action_repay(repay)
    assert e.ltv == pytest.approx(target, abs=1e-9)


# ============================================================ SimpleLending walks
@pytest.mark.core
def test_simple_lending_balance_finite_through_walk():
    e = SimpleLendingEntity()
    e.update_state(SimpleLendingGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(10_000)
    e.action_borrow(5_000)
    for bar in _bars(sigma=0.005):
        e.update_state(SimpleLendingGlobalState(**bar))
        assert e.balance == e.balance
        assert e._internal_state.collateral >= 0
        assert e._internal_state.borrowed >= 0


@pytest.mark.core
def test_aave_simple_lending_walks_match_under_canonical_inputs():
    """Identical bars + canonical prices → Aave and SimpleLending stay in lock-step."""
    a = AaveEntity()
    s = SimpleLendingEntity()
    a.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    s.update_state(SimpleLendingGlobalState(collateral_price=1.0, debt_price=1.0))
    a.action_deposit(10_000)
    s.action_deposit(10_000)
    a.action_borrow(5_000)
    s.action_borrow(5_000)
    for bar in _bars(sigma=0):
        a.update_state(AaveGlobalState(**bar))
        s.update_state(SimpleLendingGlobalState(**bar))
        assert a.balance == pytest.approx(s.balance)
        assert a._internal_state.collateral == pytest.approx(s._internal_state.collateral)
        assert a._internal_state.borrowed == pytest.approx(s._internal_state.borrowed)


# ============================================================ Cycle reuse
@pytest.mark.core
def test_aave_borrow_repay_cycle_works_multiple_times():
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(10_000)
    for _ in range(5):
        e.action_borrow(3_000)
        assert e._internal_state.borrowed == 3_000
        e.action_repay(3_000)
        assert e._internal_state.borrowed == 0
    e.action_withdraw(10_000)
    assert e._internal_state.collateral == 0
