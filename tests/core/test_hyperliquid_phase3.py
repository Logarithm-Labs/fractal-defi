"""Lock-in tests for Hyperliquid Phase 3 cleanup fixes.

* **B4 (H-6)**: ``_clearing`` preserves the **incoming** position's
  ``max_leverage`` when a flip-direction trade leaves a remainder.
  Previously hard-coded to ``self.MAX_LEVERAGE``, silently overwriting
  per-position leverage.
* **B5 (H-5)**: ``leverage`` returns ``+inf`` when ``balance <= 0`` with
  a non-zero position, instead of negative or numerical garbage.
* **B6 (H-4)**: ``action_open_position(0)`` is a no-op — does not push
  a zero-amount position onto the list, does not call ``_clearing``.
"""
import math

import pytest

from fractal.core.entities.protocols.hyperliquid import (HyperliquidEntity,
                                                         HyperLiquidGlobalState,
                                                         HyperLiquidPosition)


# ============================================================ B6 — amount=0 short-circuit
@pytest.mark.core
def test_action_open_zero_amount_is_no_op():
    e = HyperliquidEntity(trading_fee=0.001)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    assert len(e._internal_state.positions) == 0
    e.action_open_position(0)
    assert len(e._internal_state.positions) == 0
    assert e._internal_state.collateral == 1000  # no fee charged


@pytest.mark.core
def test_action_open_zero_amount_does_not_call_clearing():
    """Repeated zero-opens should not accumulate any state mutation."""
    e = HyperliquidEntity(trading_fee=0.001)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    for _ in range(5):
        e.action_open_position(0)
    assert e._internal_state.positions == []
    assert e._internal_state.collateral == 1000


@pytest.mark.core
def test_action_open_zero_amount_after_existing_position_does_not_disturb():
    """An existing position is unchanged when zero-open is called."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    e.action_open_position(1.0)
    coll_before = e._internal_state.collateral
    n_pos_before = len(e._internal_state.positions)
    e.action_open_position(0)
    assert e._internal_state.collateral == coll_before
    assert len(e._internal_state.positions) == n_pos_before
    assert e.size == 1.0


# ============================================================ B5 — leverage edge cases
@pytest.mark.core
def test_leverage_zero_when_no_position():
    e = HyperliquidEntity()
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    assert e.leverage == 0


@pytest.mark.core
def test_leverage_zero_when_no_position_even_with_zero_balance():
    """``size == 0`` short-circuits regardless of balance."""
    e = HyperliquidEntity()
    assert e.balance == 0
    assert e.leverage == 0


@pytest.mark.core
def test_leverage_inf_when_balance_negative_with_position():
    """Long position underwater (PnL drove balance < 0) → leverage = +inf,
    not a negative ratio."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(30)
    e.action_open_position(1.0)
    # Bypass update_state (which would auto-liquidate) by mutating mark directly.
    e._global_state.mark_price = 2900
    # balance = 30 + 1 × (2900 - 3000) = -70
    assert e.balance < 0
    assert e.size == 1.0
    assert math.isinf(e.leverage) and e.leverage > 0


@pytest.mark.core
def test_leverage_inf_when_balance_zero_with_position():
    """``balance == 0`` is the boundary — also returns +inf since position
    is at-or-below maintenance margin."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(0)
    # Force a position artificially with zero collateral.
    e._internal_state.positions.append(
        HyperLiquidPosition(amount=0.5, entry_price=3000.0, max_leverage=50.0)
    )
    assert e.balance == 0
    assert e.size == 0.5
    assert math.isinf(e.leverage) and e.leverage > 0


@pytest.mark.core
def test_leverage_finite_when_balance_positive():
    """Standard case: positive balance → finite, positive leverage."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    e.action_open_position(1.0)
    assert e.leverage == 3.0  # 1 × 3000 / 1000
    assert math.isfinite(e.leverage)


# ============================================================ B4 — _clearing preserves max_leverage on flip
@pytest.mark.core
def test_clearing_preserves_incoming_max_leverage_on_full_flip():
    """Open long with one max_leverage, then a larger short with a different
    max_leverage. After clearing, the remaining short carries the **short
    order's** max_leverage, not the entity default or the long's setting.
    """
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)  # entity default 50
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    # Open long 0.5 with max_lev=20 (override per-position).
    e._internal_state.positions.append(
        HyperLiquidPosition(amount=0.5, entry_price=3000.0, max_leverage=20.0)
    )
    # Send a flipping short of -1.5 with max_lev=10.
    e._internal_state.positions.append(
        HyperLiquidPosition(amount=-1.5, entry_price=3000.0, max_leverage=10.0)
    )
    e._clearing()
    # Net: short -1.0 with the incoming order's leverage = 10.
    assert len(e._internal_state.positions) == 1
    assert e._internal_state.positions[0].amount == pytest.approx(-1.0)
    assert e._internal_state.positions[0].max_leverage == 10.0


@pytest.mark.core
def test_clearing_preserves_max_leverage_on_partial_close_no_flip():
    """Same direction (no flip): partial close — base position retains
    its own max_leverage (unchanged contract)."""
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e._internal_state.positions.append(
        HyperLiquidPosition(amount=1.0, entry_price=3000.0, max_leverage=20.0)
    )
    # Smaller opposite — partial close, no flip.
    e._internal_state.positions.append(
        HyperLiquidPosition(amount=-0.3, entry_price=3000.0, max_leverage=10.0)
    )
    e._clearing()
    # Base partially closed; max_leverage unchanged.
    assert len(e._internal_state.positions) == 1
    assert e._internal_state.positions[0].amount == pytest.approx(0.7)
    assert e._internal_state.positions[0].max_leverage == 20.0


@pytest.mark.core
def test_clearing_uses_default_max_leverage_when_no_per_position_override():
    """Default path (no per-position override): flip uses entity default
    via the incoming order's leverage (which equals self.MAX_LEVERAGE)."""
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(0.5)        # uses entity default 50
    e.action_open_position(-1.5)       # flip — also default 50
    # Net: short -1.0 with entity default leverage 50.
    assert e._internal_state.positions[0].amount == pytest.approx(-1.0)
    assert e._internal_state.positions[0].max_leverage == 50.0


# ============================================================ B4 — integration with maintenance_margin
@pytest.mark.core
def test_clearing_flipped_position_uses_remainder_max_leverage_for_mm():
    """After flip, ``maintenance_margin`` reflects the new (incoming)
    leverage tier — verifies the carry-over actually drives downstream math."""
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    # Long 0.5 max_lev=50 (MMR=0.01)
    e._internal_state.positions.append(
        HyperLiquidPosition(amount=0.5, entry_price=3000.0, max_leverage=50.0)
    )
    # Flip-short -1.5 max_lev=10 (MMR=0.05) → net short 1.0 with MMR=0.05
    e._internal_state.positions.append(
        HyperLiquidPosition(amount=-1.5, entry_price=3000.0, max_leverage=10.0)
    )
    e._clearing()
    assert e._internal_state.positions[0].max_leverage == 10.0
    # MM should now be |−1.0| × 3000 × (1/(2×10)) = 1 × 3000 × 0.05 = 150
    assert e.maintenance_margin == pytest.approx(150.0)
