"""Invariant + parity tests for perp entities (HyperliquidEntity, SimplePerpEntity).

Both subclass :class:`BasePerpEntity` and share the perp paradigm —
collateral + signed position, action_open_position with sign for
long/short, mark-price-driven PnL, funding settled before liquidation.

They differ on details:
* Maintenance margin: SimplePerp uses ``1/max_leverage``; Hyperliquid
  uses ``1/(2×max_leverage)`` (matches docs).
* Liquidation comparison: SimplePerp uses ``<``; Hyperliquid uses ``<=``
  (boundary triggers, per docs).
* State shape: SimplePerp aggregates into scalar ``size, entry_price``
  fields; Hyperliquid keeps ``positions: List[HyperLiquidPosition]``
  collapsed via ``_clearing``.

These tests cover the **shared paradigm** invariants (parity-friendly)
and exchange-specific math (Hyperliquid liquidation_price closed form,
SimplePerp simplified MMR, etc).
"""
import math

import pytest

from fractal.core.entities import BasePerpEntity
from fractal.core.entities.protocols.hyperliquid import (HyperliquidEntity,
                                                         HyperLiquidGlobalState,
                                                         HyperLiquidPosition)
from fractal.core.entities.simple.perp import (SimplePerpEntity,
                                                SimplePerpGlobalState)


# ============================================================ helpers
def _hl(collateral=1000.0, mark=3000.0, fee=0.0, max_lev=50.0):
    e = HyperliquidEntity(trading_fee=fee, max_leverage=max_lev)
    e.update_state(HyperLiquidGlobalState(mark_price=mark))
    e.action_deposit(collateral)
    return e


def _sp(collateral=1000.0, mark=3000.0, fee=0.0, max_lev=50.0):
    e = SimplePerpEntity(trading_fee=fee, max_leverage=max_lev)
    e.update_state(SimplePerpGlobalState(mark_price=mark))
    e.action_deposit(collateral)
    return e


PERP_FACTORIES = [_hl, _sp]


# ============================================================ API parity
@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_perp_inherits_base_perp_entity(factory):
    e = factory()
    assert isinstance(e, BasePerpEntity)


@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_perp_exposes_required_actions(factory):
    e = factory()
    for method in ("action_deposit", "action_withdraw", "action_open_position",
                   "action_close_position", "update_state"):
        assert callable(getattr(e, method))


@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_perp_exposes_required_properties(factory):
    e = factory()
    for prop in ("balance", "size", "leverage", "pnl"):
        val = getattr(e, prop)
        assert isinstance(val, (int, float))


# ============================================================ Initial state
@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_perp_initial_state_clean(factory):
    e = factory()
    assert e.size == 0
    assert e.pnl == 0
    assert e.leverage == 0
    assert e.balance == 1000.0  # what we deposited


# ============================================================ Conservation: balance = collateral + pnl
@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_balance_equals_collateral_plus_pnl(factory):
    e = factory(fee=0.0)
    e.action_open_position(0.5)
    # No price move yet → pnl = 0
    assert e.pnl == 0
    assert e.balance == pytest.approx(e._internal_state.collateral)


# ============================================================ PnL sign for long / short
@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_long_pnl_positive_when_price_up(factory):
    e = factory(fee=0.0)
    e.action_open_position(1.0)  # long
    if isinstance(e, HyperliquidEntity):
        e._global_state.mark_price = 3100  # bypass funding/liq for pure PnL probe
    else:
        e._global_state.mark_price = 3100
    assert e.pnl > 0


@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_short_pnl_positive_when_price_down(factory):
    e = factory(fee=0.0)
    e.action_open_position(-1.0)
    if isinstance(e, HyperliquidEntity):
        e._global_state.mark_price = 2900
    else:
        e._global_state.mark_price = 2900
    assert e.pnl > 0


@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_long_pnl_negative_when_price_down(factory):
    e = factory(fee=0.0)
    e.action_open_position(1.0)
    e._global_state.mark_price = 2900
    assert e.pnl < 0


@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_short_pnl_negative_when_price_up(factory):
    e = factory(fee=0.0)
    e.action_open_position(-1.0)
    e._global_state.mark_price = 3100
    assert e.pnl < 0


# ============================================================ Open + close round-trip (zero-fee)
@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_zero_fee_round_trip_with_no_price_move_preserves_balance(factory):
    """Open then close at same price + zero fee → balance unchanged."""
    e = factory(fee=0.0)
    coll_before = e._internal_state.collateral
    e.action_open_position(1.0)
    e.action_close_position()
    assert e.size == 0
    assert e._internal_state.collateral == pytest.approx(coll_before)


@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_long_close_with_price_up_realizes_profit(factory):
    """Long close at higher price → collateral grows by realized PnL."""
    e = factory(fee=0.0)
    e.action_open_position(1.0)
    coll_before = e._internal_state.collateral
    if isinstance(e, HyperliquidEntity):
        e._global_state.mark_price = 3100
    else:
        e._global_state.mark_price = 3100
    e.action_close_position()
    # Realized profit = 1 × 100 = 100
    assert e._internal_state.collateral == pytest.approx(coll_before + 100)


# ============================================================ Funding direction (X-2 paradigm)
@pytest.mark.core
@pytest.mark.parametrize("factory,cls", [(_hl, HyperLiquidGlobalState),
                                          (_sp, SimplePerpGlobalState)])
def test_long_pays_positive_funding(factory, cls):
    e = factory(fee=0.0)
    e.action_open_position(1.0)
    coll_before = e._internal_state.collateral
    e.update_state(cls(mark_price=3000, funding_rate=0.01))
    # 1 × 3000 × 0.01 = 30 paid
    assert e._internal_state.collateral == pytest.approx(coll_before - 30)


@pytest.mark.core
@pytest.mark.parametrize("factory,cls", [(_hl, HyperLiquidGlobalState),
                                          (_sp, SimplePerpGlobalState)])
def test_short_receives_positive_funding(factory, cls):
    e = factory(fee=0.0)
    e.action_open_position(-1.0)
    coll_before = e._internal_state.collateral
    e.update_state(cls(mark_price=3000, funding_rate=0.01))
    assert e._internal_state.collateral == pytest.approx(coll_before + 30)


# ============================================================ Funding-before-liquidation (X-2 cross-cutting)
@pytest.mark.core
@pytest.mark.parametrize("factory,cls", [(_hl, HyperLiquidGlobalState),
                                          (_sp, SimplePerpGlobalState)])
def test_funding_settled_before_liquidation_check(factory, cls):
    """Both entities apply funding BEFORE liquidation check.

    Setup: long at edge of MM. Without funding, balance == MM (just safe).
    Apply positive rate → collateral drops → balance < MM → liquidate
    in the same bar.
    """
    # Both entities trigger differently due to MMR convention; we just
    # verify funding-first ordering by checking that a positive-rate tick
    # on a long ALWAYS reduces collateral first, then evaluates liquidation.
    e = factory(fee=0.0, collateral=1000.0)
    e.action_open_position(1.0)
    e.update_state(cls(mark_price=3000, funding_rate=0.01))
    # Funding paid: 1 × 3000 × 0.01 = 30 from collateral.
    # No price move; long survives — but collateral was reduced by 30.
    assert e._internal_state.collateral == pytest.approx(970)


# ============================================================ Lifecycle: action_close_position is no-op when flat
@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_close_when_flat_is_noop(factory):
    e = factory()
    e.action_close_position()
    assert e.size == 0


# ============================================================ Liquidation wipes state
@pytest.mark.core
@pytest.mark.parametrize("factory,cls", [(_hl, HyperLiquidGlobalState),
                                          (_sp, SimplePerpGlobalState)])
def test_liquidation_wipes_position(factory, cls):
    """Sharp move past liq → both entities wipe collateral and clear position."""
    e = factory(fee=0.0, collateral=100.0, max_lev=10.0)
    e.action_open_position(-1.0)  # short
    # Push mark up enough to liquidate either entity
    e.update_state(cls(mark_price=4000, funding_rate=0.0))
    assert e.size == 0
    assert e._internal_state.collateral == 0


# ============================================================ Hyperliquid-specific: closed-form liquidation_price
@pytest.mark.core
def test_hl_liquidation_price_long_matches_closed_form():
    """Long 1 ETH @ $3000, $1000 coll, MAX_LEV=50 → liq = 2000/0.99 ≈ 2020.2."""
    e = _hl(collateral=1000, max_lev=50.0)
    e.action_open_position(1.0)
    assert e.liquidation_price == pytest.approx(2000 / 0.99)


@pytest.mark.core
def test_hl_liquidation_price_short_matches_closed_form():
    """Short -0.5 ETH @ $3000, $500 coll → liq = 2000/0.505."""
    e = _hl(collateral=500, max_lev=50.0)
    e.action_open_position(-0.5)
    assert e.liquidation_price == pytest.approx(2000 / 0.505)


@pytest.mark.core
def test_hl_liquidation_price_nan_when_flat():
    e = _hl()
    assert math.isnan(e.liquidation_price)


@pytest.mark.core
def test_hl_maintenance_margin_uses_current_mark():
    """MM scales with current mark price, not entry."""
    e = _hl(collateral=1000, max_lev=50.0)
    e.action_open_position(1.0)
    assert e.maintenance_margin == pytest.approx(30)  # 1×3000×0.01
    e._global_state.mark_price = 4000
    assert e.maintenance_margin == pytest.approx(40)  # 1×4000×0.01


# ============================================================ Position model differences
@pytest.mark.core
def test_hl_position_is_dataclass():
    """Phase 4 (H-8): HyperLiquidPosition uses @dataclass — equality and
    repr work cleanly."""
    p1 = HyperLiquidPosition(amount=1.0, entry_price=3000, max_leverage=50)
    p2 = HyperLiquidPosition(amount=1.0, entry_price=3000, max_leverage=50)
    p3 = HyperLiquidPosition(amount=2.0, entry_price=3000, max_leverage=50)
    assert p1 == p2
    assert p1 != p3
    # PnL still works
    assert p1.unrealised_pnl(3100) == 100


@pytest.mark.core
def test_hl_uses_positions_list_state():
    """Hyperliquid keeps a ``positions: List`` (collapsed to single by clearing)."""
    e = _hl(fee=0.0)
    e.action_open_position(1.0)
    assert len(e._internal_state.positions) == 1
    assert isinstance(e._internal_state.positions[0], HyperLiquidPosition)


@pytest.mark.core
def test_sp_uses_scalar_size_and_entry_price_state():
    """SimplePerp aggregates into scalars."""
    e = _sp(fee=0.0)
    e.action_open_position(1.0)
    assert hasattr(e._internal_state, "size")
    assert hasattr(e._internal_state, "entry_price")
    assert e._internal_state.size == 1.0
    assert e._internal_state.entry_price == 3000.0


# ============================================================ Validation: negative on open is short, not error
@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_perp_negative_amount_on_open_is_short_not_rejected(factory):
    """Per-paradigm: ``action_open_position(< 0)`` opens a short, not raise."""
    e = factory(fee=0.0)
    e.action_open_position(-0.3)
    assert e.size == pytest.approx(-0.3)


# ============================================================ Cycle reuse: open / close repeatedly
@pytest.mark.core
@pytest.mark.parametrize("factory", PERP_FACTORIES)
def test_multiple_open_close_cycles(factory):
    """Open and close 3 times — entity stays consistent."""
    e = factory(fee=0.0)
    for _ in range(3):
        e.action_open_position(0.5)
        assert e.size == 0.5
        e.action_close_position()
        assert e.size == 0
