"""Comprehensive liquidation tests for HyperliquidEntity.

Lock-in for the math against Hyperliquid's documented mechanics:

    maintenance_margin(p) = |size| × p × MMR
    MMR                   = 1 / (2 × max_leverage)
    Liquidation triggers when balance(p) ≤ maintenance_margin(p).

Closed-form liquidation price:

    liq = (size × entry − collateral) / (size × (1 − MMR × side))

For long  (side = +1, size > 0): liq < entry, liquidated on a price drop.
For short (side = −1, size < 0): liq > entry, liquidated on a price rise.

These tests pin down both the formula and the trigger boundary, plus
edge cases (1× leverage, just-above/just-below liq, flat entity).

References:
* https://hyperliquid.gitbook.io/hyperliquid-docs/trading/liquidations
* https://hyperliquid.gitbook.io/hyperliquid-docs/trading/margin-tiers
"""
import math

import pytest

from fractal.core.entities.protocols.hyperliquid import (HyperliquidEntity,
                                                         HyperLiquidGlobalState)


def _make(collateral: float, size: float, entry: float, max_leverage: float = 50.0):
    """Build an entity with a single position at known parameters."""
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=max_leverage)
    e.update_state(HyperLiquidGlobalState(mark_price=entry))
    e.action_deposit(collateral)
    e.action_open_position(size)
    return e


def _set_mark(entity: HyperliquidEntity, mark: float):
    """Update mark price WITHOUT triggering liquidation/funding logic."""
    entity._global_state.mark_price = mark


@pytest.mark.core
def test_liquidation_price_long_at_max_leverage_default():
    """Long 1 ETH at $3000 with $1000 collateral, MAX_LEVERAGE=50.

    MMR = 1/(2×50) = 0.01.
    liq = (1×3000 − 1000) / (1×(1−0.01)) = 2000/0.99 ≈ 2020.2020.
    """
    e = _make(collateral=1000, size=1.0, entry=3000.0, max_leverage=50.0)
    assert e.liquidation_price == pytest.approx(2000 / 0.99)


@pytest.mark.core
def test_liquidation_price_short_at_max_leverage_default():
    """Short 0.5 ETH at $3000 with $500 collateral.

    liq = (−0.5×3000 − 500) / (−0.5×(1+0.01)) = −2000 / −0.505 ≈ 3960.396.
    """
    e = _make(collateral=500, size=-0.5, entry=3000.0, max_leverage=50.0)
    assert e.liquidation_price == pytest.approx(2000 / 0.505)


@pytest.mark.core
def test_liquidation_price_long_below_entry():
    e = _make(collateral=1000, size=1.0, entry=3000.0)
    assert e.liquidation_price < 3000.0


@pytest.mark.core
def test_liquidation_price_short_above_entry():
    e = _make(collateral=500, size=-0.5, entry=3000.0)
    assert e.liquidation_price > 3000.0


@pytest.mark.core
def test_liquidation_price_no_position_is_nan():
    e = HyperliquidEntity()
    assert math.isnan(e.liquidation_price)


@pytest.mark.core
def test_liquidation_price_fully_collateralized_long_is_nonpositive():
    """Collateral ≥ size×entry → liq formula returns ≤ 0 (never liquidatable)."""
    # 1 ETH at $3000 with $3000 collateral (1× leverage at entry)
    e = _make(collateral=3000.0, size=1.0, entry=3000.0)
    # liq = (3000 − 3000)/0.99 = 0 → safe at any positive price.
    assert e.liquidation_price <= 0.0


@pytest.mark.core
def test_liquidation_triggers_at_exact_liq_price_clean_fp():
    """Boundary case with clean-fp numbers (no rounding error).

    With ``max_leverage=1`` → MMR=0.5; denominator ``size·(1 − MMR·side)``
    is exact in float for a short (side=−1, denom=−1.5), so ``liq_price``
    is exactly representable. At ``mark=liq``, ``balance`` and
    ``maintenance_margin`` are bit-exact equal → ``<=`` triggers.

    Setup: short 1 unit at entry=$3, collateral=$3 (leverage 1x = max_lev,
    H4 lets it open) → balance(p)=6−p, mm(p)=0.5p, liq=4 exactly.
    """
    e = _make(collateral=3.0, size=-1.0, entry=3.0, max_leverage=1.0)
    assert e.liquidation_price == 4.0  # clean fp
    _set_mark(e, 4.0)
    assert e._check_liquidation() is True


@pytest.mark.core
def test_liquidation_does_not_trigger_just_above_liq_price_long():
    """Long: 1 cent above liq → safe."""
    e = _make(collateral=1000, size=1.0, entry=3000.0)
    _set_mark(e, e.liquidation_price + 0.01)
    assert e._check_liquidation() is False


@pytest.mark.core
def test_liquidation_triggers_just_below_liq_price_long():
    """Long: 1 cent below liq → liquidated."""
    e = _make(collateral=1000, size=1.0, entry=3000.0)
    _set_mark(e, e.liquidation_price - 0.01)
    assert e._check_liquidation() is True


@pytest.mark.core
def test_liquidation_does_not_trigger_just_below_liq_price_short():
    """Short: 1 cent below liq → safe."""
    e = _make(collateral=500, size=-0.5, entry=3000.0)
    _set_mark(e, e.liquidation_price - 0.01)
    assert e._check_liquidation() is False


@pytest.mark.core
def test_liquidation_triggers_just_above_liq_price_short():
    """Short: 1 cent above liq → liquidated."""
    e = _make(collateral=500, size=-0.5, entry=3000.0)
    _set_mark(e, e.liquidation_price + 0.01)
    assert e._check_liquidation() is True


@pytest.mark.core
def test_long_with_strong_collateral_does_not_liquidate_on_50pct_drop():
    """Pre-fix bug demonstration: at $2500 with collateral=$2500, our old
    code would have triggered liquidation prematurely for low-leverage
    longs. Correct behaviour: still safe."""
    # 1 ETH at entry=3000, collateral=2500 → liq = (3000-2500)/0.99 ≈ 505
    e = _make(collateral=2500, size=1.0, entry=3000.0)
    assert e.liquidation_price == pytest.approx(500 / 0.99)
    _set_mark(e, 2000.0)  # 33% drop, still well above liq
    assert e._check_liquidation() is False


@pytest.mark.core
def test_short_does_not_liquidate_on_small_price_rise():
    """Short doesn't liquidate before its (higher) liq_price."""
    e = _make(collateral=500, size=-0.5, entry=3000.0)
    # liq ≈ 3960; at p=3500 should be safe
    _set_mark(e, 3500.0)
    assert e._check_liquidation() is False


@pytest.mark.core
def test_leverage_change_red_test_now_green():
    """Original P0-0.1 regression. Build position, then withdraw collateral
    until balance == maintenance_margin exactly. ``_check_liquidation``
    must return ``True`` at boundary (``<=`` semantics)."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    e.action_open_position(1)
    assert e.leverage == 3
    e.action_withdraw(500)
    assert e.leverage == 6
    e.action_withdraw(470)
    assert e.leverage == 100
    # balance == maintenance_margin == 30 → boundary triggers
    assert e.balance == pytest.approx(30)
    assert e.maintenance_margin == pytest.approx(30)
    assert e._check_liquidation() is True


@pytest.mark.core
def test_update_state_wipes_position_on_short_squeeze():
    """Original ``test_check_liquidation`` scenario."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    e.action_open_position(-0.7)
    e.update_state(HyperLiquidGlobalState(mark_price=5000))
    assert e.size == 0
    assert e._internal_state.collateral == 0


@pytest.mark.core
def test_update_state_does_not_liquidate_long_in_safe_band():
    """Long survives a 10% drop with conservative leverage."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(2000)
    e.action_open_position(1)  # 1.5x leverage at entry
    e.update_state(HyperLiquidGlobalState(mark_price=2700))  # 10% drop
    assert e.size == 1
    assert e._internal_state.collateral > 0


@pytest.mark.core
def test_action_withdraw_blocked_by_current_maintenance_margin_after_drawdown():
    """After price drops, withdrawal limit shrinks because MM is computed
    from current mark (notional shrinks)."""
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    e.action_open_position(0.1)  # tiny position; MM at entry would be ~3
    # Drop price to 2500: balance = 1000 + 0.1×(2500-3000) = 950
    e.update_state(HyperLiquidGlobalState(mark_price=2500))
    # MM(current) = 0.1 × 2500 × 0.01 = 2.5
    # Withdraw up to balance − MM: 950 − 2.5 = 947.5 should pass.
    e.action_withdraw(947.5)
    # Balance = 2.5 ≥ MM(2.5). Withdrawing more should now fail.
    with pytest.raises(Exception, match="maintenance margin"):
        e.action_withdraw(0.01)


@pytest.mark.core
def test_maintenance_margin_zero_when_no_position():
    e = HyperliquidEntity()
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    assert e.maintenance_margin == 0.0


@pytest.mark.core
def test_maintenance_margin_uses_current_mark_price():
    """MM scales linearly with current mark price (size fixed)."""
    e = _make(collateral=1000, size=1.0, entry=3000.0, max_leverage=50.0)
    # MM at entry: 1×3000×0.01 = 30
    assert e.maintenance_margin == pytest.approx(30)
    _set_mark(e, 4000.0)
    # MM at new mark: 1×4000×0.01 = 40
    assert e.maintenance_margin == pytest.approx(40)


@pytest.mark.core
@pytest.mark.parametrize("max_lev,expected_mmr", [
    (50.0, 0.01),    # BTC/ETH 50x tier
    (40.0, 0.0125),  # next tier
    (25.0, 0.02),
    (20.0, 0.025),
    (10.0, 0.05),
    (3.0, 1 / 6),
])
def test_mmr_matches_doc_formula_per_tier(max_lev, expected_mmr):
    """MMR = 1/(2×max_leverage) — matches Hyperliquid docs."""
    e = _make(collateral=1000, size=1.0, entry=3000.0, max_leverage=max_lev)
    # MM at entry = size × entry × MMR
    assert e.maintenance_margin == pytest.approx(1.0 * 3000.0 * expected_mmr)
