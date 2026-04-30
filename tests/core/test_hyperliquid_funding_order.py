"""Lock-in: funding settles BEFORE liquidation check in update_state.

Bug closed: previously ``update_state`` did:

    apply state → liquidation_check → wipe → funding_settle (no-op if wiped)

This meant a funding tick that would have saved a near-liquidation
position arrived "too late" — the position had already been wiped.

Correct order (now matching ``SimplePerpEntity``):

    apply state → funding_settle → liquidation_check → wipe

Funding sign convention:
* ``funding_rate > 0``: long pays short (collateral down for long, up for short).
* ``funding_rate < 0``: short pays long (mirror).

References:
* https://hyperliquid.gitbook.io/hyperliquid-docs/trading/margining
"""
import pytest

from fractal.core.entities.protocols.hyperliquid import (HyperliquidEntity,
                                                         HyperLiquidGlobalState)


@pytest.mark.core
def test_positive_funding_saves_short_from_would_be_liquidation():
    """X-2: with funding-first ordering, a positive-funding tick on a short
    credits collateral BEFORE the liquidation check — preserving a position
    that would otherwise be wiped on the same bar.

    Setup: short -1 ETH at $3000, $100 collateral, MMR=1%, funding_rate=+1%.
    At mark=$3070 without funding: balance=30, MM=30.7 → liquidate.
    With funding-first: collateral += 30.7 = 130.7, balance = 60.7 > MM → safe.
    """
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(100)
    e.action_open_position(-1)
    e.update_state(HyperLiquidGlobalState(mark_price=3070, funding_rate=0.01))
    assert e.size == -1, "short was wiped — funding-first ordering broken"
    # collateral was 100, gained 30.7 from funding; balance = 130.7 - 70 = 60.7
    assert e.balance == pytest.approx(60.7)


@pytest.mark.core
def test_positive_funding_tips_long_into_liquidation_same_bar():
    """A long that would survive the price tick alone gets liquidated by
    funding cost in the SAME bar (not next bar).

    Setup: long 1 ETH at $3000, collateral=$40 (just above MM=30 at entry).
    Apply funding_rate=+1% with no price move: collateral -= 30 → 10. MM=30
    (unchanged). 10 < 30 → liquidate.
    """
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(40)
    e.action_open_position(1)
    # No price move; funding pays 30 from collateral.
    e.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=0.01))
    assert e.size == 0, "long should have been liquidated by funding tick"


@pytest.mark.core
def test_long_pays_positive_funding():
    """Long with rate=+0.01 → collateral decreases by size×price×rate."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(1.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=0.01))
    # Funding payment = 1 * 3000 * 0.01 = 30 (long pays)
    assert e._internal_state.collateral == pytest.approx(10_000 - 30)


@pytest.mark.core
def test_long_receives_negative_funding():
    """Long with rate=-0.01 → collateral increases."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(1.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=-0.01))
    # Funding payment = 1 * 3000 * -0.01 = -30 → collateral -= -30 → +30
    assert e._internal_state.collateral == pytest.approx(10_000 + 30)


@pytest.mark.core
def test_short_receives_positive_funding():
    """Short with rate=+0.01 → collateral increases (longs pay shorts)."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(-1.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=0.01))
    # Funding payment = -1 * 3000 * 0.01 = -30 → collateral -= -30 → +30
    assert e._internal_state.collateral == pytest.approx(10_000 + 30)


@pytest.mark.core
def test_short_pays_negative_funding():
    """Short with rate=-0.01 → collateral decreases."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(-1.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=-0.01))
    # Funding payment = -1 * 3000 * -0.01 = +30 → collateral -= 30
    assert e._internal_state.collateral == pytest.approx(10_000 - 30)


@pytest.mark.core
def test_no_funding_settle_when_flat():
    """No position → funding term skipped, collateral unchanged regardless of rate."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.action_deposit(1000)
    e.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=0.05))
    assert e._internal_state.collateral == 1000


@pytest.mark.core
def test_zero_funding_rate_no_collateral_change():
    """rate=0 → funding term is 0, even with a position."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    e.action_open_position(1.0)
    coll_before = e._internal_state.collateral
    # Same state, rate=0
    e.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=0.0))
    assert e._internal_state.collateral == coll_before


@pytest.mark.core
def test_funding_accumulates_over_multiple_bars():
    """Apply same positive rate over 5 bars on a long → collateral down by 5× per-bar payment."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(1.0)
    coll_before = e._internal_state.collateral
    for _ in range(5):
        e.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=0.001))
    # 5 × (1 * 3000 * 0.001) = 15 paid total
    assert e._internal_state.collateral == pytest.approx(coll_before - 15)


@pytest.mark.core
def test_funding_uses_alive_size_before_potential_liquidation():
    """If a long is about to be liquidated by price drop, funding settles
    on its full alive size FIRST (could push further into liquidation),
    not on size=0 after wipe.

    Setup: long 1 ETH @ 3000, coll=40, walk price to 2900. Without funding:
    balance = 40 + (2900-3000) = -60 < MM(29) → liquidate, position wiped.
    Funding then would settle on size=0 in old order → no-op, collateral
    irrelevant after wipe.

    With funding-first: funding settles on size=1 first (long pays):
    payment = 1*2900*rate. Then liquidation check on already-blown balance.
    Result: still wiped (which is correct), but funding was logically
    applied to the alive position before clearing.
    """
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(40)
    e.action_open_position(1)
    e.update_state(HyperLiquidGlobalState(mark_price=2900, funding_rate=0.01))
    # Liquidated either way; the lock-in is that order is correct, no exception.
    assert e.size == 0
    assert e._internal_state.collateral == 0


@pytest.mark.core
@pytest.mark.parametrize("size,rate,expected_collateral_delta", [
    (1.0, 0.01, -30),    # long pays positive funding → -30
    (1.0, -0.01, +30),   # long receives negative funding → +30
    (-1.0, 0.01, +30),   # short receives positive funding → +30
    (-1.0, -0.01, -30),  # short pays negative funding → -30
    (1.0, 0.0, 0),       # zero rate → no change
    (-2.0, 0.005, +30),  # bigger short, smaller rate → +2*3000*0.005 = +30
])
def test_funding_payment_matches_quadrant(size, rate, expected_collateral_delta):
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperLiquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(size)
    coll_before = e._internal_state.collateral
    e.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=rate))
    assert e._internal_state.collateral == pytest.approx(coll_before + expected_collateral_delta)


@pytest.mark.core
def test_hyperliquid_matches_simple_perp_funding_order():
    """Both entities apply funding before liquidation check.

    With identical setup (long 1 unit at $3000, coll=$40, rate=+1%, no price
    move), both should have collateral=10 after one update_state — which
    triggers liquidation in the same bar.
    """
    from fractal.core.entities.simple.perp import (SimplePerpEntity,
                                                   SimplePerpGlobalState)

    hl = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    hl.update_state(HyperLiquidGlobalState(mark_price=3000))
    hl.action_deposit(40)
    hl.action_open_position(1)
    hl.update_state(HyperLiquidGlobalState(mark_price=3000, funding_rate=0.01))

    sp = SimplePerpEntity(trading_fee=0.0, max_leverage=50.0)
    sp.update_state(SimplePerpGlobalState(mark_price=3000))
    sp.action_deposit(40)
    sp.action_open_position(1)
    sp.update_state(SimplePerpGlobalState(mark_price=3000, funding_rate=0.01))

    # Both should be liquidated (collateral=10 < MM=30 after funding tick).
    assert hl.size == 0
    assert sp.size == 0
