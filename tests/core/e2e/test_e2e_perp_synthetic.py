"""End-to-end synthetic walks for perp entities.

Walk a perp position through 30+ daily bars with realistic price walks
+ funding accrual; verify high-level invariants hold over the full
trajectory:

* balance / collateral never NaN
* funding accumulates monotonically when position survives
* liquidation triggers correctly on sharp moves
* multiple open/close cycles work with state consistent
* Hyperliquid and SimplePerp behave consistently on the shared paradigm
"""
import random

import pytest

from fractal.core.entities.protocols.hyperliquid import HyperliquidEntity, HyperliquidGlobalState
from fractal.core.entities.simple.perp import SimplePerpEntity, SimplePerpGlobalState


def _price_walk(seed=11, n=30, p0=3000.0, sigma=0.01):
    rng = random.Random(seed)
    p = p0
    out = [p0]
    for _ in range(n - 1):
        p *= 1.0 + rng.gauss(0.0, sigma)
        out.append(p)
    return out


@pytest.mark.core
def test_hl_long_survives_gentle_walk():
    """Conservative leverage + gentle walk → no liquidation, balance finite."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperliquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(1.0)  # 0.3x leverage
    for p in _price_walk(sigma=0.005):
        e.update_state(HyperliquidGlobalState(mark_price=p, funding_rate=0.0001))
        assert e.balance == e.balance, "NaN balance"
        if e.size == 0:
            pytest.fail("liquidated on a gentle walk — should not happen")


@pytest.mark.core
def test_hl_long_liquidated_by_sharp_drop():
    """High leverage + sharp drop → liquidation in same bar."""
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    e.update_state(HyperliquidGlobalState(mark_price=3000))
    e.action_deposit(100)  # tight
    e.action_open_position(1.0)  # 30x leverage
    e.update_state(HyperliquidGlobalState(mark_price=2900))  # 3.3% drop
    assert e.size == 0


@pytest.mark.core
def test_hl_long_funding_drains_collateral_over_time():
    """Long pays funding bar after bar; collateral decreases monotonically
    (until liquidation)."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperliquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(1.0)
    coll_history = [e._internal_state.collateral]
    for _ in range(20):
        e.update_state(HyperliquidGlobalState(mark_price=3000, funding_rate=0.001))
        if e.size == 0:
            break
        coll_history.append(e._internal_state.collateral)
    # Monotonically decreasing (positive funding rate, long pays)
    for i in range(1, len(coll_history)):
        assert coll_history[i] < coll_history[i - 1]


@pytest.mark.core
def test_hl_short_funding_grows_collateral_over_time():
    """Short receives positive funding → collateral grows."""
    e = HyperliquidEntity(trading_fee=0.0)
    e.update_state(HyperliquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(-1.0)
    coll_history = [e._internal_state.collateral]
    for _ in range(20):
        e.update_state(HyperliquidGlobalState(mark_price=3000, funding_rate=0.001))
        coll_history.append(e._internal_state.collateral)
    for i in range(1, len(coll_history)):
        assert coll_history[i] > coll_history[i - 1]


@pytest.mark.core
def test_hl_open_close_cycle_through_walk():
    """Open and close repeatedly through a walk — entity stays consistent."""
    e = HyperliquidEntity(trading_fee=0.001)
    e.update_state(HyperliquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    cycles = 0
    for p in _price_walk(seed=42, sigma=0.02, n=20):
        e.update_state(HyperliquidGlobalState(mark_price=p, funding_rate=0.0))
        if e.size == 0 and e._internal_state.collateral >= 100:
            e.action_open_position(0.1)
            cycles += 1
        elif e.size != 0:
            e.action_close_position()
    assert cycles >= 1
    # After all cycles, no NaN
    assert e._internal_state.collateral == e._internal_state.collateral


@pytest.mark.core
def test_hl_position_flip_through_walk():
    """Flip from long to short and back — leverage tracking holds."""
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    e.update_state(HyperliquidGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    # Long 1.0
    e.action_open_position(1.0)
    assert e.size == 1.0
    # Flip via -2.0 (closes 1.0, opens new -1.0 short)
    e.action_open_position(-2.0)
    assert e.size == pytest.approx(-1.0)
    # Should still have one position
    assert len(e._internal_state.positions) == 1


@pytest.mark.core
def test_hl_and_sp_match_pnl_through_walk():
    """Same setup, same walk, same PnL on every bar (modulo MMR/funding details)."""
    bars = _price_walk(seed=99, sigma=0.005, n=15, p0=3000)

    hl = HyperliquidEntity(trading_fee=0.0)
    sp = SimplePerpEntity(trading_fee=0.0)

    for ent, cls in ((hl, HyperliquidGlobalState), (sp, SimplePerpGlobalState)):
        ent.update_state(cls(mark_price=3000))
        ent.action_deposit(10_000)
        ent.action_open_position(1.0)

    pnls_hl, pnls_sp = [], []
    for p in bars[1:]:
        hl.update_state(HyperliquidGlobalState(mark_price=p, funding_rate=0.0))
        sp.update_state(SimplePerpGlobalState(mark_price=p, funding_rate=0.0))
        if hl.size != 0 and sp.size != 0:
            pnls_hl.append(hl.pnl)
            pnls_sp.append(sp.pnl)

    # PnL identical (positions equivalent, no fees, no funding).
    assert len(pnls_hl) == len(pnls_sp)
    for ph, ps in zip(pnls_hl, pnls_sp):
        assert ph == pytest.approx(ps)


@pytest.mark.core
def test_hl_liquidation_at_predicted_price():
    """At the moment mark_price crosses ``liquidation_price``, position is
    wiped within one ``update_state`` call."""
    e = HyperliquidEntity(trading_fee=0.0, max_leverage=50.0)
    e.update_state(HyperliquidGlobalState(mark_price=3000))
    e.action_deposit(1000)
    e.action_open_position(1.0)
    liq = e.liquidation_price
    assert liq < 3000  # long: liq below entry
    # Just below liq → liquidation
    e.update_state(HyperliquidGlobalState(mark_price=liq - 1.0))
    assert e.size == 0


@pytest.mark.core
def test_sp_long_survives_gentle_walk():
    e = SimplePerpEntity(trading_fee=0.0)
    e.update_state(SimplePerpGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    e.action_open_position(1.0)
    for p in _price_walk(sigma=0.005):
        e.update_state(SimplePerpGlobalState(mark_price=p, funding_rate=0.0))
        if e.size == 0:
            pytest.fail("SimplePerp liquidated on gentle walk")


@pytest.mark.core
def test_sp_open_close_cycle_through_walk():
    e = SimplePerpEntity(trading_fee=0.001)
    e.update_state(SimplePerpGlobalState(mark_price=3000))
    e.action_deposit(10_000)
    cycles = 0
    for p in _price_walk(seed=42, sigma=0.02, n=20):
        e.update_state(SimplePerpGlobalState(mark_price=p, funding_rate=0.0))
        if e.size == 0 and e._internal_state.collateral >= 100:
            e.action_open_position(0.1)
            cycles += 1
        elif e.size != 0:
            e.action_close_position()
    assert cycles >= 1
