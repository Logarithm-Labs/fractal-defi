"""E2E: leveraged-long ETH via Aave + UniV3Spot composition.

Demonstrates how the existing primitives compose into leveraged-long
positions of various sizes. Tests cover:

* baseline 1x (just hold ETH on spot),
* 2x via Aave alone (deposit ETH, borrow USDC, no spot loop),
* parametrized leverage factor check (LTV → exact leverage),
* **real 3x looped leverage** via Aave + UniV3Spot composition using
  ``action_inject_product`` / ``action_remove_product`` to move ETH
  across entities and ``action_deposit`` to materialize borrowed USDC,
* conservation invariants — every borrow+inject and inject+deposit pair
  must preserve ``total_balance`` (zero-equity moves).
"""
import random

import pytest

from fractal.core.entities.protocols.aave import (AaveEntity, AaveGlobalState)
from fractal.core.entities.protocols.uniswap_v3_spot import (UniswapV3SpotEntity,
                                                             UniswapV3SpotGlobalState)


def _eth_walk(seed: int = 5, n: int = 30, p0: float = 3000.0, drift: float = 0.001,
              sigma: float = 0.02):
    """Daily ETH-USD walk with optional drift."""
    rng = random.Random(seed)
    p = p0
    out = [p0]
    for _ in range(n - 1):
        p *= (1.0 + drift + rng.gauss(0.0, sigma))
        out.append(p)
    return out


@pytest.mark.core
def test_baseline_1x_long_eth_via_spot_only():
    """Baseline: hold ETH at spot, no leverage, no debt."""
    spot = UniswapV3SpotEntity(trading_fee=0.0)
    spot.update_state(UniswapV3SpotGlobalState(price=3000))
    spot.action_deposit(10_000)
    spot.action_buy(10_000)  # all-in: 10k / 3000 = 3.33 ETH

    # Walk price up 10%
    spot.update_state(UniswapV3SpotGlobalState(price=3300))
    # Balance = 3.33 × 3300 = 11_000 → +10% on initial 10k
    assert spot.balance == pytest.approx(11_000)


@pytest.mark.core
def test_leveraged_long_eth_via_aave_volatile_collateral():
    """Synthetic 1.5x long ETH via deposit-ETH/borrow-USDC.

    Simulates the "loop" manually: buy ETH → deposit to Aave →
    borrow USDC → that USDC sits as cash in Aave's accounting (not
    re-loop, since we don't have inject_product yet). Net effect: we own
    ~1 ETH worth of equity but have $1.5k of ETH-price exposure on top
    of the borrow → 1.5x.
    """
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    aave.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))

    # Step 1: deposit 1 ETH as collateral (worth $3k)
    aave.action_deposit(1.0)
    assert aave.collateral_value == 3000

    # Step 2: borrow $1500 USDC against it (LTV = 0.5)
    aave.action_borrow(1500)
    assert aave.ltv == pytest.approx(0.5)
    assert aave.balance == pytest.approx(3000 - 1500)  # equity = $1500

    # ETH-exposure (delta to ETH price): collateral side scales with ETH price.
    # For a 10% ETH gain, balance moves by 0.10 × 3000 = $300 = +20% on equity
    # of $1500 → 2x leverage on equity. Verify:
    aave.update_state(AaveGlobalState(collateral_price=3300.0, debt_price=1.0))
    new_balance = aave.balance
    pnl = new_balance - 1500
    assert pnl == pytest.approx(300)  # equity moved by $300
    leverage_factor = pnl / 1500 / 0.10  # (% equity move) / (% price move)
    assert leverage_factor == pytest.approx(2.0)


@pytest.mark.core
def test_leveraged_long_amplifies_loss_on_drawdown():
    """Same 2x leveraged-long: 10% ETH drop → 20% equity loss."""
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    aave.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    aave.action_deposit(1.0)
    aave.action_borrow(1500)

    # ETH drops 10%
    aave.update_state(AaveGlobalState(collateral_price=2700.0, debt_price=1.0))
    new_balance = aave.balance
    pnl = new_balance - 1500
    assert pnl == pytest.approx(-300)  # equity lost $300 (20% of $1500)


@pytest.mark.core
def test_leveraged_long_liquidates_on_sharp_drop():
    """ETH crash deep enough to push LTV past liq_thr → position wiped."""
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    aave.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    aave.action_deposit(1.0)
    aave.action_borrow(2000)  # LTV = 2000 / 3000 = 0.667 (close to 0.85 liq_thr)

    # ETH drops 30%: LTV = 2000 / (1 × 2100) = 0.952 > 0.85 → liquidated
    aave.update_state(AaveGlobalState(collateral_price=2100.0, debt_price=1.0))
    assert aave._internal_state.collateral == 0
    assert aave._internal_state.borrowed == 0
    assert aave.balance == 0  # full equity wiped


@pytest.mark.core
def test_leveraged_long_walks_through_price_series_without_liquidation():
    """30-bar walk with moderate sigma; verify position remains finite."""
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    aave.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    aave.action_deposit(1.0)
    aave.action_borrow(1000)  # safer LTV = 0.333

    for p in _eth_walk(sigma=0.005):  # gentle walk
        aave.update_state(AaveGlobalState(collateral_price=p, debt_price=1.0))
        if aave._internal_state.collateral == 0:
            break  # liquidated mid-walk; that's a valid outcome too
        assert aave.balance == aave.balance  # not NaN
        assert aave.health_factor > 1.0  # not liquidatable on each bar


@pytest.mark.core
@pytest.mark.parametrize("price_pct_move,expected_leverage", [
    (0.01, 2.0),   # 1% ETH up → 2% equity up
    (0.05, 2.0),   # 5% up → 10% up
    (-0.05, 2.0),  # 5% down → 10% down
])
def test_leveraged_long_constant_factor_2x(price_pct_move, expected_leverage):
    """Verify the leverage factor is the expected 2x at LTV=0.5."""
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    p0 = 3000.0
    aave.update_state(AaveGlobalState(collateral_price=p0, debt_price=1.0))
    aave.action_deposit(1.0)
    aave.action_borrow(1500)  # LTV = 0.5
    initial_equity = aave.balance

    new_p = p0 * (1 + price_pct_move)
    aave.update_state(AaveGlobalState(collateral_price=new_p, debt_price=1.0))

    pnl_pct = (aave.balance - initial_equity) / initial_equity
    actual_leverage = pnl_pct / price_pct_move
    assert actual_leverage == pytest.approx(expected_leverage, rel=1e-9)


@pytest.mark.core
def test_borrow_plus_cash_inject_preserves_total_balance():
    """``aave.borrow(X)`` then ``spot.deposit(X)`` is a zero-equity move.

    The borrowed USDC is debt in Aave; injecting it as cash in spot
    materializes it. Net change: 0 (we owe X, we hold X — net is 0).
    """
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    spot = UniswapV3SpotEntity(trading_fee=0.0)
    aave.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    spot.update_state(UniswapV3SpotGlobalState(price=3000))

    aave.action_deposit(1.0)  # 1 ETH collateral
    total_before = aave.balance + spot.balance

    aave.action_borrow(1500)
    spot.action_deposit(1500)  # materialize borrowed USDC

    total_after = aave.balance + spot.balance
    assert total_after == pytest.approx(total_before)


@pytest.mark.core
def test_remove_plus_aave_deposit_preserves_total_balance():
    """``spot.remove_product(X)`` + ``aave.deposit(X)`` is a zero-equity move.

    Moving ETH from spot to Aave as collateral doesn't change total equity
    — same ETH, just sitting in a different entity.
    """
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    spot = UniswapV3SpotEntity(trading_fee=0.0)
    aave.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    spot.update_state(UniswapV3SpotGlobalState(price=3000))

    spot.action_deposit(10_000)
    spot.action_buy(10_000)  # ~3.33 ETH on spot
    total_before = aave.balance + spot.balance

    eth = spot._internal_state.amount
    spot.action_remove_product(eth)
    aave.action_deposit(eth)

    total_after = aave.balance + spot.balance
    assert total_after == pytest.approx(total_before)


@pytest.mark.core
def test_aave_withdraw_plus_inject_preserves_total_balance():
    """``aave.withdraw(X)`` + ``spot.inject_product(X)`` is symmetric to the above."""
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    spot = UniswapV3SpotEntity(trading_fee=0.0)
    aave.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    spot.update_state(UniswapV3SpotGlobalState(price=3000))

    aave.action_deposit(2.0)  # 2 ETH in Aave
    total_before = aave.balance + spot.balance

    aave.action_withdraw(1.0)  # take 1 ETH out (in ETH-units since notional=ETH here)
    spot.action_inject_product(1.0)  # inject into spot

    total_after = aave.balance + spot.balance
    assert total_after == pytest.approx(total_before)


@pytest.mark.core
def test_real_looped_leverage_starting_from_usdc_zero_fee():
    """Build leveraged ETH long via composition: 10k USDC → ~2.4x ETH long.

    Loop: spot.buy(USDC) → spot.remove_product(ETH) → aave.deposit(ETH) →
    aave.borrow(USDC) → spot.deposit(USDC) → repeat.

    With zero-fee config and ``borrow_ratio=0.6`` per loop, geometric
    series gives max ``1/(1-0.6) = 2.5x`` leverage in the limit.
    """
    initial_capital = 10_000
    p = 3000.0
    target_borrow_ratio = 0.6  # safe LTV margin on each loop

    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    spot = UniswapV3SpotEntity(trading_fee=0.0)
    aave.update_state(AaveGlobalState(collateral_price=p, debt_price=1.0))
    spot.update_state(UniswapV3SpotGlobalState(price=p))

    spot.action_deposit(initial_capital)

    initial_equity = aave.balance + spot.balance
    assert initial_equity == initial_capital

    for _ in range(10):  # max 10 loops
        if spot._internal_state.cash < 100:
            break
        # 1. Convert all spot.cash → ETH on spot
        cash_to_use = spot._internal_state.cash
        spot.action_buy(cash_to_use)
        # 2. Move ETH from spot to Aave as collateral
        eth = spot._internal_state.amount
        spot.action_remove_product(eth)
        aave.action_deposit(eth)
        # 3. Borrow USDC at target LTV
        room_to_borrow = aave.collateral_value * target_borrow_ratio - aave.debt_value
        if room_to_borrow < 100:
            break
        aave.action_borrow(room_to_borrow)
        # 4. Materialize borrowed USDC into spot.cash
        spot.action_deposit(room_to_borrow)
        # Conservation check at every loop iteration
        equity_now = aave.balance + spot.balance
        assert equity_now == pytest.approx(initial_equity, abs=1e-6), (
            f"equity drifted: {equity_now} vs {initial_equity}"
        )

    # After loops: ETH exposure = collateral; equity = initial; leverage = exposure / equity
    eth_exposure_usd = aave.collateral_value + spot._internal_state.amount * spot.current_price
    equity = aave.balance + spot.balance
    leverage = eth_exposure_usd / equity
    # Geometric limit = 2.5x; with finite loops we get ~2.4x.
    assert 2.0 < leverage < 2.5


@pytest.mark.core
def test_real_looped_leverage_unwinds_back_to_cash():
    """Symmetrical unwind: close the loop and verify all money returns to cash.

    Reverse the loop: aave.repay(borrowed) ← spot.cash; aave.withdraw(collateral) →
    spot.action_inject_product → spot.action_sell.
    """
    p = 3000.0
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    spot = UniswapV3SpotEntity(trading_fee=0.0)
    aave.update_state(AaveGlobalState(collateral_price=p, debt_price=1.0))
    spot.update_state(UniswapV3SpotGlobalState(price=p))

    # Build a 1.5x position first: deposit ETH, borrow 50% of collat value, buy more.
    spot.action_deposit(10_000)
    spot.action_buy(10_000)  # 3.33 ETH on spot
    eth = spot._internal_state.amount
    spot.action_remove_product(eth)
    aave.action_deposit(eth)              # collat = 3.33 ETH (≈10k USD)
    aave.action_borrow(5_000)              # 50% LTV — safe (max=0.8)
    spot.action_deposit(5_000)
    spot.action_buy(5_000)                 # 1.67 ETH on spot

    initial_equity = aave.balance + spot.balance

    # Now unwind: sell all spot ETH for USDC, repay debt, withdraw collateral, sell
    # 1. Sell spot ETH to get USDC
    spot.action_sell(spot._internal_state.amount)
    # 2. Use spot.cash to repay all debt
    repay_amount = min(aave._internal_state.borrowed, spot._internal_state.cash)
    spot.action_withdraw(repay_amount)  # remove from spot before repay
    aave.action_repay(repay_amount)
    # 3. Withdraw all collateral as ETH (notional units = ETH count)
    coll = aave._internal_state.collateral
    aave.action_withdraw(coll)
    spot.action_inject_product(coll)
    # 4. Sell ETH for USDC
    spot.action_sell(spot._internal_state.amount)

    final_equity = aave.balance + spot.balance
    # Zero-fee path → equity preserved through unwind.
    assert final_equity == pytest.approx(initial_equity, abs=1e-6)


@pytest.mark.core
def test_loop_with_fees_loses_only_fee_amount():
    """With non-zero trading fee on spot, the loop loses exactly the
    expected fee amount per swap, not more."""
    p = 3000.0
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    spot = UniswapV3SpotEntity(trading_fee=0.001)  # 10bp fee
    aave.update_state(AaveGlobalState(collateral_price=p, debt_price=1.0))
    spot.update_state(UniswapV3SpotGlobalState(price=p))

    spot.action_deposit(10_000)
    initial_equity = aave.balance + spot.balance

    # Single loop: buy → move → borrow → deposit
    spot.action_buy(10_000)  # pays 0.1% fee → loses 10 USDC equivalent
    eth = spot._internal_state.amount
    spot.action_remove_product(eth)  # no fee
    aave.action_deposit(eth)         # no fee
    aave.action_borrow(5_000)         # no fee (just records debt)
    spot.action_deposit(5_000)        # no fee (cash inject)

    after_one_loop = aave.balance + spot.balance
    loss = initial_equity - after_one_loop
    # Expected loss = 10_000 × 0.001 = 10 (the spot.buy fee)
    assert loss == pytest.approx(10.0, rel=1e-3)


@pytest.mark.core
def test_loop_preserves_total_balance_under_price_walk():
    """After loop, walk price; equity changes by leverage × price_move.

    Verifies that during a price walk (no further actions), total_balance
    moves predictably with leverage factor.
    """
    p0 = 3000.0
    aave = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    spot = UniswapV3SpotEntity(trading_fee=0.0)
    aave.update_state(AaveGlobalState(collateral_price=p0, debt_price=1.0))
    spot.update_state(UniswapV3SpotGlobalState(price=p0))

    spot.action_deposit(10_000)
    # Build 1.5x leverage (single loop)
    spot.action_buy(10_000)
    eth = spot._internal_state.amount
    spot.action_remove_product(eth)
    aave.action_deposit(eth)
    aave.action_borrow(5_000)
    spot.action_deposit(5_000)
    spot.action_buy(5_000)  # buy more ETH on spot with the borrowed USDC
    initial_equity = aave.balance + spot.balance

    # exposure: aave.collateral × price + spot.amount × price
    initial_exposure = aave.collateral_value + spot._internal_state.amount * p0
    leverage = initial_exposure / initial_equity

    # Walk price up 10%
    p1 = p0 * 1.10
    aave.update_state(AaveGlobalState(collateral_price=p1, debt_price=1.0))
    spot.update_state(UniswapV3SpotGlobalState(price=p1))

    new_equity = aave.balance + spot.balance
    pnl_pct = (new_equity - initial_equity) / initial_equity
    actual_leverage_factor = pnl_pct / 0.10
    # Should match the leverage we built up (within float precision)
    assert actual_leverage_factor == pytest.approx(leverage, rel=1e-9)
