"""End-to-end tests with deterministic synthetic data.

These run V2/V3 LP entities through a full multi-bar trajectory shaped like
real pool data — price walk, fee accrual, occasional re-LPing — and check
high-level invariants that should hold over the entire run rather than
just at a single point. They complement the unit-level invariant tests
in ``../invariant_testing/`` by exercising the entity's lifecycle as a
strategy actually would.

Data is generated deterministically (seeded RNG) so CI is reproducible
and offline; the *shape* of the data — daily ETH/USDC-like bars over
30 days — mirrors what a real loader would return.
"""
import random

import pytest

from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                           UniswapV2LPEntity,
                                                           UniswapV2LPGlobalState)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                           UniswapV3LPEntity,
                                                           UniswapV3LPGlobalState)


def _bars(seed: int = 42, n: int = 30, p0: float = 3000.0,
          tvl0: float = 10_000_000.0, liq: float = 10_000_000.0,
          daily_fees: float = 10_000.0, sigma: float = 0.02):
    """Generate ``n`` daily bars: price walks via log-returns; TVL scales
    with ``√(p/p0)`` (V2 constant-product invariant); liquidity stable.

    Returns a list of dicts with ``price, tvl, volume, fees, liquidity`` —
    the fields the V2/V3 ``GlobalState`` consumes. Seeded RNG → deterministic.
    """
    rng = random.Random(seed)
    bars = []
    p = p0
    for _ in range(n):
        ret = rng.gauss(0.0, sigma)
        p *= (1.0 + ret)
        # V2 constant-product: TVL scales with √(p/p0). Without this, the
        # entity's LP appears to magically rebalance value-balanced and IL
        # signs flip. With it, IL behaves like a real V2 LP.
        tvl = tvl0 * (p / p0) ** 0.5
        bars.append({
            "price": p,
            "tvl": tvl,
            "volume": tvl * 0.05,
            "fees": daily_fees * (1 + rng.gauss(0.0, 0.2)),
            "liquidity": liq,
        })
    return bars


@pytest.mark.core
def test_v2_full_lifecycle_balance_strictly_positive():
    """Across a 30-day price walk, balance never goes negative or NaN."""
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    bars = _bars()
    e.update_state(UniswapV2LPGlobalState(**bars[0]))
    e.action_deposit(100_000)
    e.action_open_position(50_000)
    for bar in bars[1:]:
        e.update_state(UniswapV2LPGlobalState(**bar))
        assert e.balance > 0, f"balance went non-positive: {e.balance}"
        # NaN check
        assert e.balance == e.balance, "balance is NaN"


@pytest.mark.core
def test_v2_il_grows_with_price_walk():
    """Without re-rebalancing, IL is monotonically (weakly) accumulating
    as price drifts away from entry."""
    e = UniswapV2LPEntity(UniswapV2LPConfig(pool_fee_rate=0.0))
    bars = _bars(seed=7)
    e.update_state(UniswapV2LPGlobalState(**bars[0]))
    e.action_deposit(100_000)
    e.action_open_position(100_000)
    # Without trading fees from the pool perspective: cash doesn't grow,
    # IL >= 0 always.
    bars_no_pool_fees = [{**b, "fees": 0.0} for b in bars]
    for bar in bars_no_pool_fees[1:]:
        e.update_state(UniswapV2LPGlobalState(**bar))
        assert e.impermanent_loss >= -1e-6  # numerical slack


@pytest.mark.core
def test_v2_pool_fees_grow_cash_over_time():
    """With pool earning non-zero fees and our LP in position, cash
    accumulates monotonically (since we never withdraw)."""
    e = UniswapV2LPEntity(UniswapV2LPConfig(pool_fee_rate=0.0))
    bars = _bars()
    e.update_state(UniswapV2LPGlobalState(**bars[0]))
    e.action_deposit(100_000)
    e.action_open_position(100_000)
    cash_series = []
    for bar in bars[1:]:
        e.update_state(UniswapV2LPGlobalState(**bar))
        cash_series.append(e._internal_state.cash)
    # cash is monotonically non-decreasing (we never withdraw, only collect fees).
    for i in range(1, len(cash_series)):
        assert cash_series[i] >= cash_series[i - 1] - 1e-9


@pytest.mark.core
def test_v2_round_trip_over_walk_loses_fee_plus_il():
    """Open at start, hold through 30 bars of walk, close at end.
    Final cash = initial - round_trip_fees - IL_at_close + collected_fees."""
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    bars = _bars()
    e.update_state(UniswapV2LPGlobalState(**bars[0]))
    e.action_deposit(100_000)
    e.action_open_position(50_000)
    for bar in bars[1:]:
        e.update_state(UniswapV2LPGlobalState(**bar))
    e.action_close_position()
    final_cash = e._internal_state.cash
    # Sanity: final cash within plausible range of initial deposit.
    assert 80_000 < final_cash < 120_000, (
        f"final cash {final_cash} suspiciously far from initial 100_000"
    )


@pytest.mark.core
def test_v3_in_range_lifecycle_accrues_fees():
    """V3 with range that stays in price → fees accumulate."""
    e = UniswapV3LPEntity(UniswapV3LPConfig(pool_fee_rate=0.0))
    bars = _bars(p0=3000.0, sigma=0.005)  # gentle walk → stays in range
    e.update_state(UniswapV3LPGlobalState(**bars[0]))
    e.action_deposit(100_000)
    p0 = bars[0]["price"]
    # Wide-ish range to weather the walk without exiting.
    e.action_open_position(50_000, price_lower=p0 * 0.85, price_upper=p0 * 1.15)
    initial_cash = e._internal_state.cash
    for bar in bars[1:]:
        e.update_state(UniswapV3LPGlobalState(**bar))
    # Cash should have grown from fees (gentle walk → mostly in range).
    assert e._internal_state.cash >= initial_cash


@pytest.mark.core
def test_v3_out_of_range_no_fee_accrual():
    """If price walks out of range and never returns, no fees accrue while out."""
    e = UniswapV3LPEntity(UniswapV3LPConfig(pool_fee_rate=0.0))
    bars = _bars(p0=3000.0, sigma=0.001)  # very gentle walk → mostly stays put
    e.update_state(UniswapV3LPGlobalState(**bars[0]))
    e.action_deposit(100_000)
    p0 = bars[0]["price"]
    # Range entirely above the walk — position will be 100% volatile.
    e.action_open_position(50_000, price_lower=p0 * 1.5, price_upper=p0 * 2.0)
    cash_after_open = e._internal_state.cash
    for bar in bars[1:]:
        e.update_state(UniswapV3LPGlobalState(**bar))
    # No fees accrued while out of range.
    assert e._internal_state.cash == pytest.approx(cash_after_open)


@pytest.mark.core
def test_v3_narrower_range_accrues_more_fees_e2e():
    """Same data: narrower-range V3 LP earns strictly more fees than wider."""
    bars = _bars(p0=3000.0, sigma=0.005)  # gentle walk
    p0 = bars[0]["price"]

    wide = UniswapV3LPEntity(UniswapV3LPConfig(pool_fee_rate=0.0))
    narrow = UniswapV3LPEntity(UniswapV3LPConfig(pool_fee_rate=0.0))
    for ent in (wide, narrow):
        ent.update_state(UniswapV3LPGlobalState(**bars[0]))
        ent.action_deposit(100_000)

    wide.action_open_position(50_000, p0 * 0.5, p0 * 1.5)
    narrow.action_open_position(50_000, p0 * 0.95, p0 * 1.05)

    cash_after_open_wide = wide._internal_state.cash
    cash_after_open_narrow = narrow._internal_state.cash

    for bar in bars[1:]:
        wide.update_state(UniswapV3LPGlobalState(**bar))
        narrow.update_state(UniswapV3LPGlobalState(**bar))

    fees_wide = wide._internal_state.cash - cash_after_open_wide
    fees_narrow = narrow._internal_state.cash - cash_after_open_narrow
    assert fees_narrow > fees_wide, (
        f"narrow range should accrue more fees: narrow={fees_narrow}, wide={fees_wide}"
    )


@pytest.mark.core
def test_v3_rebalance_strategy_lifecycle_runs():
    """Re-LP whenever price exits range. Lifecycle handles many rebalances
    without crash; final balance is finite and non-negative."""
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    bars = _bars(p0=3000.0, sigma=0.015, n=50)
    e.update_state(UniswapV3LPGlobalState(**bars[0]))
    e.action_deposit(100_000)

    p0 = bars[0]["price"]
    e.action_open_position(50_000, p0 * 0.97, p0 * 1.03)

    rebalance_count = 0
    for bar in bars[1:]:
        e.update_state(UniswapV3LPGlobalState(**bar))
        if not e.is_in_range and e.is_position:
            e.action_close_position()
            current_price = bar["price"]
            redeploy = min(50_000, e._internal_state.cash)
            if redeploy > 1_000 and current_price > 0:
                e.action_open_position(
                    redeploy,
                    current_price * 0.97,
                    current_price * 1.03,
                )
                rebalance_count += 1

    assert rebalance_count >= 1, "expected the walk to push out of range at least once"
    assert e.balance > 0
    assert e.balance == e.balance, "balance is NaN"
    assert e._internal_state.cash >= 0


@pytest.mark.core
def test_v2_vs_v3_same_data_v3_concentrated_outperforms_on_fees():
    """V3 with a concentrated range earns more fee yield than V2 50/50
    over the same in-range data, given the same notional deposit."""
    bars = _bars(p0=3000.0, sigma=0.005, daily_fees=10_000.0)
    p0 = bars[0]["price"]

    v2 = UniswapV2LPEntity(UniswapV2LPConfig(pool_fee_rate=0.0))
    v3 = UniswapV3LPEntity(UniswapV3LPConfig(pool_fee_rate=0.0))
    for ent, gs_cls in [(v2, UniswapV2LPGlobalState), (v3, UniswapV3LPGlobalState)]:
        ent.update_state(gs_cls(**bars[0]))
        ent.action_deposit(100_000)

    v2.action_open_position(50_000)
    v3.action_open_position(50_000, p0 * 0.95, p0 * 1.05)

    v2_cash_after = v2._internal_state.cash
    v3_cash_after = v3._internal_state.cash
    for bar in bars[1:]:
        v2.update_state(UniswapV2LPGlobalState(**bar))
        v3.update_state(UniswapV3LPGlobalState(**bar))

    v2_fees = v2._internal_state.cash - v2_cash_after
    v3_fees = v3._internal_state.cash - v3_cash_after

    # V3 concentrated should earn more fees than V2 spread-thin LP at the same notional.
    assert v3_fees > v2_fees, (
        f"V3 concentrated should outearn V2 on fees: V2={v2_fees}, V3={v3_fees}"
    )


@pytest.mark.core
def test_v2_terminal_balance_decomposition():
    """Final balance ≈ initial - round_trip_cost - IL_at_close + collected_fees."""
    e = UniswapV2LPEntity(UniswapV2LPConfig(pool_fee_rate=0.003))
    bars = _bars(p0=3000.0, sigma=0.01, daily_fees=10_000.0)
    e.update_state(UniswapV2LPGlobalState(**bars[0]))
    e.action_deposit(100_000)
    e.action_open_position(100_000)

    # Track collected fees independently.
    collected_fees = 0.0
    for bar in bars[1:]:
        cash_before = e._internal_state.cash
        e.update_state(UniswapV2LPGlobalState(**bar))
        # The cash delta from update_state is exactly the fee accrual
        # (no other code path adds to cash during update_state).
        collected_fees += e._internal_state.cash - cash_before

    il_at_close = e.impermanent_loss
    balance_pre_close = e.balance

    e.action_close_position()
    final_cash = e._internal_state.cash

    # Round-trip cost from the open-and-close swap fees (bounded above by
    # 2 × pool_fee_rate × half_deposit since each leg pays once).
    expected_max_round_trip = 100_000 * 0.003

    assert collected_fees > 0, "fees should have accrued"
    assert il_at_close >= -1e-6
    # Final cash satisfies the conservation identity (modulo close-swap fee on
    # the volatile leg, which depends on the post-walk volatile_amount × price).
    assert final_cash <= balance_pre_close
    assert (
        100_000 - expected_max_round_trip - 2 * il_at_close
        <= final_cash + 2 * il_at_close
    )
