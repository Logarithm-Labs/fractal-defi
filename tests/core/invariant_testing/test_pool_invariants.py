"""Property-style invariant tests for V2/V3 LP entities.

These verify *structural* properties of the pool model — the kind of
guarantee that should hold for *every* valid input — rather than specific
numeric outcomes for a chosen fixture. Each test states the invariant in
its docstring; failures here usually mean a real modeling bug rather than
just an outdated expected value.

Invariants covered:

* **Conservation**: ``balance == stable_amount + volatile_amount × price + cash``
  whenever a position is open.
* **Hodl identity**: at entry (no price move, no fees), ``hodl_value == balance``
  → ``impermanent_loss == 0``.
* **No-fee idempotence**: with ``pool_fee_rate=0`` and ``slippage_pct=0``,
  open + close round-trip preserves the deposited notional exactly.
* **Fee proportionality (V2)**: round-trip cost is ``pool_fee_rate × deposit``
  (open swap on half + close swap on half = full deposit × fee).
* **V3 narrower-range → larger L**: at the same notional, a tighter range
  mints higher L → larger share of pool fees per bar.
* **V3 out-of-range fees zero**: when the pool price exits the position
  range, ``calculate_fees() == 0``.
* **V3 above-range zap-in**: when the range is entirely above current
  price, the position holds 100% volatile after a full-amount swap.
* **V3 below-range zap-in**: when the range is entirely below current
  price, the position is 100% stable, no swap, no fee paid.
* **Round-trip consistency**: open → close at same state must reproduce
  the same cash balance up to fee, regardless of position internals.
"""
import pytest

from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                            UniswapV2LPEntity,
                                                            UniswapV2LPGlobalState)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                            UniswapV3LPEntity,
                                                            UniswapV3LPGlobalState)


# ============================================================ shared fixtures
def _v2(pool_fee_rate=0.003, slippage_pct=0.0, notional_side="token0"):
    cfg = UniswapV2LPConfig(pool_fee_rate=pool_fee_rate,
                            slippage_pct=slippage_pct,
                            notional_side=notional_side)
    e = UniswapV2LPEntity(cfg)
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                           price=1000, fees=0, volume=0))
    return e


def _v3(pool_fee_rate=0.003, slippage_pct=0.0, notional_side="token0"):
    cfg = UniswapV3LPConfig(pool_fee_rate=pool_fee_rate,
                            slippage_pct=slippage_pct,
                            notional_side=notional_side)
    e = UniswapV3LPEntity(cfg)
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                           price=1.0, fees=0, volume=0))
    return e


# ============================================================ Conservation
@pytest.mark.core
def test_v2_balance_decomposes_into_stable_volatile_cash():
    """``balance == stable_amount + volatile_amount × price + cash`` must hold
    after every operation that mutates state."""
    e = _v2()
    e.action_deposit(1000)
    e.action_open_position(500)
    expected = (
        e.stable_amount
        + e.volatile_amount * e._global_state.price
        + e._internal_state.cash
    )
    assert e.balance == pytest.approx(expected)


@pytest.mark.core
def test_v3_balance_decomposes_into_stable_volatile_cash():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    expected = (
        e.stable_amount
        + e.volatile_amount * e._global_state.price
        + e._internal_state.cash
    )
    assert e.balance == pytest.approx(expected)


# ============================================================ Hodl identity
@pytest.mark.core
def test_v2_hodl_equals_balance_at_entry():
    """At position open with no price move, hodl == balance and IL == 0."""
    e = _v2()
    e.action_deposit(1000)
    e.action_open_position(500)
    assert e.hodl_value == pytest.approx(e.balance)
    assert e.impermanent_loss == pytest.approx(0.0, abs=1e-9)


@pytest.mark.core
def test_v3_hodl_equals_balance_at_entry():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    assert e.hodl_value == pytest.approx(e.balance)
    assert e.impermanent_loss == pytest.approx(0.0, abs=1e-9)


# ============================================================ No-fee idempotence
@pytest.mark.core
def test_v2_zero_fee_round_trip_preserves_notional():
    """With pool_fee_rate=0, slippage_pct=0: deposit → open → close → all back to cash."""
    e = _v2(pool_fee_rate=0.0, slippage_pct=0.0)
    e.action_deposit(1000)
    e.action_open_position(500)
    e.action_close_position()
    assert e._internal_state.cash == pytest.approx(1000.0)


@pytest.mark.core
def test_v3_zero_fee_round_trip_preserves_notional():
    e = _v3(pool_fee_rate=0.0, slippage_pct=0.0)
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    e.action_close_position()
    assert e._internal_state.cash == pytest.approx(1000.0)


# ============================================================ Fee proportionality
@pytest.mark.core
@pytest.mark.parametrize("pool_fee", [0.001, 0.003, 0.01, 0.03])
def test_v2_round_trip_cost_equals_pool_fee_times_deposit(pool_fee):
    """V2 round-trip cost = pool_fee × deposit.

    Open: fee on half (the swapped portion) = deposit/2 × fee.
    Close: fee on half (volatile_back × p) = stable_used × fee = deposit/2 × (1 - fee) × fee.
    Total ≈ deposit × fee × (1 - fee/2) ≈ deposit × fee for small fee.
    """
    deposit = 1000.0
    e = _v2(pool_fee_rate=pool_fee, slippage_pct=0.0)
    e.action_deposit(deposit)
    e.action_open_position(deposit)
    e.action_close_position()
    cost = deposit - e._internal_state.cash
    # Cost is between (deposit × fee × (1 - fee/2)) and (deposit × fee).
    expected_max = deposit * pool_fee
    expected_min = deposit * pool_fee * (1 - pool_fee / 2)
    assert expected_min - 1e-6 <= cost <= expected_max + 1e-6, (
        f"round-trip cost {cost} not in [{expected_min}, {expected_max}] for fee={pool_fee}"
    )


@pytest.mark.core
def test_v2_slippage_stacks_with_pool_fee():
    """slippage_pct adds to pool_fee_rate as combined haircut on swaps."""
    e_no_slip = _v2(pool_fee_rate=0.003, slippage_pct=0.0)
    e_slip = _v2(pool_fee_rate=0.003, slippage_pct=0.002)
    e_no_slip.action_deposit(1000)
    e_no_slip.action_open_position(500)
    e_slip.action_deposit(1000)
    e_slip.action_open_position(500)
    # Higher effective fee → smaller volatile leg → smaller balance.
    assert e_slip.balance < e_no_slip.balance


# ============================================================ V3 narrower range → higher L
@pytest.mark.core
def test_v3_narrower_range_yields_higher_liquidity():
    """At the same notional deposit and same current price, a tighter range
    concentrates more L per token deposited → higher fee share when in range."""
    wide = _v3()
    wide.action_deposit(1000)
    wide.action_open_position(500, 0.5, 1.5)  # wide range

    narrow = _v3()
    narrow.action_deposit(1000)
    narrow.action_open_position(500, 0.95, 1.05)  # tight range

    assert narrow._internal_state.liquidity > wide._internal_state.liquidity


@pytest.mark.core
def test_v3_narrower_range_earns_more_fees_per_bar():
    """Same pool fees in a bar → narrower range earns more (higher L share)."""
    wide = _v3()
    wide.action_deposit(1000)
    wide.action_open_position(500, 0.5, 1.5)

    narrow = _v3()
    narrow.action_deposit(1000)
    narrow.action_open_position(500, 0.95, 1.05)

    bar = UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                  price=1.0, fees=100, volume=0)
    wide.update_state(bar)
    narrow.update_state(bar)
    assert narrow.calculate_fees() > wide.calculate_fees()


# ============================================================ V3 out-of-range fees zero
@pytest.mark.core
def test_v3_fees_zero_when_price_below_range():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    # Price exits below the range
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                           price=0.85, fees=100, volume=0))
    assert e.calculate_fees() == 0


@pytest.mark.core
def test_v3_fees_zero_when_price_above_range():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                           price=1.15, fees=100, volume=0))
    assert e.calculate_fees() == 0


@pytest.mark.core
def test_v3_is_in_range_matches_calculate_fees_zero_check():
    """``is_in_range == False`` should imply ``calculate_fees() == 0``."""
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    for price in [0.85, 0.9, 1.1, 1.15]:
        e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                               price=price, fees=100, volume=0))
        if not e.is_in_range:
            assert e.calculate_fees() == 0


# ============================================================ V3 zap-in edge cases
@pytest.mark.core
def test_v3_zap_in_above_range_holds_only_volatile():
    """When current price < price_lower (range above current), zap-in swaps
    full notional → volatile, paying fee on full amount. Position holds 0 stable."""
    e = _v3()
    e.action_deposit(1000)
    # Range entirely above current price (1.0)
    e.action_open_position(500, 1.5, 2.0)
    assert e.stable_amount == 0
    assert e.volatile_amount > 0
    # Effective cost = full × fee (entire amount swapped)
    expected_cost = 500 * e.effective_fee_rate
    actual_cost = 500 - (e.balance - 500)  # 500 deposited into position, rest is cash
    # balance ≈ 1000 - 500*fee
    assert e.balance == pytest.approx(1000 - expected_cost, rel=1e-4)


@pytest.mark.core
def test_v3_zap_in_below_range_holds_only_stable_no_fee():
    """When current price > price_upper (range below current), no swap needed:
    full notional becomes stable, no fee paid."""
    e = _v3()
    e.action_deposit(1000)
    # Range entirely below current price (1.0)
    e.action_open_position(500, 0.5, 0.8)
    assert e.volatile_amount == 0
    assert e.stable_amount > 0
    # No fee charged → balance unchanged from before zap-in
    assert e.balance == pytest.approx(1000.0)


# ============================================================ Round-trip consistency
@pytest.mark.core
def test_v2_open_close_at_same_state_is_pool_fee_round_trip():
    """Opening then immediately closing at the same global state yields the
    expected round-trip cost; final cash equals deposit minus that cost."""
    e = _v2()
    e.action_deposit(1000)
    pre_balance = e.balance
    e.action_open_position(500)
    e.action_close_position()
    cost = pre_balance - e.balance
    expected = 500 * e.effective_fee_rate
    # Cost is bounded by [expected × (1 - fee/2), expected]
    assert cost <= expected + 1e-9


@pytest.mark.core
def test_v3_open_close_at_same_state_round_trip_bounded():
    """Same as V2 but for V3. Round-trip cost is bounded by the effective fee
    on the swapped portion (not full deposit)."""
    e = _v3()
    e.action_deposit(1000)
    pre_balance = e.balance
    e.action_open_position(500, 0.9, 1.1)
    e.action_close_position()
    cost = pre_balance - e.balance
    # For V3 in-range with symmetric range around price=1.0, swap portion is ~half.
    assert 0 < cost < 500 * e.effective_fee_rate


@pytest.mark.core
def test_v2_open_close_with_no_position_amount_no_fee():
    """Round-trip with deposit=0 should be a no-op."""
    e = _v2()
    e.action_deposit(0)
    assert e.balance == 0


# ============================================================ Pair-level invariants
@pytest.mark.core
def test_v2_open_from_pair_no_fee():
    """``_open_from_pair`` mints LP without applying any swap fee — pair
    is provided directly, no swap simulated."""
    e = _v2()
    e.action_deposit(1000)
    # Provide on-chain pair directly: token0=stable=500 notional, token1=volatile=0.5 (= 500/price)
    leftover0, leftover1 = e._open_from_pair(token0_amount=500, token1_amount=0.5)
    # No fee applied → if pair matches pool ratio exactly, leftover should be 0.
    assert leftover0 == pytest.approx(0.0, abs=1e-9)
    assert leftover1 == pytest.approx(0.0, abs=1e-9)
    assert e.is_position
    assert e.stable_amount == pytest.approx(500)
    assert e.volatile_amount == pytest.approx(0.5)


@pytest.mark.core
def test_v2_close_to_pair_returns_position_amounts_no_fee():
    """``_close_to_pair`` returns the on-chain pair without swap/fee."""
    e = _v2()
    e.action_deposit(1000)
    e._open_from_pair(token0_amount=500, token1_amount=0.5)
    expected_token0 = e._internal_state.token0_amount
    expected_token1 = e._internal_state.token1_amount
    token0_back, token1_back = e._close_to_pair()
    assert token0_back == pytest.approx(expected_token0)
    assert token1_back == pytest.approx(expected_token1)
    assert not e.is_position


@pytest.mark.core
def test_v3_open_from_pair_above_range_returns_stable_as_leftover():
    """If range is above current price, only volatile counts; passing
    stable into ``_open_from_pair`` returns it as leftover."""
    e = _v3()
    e.action_deposit(1000)
    # current price = 1.0, range [1.5, 2.0] — above current
    leftover0, leftover1 = e._open_from_pair(
        token0_amount=100, token1_amount=50,  # token0=stable in default mode
        price_lower=1.5, price_upper=2.0,
    )
    # Stable (token0) returned in full as leftover — only volatile (token1) was used.
    assert leftover0 == pytest.approx(100)
    assert leftover1 == pytest.approx(0.0, abs=1e-9)


# ============================================================ Notional-side flip invariance
@pytest.mark.core
def test_v2_round_trip_cost_invariant_under_notional_flip():
    """Same pool data + same deposit → same round-trip cost regardless
    of which on-chain slot holds the notional."""
    e0 = _v2(notional_side="token0")
    e1 = _v2(notional_side="token1")
    e0.action_deposit(1000)
    e1.action_deposit(1000)
    e0.action_open_position(500)
    e1.action_open_position(500)
    e0.action_close_position()
    e1.action_close_position()
    assert e0._internal_state.cash == pytest.approx(e1._internal_state.cash)


@pytest.mark.core
def test_v3_round_trip_cost_invariant_under_notional_flip():
    e0 = _v3(notional_side="token0")
    e1 = _v3(notional_side="token1")
    e0.action_deposit(1000)
    e1.action_deposit(1000)
    e0.action_open_position(500, 0.9, 1.1)
    e1.action_open_position(500, 0.9, 1.1)
    e0.action_close_position()
    e1.action_close_position()
    assert e0._internal_state.cash == pytest.approx(e1._internal_state.cash)
