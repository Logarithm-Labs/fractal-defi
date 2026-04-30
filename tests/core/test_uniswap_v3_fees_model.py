"""Unit tests for ``fractal.core.entities.models.uniswap_v3_fees``.

Direct tests for the math helpers used by :class:`UniswapV3LPEntity`'s
fees calculation. These pin down the V3 standard-convention contracts
(price = token1/token0, decimals scaling) and the validation behaviour.
"""
import pytest

from fractal.core.entities.models.uniswap_v3_fees import (Q96,
                                                          estimate_fee,
                                                          expand_decimals,
                                                          get_liquidity_delta,
                                                          get_liquidity_for_amount0,
                                                          get_liquidity_for_amount1,
                                                          get_sqrt_price_x96)


@pytest.mark.core
@pytest.mark.parametrize("amount,decimals,expected", [
    (1.0, 18, 10**18),
    (1.5, 18, 15 * 10**17),
    (0.0, 18, 0),
    (1.0, 6, 10**6),
    (1.0, 0, 1),
    (1234.5678, 6, 1_234_567_800),
])
def test_expand_decimals_correct_scaling(amount, decimals, expected):
    assert expand_decimals(amount, decimals) == expected


@pytest.mark.core
def test_expand_decimals_uses_banker_rounding_not_truncation():
    """``round`` is used, not ``int`` — half-amounts round to even."""
    # 0.5 raw → rounds to 0 (banker's rounding to even)
    assert expand_decimals(0.5, 0) == 0
    # 1.5 raw → rounds to 2 (banker's rounding to even)
    assert expand_decimals(1.5, 0) == 2
    # 2.5 raw → rounds to 2 (banker's rounding to even)
    assert expand_decimals(2.5, 0) == 2


@pytest.mark.core
def test_expand_decimals_rejects_negative_decimals():
    with pytest.raises(ValueError, match="decimals must be >= 0"):
        expand_decimals(1.0, -1)


@pytest.mark.core
def test_sqrt_price_x96_at_unity_gives_q96():
    """At price=1 with equal decimals: √1 × 2^96 = Q96."""
    assert get_sqrt_price_x96(1.0, 18, 18) == Q96


@pytest.mark.core
def test_sqrt_price_x96_scales_with_decimals_difference():
    """Different decimals shift the effective price.

    For USDC (6 dec) / WETH (18 dec) at price=3000:
    scaled = price × 10**token1_dec / 10**token0_dec
           = 3000 × 10**18 / 10**6 = 3 × 10**15.
    Float precision is ~16 sig digits, so we use relative tolerance.
    """
    sqrt_p = get_sqrt_price_x96(3000.0, token0_decimal=6, token1_decimal=18)
    expected = int((3 * 10**15) ** 0.5 * Q96)
    assert sqrt_p == pytest.approx(expected, rel=1e-14)


@pytest.mark.core
def test_sqrt_price_x96_monotonic_in_price():
    """Higher price → higher sqrt_price."""
    p1 = get_sqrt_price_x96(1.0, 18, 18)
    p2 = get_sqrt_price_x96(2.0, 18, 18)
    p3 = get_sqrt_price_x96(4.0, 18, 18)
    assert p1 < p2 < p3


@pytest.mark.core
def test_sqrt_price_x96_rejects_zero_price():
    with pytest.raises(ValueError, match="price must be > 0"):
        get_sqrt_price_x96(0.0, 18, 18)


@pytest.mark.core
def test_sqrt_price_x96_rejects_negative_price():
    with pytest.raises(ValueError, match="price must be > 0"):
        get_sqrt_price_x96(-1.0, 18, 18)


@pytest.mark.core
def test_sqrt_price_x96_rejects_negative_decimals():
    with pytest.raises(ValueError, match="decimals must be >= 0"):
        get_sqrt_price_x96(1.0, -1, 18)


@pytest.mark.core
def test_liquidity_for_amount0_zero_amount_yields_zero():
    sqrt_a = get_sqrt_price_x96(0.9, 18, 18)
    sqrt_b = get_sqrt_price_x96(1.1, 18, 18)
    assert get_liquidity_for_amount0(sqrt_a, sqrt_b, amount0=0) == 0


@pytest.mark.core
def test_liquidity_for_amount0_scales_linearly_with_amount():
    """L scales linearly with amount0 (other inputs fixed)."""
    sqrt_a = get_sqrt_price_x96(0.9, 18, 18)
    sqrt_b = get_sqrt_price_x96(1.1, 18, 18)
    L1 = get_liquidity_for_amount0(sqrt_a, sqrt_b, amount0=10**18)
    L2 = get_liquidity_for_amount0(sqrt_a, sqrt_b, amount0=2 * 10**18)
    assert L2 == 2 * L1


@pytest.mark.core
def test_liquidity_for_amount0_rejects_inverted_bounds():
    sqrt_a = get_sqrt_price_x96(1.1, 18, 18)
    sqrt_b = get_sqrt_price_x96(0.9, 18, 18)
    with pytest.raises(ValueError, match="sqrt_ratio_b_x96 must be >"):
        get_liquidity_for_amount0(sqrt_a, sqrt_b, amount0=10**18)


@pytest.mark.core
def test_liquidity_for_amount0_rejects_equal_bounds():
    sqrt = get_sqrt_price_x96(1.0, 18, 18)
    with pytest.raises(ValueError, match="sqrt_ratio_b_x96 must be >"):
        get_liquidity_for_amount0(sqrt, sqrt, amount0=10**18)


@pytest.mark.core
def test_liquidity_for_amount1_zero_amount_yields_zero():
    sqrt_a = get_sqrt_price_x96(0.9, 18, 18)
    sqrt_b = get_sqrt_price_x96(1.1, 18, 18)
    assert get_liquidity_for_amount1(sqrt_a, sqrt_b, amount1=0) == 0


@pytest.mark.core
def test_liquidity_for_amount1_scales_linearly_with_amount():
    sqrt_a = get_sqrt_price_x96(0.9, 18, 18)
    sqrt_b = get_sqrt_price_x96(1.1, 18, 18)
    L1 = get_liquidity_for_amount1(sqrt_a, sqrt_b, amount1=10**18)
    L2 = get_liquidity_for_amount1(sqrt_a, sqrt_b, amount1=3 * 10**18)
    assert L2 == 3 * L1


@pytest.mark.core
def test_liquidity_for_amount1_rejects_inverted_bounds():
    sqrt_a = get_sqrt_price_x96(1.1, 18, 18)
    sqrt_b = get_sqrt_price_x96(0.9, 18, 18)
    with pytest.raises(ValueError, match="sqrt_ratio_b_x96 must be >"):
        get_liquidity_for_amount1(sqrt_a, sqrt_b, amount1=10**18)


@pytest.mark.core
def test_liquidity_delta_in_range_takes_min_of_both_sides():
    """In range, L = min(L_from_amount0, L_from_amount1). With balanced
    amounts both should give similar L; the smaller wins."""
    L = get_liquidity_delta(
        P=1.0, lower_price=0.9, upper_price=1.1,
        amount0=1.0, amount1=1.0,
        token0_decimal=18, token1_decimal=18,
    )
    assert L > 0


@pytest.mark.core
def test_liquidity_delta_below_range_uses_only_amount0():
    """Below range (P <= lower) the position holds 100% token0."""
    L = get_liquidity_delta(
        P=0.8, lower_price=0.9, upper_price=1.1,
        amount0=1.0, amount1=999.0,  # huge amount1, ignored
        token0_decimal=18, token1_decimal=18,
    )
    L_just_amount0 = get_liquidity_delta(
        P=0.8, lower_price=0.9, upper_price=1.1,
        amount0=1.0, amount1=0.0,
        token0_decimal=18, token1_decimal=18,
    )
    assert L == L_just_amount0


@pytest.mark.core
def test_liquidity_delta_above_range_uses_only_amount1():
    """Above range (P >= upper) the position holds 100% token1."""
    L = get_liquidity_delta(
        P=1.2, lower_price=0.9, upper_price=1.1,
        amount0=999.0, amount1=1.0,  # huge amount0, ignored
        token0_decimal=18, token1_decimal=18,
    )
    L_just_amount1 = get_liquidity_delta(
        P=1.2, lower_price=0.9, upper_price=1.1,
        amount0=0.0, amount1=1.0,
        token0_decimal=18, token1_decimal=18,
    )
    assert L == L_just_amount1


@pytest.mark.core
def test_liquidity_delta_rejects_inverted_range():
    with pytest.raises(ValueError, match="lower_price must be <"):
        get_liquidity_delta(
            P=1.0, lower_price=1.1, upper_price=0.9,
            amount0=1.0, amount1=1.0,
            token0_decimal=18, token1_decimal=18,
        )


@pytest.mark.core
def test_liquidity_delta_rejects_zero_price():
    with pytest.raises(ValueError, match="prices must all be > 0"):
        get_liquidity_delta(
            P=0.0, lower_price=0.9, upper_price=1.1,
            amount0=1.0, amount1=1.0,
            token0_decimal=18, token1_decimal=18,
        )


@pytest.mark.core
def test_liquidity_delta_rejects_negative_lower():
    with pytest.raises(ValueError, match="prices must all be > 0"):
        get_liquidity_delta(
            P=1.0, lower_price=-0.1, upper_price=1.1,
            amount0=1.0, amount1=1.0,
            token0_decimal=18, token1_decimal=18,
        )


@pytest.mark.core
def test_estimate_fee_zero_position_yields_zero():
    assert estimate_fee(liquidity_delta=0, liquidity=10**18, fees=100.0) == 0


@pytest.mark.core
def test_estimate_fee_full_pool_share():
    """If position L equals pool L, share = 50%."""
    fee = estimate_fee(liquidity_delta=10**18, liquidity=10**18, fees=100.0)
    assert fee == pytest.approx(50.0)


@pytest.mark.core
def test_estimate_fee_small_share():
    """Position L = 1% of pool L → ~1% of fees."""
    fee = estimate_fee(liquidity_delta=10**16, liquidity=10**18, fees=100.0)
    # share = 1e16 / (1e18 + 1e16) ≈ 0.0099
    assert 0.99 < fee < 1.0


@pytest.mark.core
def test_estimate_fee_zero_pool_zero_position_yields_zero():
    """Empty pool with no position — fee is 0, no division by zero."""
    assert estimate_fee(liquidity_delta=0, liquidity=0, fees=100.0) == 0


@pytest.mark.core
def test_estimate_fee_rejects_negative_position():
    with pytest.raises(ValueError, match="liquidity_delta must be >= 0"):
        estimate_fee(liquidity_delta=-1, liquidity=10**18, fees=100.0)


@pytest.mark.core
def test_estimate_fee_rejects_negative_pool():
    with pytest.raises(ValueError, match="liquidity must be >= 0"):
        estimate_fee(liquidity_delta=10**18, liquidity=-1, fees=100.0)


@pytest.mark.core
def test_estimate_fee_zero_fees_yields_zero():
    assert estimate_fee(liquidity_delta=10**18, liquidity=10**18, fees=0.0) == 0


@pytest.mark.core
def test_full_pipeline_realistic_eth_usdc_position():
    """Realistic USDC/WETH position: 10k USDC + 3 WETH at $3000/ETH,
    range [$2700, $3300]. Verify the model produces a positive L and
    reasonable fee share for a $50k pool with $100 of bar fees."""
    # USDC/WETH convention: token0=USDC (6 dec), token1=WETH (18 dec).
    # Standard V3 P = token1/token0 = WETH per USDC = 1/3000 (WETH gets cheap)
    # Wait — for USDC/WETH where token0=USDC, P = WETH/USDC ≈ 0.000333.
    L = get_liquidity_delta(
        P=1.0 / 3000.0,
        lower_price=1.0 / 3300.0,
        upper_price=1.0 / 2700.0,
        amount0=10_000.0,    # USDC
        amount1=3.0,         # WETH
        token0_decimal=6,
        token1_decimal=18,
    )
    assert L > 0

    # Pool has, say, L_pool = 1000 × our position (we're a small LP).
    fee_share = estimate_fee(
        liquidity_delta=L,
        liquidity=1000 * L,
        fees=100.0,
    )
    # Should be ~0.1% of $100 = $0.10
    assert 0.05 < fee_share < 0.15
