"""Uniswap V3 fees / liquidity model.

Pure-math helpers used by :class:`UniswapV3LPEntity.calculate_fees` to
compute an LP position's share of pool swap-fees. The functions here
mirror the on-chain V3 math (see Uniswap V3 whitepaper §6 and the
``LiquidityAmounts`` reference library) but operate on Python floats
plus ``Q96`` fixed-point ints — they are *not* bit-exact with on-chain
arithmetic. Float precision is ~15-16 significant digits, which gives
sub-bp accuracy on realistic LP positions; do not rely on this module
for on-chain transaction simulation.

All functions assume **standard V3 convention**: ``P = token1 / token0``,
amounts in their respective on-chain decimals.
"""
Q96 = 1 << 96  # 2**96, used as Q96 fixed-point scale


def expand_decimals(amount: float, decimals: int) -> int:
    """Convert a human-readable token amount to its raw on-chain integer.

    ``int(amount × 10**decimals)`` truncates toward zero. With float input
    the multiplication is subject to ~16-digit precision; we use
    :func:`round` (banker's rounding) instead of plain truncation to
    minimise systematic bias.

    Args:
        amount: Token amount in human units (e.g. ``1.5`` ETH).
        decimals: Token decimals (e.g. 18 for WETH, 6 for USDC).

    Returns:
        Raw integer amount (e.g. ``1.5`` × ``10**18``).
    """
    if decimals < 0:
        raise ValueError(f"decimals must be >= 0, got {decimals}")
    return round(amount * 10 ** decimals)


def get_sqrt_price_x96(price: float, token0_decimal: int, token1_decimal: int) -> int:
    """Compute ``√(price)`` in Q96 fixed-point.

    ``price`` is interpreted as **token1 per token0** (standard Uniswap V3
    convention). The decimals scaling adjusts for the unit difference
    between the two tokens before taking the square root.

    Args:
        price: Spot price as ``token1 / token0``.
        token0_decimal: Decimals of token0.
        token1_decimal: Decimals of token1.

    Returns:
        ``√(price_scaled) × 2**96`` as int.
    """
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")
    if token0_decimal < 0 or token1_decimal < 0:
        raise ValueError(
            f"decimals must be >= 0, got token0={token0_decimal}, token1={token1_decimal}"
        )
    scaled = (price * 10 ** token1_decimal) / 10 ** token0_decimal
    return int(scaled ** 0.5 * Q96)


def get_liquidity_for_amount0(
    sqrt_ratio_a_x96: int, sqrt_ratio_b_x96: int, amount0: int
) -> int:
    """L derived from a token0 amount over a sqrt-price range.

    Standard V3 identity: ``amount0 = L × (1/√pa - 1/√pb)`` so
    ``L = amount0 × √pa × √pb / (√pb - √pa)``.

    Args:
        sqrt_ratio_a_x96: Lower sqrt-price bound, Q96 fixed-point.
        sqrt_ratio_b_x96: Upper sqrt-price bound, Q96 fixed-point.
        amount0: Token0 amount, raw integer.

    Returns:
        Liquidity ``L`` corresponding to ``amount0`` over ``[a, b]``.
    """
    if sqrt_ratio_b_x96 <= sqrt_ratio_a_x96:
        raise ValueError(
            f"sqrt_ratio_b_x96 must be > sqrt_ratio_a_x96, "
            f"got a={sqrt_ratio_a_x96}, b={sqrt_ratio_b_x96}"
        )
    inter = (sqrt_ratio_a_x96 * sqrt_ratio_b_x96) >> 96
    return amount0 * inter // (sqrt_ratio_b_x96 - sqrt_ratio_a_x96)


def get_liquidity_for_amount1(
    sqrt_ratio_a_x96: int, sqrt_ratio_b_x96: int, amount1: int
) -> int:
    """L derived from a token1 amount over a sqrt-price range.

    Standard V3 identity: ``amount1 = L × (√pb - √pa)`` so
    ``L = amount1 × Q96 / (√pb - √pa)``.

    Args:
        sqrt_ratio_a_x96: Lower sqrt-price bound, Q96 fixed-point.
        sqrt_ratio_b_x96: Upper sqrt-price bound, Q96 fixed-point.
        amount1: Token1 amount, raw integer.

    Returns:
        Liquidity ``L`` corresponding to ``amount1`` over ``[a, b]``.
    """
    if sqrt_ratio_b_x96 <= sqrt_ratio_a_x96:
        raise ValueError(
            f"sqrt_ratio_b_x96 must be > sqrt_ratio_a_x96, "
            f"got a={sqrt_ratio_a_x96}, b={sqrt_ratio_b_x96}"
        )
    return amount1 * Q96 // (sqrt_ratio_b_x96 - sqrt_ratio_a_x96)


def get_liquidity_delta(
    P: float,
    lower_price: float,
    upper_price: float,
    amount0: float,
    amount1: float,
    token0_decimal: int,
    token1_decimal: int,
) -> int:
    """Liquidity ``L`` of a position holding ``(amount0, amount1)`` over ``[lower, upper]``.

    Branches on ``P`` relative to range: below range → all token0, in range
    → ``min(L_from_amount0, L_from_amount1)``, above → all token1.

    Args:
        P: Current pool price as ``token1 / token0``.
        lower_price: Lower bound of position, same convention as ``P``.
        upper_price: Upper bound of position, same convention.
        amount0: Token0 amount in human units.
        amount1: Token1 amount in human units.
        token0_decimal: Token0 decimals.
        token1_decimal: Token1 decimals.

    Returns:
        Position liquidity ``L`` as int.
    """
    if lower_price >= upper_price:
        raise ValueError(
            f"lower_price must be < upper_price, got lower={lower_price}, upper={upper_price}"
        )
    if P <= 0 or lower_price <= 0 or upper_price <= 0:
        raise ValueError(
            f"prices must all be > 0, got P={P}, lower={lower_price}, upper={upper_price}"
        )

    amt0 = expand_decimals(amount0, token0_decimal)
    amt1 = expand_decimals(amount1, token1_decimal)

    sqrt_ratio_x96 = get_sqrt_price_x96(P, token0_decimal, token1_decimal)
    sqrt_ratio_a_x96 = get_sqrt_price_x96(lower_price, token0_decimal, token1_decimal)
    sqrt_ratio_b_x96 = get_sqrt_price_x96(upper_price, token0_decimal, token1_decimal)

    if sqrt_ratio_x96 <= sqrt_ratio_a_x96:
        # Below range — all liquidity is in token0.
        return get_liquidity_for_amount0(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amt0)
    if sqrt_ratio_x96 < sqrt_ratio_b_x96:
        # In range — limited by smaller side.
        liquidity0 = get_liquidity_for_amount0(sqrt_ratio_x96, sqrt_ratio_b_x96, amt0)
        liquidity1 = get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_x96, amt1)
        return min(liquidity0, liquidity1)
    # Above range — all liquidity is in token1.
    return get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amt1)


def tick_to_price(tick: float, token0_decimal: int, token1_decimal: int) -> float:
    """Human price of token0 in token1 units:
    ``1.0001^tick × 10^(decimals0 − decimals1)``. Keep every scalar
    conversion on this helper — the decimals sign is the classic V3
    foot-gun.
    """
    if token0_decimal < 0 or token1_decimal < 0:
        raise ValueError(
            f"decimals must be >= 0, got token0={token0_decimal}, token1={token1_decimal}"
        )
    return 1.0001 ** tick * 10 ** (token0_decimal - token1_decimal)


def liquidity_to_onchain(liquidity: float, token0_decimal: int, token1_decimal: int) -> float:
    """Human-unit position ``L`` → on-chain units:
    ``L × 10^((decimals0 + decimals1) / 2)`` (the sqrt-price decimals
    adjustment applied to both legs).
    """
    if token0_decimal < 0 or token1_decimal < 0:
        raise ValueError(
            f"decimals must be >= 0, got token0={token0_decimal}, token1={token1_decimal}"
        )
    return liquidity * 10 ** ((token0_decimal + token1_decimal) / 2)


def fees_from_growth(fee_growth: float, liquidity_onchain: float, token_decimal: int) -> float:
    """One leg's fees from a feeGrowth counter delta: ``delta × L``,
    scaled to human token units. The counters accumulate
    ``fee / L(swap)`` per swap — the chain's own liquidity weighting,
    net of any protocol-fee cut.
    """
    if fee_growth < 0:
        raise ValueError(f"fee_growth must be >= 0, got {fee_growth}")
    if liquidity_onchain < 0:
        raise ValueError(f"liquidity_onchain must be >= 0, got {liquidity_onchain}")
    return fee_growth * liquidity_onchain / 10 ** token_decimal


def position_fees_from_growth(
    fee_growth0: float,
    fee_growth1: float,
    liquidity: float,
    token0_decimal: int,
    token1_decimal: int,
    pool_liquidity: float = 0.0,
) -> tuple:
    """Per-leg position fees for one bar of feeGrowth deltas.

    Like :func:`estimate_fee`, the position is treated as additional
    liquidity: with ``pool_liquidity`` given (on-chain ``L``, excluding
    the hypothetical position) both legs are diluted by
    ``pool_liquidity / (pool_liquidity + L_position)`` — ~1 for
    marginal positions; slightly undercounts when replicating a real
    position already included in the counter. ``pool_liquidity=0``
    skips dilution.

    Returns:
        ``(fees_token0, fees_token1)`` in human token units.
    """
    liquidity_onchain = liquidity_to_onchain(liquidity, token0_decimal, token1_decimal)
    dilution = 1.0
    if pool_liquidity > 0:
        dilution = pool_liquidity / (pool_liquidity + liquidity_onchain)
    return (
        fees_from_growth(fee_growth0, liquidity_onchain, token0_decimal) * dilution,
        fees_from_growth(fee_growth1, liquidity_onchain, token1_decimal) * dilution,
    )


def estimate_fee(liquidity_delta: int, liquidity: int, fees: float) -> float:
    """Pro-rate pool swap fees by an LP position's L-share of total liquidity.

    Formula: ``fees × L_position / (L_pool + L_position)``. Treats the
    position as **additional** liquidity joining the pool — appropriate
    for back-testing an LP that didn't actually exist at the time of the
    snapshot, since the pool's reported ``liquidity`` does not include
    the hypothetical position.

    Args:
        liquidity_delta: LP position's ``L``.
        liquidity: Pool's reported active ``L`` (excludes the position).
        fees: Total pool swap-fees collected over the bar (notional).

    Returns:
        Fees attributable to the position (notional).
    """
    if liquidity_delta < 0:
        raise ValueError(f"liquidity_delta must be >= 0, got {liquidity_delta}")
    if liquidity < 0:
        raise ValueError(f"liquidity must be >= 0, got {liquidity}")
    denom = liquidity + liquidity_delta
    if denom == 0:
        return 0.0
    return fees * (liquidity_delta / denom)
