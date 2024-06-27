Q96 = 2 ** 96


def expand_decimals(amount: float, decimals: int) -> float:
    return amount * (10 ** decimals)


def get_liquidity_for_amount0(
        sqrt_ratio_a_x96: float, sqrt_ratio_b_x96: float,
        amount0: float
) -> float:
    # amount0 * (sqrt(upper) * sqrt(lower)) / (sqrt(upper) - sqrt(lower))
    inter = sqrt_ratio_b_x96 * sqrt_ratio_a_x96 / Q96
    return amount0 * inter / (sqrt_ratio_b_x96 - sqrt_ratio_a_x96)


def get_liquidity_for_amount1(
        sqrt_ratio_a_x96: float, sqrt_ratio_b_x96: float,
        amount1: float
) -> float:
    # amount1 / (sqrt(upper) - sqrt(lower))
    return amount1 * Q96 / (sqrt_ratio_b_x96 - sqrt_ratio_a_x96)


def get_sqrt_price_x96(price: float, token0_decimal: int, token1_decimal: int) -> float:
    token0 = expand_decimals(price, token0_decimal)
    token1 = expand_decimals(1, token1_decimal)
    return ((token0 / token1)**0.5) * Q96


def get_liquidity_delta(
        P: float, lower_price: float, upper_price: float,
        amount0: float, amount1: float,
        token0_decimal: int, token1_decimal: int
) -> float:
    amt0 = expand_decimals(amount0, token1_decimal)
    amt1 = expand_decimals(amount1, token0_decimal)

    sqrt_ratio_x96 = get_sqrt_price_x96(P, token0_decimal, token1_decimal)
    sqrt_ratio_a_x96 = get_sqrt_price_x96(lower_price, token0_decimal, token1_decimal)
    sqrt_ratio_b_x96 = get_sqrt_price_x96(upper_price, token0_decimal, token1_decimal)

    if sqrt_ratio_x96 <= sqrt_ratio_a_x96:
        liquidity = get_liquidity_for_amount0(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amt0)
    elif sqrt_ratio_x96 < sqrt_ratio_b_x96:
        liquidity0 = get_liquidity_for_amount0(sqrt_ratio_x96, sqrt_ratio_b_x96, amt0)
        liquidity1 = get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_x96, amt1)
        liquidity = min(liquidity0, liquidity1)
    else:
        liquidity = get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amt1)

    return liquidity


def estimate_fee(
        liquidity_delta: float, liquidity: float,
        fees: float,
) -> float:
    liquidity_percentage = liquidity_delta / (liquidity + liquidity_delta)
    return fees * liquidity_percentage
