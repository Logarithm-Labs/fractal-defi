Q96 = 1 << 96  # same as 2**96, but as an int

def expand_decimals(amount: float, decimals: int) -> int:
    # scale token amount into its integer representation
    return int(amount * 10**decimals)

def get_liquidity_for_amount0(
        sqrt_ratio_a_x96: int, sqrt_ratio_b_x96: int,
        amount0: int
) -> int:
    # amount0 * (sqrt(b)*sqrt(a)/Q96) / (sqrt(b) - sqrt(a))
    # all inputs are Q96-scaled ints
    inter = (sqrt_ratio_a_x96 * sqrt_ratio_b_x96) >> 96
    return amount0 * inter // (sqrt_ratio_b_x96 - sqrt_ratio_a_x96)

def get_liquidity_for_amount1(
        sqrt_ratio_a_x96: int, sqrt_ratio_b_x96: int,
        amount1: int
) -> int:
    # amount1 * Q96 / (sqrt(b) - sqrt(a))
    return amount1 * Q96 // (sqrt_ratio_b_x96 - sqrt_ratio_a_x96)

def get_sqrt_price_x96(
        price: float, token0_decimal: int, token1_decimal: int
) -> int:
    """
    Compute sqrt(price) in Q96 fixed-point.
    price is token1 per token0.
    """
    # scale price to fixed-point
    scaled = (price * 10**token1_decimal) / 10**token0_decimal
    return int(scaled**0.5 * Q96)

def get_liquidity_delta(
        P: float, lower_price: float, upper_price: float,
        amount0: float, amount1: float,
        token0_decimal: int, token1_decimal: int
) -> int:
    # first, expand amounts correctly
    amt0 = expand_decimals(amount0, token0_decimal)
    amt1 = expand_decimals(amount1, token1_decimal)

    sqrt_ratio_x96   = get_sqrt_price_x96(P,           token0_decimal, token1_decimal)
    sqrt_ratio_a_x96 = get_sqrt_price_x96(lower_price, token0_decimal, token1_decimal)
    sqrt_ratio_b_x96 = get_sqrt_price_x96(upper_price, token0_decimal, token1_decimal)

    if sqrt_ratio_b_x96 <= sqrt_ratio_a_x96:
        raise ValueError("upper_price must be greater than lower_price")

    if sqrt_ratio_x96 <= sqrt_ratio_a_x96:
        liquidity = get_liquidity_for_amount0(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amt0)
    elif sqrt_ratio_x96 < sqrt_ratio_b_x96:
        liquidity0 = get_liquidity_for_amount0(sqrt_ratio_x96,   sqrt_ratio_b_x96, amt0)
        liquidity1 = get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_x96,   amt1)
        liquidity = min(liquidity0, liquidity1)
    else:
        liquidity = get_liquidity_for_amount1(sqrt_ratio_a_x96, sqrt_ratio_b_x96, amt1)

    return liquidity

def estimate_fee(
        liquidity_delta: int, liquidity: int,
        fees: float,
) -> float:
    """
    Pro-rate collected swap fees by the LP’s share of total liquidity.
    fees must be the total fees collected from swaps.
    """
    return fees * (liquidity_delta / (liquidity + liquidity_delta))
