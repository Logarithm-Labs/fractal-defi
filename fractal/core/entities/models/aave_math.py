"""Pure Aave V3 liquidation maths.

Source: ``aave-v3-origin`` @ ``8305565``,
``src/contracts/protocol/libraries/logic/LiquidationLogic.sol``
(``executeLiquidationCall``, ``_calculateAvailableCollateralToLiquidate``)
and ``ValidationLogic.sol`` (``HEALTH_FACTOR_LIQUIDATION_THRESHOLD``).

Values are in the accounting unit of the prices (Aave's base currency is
USD with 8 decimals; the dollar thresholds below assume USD prices).
The liquidation protocol fee only splits the seized collateral between the
liquidator and the treasury, so it does not change what the borrower loses
and is not modelled here.
"""

DEFAULT_LIQUIDATION_CLOSE_FACTOR = 0.5         # LiquidationLogic.DEFAULT_LIQUIDATION_CLOSE_FACTOR (0.5e4 bps)
CLOSE_FACTOR_HF_THRESHOLD = 0.95               # LiquidationLogic.CLOSE_FACTOR_HF_THRESHOLD (0.95e18)
MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD = 2000.0   # LiquidationLogic.MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD ($2000, 1e8)
MIN_LEFTOVER_BASE = MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD / 2  # LiquidationLogic.MIN_LEFTOVER_BASE ($1000)

__all__ = [
    "CLOSE_FACTOR_HF_THRESHOLD",
    "DEFAULT_LIQUIDATION_CLOSE_FACTOR",
    "MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD",
    "MIN_LEFTOVER_BASE",
    "liquidation_call",
]


def liquidation_call(
    collateral: float,
    debt: float,
    collateral_price: float,
    debt_price: float,
    *,
    liq_thr: float,
    liquidation_bonus: float,
) -> tuple[float, float]:
    """One ``liquidationCall`` on a single-collateral, single-debt position.

    The liquidator is assumed to cover the largest debt the protocol
    allows (the worst case for the borrower within the protocol rules):

    * nothing happens unless ``HF = liq_thr · collateral_value / debt_value < 1``;
    * up to 50% of the debt may be repaid, or all of it when ``HF ≤ 0.95``
      or when the collateral or debt value is below $2000;
    * collateral worth ``repaid × debt_price × liquidation_bonus`` is seized;
      if that exceeds the collateral, all collateral is seized and the
      debt left over is written off as reserve deficit;
    * a call that clears neither all debt nor all collateral may not leave
      under $1000 of either; the repaid amount is reduced until it does not.

    Args:
        collateral: Collateral units held by the borrower.
        debt: Debt units owed.
        collateral_price: Price of one collateral unit in the accounting unit.
        debt_price: Price of one debt unit in the accounting unit.
        liq_thr: Liquidation threshold of the collateral (e.g. ``0.85``).
        liquidation_bonus: Collateral bonus as a multiplier (``1.05`` = 5%).

    Returns:
        ``(debt_removed, collateral_seized)`` in token units. ``debt_removed``
        includes any written-off deficit; ``(0.0, 0.0)`` when the position
        is healthy.
    """
    if debt <= 0 or debt_price <= 0:
        return 0.0, 0.0
    collateral_value = max(collateral * collateral_price, 0.0)
    debt_value = debt * debt_price
    health_factor = liq_thr * collateral_value / debt_value
    if health_factor >= 1:
        return 0.0, 0.0
    if collateral_value == 0:
        return debt, collateral

    repaid = debt
    if (
        collateral_value >= MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD
        and debt_value >= MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD
        and health_factor > CLOSE_FACTOR_HF_THRESHOLD
    ):
        repaid = debt * DEFAULT_LIQUIDATION_CLOSE_FACTOR

    seized = repaid * debt_price / collateral_price * liquidation_bonus
    if seized >= collateral:
        return debt, collateral

    if repaid < debt and (
        (debt - repaid) * debt_price < MIN_LEFTOVER_BASE
        or (collateral - seized) * collateral_price < MIN_LEFTOVER_BASE
    ):
        repaid = min(
            repaid,
            debt - MIN_LEFTOVER_BASE / debt_price,
            (collateral_value - MIN_LEFTOVER_BASE) / (debt_price * liquidation_bonus),
        )
        if repaid <= 0:
            return 0.0, 0.0
        seized = repaid * debt_price / collateral_price * liquidation_bonus
    return repaid, seized
