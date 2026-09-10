"""Pure Pendle V2 maths: PT pricing, lender oracles and the PT/SY AMM.

Everything here is stateless and unit-free apart from time, which is
always **years, ACT/365** (``seconds / SECONDS_PER_YEAR``). Prices are
"accounting asset per PT" (the asset one PT redeems to at expiry: USDe,
stETH, …), never USD and never SY — see :class:`PendlePTEntity` for
how those are bridged.

Pricing convention (verified against ``PendleMarket.readState`` and the
``PendlePYLpOracle``): Pendle stores ``lnImpliedRate`` and defines

    exchangeRate = exp(lnImpliedRate * t)          # asset per PT, >= 1
    ptPrice      = 1 / exchangeRate = (1 + impliedApy) ** (-t)
    impliedApy   = exp(lnImpliedRate) - 1

i.e. compounded, not the linear ``1 - apy * t`` discount that lender
oracles use (:func:`linear_discount_oracle_price`).

The AMM functions replay ``MarketMathCore.calcTrade`` from
``pendle-core-v2``: a logit curve in the pool's PT proportion whose
steepness ``rateScalar = scalarRoot / t`` grows toward expiry, a
``rateAnchor`` re-anchored to the last implied rate before every trade,
and a fee that is a *spread on the implied rate*
(``feeRate = exp(lnFeeRateRoot * t)``), so both fee and impact shrink
with time to expiry. ``rate_spread_*`` is a lighter approximation with
the same time scaling for markets where the pool state is not loaded.
"""
import math
from dataclasses import dataclass

from fractal.core.base.time import SECONDS_PER_YEAR

MAX_MARKET_PROPORTION = 0.96  # PendleMarket: PT share of the pool after a trade
_BISECTION_TOL = 1e-12
_BISECTION_MAX_ITER = 200

__all__ = [
    "MAX_MARKET_PROPORTION",
    "AmmQuote",
    "ln_rate",
    "pt_price_from_apy",
    "apy_from_pt_price",
    "linear_discount_oracle_price",
    "twap_oracle_price",
    "rate_scalar",
    "rate_anchor",
    "exchange_rate",
    "amm_swap_exact_pt",
    "amm_swap_exact_asset_in",
    "rate_spread_buy",
    "rate_spread_sell",
]


# ------------------------------------------------------------------ pricing
def ln_rate(implied_apy: float) -> float:
    """``ln(1 + impliedApy)`` — Pendle's stored ``lnImpliedRate``."""
    if implied_apy <= -1.0:
        raise ValueError(f"implied_apy must be > -1, got {implied_apy}")
    return math.log1p(implied_apy)


def pt_price_from_apy(implied_apy: float, years: float) -> float:
    """Accounting asset per PT: ``(1 + apy) ** (-years)``; ``1.0`` at/after expiry."""
    if years <= 0.0:
        return 1.0
    return math.exp(-ln_rate(implied_apy) * years)


def apy_from_pt_price(pt_price: float, years: float) -> float:
    """Inverse of :func:`pt_price_from_apy`: ``pt_price ** (-1/years) - 1``."""
    if pt_price <= 0.0:
        raise ValueError(f"pt_price must be > 0, got {pt_price}")
    if years <= 0.0:
        raise ValueError(f"years must be > 0 to imply an APY, got {years}")
    return math.exp(-math.log(pt_price) / years) - 1.0


def linear_discount_oracle_price(base_discount_per_year: float, seconds_to_expiry: float) -> float:
    """``PendleSparkLinearDiscountOracle``: ``1 - d * timeLeft / 365d``, clamped to ``[0, 1]``.

    This is the conservative valuation Morpho/Spark markets use for PT
    collateral, not the market price. ``1.0`` at/after expiry.
    """
    if base_discount_per_year < 0.0:
        raise ValueError(f"base_discount_per_year must be >= 0, got {base_discount_per_year}")
    time_left = max(seconds_to_expiry, 0.0)
    return min(1.0, max(0.0, 1.0 - base_discount_per_year * time_left / SECONDS_PER_YEAR))


def twap_oracle_price(ln_rate_twap: float, seconds_to_expiry: float) -> float:
    """``PendlePYOracleLib.getPtToAssetRate`` core: ``exp(-lnImpliedRate_TWAP * t)``; ``1.0`` after expiry."""
    time_left = max(seconds_to_expiry, 0.0)
    return math.exp(-ln_rate_twap * time_left / SECONDS_PER_YEAR)


# ---------------------------------------------------------------------- AMM
@dataclass(frozen=True)
class AmmQuote:
    """Result of one AMM trade, signed from the trader's point of view.

    Attributes:
        net_pt_to_account (float): PT received (``> 0`` buy) or given (``< 0`` sell).
        net_asset_to_account (float): accounting asset received (``> 0``)
            or paid (``< 0``), **after** fees.
        fee_asset (float): fee in accounting asset (``>= 0``).
        post_ln_rate (float): the pool's ``lnImpliedRate`` after the trade.
    """
    net_pt_to_account: float
    net_asset_to_account: float
    fee_asset: float
    post_ln_rate: float


def rate_scalar(scalar_root: float, years: float) -> float:
    """``scalarRoot * YEAR / timeToExpiry`` — curve steepness, grows toward expiry."""
    if scalar_root <= 0.0:
        raise ValueError(f"scalar_root must be > 0, got {scalar_root}")
    if years <= 0.0:
        raise ValueError("cannot price an AMM trade at or after expiry")
    return scalar_root / years


def _proportion(total_pt: float, total_asset: float, net_pt_to_account: float = 0.0) -> float:
    total = total_pt + total_asset
    if total_pt <= 0.0 or total_asset <= 0.0:
        raise ValueError(f"pool reserves must be > 0, got total_pt={total_pt}, total_asset={total_asset}")
    proportion = (total_pt - net_pt_to_account) / total
    if proportion <= 0.0:
        raise ValueError("trade would drain the pool's PT reserve")
    if proportion > MAX_MARKET_PROPORTION:
        raise ValueError(
            f"trade pushes the pool's PT share to {proportion:.4f} > {MAX_MARKET_PROPORTION}"
        )
    return proportion


def rate_anchor(total_pt: float, total_asset: float, scalar: float, ln_implied_rate: float, years: float) -> float:
    """``MarketMathCore._getRateAnchor``: re-anchor the curve so the pre-trade rate is the last implied rate."""
    proportion = _proportion(total_pt, total_asset)
    last_exchange_rate = math.exp(ln_implied_rate * years)
    if last_exchange_rate < 1.0:
        raise ValueError("exchange rate below one: negative implied rate is not supported")
    return last_exchange_rate - math.log(proportion / (1.0 - proportion)) / scalar


def exchange_rate(
    total_pt: float,
    total_asset: float,
    scalar: float,
    anchor: float,
    net_pt_to_account: float,
) -> float:
    """``MarketMathCore._getExchangeRate``: asset per PT for a trade of ``net_pt_to_account``."""
    proportion = _proportion(total_pt, total_asset, net_pt_to_account)
    rate = math.log(proportion / (1.0 - proportion)) / scalar + anchor
    if rate < 1.0:
        raise ValueError("exchange rate below one: the trade is not executable on the curve")
    return rate


def amm_swap_exact_pt(
    net_pt_to_account: float,
    *,
    total_pt: float,
    total_asset: float,
    scalar_root: float,
    ln_fee_rate_root: float,
    implied_apy: float,
    years: float,
) -> AmmQuote:
    """``MarketMathCore.calcTrade`` for an exact PT amount (buy ``> 0``, sell ``< 0``).

    Args:
        net_pt_to_account: PT the trader receives (positive) or sells (negative).
        total_pt: pool PT reserve.
        total_asset: pool SY reserve **in accounting-asset terms** (``totalSy * syExchangeRate``).
        scalar_root: market immutable ``scalarRoot``.
        ln_fee_rate_root: market ``lnFeeRateRoot`` (router-specific; ~5e-4 on live markets).
        implied_apy: current implied APY (sets ``lnImpliedRate`` for the anchor).
        years: time to expiry in years.
    """
    if net_pt_to_account == 0.0:
        return AmmQuote(0.0, 0.0, 0.0, ln_rate(implied_apy))
    if ln_fee_rate_root < 0.0:
        raise ValueError(f"ln_fee_rate_root must be >= 0, got {ln_fee_rate_root}")
    scalar = rate_scalar(scalar_root, years)
    current_ln_rate = ln_rate(implied_apy)
    anchor = rate_anchor(total_pt, total_asset, scalar, current_ln_rate, years)
    pre_fee_exchange_rate = exchange_rate(total_pt, total_asset, scalar, anchor, net_pt_to_account)
    pre_fee_asset_to_account = -net_pt_to_account / pre_fee_exchange_rate
    fee_rate = math.exp(ln_fee_rate_root * years)
    if net_pt_to_account > 0.0:
        if pre_fee_exchange_rate / fee_rate < 1.0:
            raise ValueError("exchange rate below one after fees: the buy is not executable")
        fee = pre_fee_asset_to_account * (1.0 - fee_rate)
    else:
        fee = -(pre_fee_asset_to_account * (1.0 - fee_rate)) / fee_rate
    net_asset_to_account = pre_fee_asset_to_account - fee
    # Post-trade implied rate: re-read the curve at the new proportion.
    post_proportion = _proportion(total_pt, total_asset, net_pt_to_account)
    post_exchange_rate = math.log(post_proportion / (1.0 - post_proportion)) / scalar + anchor
    post_ln_rate = math.log(post_exchange_rate) / years
    return AmmQuote(
        net_pt_to_account=net_pt_to_account,
        net_asset_to_account=net_asset_to_account,
        fee_asset=fee,
        post_ln_rate=post_ln_rate,
    )


def amm_swap_exact_asset_in(
    asset_in: float,
    *,
    total_pt: float,
    total_asset: float,
    scalar_root: float,
    ln_fee_rate_root: float,
    implied_apy: float,
    years: float,
) -> AmmQuote:
    """Buy PT with an exact accounting-asset amount (router ``swapExactSyForPt``).

    Bisects :func:`amm_swap_exact_pt` on the PT amount until the asset
    paid matches ``asset_in``; the cost is monotone in the PT amount so
    the search is well posed. Raises when no executable size exists.
    """
    if asset_in < 0.0:
        raise ValueError(f"asset_in must be >= 0, got {asset_in}")
    if asset_in == 0.0:
        return AmmQuote(0.0, 0.0, 0.0, ln_rate(implied_apy))
    kwargs = {
        "total_pt": total_pt, "total_asset": total_asset, "scalar_root": scalar_root,
        "ln_fee_rate_root": ln_fee_rate_root, "implied_apy": implied_apy, "years": years,
    }

    def cost(pt: float):
        try:
            return -amm_swap_exact_pt(pt, **kwargs).net_asset_to_account
        except ValueError:
            return math.inf

    lo, hi = 0.0, total_pt * (1.0 - 1e-9)
    if cost(hi) < asset_in:
        raise ValueError("asset_in exceeds what the pool can sell: the buy is not executable")
    for _ in range(_BISECTION_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if cost(mid) < asset_in:
            lo = mid
        else:
            hi = mid
        if hi - lo <= _BISECTION_TOL * max(1.0, hi):
            break
    return amm_swap_exact_pt(lo, **kwargs)


# --------------------------------------------------------------- fallback
def _effective_ln_rate(implied_apy: float, fee_ln_rate: float, impact_ln_rate_per_share: float,
                       share: float, sign: float) -> float:
    if fee_ln_rate < 0.0 or impact_ln_rate_per_share < 0.0 or share < 0.0:
        raise ValueError("fee_ln_rate, impact_ln_rate_per_share and share must be >= 0")
    return ln_rate(implied_apy) + sign * (fee_ln_rate + impact_ln_rate_per_share * share)


def rate_spread_buy(asset_in: float, implied_apy: float, years: float, fee_ln_rate: float,
                    impact_ln_rate_per_share: float, pool_asset: float) -> float:
    """PT received for ``asset_in`` when fee and impact are spreads *below* the implied rate.

    ``share = asset_in / pool_asset`` (``0`` when the pool size is unknown);
    ``ln_eff = ln(1+apy) - fee - impact * share``; ``price = exp(-ln_eff * years)``.
    Both costs scale with ``years`` by construction, like the real AMM.
    """
    if asset_in < 0.0:
        raise ValueError(f"asset_in must be >= 0, got {asset_in}")
    if years <= 0.0:
        raise ValueError("cannot trade at or after expiry")
    share = asset_in / pool_asset if pool_asset > 0.0 else 0.0
    price = math.exp(-_effective_ln_rate(implied_apy, fee_ln_rate, impact_ln_rate_per_share, share, -1.0) * years)
    return asset_in / price


def rate_spread_sell(pt_in: float, implied_apy: float, years: float, fee_ln_rate: float,
                     impact_ln_rate_per_share: float, pool_asset: float) -> float:
    """Accounting asset received for ``pt_in`` when fee and impact are spreads *above* the implied rate."""
    if pt_in < 0.0:
        raise ValueError(f"pt_in must be >= 0, got {pt_in}")
    if years <= 0.0:
        raise ValueError("cannot trade at or after expiry")
    mid_price = pt_price_from_apy(implied_apy, years)
    share = pt_in * mid_price / pool_asset if pool_asset > 0.0 else 0.0
    price = math.exp(-_effective_ln_rate(implied_apy, fee_ln_rate, impact_ln_rate_per_share, share, +1.0) * years)
    return pt_in * price
