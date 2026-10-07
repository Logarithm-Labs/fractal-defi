"""Closed-form and fixture tests for :mod:`fractal.core.entities.models.pendle_math`.

Fixtures come from live reads of Ethereum mainnet on 2026-09-10
(``PendleMarket.readState`` / ``PendlePYLpOracle``), recorded in
``research/pendle_boros_morpho/CONTEXT.md``.
"""
import math

import pytest

from fractal.core.base.time import SECONDS_PER_DAY, SECONDS_PER_YEAR
from fractal.core.entities.models.pendle_math import (
    MAX_MARKET_PROPORTION,
    amm_swap_exact_asset_in,
    amm_swap_exact_pt,
    apy_from_pt_price,
    linear_discount_oracle_price,
    ln_rate,
    pt_price_from_apy,
    rate_spread_buy,
    rate_spread_sell,
    twap_oracle_price,
)

# PT-sUSDe-26NOV2026 on 2026-09-10: lnImpliedRate 0.0480287 (4.920 % APY),
# 76 days to expiry, oracle ptToAsset 0.99004, scalarRoot 57.60,
# lnFeeRateRoot 9.8686e-4 via the Pendle router, PT share ~25 %,
# pool ~2.7 M USDe in asset terms.
SUSDE_LN_RATE = 0.0480287
SUSDE_APY = math.expm1(SUSDE_LN_RATE)
SUSDE_YEARS = 76 * SECONDS_PER_DAY / SECONDS_PER_YEAR
SUSDE_SCALAR_ROOT = 57.60
SUSDE_LN_FEE = 9.8686e-4
SUSDE_TOTAL_ASSET = 2.71e6
SUSDE_TOTAL_PT = SUSDE_TOTAL_ASSET / 3.0  # p = pt / (pt + asset) = 0.25

AMM = {
    "total_pt": SUSDE_TOTAL_PT, "total_asset": SUSDE_TOTAL_ASSET,
    "scalar_root": SUSDE_SCALAR_ROOT, "ln_fee_rate_root": SUSDE_LN_FEE,
    "implied_apy": SUSDE_APY, "years": SUSDE_YEARS,
}


@pytest.mark.core
def test_pt_price_matches_onchain_susde_fixture():
    """(1 + apy) ** (-t) reproduces the oracle's ptToAsset to 1e-4."""
    assert pt_price_from_apy(SUSDE_APY, SUSDE_YEARS) == pytest.approx(0.99004, abs=1e-4)
    assert ln_rate(SUSDE_APY) == pytest.approx(SUSDE_LN_RATE)


@pytest.mark.core
def test_pt_price_apy_round_trip_and_expiry():
    for apy in (0.02, 0.10, 0.20):
        for years in (0.1, 0.5, 1.0):
            price = pt_price_from_apy(apy, years)
            assert apy_from_pt_price(price, years) == pytest.approx(apy, rel=1e-12)
    assert pt_price_from_apy(0.5, 0.0) == 1.0
    assert pt_price_from_apy(0.5, -1.0) == 1.0
    with pytest.raises(ValueError):
        ln_rate(-1.0)


@pytest.mark.core
def test_linear_discount_underprices_compounded_pt():
    """Digest table: 10 % / 6 mo → linear price ~36 bp below compounded."""
    compounded = pt_price_from_apy(0.10, 0.5)
    linear = linear_discount_oracle_price(0.10, 0.5 * SECONDS_PER_YEAR)
    assert (compounded - linear) * 1e4 == pytest.approx(36, abs=2)
    assert linear_discount_oracle_price(0.06, 0.0) == 1.0
    assert linear_discount_oracle_price(0.06, -10.0) == 1.0
    assert linear_discount_oracle_price(2.0, SECONDS_PER_YEAR) == 0.0


@pytest.mark.core
def test_twap_oracle_price_matches_compounded_formula():
    assert twap_oracle_price(SUSDE_LN_RATE, 76 * SECONDS_PER_DAY) == pytest.approx(
        pt_price_from_apy(SUSDE_APY, SUSDE_YEARS), rel=1e-12
    )
    assert twap_oracle_price(SUSDE_LN_RATE, 0.0) == 1.0


@pytest.mark.core
def test_amm_zero_trade_is_free_and_rate_neutral():
    quote = amm_swap_exact_pt(0.0, **AMM)
    assert quote.net_asset_to_account == 0.0 and quote.fee_asset == 0.0
    assert quote.post_ln_rate == pytest.approx(SUSDE_LN_RATE)


@pytest.mark.core
def test_amm_buy_impact_matches_live_simulation():
    """Digest: buying 1 % of the pool moves the price ~1.5 bp, 10 % ~16 bp, APY 4.92 → ~4.11 %."""
    mid = pt_price_from_apy(SUSDE_APY, SUSDE_YEARS)
    one_pct = amm_swap_exact_asset_in(0.01 * SUSDE_TOTAL_ASSET, **AMM)
    ten_pct = amm_swap_exact_asset_in(0.10 * SUSDE_TOTAL_ASSET, **AMM)
    for quote, asset_in in ((one_pct, 0.01 * SUSDE_TOTAL_ASSET), (ten_pct, 0.10 * SUSDE_TOTAL_ASSET)):
        assert -quote.net_asset_to_account == pytest.approx(asset_in, rel=1e-9)
        paid_price = asset_in / quote.net_pt_to_account
        assert paid_price > mid
    slip_1 = (0.01 * SUSDE_TOTAL_ASSET / one_pct.net_pt_to_account / mid - 1) * 1e4
    slip_10 = (0.10 * SUSDE_TOTAL_ASSET / ten_pct.net_pt_to_account / mid - 1) * 1e4
    assert 2.5 < slip_1 < 5.5          # ~1.5 bp impact + ~2.1 bp fee
    assert 13 < slip_10 < 22           # ~16 bp impact + ~2.1 bp fee
    assert math.expm1(ten_pct.post_ln_rate) == pytest.approx(0.0411, abs=0.003)


@pytest.mark.core
def test_amm_sell_moves_rate_up_and_charges_fee():
    quote = amm_swap_exact_pt(-0.10 * SUSDE_TOTAL_PT, **AMM)
    assert quote.net_asset_to_account > 0
    assert quote.fee_asset > 0
    assert quote.post_ln_rate > SUSDE_LN_RATE
    mid = pt_price_from_apy(SUSDE_APY, SUSDE_YEARS)
    received_price = quote.net_asset_to_account / (0.10 * SUSDE_TOTAL_PT)
    assert received_price < mid


@pytest.mark.core
def test_amm_fee_scales_with_time_to_expiry():
    """Fee is a spread on the rate: cost ≈ lnFeeRateRoot × years of notional."""
    trade = 1_000.0
    near = amm_swap_exact_pt(trade, **{**AMM, "years": 10 / 365})
    far = amm_swap_exact_pt(trade, **{**AMM, "years": 300 / 365})
    assert far.fee_asset / near.fee_asset == pytest.approx(30, rel=0.05)
    assert near.fee_asset / (trade * pt_price_from_apy(SUSDE_APY, 10 / 365)) == pytest.approx(
        SUSDE_LN_FEE * 10 / 365, rel=0.05
    )


@pytest.mark.core
def test_amm_exact_asset_in_inverts_exact_pt():
    quote_pt = amm_swap_exact_pt(50_000.0, **AMM)
    quote_asset = amm_swap_exact_asset_in(-quote_pt.net_asset_to_account, **AMM)
    assert quote_asset.net_pt_to_account == pytest.approx(50_000.0, rel=1e-8)


@pytest.mark.core
def test_amm_rejects_trades_past_the_pool_share_cap_or_reserves():
    sell_too_much = -(MAX_MARKET_PROPORTION * (SUSDE_TOTAL_PT + SUSDE_TOTAL_ASSET) - SUSDE_TOTAL_PT + 1.0)
    with pytest.raises(ValueError, match="PT share"):
        amm_swap_exact_pt(sell_too_much, **AMM)
    with pytest.raises(ValueError, match="drain"):
        amm_swap_exact_pt(SUSDE_TOTAL_PT, **AMM)
    with pytest.raises(ValueError, match="expiry"):
        amm_swap_exact_pt(1.0, **{**AMM, "years": 0.0})


@pytest.mark.core
def test_rate_spread_costs_scale_with_time_and_size():
    mid_near = pt_price_from_apy(0.05, 10 / 365)
    mid_far = pt_price_from_apy(0.05, 300 / 365)
    near = 100.0 / rate_spread_buy(100.0, 0.05, 10 / 365, 0.001, 0.075, 1e6) / mid_near - 1
    far = 100.0 / rate_spread_buy(100.0, 0.05, 300 / 365, 0.001, 0.075, 1e6) / mid_far - 1
    assert far / near == pytest.approx(30, rel=0.02)
    small = 1_000.0 / rate_spread_buy(1_000.0, 0.05, 0.5, 0.0, 0.075, 1e6)
    big = 100_000.0 / rate_spread_buy(100_000.0, 0.05, 0.5, 0.0, 0.075, 1e6)
    assert big > small  # larger trade → worse price
    sold = rate_spread_sell(1_000.0, 0.05, 0.5, 0.001, 0.075, 1e6)
    assert sold < 1_000.0 * pt_price_from_apy(0.05, 0.5)
    with pytest.raises(ValueError):
        rate_spread_buy(1.0, 0.05, 0.0, 0.001, 0.075, 1e6)
