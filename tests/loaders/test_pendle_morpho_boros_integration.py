"""Live smoke tests for the Pendle, Morpho and Boros loaders (no keys needed).

Pinned to markets that exist on 2026-09-11; when they expire, swap the
ids for live ones from the APIs (see research/pendle_boros_morpho).
"""
from datetime import timedelta

import pytest

from fractal.loaders import (
    BinanceFundingLoader,
    BorosMarketLoader,
    LoaderType,
    MorphoMarketLoader,
    PendleMarketLoader,
    PendleOHLCVLoader,
)
from fractal.loaders._dt import annualise_funding, utcnow
from fractal.loaders.boros import get_market_info as boros_market_info
from fractal.loaders.morpho import get_market_info as morpho_market_info
from fractal.loaders.pendle import get_market_info as pendle_market_info

PENDLE_SUSDE_26NOV2026 = "0x47ad2cd1dd15739a7a035b9d3b7828d916fef77e"
MORPHO_PT_REUSD_USDC = "0x1e9d614631a7df0ec07fb05b2c8cb2491575fd1a63a33bf187a6afb295a4fc64"
BOROS_BINANCE_BTC_25SEP2026 = 130


@pytest.mark.integration
def test_pendle_market_info_and_hourly_history_live():
    info = pendle_market_info(1, PENDLE_SUSDE_26NOV2026)
    assert info.expiry.year >= 2026 and info.pt and info.sy
    end = utcnow() - timedelta(hours=2)
    loader = PendleMarketLoader(PENDLE_SUSDE_26NOV2026, 1, end - timedelta(days=3), end, expiry=info.expiry,
                                loader_type=LoaderType.CSV)
    history = loader.read(with_run=True)
    assert len(history) >= 48
    assert history["implied_apy"].between(0.0, 1.0).all()
    assert history["pt_price_asset"].between(0.8, 1.0).all()
    assert history["total_pt"].gt(0).all() and history["total_sy"].gt(0).all()
    assert history["seconds_to_expiry"].is_monotonic_decreasing


@pytest.mark.integration
def test_pendle_ohlcv_live():
    info = pendle_market_info(1, PENDLE_SUSDE_26NOV2026)
    end = utcnow() - timedelta(hours=2)
    klines = PendleOHLCVLoader(info.pt, 1, end - timedelta(days=2), end).read(with_run=True)
    assert len(klines) >= 40
    assert klines["close"].between(0.8, 1.05).all()


@pytest.mark.integration
def test_morpho_market_info_and_history_live():
    info = morpho_market_info(MORPHO_PT_REUSD_USDC, 1)
    assert info.lltv in (0.86, 0.915, 0.945, 0.965)
    assert info.collateral_symbol.startswith("PT-")
    end = utcnow() - timedelta(hours=2)
    loader = MorphoMarketLoader(MORPHO_PT_REUSD_USDC, 1, end - timedelta(days=2), end)
    history = loader.read(with_run=True)
    assert len(history) >= 40
    assert history["borrow_apy"].between(0.0, 2.0).all()
    assert history["utilization"].between(0.0, 1.0).all()
    assert (history["borrowing_rate"] > 0).all()
    assert len(loader.read_collateral_price()) >= 40


@pytest.mark.integration
def test_boros_market_info_history_and_binance_parity_live():
    info = boros_market_info(BOROS_BINANCE_BTC_25SEP2026)
    assert info.payment_period_seconds == 8 * 3600
    assert 0 < info.k_im < 1 and 0 < info.k_mm <= info.k_im
    assert 0.03 < info.rate_floor < 0.15
    end = utcnow() - timedelta(hours=2)
    loader = BorosMarketLoader(BOROS_BINANCE_BTC_25SEP2026, end - timedelta(days=2), end, maturity=info.maturity,
                               underlying=("BTC", "Binance"))
    history = loader.read(with_run=True)
    assert len(history) >= 40
    assert history["mark_apr_close"].between(-1.0, 3.0).all()
    settled = history["settlement_apr"].dropna()
    assert len(settled) >= 4
    # settlement_apr is the venue's raw funding annualised by the payment period:
    # Boros settles Binance BTCUSDT funding * 1095 at the same fundingTime.
    funding = BinanceFundingLoader("BTCUSDT", start_time=end - timedelta(days=2), end_time=end).read(with_run=True)
    common = settled.index.intersection(funding.index)
    assert len(common) >= 3
    for ts in common:
        assert settled[ts] == pytest.approx(annualise_funding(float(funding.loc[ts, "rate"]), 8 * 3600), rel=1e-6)
