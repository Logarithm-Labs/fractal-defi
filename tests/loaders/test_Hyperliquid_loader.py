"""Real-API tests for the Hyperliquid loaders.

These tests hit the public ``https://api.hyperliquid.xyz/info`` endpoint
directly (no third-party SDK). They are marked ``integration`` and ``slow``
so CI can opt out via ``-m "not integration"``.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders import FundingHistory, KlinesHistory, PriceHistory
from fractal.loaders.hyperliquid import (HyperliquidFundingRatesLoader,
                                         HyperliquidPerpsKlinesLoader,
                                         HyperLiquidPerpsPricesLoader)

UTC = timezone.utc


# ----------------------------------------------------------- funding history
@pytest.mark.integration
@pytest.mark.slow
def test_hyperliquid_fundings_loader():
    """Basic shape: with_run=True populates non-empty FundingHistory; cache reads back identical."""
    end = datetime(2025, 2, 1, tzinfo=UTC)
    start = end - timedelta(days=14)
    loader = HyperliquidFundingRatesLoader(ticker="ETH", start_time=start, end_time=end)
    data: FundingHistory = loader.read(with_run=True)
    assert isinstance(data, FundingHistory)
    assert len(data) > 0
    assert data["rate"].dtype == "float64"
    assert data.index.tz is not None  # UTC-aware
    # Cache round-trip via a fresh loader instance with the same window.
    fresh = HyperliquidFundingRatesLoader(ticker="ETH", start_time=start, end_time=end)
    cached: FundingHistory = fresh.read()
    assert len(cached) == len(data)


@pytest.mark.integration
def test_hyperliquid_fundings_loader_with_time_ranges():
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = datetime(2024, 2, 1, tzinfo=UTC)
    loader = HyperliquidFundingRatesLoader(
        ticker="BTC", start_time=start_time, end_time=end_time,
    )
    data: FundingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["rate"].dtype == "float64"
    assert data.index.min() >= start_time
    assert data.index.max() <= end_time


@pytest.mark.integration
def test_hyperliquid_fundings_loader_handles_empty_window():
    """Future window must not crash; loader returns an empty FundingHistory."""
    far_future_start = datetime(2099, 1, 1, tzinfo=UTC)
    far_future_end = datetime(2099, 1, 2, tzinfo=UTC)
    loader = HyperliquidFundingRatesLoader(
        ticker="BTC", start_time=far_future_start, end_time=far_future_end,
    )
    data: FundingHistory = loader.read(with_run=True)
    assert isinstance(data, FundingHistory)
    assert len(data) == 0
    assert list(data.columns) == ["rate"]


@pytest.mark.integration
def test_hyperliquid_fundings_loader_pagination():
    """A 60-day window forces pagination (Hyperliquid caps at 500 entries per call)."""
    end_time = datetime(2024, 3, 1, tzinfo=UTC)
    start_time = end_time - timedelta(days=60)
    loader = HyperliquidFundingRatesLoader(
        ticker="ETH", start_time=start_time, end_time=end_time,
    )
    data: FundingHistory = loader.read(with_run=True)
    # Funding fires hourly on Hyperliquid → ~24*60 = 1440 entries expected.
    assert len(data) > 500
    # Strictly monotonic timestamps after dedup.
    assert data.index.is_monotonic_increasing


# -------------------------------------------------------------- perp prices
@pytest.mark.integration
def test_hyperliquid_perp_prices_loader():
    end = datetime(2025, 2, 1, tzinfo=UTC)
    start = end - timedelta(days=14)
    loader = HyperLiquidPerpsPricesLoader(
        ticker="ETH", interval="1d", start_time=start, end_time=end,
    )
    data: PriceHistory = loader.read(with_run=True)
    assert isinstance(data, PriceHistory)
    assert len(data) > 0
    assert data["price"].dtype == "float64"
    fresh = HyperLiquidPerpsPricesLoader(
        ticker="ETH", interval="1d", start_time=start, end_time=end,
    )
    cached: PriceHistory = fresh.read()
    assert len(cached) == len(data)


@pytest.mark.integration
def test_hyperliquid_perp_prices_loader_rejects_unknown_interval():
    with pytest.raises(ValueError):
        HyperLiquidPerpsPricesLoader(ticker="ETH", interval="42m")


@pytest.mark.integration
def test_hyperliquid_perp_prices_loader_handles_empty_window():
    far_future_start = datetime(2099, 1, 1, tzinfo=UTC)
    far_future_end = datetime(2099, 1, 2, tzinfo=UTC)
    loader = HyperLiquidPerpsPricesLoader(
        ticker="BTC", interval="1d",
        start_time=far_future_start, end_time=far_future_end,
    )
    data: PriceHistory = loader.read(with_run=True)
    assert isinstance(data, PriceHistory)
    assert len(data) == 0
    assert list(data.columns) == ["price"]


# --------------------------------------------------------------- perp klines
@pytest.mark.integration
def test_hyperliquid_perp_klines_loader():
    end = datetime(2025, 2, 1, tzinfo=UTC)
    start = end - timedelta(days=7)
    loader = HyperliquidPerpsKlinesLoader(
        ticker="ETH", interval="1d", start_time=start, end_time=end,
    )
    data: KlinesHistory = loader.read(with_run=True)
    assert isinstance(data, KlinesHistory)
    assert len(data) > 0
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in data.columns
        assert data[col].dtype == "float64"
    fresh = HyperliquidPerpsKlinesLoader(
        ticker="ETH", interval="1d", start_time=start, end_time=end,
    )
    cached: KlinesHistory = fresh.read()
    assert len(cached) == len(data)


@pytest.mark.integration
def test_hyperliquid_perp_klines_loader_with_time_ranges():
    start_time = datetime(2025, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 2, 1, tzinfo=UTC)
    loader = HyperliquidPerpsKlinesLoader(
        ticker="ETH", interval="1d", start_time=start_time, end_time=end_time,
    )
    data: KlinesHistory = loader.read(with_run=True)
    assert len(data) > 0
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in data.columns
    # Index is the timestamp; check both endpoints are within the requested window.
    assert data.index.min() >= start_time
    assert data.index.max() <= end_time
    cached = loader.read()
    assert len(cached) == len(data)


@pytest.mark.integration
def test_hyperliquid_perp_klines_loader_no_gaps_for_1d():
    """Within a 30-day window, daily klines should produce no gaps."""
    start_time = datetime(2025, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 1, 31, tzinfo=UTC)
    loader = HyperliquidPerpsKlinesLoader(
        ticker="BTC", interval="1d", start_time=start_time, end_time=end_time,
    )
    data: KlinesHistory = loader.read(with_run=True)
    assert len(data) >= 29  # tolerate 1 missing edge bucket
    diffs = data.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all()
