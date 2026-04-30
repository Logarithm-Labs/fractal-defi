"""Real-API tests for Binance USDT-M futures loaders."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders.binance import (BinanceDayPriceLoader,
                                     BinanceFundingLoader,
                                     BinanceHourPriceLoader,
                                     BinanceKlinesLoader, BinancePriceLoader)
from fractal.loaders.structs import FundingHistory, KlinesHistory, PriceHistory

UTC = timezone.utc


def _assert_time_bounds(df: pd.DataFrame, time_col: str, start: datetime, end: datetime):
    assert not df.empty, "DataFrame is empty — expected data in the requested range"
    assert pd.api.types.is_datetime64tz_dtype(df[time_col].dtype)
    tmin = df[time_col].min()
    tmax = df[time_col].max()
    assert tmin >= start, f"Earliest {time_col} {tmin} < requested start {start}"
    assert tmax <= end, f"Latest {time_col} {tmax} > requested end {end}"
    assert df[time_col].is_monotonic_increasing, f"{time_col} should be sorted ascending"


@pytest.mark.integration
@pytest.mark.slow
def test_binance_funding_loader():
    end = datetime(2025, 2, 1, tzinfo=UTC)
    start = end - timedelta(days=14)
    loader = BinanceFundingLoader(ticker="BTCUSDT", start_time=start, end_time=end)
    data: FundingHistory = loader.read(with_run=True)
    assert isinstance(data, FundingHistory)
    assert len(data) > 0
    assert data["rate"].dtype == "float64"
    assert data.index.dtype == "datetime64[ns, UTC]"
    # Cache round-trip via fresh instance with same window.
    fresh = BinanceFundingLoader(ticker="BTCUSDT", start_time=start, end_time=end)
    cached: FundingHistory = fresh.read()
    assert len(cached) == len(data)


@pytest.mark.integration
def test_binance_funding_loader_with_time_ranges():
    start_time = datetime(2025, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 2, 1, tzinfo=UTC)
    loader = BinanceFundingLoader(ticker="BTCUSDT", start_time=start_time, end_time=end_time)
    data: FundingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["rate"].dtype == "float64"
    assert data.index.min() >= start_time
    assert data.index.max() <= end_time


@pytest.mark.integration
@pytest.mark.slow
def test_binance_price_loader():
    end = datetime(2025, 2, 1, tzinfo=UTC)
    start = end - timedelta(days=14)
    loader = BinanceDayPriceLoader(ticker="BTCUSDT", start_time=start, end_time=end)
    data: PriceHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["price"].dtype == "float64"
    assert data.index.dtype == "datetime64[ns, UTC]"
    fresh = BinanceDayPriceLoader(ticker="BTCUSDT", start_time=start, end_time=end)
    cached: PriceHistory = fresh.read()
    assert len(cached) == len(data)


@pytest.mark.integration
@pytest.mark.slow
def test_binance_klines_loader():
    end = datetime(2025, 2, 1, tzinfo=UTC)
    start = end - timedelta(days=14)
    loader = BinanceKlinesLoader(ticker="BTCUSDT", interval="1d", start_time=start, end_time=end)
    data: KlinesHistory = loader.read(with_run=True)
    assert len(data) > 0
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in data.columns, f"{col} not found in data"
        assert data[col].dtype == "float64", f"{col} is not of type float64"
    assert data.index.dtype == "datetime64[ns, UTC]"
    fresh = BinanceKlinesLoader(ticker="BTCUSDT", interval="1d", start_time=start, end_time=end)
    cached: KlinesHistory = fresh.read()
    assert len(cached) == len(data)


@pytest.mark.integration
def test_binance_klines_loader_with_time_ranges():
    start_time = datetime(2025, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 2, 1, tzinfo=UTC)
    loader = BinanceKlinesLoader(ticker="BTCUSDT", interval="1d", start_time=start_time, end_time=end_time)
    data: KlinesHistory = loader.read(with_run=True)
    assert len(data) > 0
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in data.columns
        assert data[col].dtype == "float64"
    assert data.index.min() >= start_time
    assert data.index.max() <= end_time


@pytest.mark.slow
@pytest.mark.integration
def test_funding_btcusdt_bounds_2020_2025():
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)
    ldr = BinanceFundingLoader("BTCUSDT", start_time=start, end_time=end)
    ldr.extract()
    ldr.transform()
    df = ldr._data
    _assert_time_bounds(df, "fundingTime", start, end)


@pytest.mark.slow
@pytest.mark.integration
def test_funding_taousdt_bounds_2020_2025():
    start = datetime(2020, 3, 14, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)
    ldr = BinanceFundingLoader("TAOUSDT", start_time=start, end_time=end)
    ldr.extract()
    ldr.transform()
    df = ldr._data
    _assert_time_bounds(df, "fundingTime", start, end)


@pytest.mark.integration
def test_funding_hypeusdt_bounds():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 8, 1, tzinfo=UTC)
    df = BinanceFundingLoader("HYPEUSDT", start_time=start, end_time=end).read(with_run=True)
    assert len(df) > 0
    assert df.index.min() >= start
    assert df.index.max() <= end
    assert df.index.is_monotonic_increasing


@pytest.mark.slow
@pytest.mark.integration
def test_klines_btcusdt_1h_bounds_2024_2025():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)
    ldr = BinanceHourPriceLoader("BTCUSDT", start_time=start, end_time=end)
    ldr.extract()
    ldr.transform()
    df = ldr._data
    _assert_time_bounds(df, "openTime", start, end)


@pytest.mark.slow
@pytest.mark.integration
def test_klines_ethusdt_1h_bounds_2024_2025():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)
    ldr = BinanceHourPriceLoader("ETHUSDT", start_time=start, end_time=end)
    ldr.extract()
    ldr.transform()
    df = ldr._data
    _assert_time_bounds(df, "openTime", start, end)


@pytest.mark.integration
def test_invalid_interval_raises_valueerror():
    with pytest.raises(ValueError):
        BinancePriceLoader("BTCUSDT", interval="15x")  # invalid unit
    with pytest.raises(ValueError):
        BinancePriceLoader("BTCUSDT", interval="x1h")  # invalid format


@pytest.mark.integration
def test_klines_empty_range_returns_empty_df():
    """If start_time == end_time, loader should return an empty, well-formed DataFrame."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)
    ldr = BinanceHourPriceLoader("BTCUSDT", start_time=start, end_time=end)
    ldr.extract()
    ldr.transform()
    df = ldr._data
    assert len(df) <= 1
    assert list(df.columns) == ["openTime", "open", "high", "low", "close", "volume"]


@pytest.mark.integration
def test_klines_future_range_returns_empty():
    """A purely future window must not crash; loader returns empty PriceHistory."""
    start = datetime(2099, 1, 1, tzinfo=UTC)
    end = datetime(2099, 1, 2, tzinfo=UTC)
    data = BinanceDayPriceLoader("BTCUSDT", start_time=start, end_time=end).read(with_run=True)
    assert isinstance(data, PriceHistory)
    assert len(data) == 0
    assert list(data.columns) == ["price"]


@pytest.mark.integration
def test_funding_future_range_returns_empty():
    start = datetime(2099, 1, 1, tzinfo=UTC)
    end = datetime(2099, 1, 2, tzinfo=UTC)
    data = BinanceFundingLoader("BTCUSDT", start_time=start, end_time=end).read(with_run=True)
    assert isinstance(data, FundingHistory)
    assert len(data) == 0
    assert list(data.columns) == ["rate"]
