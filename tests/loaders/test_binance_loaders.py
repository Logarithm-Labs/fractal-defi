from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from fractal.loaders.binance import (BinanceDayPriceLoader,
                                     BinanceFundingLoader,
                                     BinanceHourPriceLoader,
                                     BinanceKlinesLoader, BinancePriceLoader)
from fractal.loaders.structs import FundingHistory, KlinesHistory, PriceHistory


def _assert_time_bounds(df: pd.DataFrame, time_col: str, start: datetime, end: datetime):
    assert not df.empty, "DataFrame is empty — expected data in the requested range"
    # Ensure timezone-aware timestamps
    assert pd.api.types.is_datetime64tz_dtype(df[time_col].dtype)

    tmin = df[time_col].min()
    tmax = df[time_col].max()
    assert tmin >= start, (
        f"Earliest {time_col} {tmin} is before requested start {start}"
    )
    assert tmax <= end, (
        f"Latest {time_col} {tmax} is after requested end {end}"
    )

    # Sorted and monotonic increasing
    assert df[time_col].is_monotonic_increasing, f"{time_col} should be sorted ascending"


@pytest.mark.integration
@pytest.mark.slow
def test_binance_funding_loader():
    loader = BinanceFundingLoader(ticker="BTCUSDT")
    data: FundingHistory = loader.read(with_run=True)
    # Check that we received some data.
    assert len(data) > 0
    # Verify that funding rates are floats.
    assert data["rate"].dtype == "float64"

    # Read again without calling run() to ensure consistency.
    read_data = loader.read()
    assert len(read_data) > 0
    assert read_data["rate"].dtype == "float64"
    # Check that both read methods return the same number of records.
    assert len(data) == len(read_data), "Data length mismatch between read and with_run"
    # Ensure the time index is timezone-aware.
    assert data.index.dtype == "datetime64[ns, UTC]"


@pytest.mark.integration
def test_binance_funding_loader_with_time_ranges():
    start_time = datetime(2025, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 2, 1, tzinfo=UTC)
    loader = BinanceFundingLoader(ticker="BTCUSDT", start_time=start_time, end_time=end_time)
    data: FundingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["rate"].dtype == "float64"

    # Reset index to verify the time boundaries.
    data_reset = data.reset_index()
    t0 = data_reset.iloc[0]["fundingTime"]
    t_last = data_reset.iloc[-1]["fundingTime"]

    assert t0 >= start_time, "Start time is not respected"
    assert t_last <= end_time, "End time is not respected"


@pytest.mark.integration
@pytest.mark.slow
def test_binance_price_loader():
    # Using the hourly price loader as an example.
    loader = BinanceDayPriceLoader(ticker="BTCUSDT")
    data: PriceHistory = loader.read(with_run=True)
    assert len(data) > 0
    # Verify that price data is float.
    assert data["price"].dtype == "float64"

    read_data = loader.read()
    assert len(read_data) > 0
    assert read_data["price"].dtype == "float64"
    assert len(data) == len(read_data), "Data length mismatch between read and with_run"
    # Ensure the time index is timezone-aware.
    assert data.index.dtype == "datetime64[ns, UTC]"


@pytest.mark.integration
@pytest.mark.slow
def test_binance_klines_loader():
    loader = BinanceKlinesLoader(ticker="BTCUSDT", interval="1d")
    data: KlinesHistory = loader.read(with_run=True)
    assert len(data) > 0
    # Verify that all OHLC columns exist and are of type float.
    for col in ["open", "high", "low", "close"]:
        assert col in data.columns, f"{col} not found in data"
        assert data[col].dtype == "float64", f"{col} is not of type float64"

    read_data = loader.read()
    assert len(read_data) > 0
    assert len(data) == len(read_data), "Data length mismatch between read and with_run"
    # Ensure the time index is timezone-aware.
    assert data.index.dtype == "datetime64[ns, UTC]"


@pytest.mark.integration
def test_binance_klines_loader_with_time_ranges():
    start_time = datetime(2025, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 2, 1, tzinfo=UTC)
    loader = BinanceKlinesLoader(ticker="BTCUSDT", start_time=start_time, end_time=end_time)
    data: KlinesHistory = loader.read(with_run=True)
    assert len(data) > 0
    for col in ["open", "high", "low", "close"]:
        assert col in data.columns, f"{col} not found in data"
        assert data[col].dtype == "float64", f"{col} is not of type float64"

    data_reset = data.reset_index()
    t0 = data_reset.iloc[0]["openTime"]
    t_last = data_reset.iloc[-1]["openTime"]

    assert t0 >= start_time, "Start time is not respected"
    assert t_last <= end_time, "End time is not respected"


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
    df = df.reset_index()
    _assert_time_bounds(df, "fundingTime", start, end)



@pytest.mark.slow
@pytest.mark.integration
def test_klines_btcusdt_1h_bounds_2020_2025():
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2025, 1, 1, tzinfo=UTC)

    ldr = BinanceHourPriceLoader("BTCUSDT", start_time=start, end_time=end)
    ldr.extract()
    ldr.transform()

    df = ldr._data
    _assert_time_bounds(df, "openTime", start, end)


@pytest.mark.slow
@pytest.mark.integration
def test_klines_ethusdt_1h_bounds_2020_2025():
    start = datetime(2020, 1, 1, tzinfo=UTC)
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
        BinancePriceLoader("BTCUSDT", interval="x1h")   # invalid format


@pytest.mark.integration
def test_klines_empty_range_returns_empty_df():
    """If start_time > end_time, loader should return an empty, well-formed DataFrame."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)

    ldr = BinanceHourPriceLoader("BTCUSDT", start_time=start, end_time=end)
    ldr.extract(); ldr.transform()

    df = ldr._data
    assert len(df) <= 1
    # Columns present and in expected order
    assert list(df.columns) == ["openTime", "open", "high", "low", "close", "volume"]
