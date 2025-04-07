from datetime import UTC, datetime

import pytest

from fractal.loaders.binance import (BinanceDayPriceLoader,
                                     BinanceFundingLoader, BinanceKlinesLoader)
from fractal.loaders.structs import FundingHistory, KlinesHistory, PriceHistory

# --- Funding Loader Tests ---

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


# --- Price Loader Tests ---
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


# --- Klines Loader Tests ---
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
