
from datetime import UTC, datetime

import pytest

from fractal.loaders import FundingHistory, PriceHistory
from fractal.loaders.hyperliquid import (HyperliquidFundingRatesLoader,
                                         HyperliquidPerpsKlinesLoader,
                                         HyperLiquidPerpsPricesLoader)


@pytest.mark.integration
@pytest.mark.slow
def test_hyperliquid_fundings_loader():
    loader: HyperliquidFundingRatesLoader = HyperliquidFundingRatesLoader(ticker="ETH")
    data: FundingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["rate"].dtype == "float64"

    read_data = loader.read()
    assert len(read_data) > 0
    assert data["rate"].dtype == "float64"
    assert data['rate'].iloc[-1] != 0
    assert data['rate'].iloc[0] != 0
    assert data.index.dtype == "datetime64[ns, UTC]"
    assert len(data) == len(read_data), "Data length mismatch between read and with_run"


@pytest.mark.integration
def test_hyperliquid_fundings_loader_with_time_ranges():
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 1, 1, tzinfo=UTC)
    loader: HyperliquidFundingRatesLoader = HyperliquidFundingRatesLoader(
        ticker="BTC",
        start_time=start_time,
        end_time=end_time,
    )
    data: FundingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["rate"].dtype == "float64"
    assert data['rate'].iloc[-1] != 0
    assert data['rate'].iloc[0] != 0

    data = data.reset_index()
    t0 = data.iloc[0].to_dict()['time']
    t_last = data.iloc[-1].to_dict()['time']

    assert t0 >= start_time, "Start time is not respected"
    assert t_last <= end_time, "End time is not respected"


@pytest.mark.integration
def test_hyperliquid_perp_prices_loader():
    loader: HyperLiquidPerpsPricesLoader = HyperLiquidPerpsPricesLoader(
        ticker="ETH",
        interval="1d",

    data: PriceHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["price"].dtype == "float64"
    read_data = loader.read()
    assert len(read_data) > 0
    assert data["price"].dtype == "float64"
    assert len(data) == len(read_data), "Data length mismatch between read and with_run"


@pytest.mark.integration
def test_hyperliquid_perp_klines_loader():
    loader: HyperliquidPerpsKlinesLoader = HyperliquidPerpsKlinesLoader(
        ticker="ETH",
        interval="1d",
    )
    data: PriceHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["close"].dtype == "float64"
    for col in ["open", "high", "low", "close"]:
        assert col in data.columns, f"{col} not found in data"

    read_data = loader.read()
    assert len(read_data) > 0
    assert data["close"].dtype == "float64"
    assert len(data) == len(read_data), "Data length mismatch between read and with_run"


@pytest.mark.integration
def test_hyperliquid_perp_klines_loader_with_time_ranges():
    start_time = datetime(2025, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 2, 1, tzinfo=UTC)
    loader: HyperliquidPerpsKlinesLoader = HyperliquidPerpsKlinesLoader(
        ticker="ETH",
        interval="1d",
        start_time=start_time,
        end_time=end_time,
    )
    data: PriceHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["close"].dtype == "float64"
    for col in ["open", "high", "low", "close"]:
        assert col in data.columns, f"{col} not found in data"

    data = data.reset_index()
    t0 = data.iloc[0].to_dict()['open_time']
    t_last = data.iloc[-1].to_dict()['open_time']
    assert t0 >= start_time, "Start time is not respected"
    assert t_last <= end_time, "End time is not respected"
    read_data = loader.read()
    assert len(read_data) > 0
