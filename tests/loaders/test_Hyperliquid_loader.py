from datetime import datetime

from fractal.loaders import FundingHistory, PriceHistory
from fractal.loaders.hyperliquid import (HyperliquidFundingRatesLoader,
                                         HyperLiquidPerpsPricesLoader)


def test_hyperliquid_fundings_loader():
    loader: HyperliquidFundingRatesLoader = HyperliquidFundingRatesLoader(
        ticker="ETH",
        start_time=datetime(2025, 1, 1),
        end_time=datetime(2025, 6, 1),
    )
    data: FundingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["rate"].dtype == "float64"


def test_hyperliquid_klines_loader():
    loader: HyperLiquidPerpsPricesLoader = HyperLiquidPerpsPricesLoader(
        ticker="ETH",
        interval="15m",
    )
    data: PriceHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["price"].dtype == "float64"
    read_data = loader.read()
    assert len(read_data) > 0
