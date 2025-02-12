from datetime import datetime

from fractal.loaders.hyperliquid_loader import HyperliquidFundingRatesLoader, HyperLiquidPerpsPricesLoader

from fractal.loaders import (FundingHistory, PriceHistory)


def test_hyperliquid_fundings_loader():
    loader: HyperliquidFundingRatesLoader = HyperliquidFundingRatesLoader(
        ticker="ETH",
        start_time=datetime(2023, 1, 1),
        end_time=datetime(2023, 6, 1),
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
