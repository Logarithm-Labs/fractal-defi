from datetime import datetime

from fractal.loaders.Hyperliquid_Loader import HyperliquidFundingRatesLoader, HyperLiquidPerpsPricesLoader

from fractal.loaders import (FundingHistory, PriceHistory)


def test_Hyperliquid_Funding_loader():
    loader: HyperliquidFundingRatesLoader = HyperliquidFundingRatesLoader(
        ticker="ETH",
        start_time=datetime(2023, 1, 1),
        end_time=datetime(2023, 6, 1),
    )
    data: FundingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["fundingRate"].dtype == "float64"


def test_Hyperliquid_Klines_loader():
    loader: HyperLiquidPerpsPricesLoader = HyperLiquidPerpsPricesLoader(
        ticker="ETH",
        interval="15m",
        start_time=datetime(2023, 1, 1),
        end_time=datetime(2023, 6, 1),
    )
    data: PriceHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["close_price"].dtype == "float64"
