"""Real-API tests for the Uniswap V3 hourly spot-price loaders."""
from datetime import datetime, timedelta, timezone

import pytest

from fractal.loaders import LoaderType, PriceHistory, UniswapV3ArbitrumPricesLoader, UniswapV3EthereumPricesLoader

UTC = timezone.utc

ETH_USDC_USDT = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
ARB_USDC_WETH = "0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443"


@pytest.mark.integration
def test_uniswap_v3_arbitrum_prices_loader(THE_GRAPH_API_KEY: str):
    loader = UniswapV3ArbitrumPricesLoader(
        api_key=THE_GRAPH_API_KEY,
        pool=ARB_USDC_WETH,
        loader_type=LoaderType.CSV,
        decimals=12,
    )
    data: PriceHistory = loader.read(with_run=True)
    assert isinstance(data, PriceHistory)
    assert len(data) > 0
    assert data["price"].dtype == "float64"
    assert data["price"].iloc[-1] > 0


@pytest.mark.integration
def test_uniswap_v3_arbitrum_prices_loader_window(THE_GRAPH_API_KEY: str):
    end = datetime(2025, 2, 1, tzinfo=UTC)
    start = end - timedelta(days=7)
    loader = UniswapV3ArbitrumPricesLoader(
        api_key=THE_GRAPH_API_KEY, pool=ARB_USDC_WETH,
        decimals=12, start_time=start, end_time=end,
    )
    data = loader.read(with_run=True)
    assert len(data) > 0
    assert data.index.min() >= start
    assert data.index.max() <= end


@pytest.mark.integration
def test_uniswap_v3_ethereum_prices_loader_window(THE_GRAPH_API_KEY: str):
    end = datetime(2025, 1, 31, tzinfo=UTC)
    start = end - timedelta(days=7)
    loader = UniswapV3EthereumPricesLoader(
        api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_USDT,
        start_time=start, end_time=end,
    )
    data = loader.read(with_run=True)
    assert len(data) > 0
    assert data["price"].iloc[-1] > 0


@pytest.mark.integration
def test_uniswap_v3_prices_empty_window_returns_empty(THE_GRAPH_API_KEY: str):
    far_past_start = datetime(2000, 1, 1, tzinfo=UTC)
    far_past_end = datetime(2000, 1, 2, tzinfo=UTC)
    loader = UniswapV3EthereumPricesLoader(
        api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_USDT,
        start_time=far_past_start, end_time=far_past_end,
    )
    data = loader.read(with_run=True)
    assert isinstance(data, PriceHistory)
    assert len(data) == 0
    assert list(data.columns) == ["price"]
