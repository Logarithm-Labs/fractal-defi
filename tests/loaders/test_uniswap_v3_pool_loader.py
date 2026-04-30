"""Real-API tests for the Uniswap V3 pool-data loaders (Ethereum + Arbitrum)."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders import (Loader, LoaderType,
                             UniswapV3ArbitrumPoolDayDataLoader,
                             UniswapV3ArbitrumPoolHourDataLoader,
                             UniswapV3EthereumPoolDayDataLoader,
                             UniswapV3EthereumPoolHourDataLoader)

UTC = timezone.utc

ETH_USDC_USDT = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
ARB_USDC_WETH = "0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443"


@pytest.mark.integration
@pytest.mark.slow
def test_uniswap_v3_loaders(THE_GRAPH_API_KEY: str):
    loaders = [
        UniswapV3EthereumPoolDayDataLoader(
            api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_USDT, loader_type=LoaderType.CSV,
        ),
        UniswapV3ArbitrumPoolDayDataLoader(
            api_key=THE_GRAPH_API_KEY, pool=ARB_USDC_WETH, loader_type=LoaderType.CSV,
        ),
        UniswapV3EthereumPoolHourDataLoader(
            api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_USDT, loader_type=LoaderType.CSV,
        ),
        UniswapV3ArbitrumPoolHourDataLoader(
            api_key=THE_GRAPH_API_KEY, pool=ARB_USDC_WETH, loader_type=LoaderType.CSV,
        ),
    ]
    for loader in loaders:
        data = loader.read(with_run=True)
        assert len(data) > 0, f"empty data for {type(loader).__name__}"
        assert data["tvl"].dtype == "float64"
        assert data["volume"].dtype == "float64"
        assert data["fees"].dtype == "float64"
        assert data["liquidity"].dtype == "float64"
        assert data["tvl"].iloc[-1] > 0
        assert data.index.tz is not None


@pytest.mark.integration
@pytest.mark.slow
def test_uniswap_v3_get_pool_decimals(THE_GRAPH_API_KEY: str):
    loader: UniswapV3EthereumPoolDayDataLoader = UniswapV3EthereumPoolDayDataLoader(
        api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_USDT, loader_type=LoaderType.CSV,
    )
    decimals = loader.get_pool_decimals(ETH_USDC_USDT)
    assert decimals == (6, 18)

    arb_loader = UniswapV3ArbitrumPoolDayDataLoader(
        api_key=THE_GRAPH_API_KEY, pool=ARB_USDC_WETH, loader_type=LoaderType.CSV,
    )
    decimals = arb_loader.get_pool_decimals(ARB_USDC_WETH)
    assert decimals == (18, 6)


@pytest.mark.integration
@pytest.mark.slow
def test_uniswap_v3_data_types(THE_GRAPH_API_KEY: str):
    loader = UniswapV3EthereumPoolHourDataLoader(
        api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_USDT, loader_type=LoaderType.CSV,
    )
    data = loader.read(with_run=True)
    assert len(data) > 0
    assert data["tvl"].dtype == "float64"
    assert data["volume"].dtype == "float64"
    assert data["fees"].dtype == "float64"
    assert data.index.dtype == "datetime64[ns, UTC]"
    cached = loader.read(with_run=False)
    assert len(cached) == len(data)


@pytest.mark.integration
def test_uniswap_v3_pool_window_filter(THE_GRAPH_API_KEY: str):
    end = datetime(2025, 1, 31, tzinfo=UTC)
    start = end - timedelta(days=14)
    loader = UniswapV3EthereumPoolDayDataLoader(
        api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_USDT,
        start_time=start, end_time=end,
    )
    data = loader.read(with_run=True)
    assert len(data) > 0
    assert data.index.min() >= start
    assert data.index.max() <= end


@pytest.mark.integration
def test_uniswap_v3_pool_empty_window(THE_GRAPH_API_KEY: str):
    far_past_start = datetime(2000, 1, 1, tzinfo=UTC)
    far_past_end = datetime(2000, 1, 2, tzinfo=UTC)
    loader = UniswapV3EthereumPoolDayDataLoader(
        api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_USDT,
        start_time=far_past_start, end_time=far_past_end,
    )
    data = loader.read(with_run=True)
    assert len(data) == 0
    assert list(data.columns) == ["tvl", "volume", "fees", "liquidity"]


@pytest.mark.integration
def test_uniswap_v3_hour_no_gaps(THE_GRAPH_API_KEY: str):
    """Hourly stretching of daily data must produce continuous 1-hour buckets."""
    end = datetime(2025, 1, 7, tzinfo=UTC)
    start = end - timedelta(days=3)
    loader = UniswapV3EthereumPoolHourDataLoader(
        api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_USDT,
        start_time=start, end_time=end,
    )
    data = loader.read(with_run=True)
    # 3-day window stretched daily→hourly produces ~49 hourly buckets.
    assert len(data) >= 24 * 2
    diffs = data.index.to_series().diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()
