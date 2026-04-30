"""Real-API tests for the Uniswap V2 hourly pool-data loader."""
from datetime import datetime, timedelta, timezone

import pytest

from fractal.loaders import (EthereumUniswapV2PoolDataLoader, LoaderType,
                             PoolHistory)

UTC = timezone.utc

# USDC/WETH 0.3% pair on Ethereum mainnet
USDC_WETH_PAIR = "0xa43fe16908251ee70ef74718545e4fe6c5ccec9f"


@pytest.mark.integration
def test_uniswap_v2_lp(THE_GRAPH_API_KEY: str):
    loader = EthereumUniswapV2PoolDataLoader(
        pool=USDC_WETH_PAIR,
        fee_tier=0.003,
        api_key=THE_GRAPH_API_KEY,
        loader_type=LoaderType.CSV,
    )
    data: PoolHistory = loader.read(with_run=True)
    assert isinstance(data, PoolHistory)
    assert len(data) > 0
    assert data["volume"].dtype == "float64"
    assert data["liquidity"].dtype == "float64"
    assert data["tvl"].iloc[-1] > 0
    assert data.index.tz is not None


@pytest.mark.integration
def test_uniswap_v2_lp_window_filter(THE_GRAPH_API_KEY: str):
    end = datetime(2025, 1, 31, tzinfo=UTC)
    start = end - timedelta(days=7)
    loader = EthereumUniswapV2PoolDataLoader(
        pool=USDC_WETH_PAIR,
        fee_tier=0.003,
        api_key=THE_GRAPH_API_KEY,
        start_time=start,
        end_time=end,
    )
    data = loader.read(with_run=True)
    assert len(data) > 0
    assert data.index.min() >= start
    assert data.index.max() <= end


@pytest.mark.integration
def test_uniswap_v2_lp_empty_window_returns_empty(THE_GRAPH_API_KEY: str):
    far_past_start = datetime(2000, 1, 1, tzinfo=UTC)
    far_past_end = datetime(2000, 1, 2, tzinfo=UTC)
    loader = EthereumUniswapV2PoolDataLoader(
        pool=USDC_WETH_PAIR,
        fee_tier=0.003,
        api_key=THE_GRAPH_API_KEY,
        start_time=far_past_start,
        end_time=far_past_end,
    )
    data = loader.read(with_run=True)
    assert isinstance(data, PoolHistory)
    assert len(data) == 0
    assert list(data.columns) == ["tvl", "volume", "fees", "liquidity"]
