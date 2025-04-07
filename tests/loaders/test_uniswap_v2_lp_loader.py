import pytest

from fractal.loaders import (EthereumUniswapV2PoolDataLoader, LoaderType,
                             PoolHistory)


@pytest.mark.integration
def test_uniswap_v2_lp(THE_GRAPH_API_KEY: str):
    loader = EthereumUniswapV2PoolDataLoader(
        pool="0xa43fe16908251ee70ef74718545e4fe6c5ccec9f",
        fee_tier=0.003,
        api_key=THE_GRAPH_API_KEY,
        loader_type=LoaderType.CSV
    )
    data: PoolHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["volume"].dtype == "float64"
    assert data["liquidity"].dtype == "float64"
    assert data["volume"].iloc[-1] > 0
    assert data["volume"].iloc[0] > 0
