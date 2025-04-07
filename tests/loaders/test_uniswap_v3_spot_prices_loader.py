import pytest

from fractal.loaders import LoaderType, UniswapV3ArbitrumPricesLoader


@pytest.mark.integration
def test_uniswap_v3_arbitrum_prices_loader(THE_GRAPH_API_KEY: str):
    loader = UniswapV3ArbitrumPricesLoader(
        api_key=THE_GRAPH_API_KEY,
        pool="0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443",  # USDC/ETH
        loader_type=LoaderType.CSV,
        decimals=12,
    )
    data = loader.read(with_run=True)
    assert len(data) > 0
    assert data["price"].dtype == "float64"
    assert data["price"].iloc[-1] > 0
