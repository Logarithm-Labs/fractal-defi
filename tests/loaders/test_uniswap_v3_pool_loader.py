from fractal.loaders import (Loader, LoaderType,
                             UniswapV3ArbitrumPoolDayDataLoader,
                             UniswapV3ArbitrumPoolHourDataLoader,
                             UniswapV3EthereumPoolDayDataLoader,
                             UniswapV3EthereumPoolHourDataLoader)


def test_uniswap_v3_loaders(THE_GRAPH_API_KEY: str):
    loaders: Loader = [
        UniswapV3EthereumPoolDayDataLoader(
            api_key=THE_GRAPH_API_KEY,
            pool="0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",  # USDC/ETH
            loader_type=LoaderType.CSV
        ),
        UniswapV3ArbitrumPoolDayDataLoader(
            api_key=THE_GRAPH_API_KEY,
            pool="0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443",  # USDC/ETH
            loader_type=LoaderType.CSV
        ),
        UniswapV3EthereumPoolHourDataLoader(
            api_key=THE_GRAPH_API_KEY,
            pool="0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",  # USDC/ETH
            loader_type=LoaderType.CSV
        ),
        UniswapV3ArbitrumPoolHourDataLoader(
            api_key=THE_GRAPH_API_KEY,
            pool="0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443",  # USDC/ETH
            loader_type=LoaderType.CSV
        )
    ]
    for loader in loaders:
        data = loader.read(with_run=True)
        assert len(data) > 0
        assert data["tvl"].dtype == "float64"
        assert data["volume"].dtype == "float64"
        assert data["fees"].dtype == "float64"
        assert data["liquidity"].dtype == "float64"
        assert data["tvl"].iloc[-1] > 0
