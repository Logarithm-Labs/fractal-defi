import pytest

from fractal.loaders import (Loader, LoaderType,
                             UniswapV3ArbitrumPoolDayDataLoader,
                             UniswapV3ArbitrumPoolHourDataLoader,
                             UniswapV3EthereumPoolDayDataLoader,
                             UniswapV3EthereumPoolHourDataLoader)


@pytest.mark.integration
@pytest.mark.slow
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



@pytest.mark.integration
@pytest.mark.slow
def test_uniswap_v3_get_pool_decimals(THE_GRAPH_API_KEY: str):
    loader: UniswapV3EthereumPoolDayDataLoader = UniswapV3EthereumPoolDayDataLoader(
            api_key=THE_GRAPH_API_KEY,
            pool="0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",  # USDC/ETH
            loader_type=LoaderType.CSV
    )
    decimals = loader.get_pool_decimals("0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8")
    assert decimals == (6, 18)

    loader: UniswapV3ArbitrumPoolDayDataLoader = UniswapV3ArbitrumPoolDayDataLoader(
            api_key=THE_GRAPH_API_KEY,
            pool="0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443",  # USDC/ETH
            loader_type=LoaderType.CSV
    )
    decimals = loader.get_pool_decimals("0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443")
    assert decimals == (18, 6)


@pytest.mark.integration
@pytest.mark.slow
def test_uniswap_v3_data_types(THE_GRAPH_API_KEY: str):
    loader =UniswapV3EthereumPoolHourDataLoader(
            api_key=THE_GRAPH_API_KEY,
            pool="0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",  # USDC/ETH
            loader_type=LoaderType.CSV)

    data = loader.read(with_run=True)
    assert data is not None
    assert len(data) > 0
    assert data["tvl"].dtype == "float64"
    assert data["volume"].dtype == "float64"
    assert data["fees"].dtype == "float64"
    data_reset = data.reset_index()
    assert data_reset["date"].dtype == "datetime64[ns, UTC]"

    read_data = loader.read(with_run=False)
    assert read_data is not None
    assert len(read_data) > 0
    assert read_data["tvl"].dtype == "float64"
    assert read_data["volume"].dtype == "float64"
    assert read_data["fees"].dtype == "float64"
    read_data_reset = read_data.reset_index()
    assert read_data_reset["date"].dtype == "datetime64[ns, UTC]"
    assert len(read_data_reset) == len(data_reset), "Data length mismatch"
