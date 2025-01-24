import os

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.thegraph.uniswap_v3 import (
    UniswapV3EthereumPoolHourDataLoader, UniswapV3EthereumPricesLoader, UniswapV3ArbitrumPoolDayDataLoader
)

api_key = os.getenv('THE_GRAPH_API_KEY')


# Load hourly pool data for Uniswap V3 Ethereum pool
pool_data = UniswapV3ArbitrumPoolDayDataLoader(
    api_key=api_key, pool="0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8", loader_type=LoaderType.CSV)
prices = UniswapV3EthereumPricesLoader(
    api_key=api_key, pool="0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8", loader_type=LoaderType.CSV)


# Load the data
print(pool_data.read(True))
