from fractal.loaders.aave import AaveV2EthereumLoader, AaveV3ArbitrumLoader, AaveV3EthereumLoader, AaveV3RatesLoader
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.binance import (
    BinanceDayPriceLoader,
    BinanceFundingLoader,
    BinanceHourPriceLoader,
    BinanceKlinesLoader,
    BinancePriceLoader,
    BinanceSpotPriceLoader,
)
from fractal.loaders.gmx_v1 import GMXV1FundingLoader
from fractal.loaders.hyperliquid import (  # Pre-1.3.0 alias.
    HyperliquidFundingRatesLoader,
    HyperliquidPerpsKlinesLoader,
    HyperliquidPerpsPricesLoader,
    HyperLiquidPerpsPricesLoader,
)
from fractal.loaders.simulations import ConstantFundingsLoader, MonteCarloHourPriceLoader, MonteCarloPriceLoader
from fractal.loaders.structs import (
    FundingHistory,
    KlinesHistory,
    LendingHistory,
    PoolHistory,
    PriceHistory,
    RateHistory,
    TrajectoryBundle,
)
from fractal.loaders.thegraph import (
    ArbitrumGraphLoader,
    BaseGraphLoader,
    EthereumUniswapV2PoolDataLoader,
    GraphLoaderException,
    StETHLoader,
    UniswapV3ArbitrumPoolDayDataLoader,
    UniswapV3ArbitrumPoolHourDataLoader,
    UniswapV3ArbitrumPricesLoader,
    UniswapV3EthereumPoolDayDataLoader,
    UniswapV3EthereumPoolHourDataLoader,
    UniswapV3EthereumPoolMinuteDataLoader,
    UniswapV3EthereumPricesLoader,
)

__all__ = [
    "Loader",
    "LoaderType",
    "FundingHistory",
    "PoolHistory",
    "PriceHistory",
    "RateHistory",
    "LendingHistory",
    "KlinesHistory",
    "TrajectoryBundle",
    "AaveV2EthereumLoader",
    "AaveV3ArbitrumLoader",
    "AaveV3EthereumLoader",
    "AaveV3RatesLoader",
    "BinanceDayPriceLoader",
    "BinanceFundingLoader",
    "BinanceHourPriceLoader",
    "GMXV1FundingLoader",
    "MonteCarloPriceLoader",
    "MonteCarloHourPriceLoader",
    "UniswapV3ArbitrumPoolDayDataLoader",
    "UniswapV3ArbitrumPoolHourDataLoader",
    "UniswapV3EthereumPoolDayDataLoader",
    "UniswapV3EthereumPoolHourDataLoader",
    "StETHLoader",
    "ConstantFundingsLoader",
    "EthereumUniswapV2PoolDataLoader",
    "BaseGraphLoader",
    "GraphLoaderException",
    "ArbitrumGraphLoader",
    "UniswapV3ArbitrumPricesLoader",
    "UniswapV3EthereumPricesLoader",
    "UniswapV3EthereumPoolMinuteDataLoader",
    "HyperliquidFundingRatesLoader",
    "HyperliquidPerpsPricesLoader",
    "HyperliquidPerpsKlinesLoader",
    # Pre-1.3.0 alias (deprecated; will be removed in a future major release).
    "HyperLiquidPerpsPricesLoader",
    "BinanceKlinesLoader",
    "BinancePriceLoader",
    "BinanceSpotPriceLoader",
]
