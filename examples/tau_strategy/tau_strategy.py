import os

from typing import List
from datetime import datetime, UTC

import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3 import (
    UniswapV3EthereumPoolHourDataLoader, EthereumUniswapV3Loader, UniswapV3EthereumPoolMinuteDataLoader
)
from fractal.loaders.binance import BinanceHourPriceLoader, BinanceMinutePriceLoader
from fractal.loaders.structs import PriceHistory, PoolHistory

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState
from fractal.strategies.tau_reset_strategy import TauResetParams, TauResetStrategy


THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')


def get_observations(
        pool_data: PoolHistory, price_data: PriceHistory,
        start_time: datetime = None, end_time: datetime = None
    ) -> List[Observation]:
    """
    Get observations from the pool and price data for the TauResetStrategy.

    Returns:
        List[Observation]: The observation list for TauResetStrategy.
    """
    observations_df: pd.DataFrame = pool_data.join(price_data)
    observations_df = observations_df.dropna()
    observations_df = observations_df.loc[start_time:end_time]
    if start_time is None:
        start_time = observations_df.index.min()
    if end_time is None:
        end_time = observations_df.index.max()
    observations_df = observations_df[observations_df.tvl > 0]
    observations_df = observations_df.sort_index()
    return [
        Observation(
            timestamp=timestamp,
            states={
                'UNISWAP_V3': UniswapV3LPGlobalState(price=price, tvl=tvls, volume=volume, fees=fees, liquidity=liquidity),
            }
        ) for timestamp, (tvls, volume, fees, liquidity, price) in observations_df.iterrows()
    ]


def build_observations(
        ticker: str, pool_address: str, api_key: str,
        start_time: datetime = None, end_time: datetime = None, fidelity: str = 'hour',
    ) -> List[Observation]:
    """
    Build observations for the TauResetStrategy from the given start and end time.
    """
    if fidelity == 'hour':
        pool_data: PoolHistory = UniswapV3EthereumPoolHourDataLoader(
            api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        binance_prices: PriceHistory = BinanceHourPriceLoader(ticker, loader_type=LoaderType.CSV).read(with_run=True)
    elif fidelity == 'minute':
        pool_data: PoolHistory = UniswapV3EthereumPoolMinuteDataLoader(
            api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        binance_prices: PriceHistory = BinanceMinutePriceLoader(ticker, loader_type=LoaderType.CSV,
                                                                start_time=start_time, end_time=end_time).read(with_run=True)
    else:
        raise ValueError("Fidelity must be either 'hour' or 'minute'.")
    return get_observations(pool_data, binance_prices, start_time, end_time)


if __name__ == '__main__':
    # Set up
    ticker: str = 'ETHUSDT'
    pool_address: str = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')

    # Load data
    token0_decimals, token1_decimals = EthereumUniswapV3Loader(
        THE_GRAPH_API_KEY, loader_type=LoaderType.CSV).get_pool_decimals(pool_address)

    # Init the strategy
    params: TauResetParams = TauResetParams(TAU=90, INITIAL_BALANCE=1_000_000)
    TauResetStrategy.token0_decimals = token0_decimals
    TauResetStrategy.token1_decimals = token1_decimals
    TauResetStrategy.tick_spacing = 60
    strategy: TauResetStrategy = TauResetStrategy(debug=True, params=params)

    # Build observations
    entities = strategy.get_all_available_entities().keys()
    observations: List[Observation] = build_observations(
        ticker=ticker, pool_address=pool_address, api_key=THE_GRAPH_API_KEY,
        start_time=datetime(2025, 1, 11, tzinfo=UTC), end_time=datetime(2025, 2, 11, tzinfo=UTC),
        fidelity='hour'
    )
    observation0 = observations[0]
    # check if the observation has the right entities
    assert all(entity in observation0.states for entity in entities)

    # Run the strategy
    result = strategy.run(observations)
    print(result.get_default_metrics())  # show metrics
    result.to_dataframe().to_csv('tau_strategy_result.csv')  # save results of strategy states
    print(result.to_dataframe().iloc[-1])  # show the last state of the strategy
