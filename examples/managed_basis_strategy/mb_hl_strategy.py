from typing import List
from datetime import datetime, UTC
from dataclasses import dataclass

import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.hyperliquid import HyperliquidFundingRatesLoader, HyperLiquidPerpsPricesLoader
from fractal.loaders.binance import BinancePriceLoader
from fractal.loaders.structs import PriceHistory, RateHistory

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState, HyperLiquidGlobalState
from fractal.strategies.basis_trading_strategy import BasisTradingStrategyHyperparams
from fractal.strategies.hyperliquid_basis import HyperliquidBasis


@dataclass
class HyperliquidBasisParams(BasisTradingStrategyHyperparams):
    """
    Parameters for the HyperliquidBasis strategy.
    """
    EXECUTION_COST: float


def get_observations(
        rate_data: RateHistory, price_data: PriceHistory,
        start_time: datetime = None, end_time: datetime = None
    ) -> List[Observation]:
    """
    Get observations from the pool and price data for the ManagedBasisStrategy.

    Returns:
        List[Observation]: The observation list for ManagedBasisStrategy.
    """
    observations_df: pd.DataFrame = price_data.join(rate_data)
    observations_df['rate'] = observations_df['rate'].fillna(0)
    observations_df = observations_df.loc[start_time:end_time]
    observations_df = observations_df.dropna()
    start_time = observations_df.index.min()
    if end_time is None:
        end_time = observations_df.index.max()
    observations_df = observations_df.sort_index()
    return [
        Observation(
            timestamp=timestamp,
            states={
                'SPOT': UniswapV3LPGlobalState(price=price, tvl=0, volume=0, fees=0, liquidity=0),
                'HEDGE': HyperLiquidGlobalState(mark_price=price, funding_rate=rate)
            }
        ) for timestamp, (price, rate) in observations_df.iterrows()
    ]


def build_observations(
        ticker: str, start_time: datetime = None, end_time: datetime = None, fidelity: str = '1h',
    ) -> List[Observation]:
    """
    Build observations for the ManagedBasisStrategy from the given start and end time.
    """
    rate_data: RateHistory = HyperliquidFundingRatesLoader(
        ticker, start_time=start_time, end_time=end_time).read(with_run=True)
    if fidelity == '1d':
        rate_data = rate_data.resample(fidelity).sum()
    # use binance perp price because hyperliquid has limitations for klines limit
    prices: PriceHistory = BinancePriceLoader(
        ticker+'USDT', interval=fidelity, loader_type=LoaderType.CSV,
        start_time=start_time, end_time=end_time).read(with_run=True)
    return get_observations(rate_data, prices, start_time, end_time)


if __name__ == '__main__':
    # Set up
    ticker: str = 'BTC'
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 1, 1, tzinfo=UTC)
    fidelity = '1d'
    MIN_LVG = 1
    TARGET_LVG = 4
    MAX_LVG = 8
    HyperliquidBasis.MAX_LEVERAGE = 45

    # Init the strategy
    params: HyperliquidBasisParams = HyperliquidBasisParams(
        MIN_LEVERAGE=MIN_LVG,
        MAX_LEVERAGE=MAX_LVG,
        TARGET_LEVERAGE=TARGET_LVG,
        INITIAL_BALANCE=1_000_000,
        EXECUTION_COST=0.005,
    )
    strategy: HyperliquidBasis = HyperliquidBasis(debug=True, params=params)

    # Build observations
    entities = strategy.get_all_available_entities().keys()
    observations: List[Observation] = build_observations(
        ticker=ticker,
        start_time=start_time,
        end_time=end_time,
        fidelity=fidelity
    )
    observation0 = observations[0]
    # check if the observation has the right entities
    assert all(entity in observation0.states for entity in entities)

    # Run the strategy
    result = strategy.run(observations)
    print(result.get_default_metrics())  # show metrics
    result.to_dataframe().to_csv(f'{ticker}.csv')  # save results of strategy states
