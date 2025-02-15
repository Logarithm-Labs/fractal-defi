from typing import List
from datetime import datetime
from dataclasses import dataclass

import pandas as pd

from fractal.core.base.strategy import NamedEntity
from fractal.loaders.base_loader import LoaderType

from fractal.loaders.hyperliquid import HyperliquidFundingRatesLoader, HyperLiquidPerpsPricesLoader
from fractal.loaders.structs import PriceHistory, RateHistory

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState, UniswapV3SpotEntity, HyperliquidEntity, HyperLiquidGlobalState
from fractal.strategies.basis_trading_strategy import BasisTradingStrategy, BasisTradingStrategyHyperparams


@dataclass
class HyperliquidBasisParams(BasisTradingStrategyHyperparams):
    """
    Parameters for the HyperliquidBasis strategy.
    """
    EXECUTION_COST: float


class HyperliquidBasis(BasisTradingStrategy):

    def set_up(self):
        """
        Set up the strategy by registering the hedge and spot entities.
        """
        # include execution cost for spread
        self.register_entity(NamedEntity(entity_name='HEDGE', entity=HyperliquidEntity(trading_fee=self._params.EXECUTION_COST)))
        self.register_entity(NamedEntity(entity_name='SPOT', entity=UniswapV3SpotEntity(trading_fee=self._params.EXECUTION_COST)))
        super().set_up()


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
    observations_df = observations_df.dropna()
    observations_df = observations_df.loc[start_time:end_time]
    if start_time is None:
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
        ticker: str, start_time: datetime = None, end_time: datetime = None, fidelity: str = 'hour',
    ) -> List[Observation]:
    """
    Build observations for the ManagedBasisStrategy from the given start and end time.
    """
    rate_data: RateHistory = HyperliquidFundingRatesLoader(
        ticker, loader_type=LoaderType.CSV).read(with_run=True)
    if fidelity == 'day':
        interval = '1d'
        rate_data = rate_data.resample(interval).sum()
    elif fidelity == 'hour':
        interval = '1h'
    elif fidelity == 'minute':
        interval = '1m'
    prices: PriceHistory = HyperLiquidPerpsPricesLoader(
        ticker, interval=interval, loader_type=LoaderType.CSV,
        start_time=start_time, end_time=end_time).read(with_run=True)
    return get_observations(rate_data, prices, start_time, end_time)


if __name__ == '__main__':
    # Set up
    ticker: str = 'BTC'

    # Init the strategy
    params: HyperliquidBasisParams = HyperliquidBasisParams(
        MIN_LEVERAGE=1,
        MAX_LEVERAGE=10,
        TARGET_LEVERAGE=2.5,
        INITIAL_BALANCE=1_000_000,
        EXECUTION_COST=0.005,
    )
    strategy: HyperliquidBasis = HyperliquidBasis(debug=True, params=params)

    # Build observations
    entities = strategy.get_all_available_entities().keys()
    observations: List[Observation] = build_observations(
        ticker=ticker,
        start_time=datetime(2025, 1, 1),
        end_time=datetime(2025, 2, 1),
        fidelity='day'
    )
    observation0 = observations[0]
    # check if the observation has the right entities
    assert all(entity in observation0.states for entity in entities)

    # Run the strategy
    result = strategy.run(observations)
    print(result.get_default_metrics())  # show metrics
    result.to_dataframe().to_csv('basis.csv')  # save results of strategy states
