from datetime import UTC, datetime
from typing import List

import pandas as pd

from fractal.core.base import Observation
from fractal.core.entities import HyperliquidGlobalState, UniswapV3LPGlobalState
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.binance import BinanceFundingLoader, BinancePriceLoader
from fractal.loaders.hyperliquid import HyperliquidFundingRatesLoader, HyperliquidPerpsPricesLoader
from fractal.loaders.structs import PriceHistory, RateHistory
from fractal.strategies.hyperliquid_basis import HyperliquidBasis, HyperliquidBasisParams


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
                'HEDGE': HyperliquidGlobalState(mark_price=price, funding_rate=rate)
            }
        ) for timestamp, (price, rate) in observations_df.iterrows()
    ]


def build_observations(
        ticker: str, start_time: datetime = None, end_time: datetime = None,
        fidelity: str = '1h', use_binance_data: bool = True,
    ) -> List[Observation]:
    """Build observations for the basis strategy from the given window.

    Two data sources:

    * ``use_binance_data=True`` (default) — Binance public REST for both
      perp prices and funding rates. Has multi-year history and no rate
      limit on tickers, so it's the safe choice for any window.
    * ``use_binance_data=False`` — Hyperliquid's ``candleSnapshot`` and
      ``fundingHistory`` info endpoints. ``candleSnapshot`` only retains
      the last few hundred days (asset-dependent); requests further back
      return empty.
    """
    if use_binance_data:
        prices: PriceHistory = BinancePriceLoader(
            ticker+'USDT', interval=fidelity, loader_type=LoaderType.CSV,
            start_time=start_time, end_time=end_time).read(with_run=True)
        rate_data: RateHistory = BinanceFundingLoader(
            ticker=ticker+'USDT', start_time=start_time, end_time=end_time).read(with_run=True)
    else:
        prices: PriceHistory = HyperliquidPerpsPricesLoader(
            ticker=ticker, interval=fidelity,
            start_time=start_time, end_time=end_time).read(with_run=True)
        rate_data: RateHistory = HyperliquidFundingRatesLoader(
            ticker=ticker, start_time=start_time, end_time=end_time).read(with_run=True)

    if fidelity == '1d':
        rate_data = rate_data.resample(fidelity).sum()

    return get_observations(rate_data, prices, start_time, end_time)


if __name__ == '__main__':
    # Set up
    ticker: str = 'LINK'
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 5, 1, tzinfo=UTC)
    fidelity = '1h'
    MIN_LVG = 1
    TARGET_LVG = 3
    MAX_LVG = 5
    # Hyperliquid per-asset margin cap (entity config). Default is
    # also 10 — this line is for explicitness; remove it to use the
    # default.
    HyperliquidBasis.MAX_LEVERAGE = 10

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
