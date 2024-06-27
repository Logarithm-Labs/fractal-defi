import warnings
warnings.filterwarnings('ignore')

from typing import List

import numpy as np

from sklearn.model_selection import ParameterGrid

from fractal.loaders import BinanceDayPriceLoader, LoaderType
from fractal.core.base import Observation
from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from binance_entity import BinanceGlobalState
from holder_strategy import BinanceHodlerStrategy


# Load prices from Binance and build observations
def build_observations() -> List[Observation]:
    # Load prices from Binance
    binance_prices = BinanceDayPriceLoader('BTCUSDT', loader_type=LoaderType.CSV).read(with_run=True)

    # Build observations list
    observations: List[Observation] = [
        Observation(timestamp=timestamp, states={'exchange': BinanceGlobalState(price=price)})
        for timestamp, price in zip(binance_prices.index, binance_prices['price'])
    ]
    return observations


# Build a grid of parameters to search
def build_grid() -> ParameterGrid:
    grid = ParameterGrid({
        'BUY_PRICE': np.linspace(50_000, 60_000, 3),
        'SELL_PRICE': np.linspace(60_000, 70_000, 3),
        'TRADE_SHARE': np.linspace(0.1, 0.9, 3),
        'INITIAL_BALANCE': [100_000]
    })
    return grid


if __name__ == '__main__':
    # Define MLFlow and Experiment configurations
    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri='http://127.0.01:5000',
        experiment_name='binance_hodler_btc_0'
    )
    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=BinanceHodlerStrategy,
        fractal_observations=build_observations(),
        window_size=24,
        params_grid=build_grid(),
        debug=True,
    )
    pipeline: DefaultPipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    pipeline.run()
