import warnings
warnings.filterwarnings('ignore')

from typing import List
from datetime import datetime
from sklearn.model_selection import ParameterGrid

from fractal.loaders import LoaderType
from fractal.loaders.binance import BinanceKlinesLoader
from fractal.core.base import Observation
from fractal.core.base.observations import SQLiteObservationsStorage
from fractal.core.entities.single_spot_exchange import SingleSpotExchangeGlobalState
from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from agent_strategy import AgentTradingStrategy
from prompts import BULLISH_PROMPT, BEARISH_PROMPT, NEUTRAL_PROMPT


# Load prices from Binance and build observations
def build_observations() -> List[Observation]:
    # Load prices from Binance
    binance_klines = BinanceKlinesLoader('BTCUSDT', interval='1d',
                                         start_time=datetime(2024, 1, 1),
                                         end_time=datetime(2025, 1, 1),
                                         loader_type=LoaderType.CSV).read(with_run=True)

    # Build observations list
    observations: List[Observation] = [
        Observation(timestamp=timestamp, states={'exchange': SingleSpotExchangeGlobalState(open=o, high=h, low=l, close=c)})
        for timestamp, o, h, l, c in zip(binance_klines.index, binance_klines['open'], binance_klines['high'],
                                         binance_klines['low'], binance_klines['close'])
    ]
    return observations


# Build a grid of parameters to search
def build_grid() -> ParameterGrid:
    grid = ParameterGrid({
        'PROMPT': [BULLISH_PROMPT, BEARISH_PROMPT, NEUTRAL_PROMPT],
        'MODEL': ['gpt-4o', 'o3-mini'],
        'WINDOW_SIZE': [7, 14, 30],
        'INITIAL_BALANCE': [100_000]
    })
    return grid


if __name__ == '__main__':
    # Define MLFlow and Experiment configurations
    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri='http://127.0.01:8080',
        experiment_name=f'agent_trader_btc_v0.1-2024'
    )
    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=AgentTradingStrategy,
        backtest_observations=build_observations(),
        observations_storage_type=SQLiteObservationsStorage,
        params_grid=build_grid(),
        debug=True,
    )
    pipeline: DefaultPipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    pipeline.run()
