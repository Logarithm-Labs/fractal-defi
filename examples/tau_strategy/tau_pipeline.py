import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from datetime import datetime
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from tau_strategy import TauResetStrategy, build_observations, THE_GRAPH_API_KEY


def build_grid():
    return ParameterGrid({
        'TAU': np.linspace(start=1, stop=15, num=15, dtype=int),
        'INITIAL_BALANCE': [1_000_000]
    })


if __name__ == '__main__':
    ticker: str = 'ETHUSDT'
    pool_address: str = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    start_time=datetime(2022, 1, 1)
    end_time=datetime(2025, 1, 1)
    experiment_name = f'tau_{ticker}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'
    TauResetStrategy.token0_decimals = 6
    TauResetStrategy.token1_decimals = 18
    TauResetStrategy.tick_spacing = 60
    # Define MLFlow and Experiment configurations
    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri='http://127.0.01:5000',
        experiment_name=experiment_name,
    )
    observations = build_observations(ticker, pool_address, THE_GRAPH_API_KEY, start_time, end_time)
    assert len(observations) > 0
    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=TauResetStrategy,
        backtest_observations=observations,
        window_size=24,
        params_grid=build_grid(),
        debug=True,
    )
    pipeline: DefaultPipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    pipeline.run()
