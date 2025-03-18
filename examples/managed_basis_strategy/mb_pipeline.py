import os
import warnings

import numpy as np
from datetime import datetime
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from mb_hl_strategy import HyperliquidBasis, build_observations

warnings.filterwarnings('ignore')


def build_grid():
    raw_grid = ParameterGrid({
        'MIN_LEVERAGE': np.arange(1, 12, 1).tolist(),
        'TARGET_LEVERAGE': np.arange(1, 12, 1).tolist(),
        'MAX_LEVERAGE': np.arange(1, 12, 1).tolist(),
        'EXECUTION_COST': [0.002],
        'INITIAL_BALANCE': [1_000_000]
    })

    valid_grid = [
        params for params in raw_grid
        if round(params['MIN_LEVERAGE'], 1) < round(params['TARGET_LEVERAGE'], 1) < round(params['MAX_LEVERAGE'], 1)
    ]
    print(f'Length of valid grid: {len(valid_grid)}')
    return valid_grid


if __name__ == '__main__':
    ticker: str = 'BTC'
    start_time = datetime(2022, 1, 1)
    end_time = datetime(2025, 1, 1)
    fidelity = '1h'
    experiment_name = f'mb_binance_{fidelity}_{ticker}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'

    # Define MLFlow and Experiment configurations
    mlflow_uri = os.getenv('MLFLOW_URI')
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not mlflow_uri:
        raise ValueError("MLFLOW_URI isn't set.")

    if not aws_key or not aws_secret:
        warnings.warn("AWS_ACCESS_KEY_ID или AWS_SECRET_ACCESS_KEY are not set", RuntimeWarning)

    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )

    observations = build_observations(ticker, start_time, end_time, fidelity=fidelity)
    assert len(observations) > 0
    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=HyperliquidBasis,
        backtest_observations=observations,
        window_size=24 * 30,
        params_grid=build_grid(),
        debug=True,
    )
    pipeline: DefaultPipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    pipeline.run()
