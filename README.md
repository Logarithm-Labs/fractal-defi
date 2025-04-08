# Fractal
[![PyPI version](https://badge.fury.io/py/fractal-defi.svg)](https://badge.fury.io/py/fractal-defi)
[![Python Versions](https://img.shields.io/pypi/pyversions/fractal-defi.svg)](https://pypi.org/project/fractal-defi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/BSD)

Fractal is the ultimate DeFi research library for strategies development and backtesting created by Logarithm Labs.

## Overview

Fractal offers a modular architecture where each component plays a crucial role in constructing and managing complex DeFi strategies. Whether you’re experimenting with automated market-making, liquidity management, or yield strategies, Fractal provides the building blocks and pipelines necessary for thorough analysis and execution.

- **[Entities](https://github.com/Logarithm-Labs/fractal-defi/tree/main/fractal/core/entities)** represent a scope of DeFi primitives with behavior replicated in Python. Each entity holds global market states (e.g., prices, yield rates) along with its own internal states (e.g., balances, open positions, LP positions).
- **[Strategies](https://github.com/Logarithm-Labs/fractal-defi/tree/main/fractal/strategies)** contain management logic on top of the entites with actions execution.
- **[Loaders](https://github.com/Logarithm-Labs/fractal-defi/tree/main/fractal/loaders)** simplify the process of gathering and preparing market data from various DeFi protocols. Fractal provides a collection of ad-hoc ETL tools to quickly load data, making it easier to construct observations for dynamic strategy execution.
- Fractal’s engine includes robust backtesting and simulation **[pipelines](https://github.com/Logarithm-Labs/fractal-defi/blob/main/fractal/core/pipeline.py)** for strategy optimization powered by [MLFlow](https://mlflow.org/):

## Quick Start

Install:

```bash
pip install fractal-defi
```

Start with a built-in strategies:
Run MLFlow server:
```bash
mlflow server --host 127.0.0.1 --port 8080
```

Set up your experiment:
```python
import numpy as np
import pandas as pd

from typing import List
from datetime import datetime, UTC
from sklearn.model_selection import ParameterGrid

from fractal.loaders import PriceHistory, RateHistory
from fractal.loaders import HyperliquidFundingRatesLoader, HyperLiquidPerpsPricesLoader

from fractal.core.base import Observation
from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)
from fractal.core.entities import UniswapV3LPGlobalState, HyperLiquidGlobalState

from fractal.strategies.hyperliquid_basis import HyperliquidBasis


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
                'SPOT': UniswapV3LPGlobalState(price=price, tvl=0, volume=0, fees=0, liquidity=0),  # we need only spot price
                'HEDGE': HyperLiquidGlobalState(mark_price=price, funding_rate=rate)
            }
        ) for timestamp, (price, rate) in observations_df.iterrows()
    ]


def build_observations(
        ticker: str, start_time: datetime = None, end_time: datetime = None,
    ) -> List[Observation]:
    """
    Build observations for the ManagedBasisStrategy from the given start and end time.
    """
    rate_data: RateHistory = HyperliquidFundingRatesLoader(
          ticker, start_time=start_time, end_time=end_time).read(with_run=True)
    prices: PriceHistory = HyperLiquidPerpsPricesLoader(
          ticker, interval='1h', start_time=start_time, end_time=end_time).read(with_run=True)
    return get_observations(rate_data, prices, start_time, end_time)


def build_grid():
    raw_grid = ParameterGrid({
        'MIN_LEVERAGE': np.arange(1, 12, 1).tolist(),
        'TARGET_LEVERAGE': np.arange(1, 12, 1).tolist(),
        'MAX_LEVERAGE': np.arange(1, 12, 1).tolist(),
        'INITIAL_BALANCE': [1_000_000]
    })

    valid_grid = [
        params for params in raw_grid
        if round(params['MIN_LEVERAGE'], 1) < round(params['TARGET_LEVERAGE'], 1) < round(params['MAX_LEVERAGE'], 1)
    ]
    return valid_grid


if __name__ == '__main__':
    # Strategy environment
    ticker: str = 'BTC'
    start_time = datetime(2025, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 3, 1, tzinfo=UTC)
    experiment_name = f'hl_basis_{ticker}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'
    HyperliquidBasis.MAX_LEVERAGE = 45
    
    # Mlflow setup
    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri='http://127.0.01:8080',
        experiment_name=experiment_name,
    )

    # Load data and build observations
    observations = build_observations(ticker, start_time, end_time)
    assert len(observations) > 0

    # Experiment setup
    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=HyperliquidBasis,
        backtest_observations=observations,
        window_size=24,  # number of scenarios from history
        params_grid=build_grid(),
        debug=True,
    )

    # Run the DefualtPipeline
    pipeline: DefaultPipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    pipeline.run()

```
Run your pipeline:
```bash
╰─➤  python3 fractal_basis.py
🏃 View run whimsical-sheep-752 at: http://127.0.01:8080/#/experiments/743858278487100844/runs/eeb3db5833b54f38aa9eb4b31990f6e2
🧪 View experiment at: http://127.0.01:8080/#/experiments/743858278487100844
🏃 View run useful-crab-883 at: http://127.0.01:8080/#/experiments/743858278487100844/runs/d728d6f94b1d4f708e96e628111f215e
🧪 View experiment at: http://127.0.01:8080/#/experiments/743858278487100844
🏃 View run bittersweet-goat-277 at: http://127.0.01:8080/#/experiments/743858278487100844/runs/31fc73bfef6d47e296dea8880f161821
🧪 View experiment at: http://127.0.01:8080/#/experiments/743858278487100844
🏃 View run gifted-gnu-901 at: http://127.0.01:8080/#/experiments/743858278487100844/runs/c88004f211c74be8964e992a168addf6
🧪 View experiment at: http://127.0.01:8080/#/experiments/743858278487100844
```

## Examples

Explore our example strategies to jumpstart your DeFi research:

- **HODL Strategy:** Start with a straightforward HODL.  
  [View Example](https://github.com/Logarithm-Labs/fractal-defi/tree/main/examples/holder)

- **Tau Reset Strategy:** Focused on active liquidity management within Uniswap V3.  
  [View Example](https://github.com/Logarithm-Labs/fractal-defi/tree/main/examples/tau_strategy)

- **Managed Basis Strategy:** Strategies addressing basis management.  
  [View Example](https://github.com/Logarithm-Labs/fractal-defi/tree/main/examples/managed_basis_strategy)

- **Agent Trading Backtesting:** A comprehensive guide for backtesting AI agent trading strategies.  
  [View Example](https://github.com/Logarithm-Labs/fractal-defi/tree/main/examples/agent_trader)

## Documentation

For detailed technical documentation and comprehensive code references, please explore the following resources:

- **Fractal Tech Docs:**  
  Dive into in-depth guides, technical details, and conceptual overviews at our [Tech Docs site](https://logarithm-labs.gitbook.io/fractal).

- **Code References:**  
  Access comprehensive API documentation and code references at our [Code References page](https://logarithm-labs.gitbook.io/fractal).
