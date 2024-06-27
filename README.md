# Fractal
Fractal is an ultimate DeFi research library for strategy development and fractaling created by Logarithm Labs.

## Docs
[Technical docs.](https://logarithm-labs.gitbook.io/fractal/)


## How to install?
```
pip install fractal-defi
```

## Quick start with ready-to-go strategies

Run MLFlow locally or self-hosted:
```
pip install mlflow
mlflow io
```

Than just import pre-built strategies and run the pipeline:
```python
from typing import List

from sklearn.model_selection import ParameterGrid

from fractal.strategies import GMXV2UniswapV3Basis, BasisTradingStrategyHyperparams
from fractal.loaders import BinanceDayPriceLoader, LoaderType

from fractal.core.pipeline import DefaultPipeline, MLFlowConfig, ExperimentConfig
from fractal.core.base import Observation


# Load data and build observations
def build_observations() -> List[Observation]:
  ...


# Build a grid of parameters to search
def build_grid() -> ParameterGrid:
  ...


# Define MLFlow and Experiment configurations
mlflow_config: MLFlowConfig = MLFlowConfig(
    mlflow_uri='http://127.0.01:5000',
    experiment_name='my_strategy'
)
experiment_config: ExperimentConfig = ExperimentConfig(
    strategy=GMXV2UniswapV3Basis,
    fractal_observations=build_observations(),
    window_size=24,
    params_grid=build_grid(),
    debug=True
)
pipeline: DefaultPipeline = DefaultPipeline(
    experiment_config=experiment_config,
    mlflow_config=mlflow_config
)
pipeline.run()
```
### See more detailes examples in /examples directory

## Changelog
See our discord dev channel for updates.
