"""Deterministic, offline benchmarks for representative backtest hot paths."""
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from fractal.core import pipeline as pipeline_module
from fractal.core.base import BaseStrategy, BaseStrategyParams, NamedEntity, Observation
from fractal.core.entities.protocols.uniswap_v3_lp import UniswapV3LPConfig, UniswapV3LPEntity, UniswapV3LPGlobalState
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeGlobalState
from fractal.core.pipeline import DefaultPipeline, ExperimentConfig, MLflowConfig

pytestmark = pytest.mark.benchmark
UTC = timezone.utc
BARS_PER_YEAR = 365 * 24


@dataclass
class _Params(BaseStrategyParams):
    variant: int = 0


class _HoldStrategy(BaseStrategy[_Params]):
    """Minimal strategy: benchmark engine/state-copy overhead, not a model."""

    def set_up(self) -> None:
        self.register_entity(NamedEntity("SPOT", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


def _observations(count: int = BARS_PER_YEAR):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Observation(
            timestamp=start + timedelta(hours=index),
            states={"SPOT": SimpleSpotExchangeGlobalState(close=2_000.0 + index % 100)},
        )
        for index in range(count)
    ]


def test_strategy_run_one_year(benchmark):
    observations = _observations()

    def run():
        return _HoldStrategy(params=_Params()).run(observations)

    result = benchmark(run)
    assert len(result.timestamps) == BARS_PER_YEAR


def test_uniswap_v3_update_state_one_year(benchmark):
    states = [
        UniswapV3LPGlobalState(
            price=1.0 + (index % 100 - 50) / 10_000,
            tvl=1_000_000.0,
            volume=100_000.0,
            fees=300.0,
            liquidity=10_000_000.0,
        )
        for index in range(BARS_PER_YEAR)
    ]

    def run():
        entity = UniswapV3LPEntity(config=UniswapV3LPConfig(fee_model="aggregate"))
        entity.update_state(states[0])
        entity.action_deposit(10_000.0)
        entity.action_open_position(9_000.0, 0.9, 1.1)
        for state in states[1:]:
            entity.update_state(state)
        return entity

    entity = benchmark(run)
    assert entity.balance > 0


def test_pipeline_sixteen_cell_grid(benchmark, monkeypatch):
    """Exercise the full grid/backtest path while excluding external MLflow I/O."""
    observations = _observations(24 * 7)
    params_grid = [_Params(variant=index) for index in range(16)]
    pipeline = DefaultPipeline(
        MLflowConfig(experiment_name="benchmark", mlflow_uri="stub://offline"),
        ExperimentConfig(
            strategy_type=_HoldStrategy,
            params_grid=params_grid,
            backtest_observations=observations,
        ),
    )

    monkeypatch.setattr(pipeline, "_connect_mlflow", lambda: None)
    monkeypatch.setattr(pipeline_module.mlflow, "start_run", lambda **_kwargs: nullcontext())
    for name in ("end_run", "log_params", "log_metrics", "log_text", "log_artifact"):
        monkeypatch.setattr(pipeline_module.mlflow, name, lambda *_args, **_kwargs: None)

    benchmark(pipeline.run)
    assert pipeline._connected is True
