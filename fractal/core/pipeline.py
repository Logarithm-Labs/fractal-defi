"""Pipeline machinery for running parameterised strategy experiments.

A :class:`Pipeline` couples MLflow tracking with a :class:`Launcher` that
runs a strategy across a grid of hyperparameters. ``DefaultPipeline``
implements the standard flow: per-grid-point single-trajectory backtest,
optional Monte-Carlo trajectory bundle, and optional sliding-window
scenarios — each writes its artifacts into one MLflow run.

Design notes:

* MLflow is **lazily connected** on the first call to ``run`` or
  ``grid_step`` (P2-5.5). Construction does not require network access,
  which makes the pipeline trivially testable and importable.
* AWS credentials in ``MLFlowConfig`` are only injected into the
  environment when explicitly provided — they no longer overwrite
  pre-existing ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` with
  empty strings (P1-5.1).
* ``mlflow.log_params`` requires a ``Mapping``; we coerce
  ``BaseStrategyParams`` (and its dataclass subclasses) via
  ``_params_to_dict`` so logging never crashes (P1-5.2).
* ``ExperimentConfig.step_size`` exposes the sliding-window stride that
  was previously hardcoded inside ``Launcher.run_scenario`` (P2-5.4).
"""
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from io import StringIO
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Type, Union)

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from fractal.core.base.observations import Observation, ObservationsStorage
from fractal.core.base.strategy import (BaseStrategy, BaseStrategyParams,
                                        StrategyMetrics, StrategyResult)
from fractal.core.launcher import Launcher


@dataclass
class MLFlowConfig:
    """
    MLFlow Configuration for the pipeline.

    Attributes:
        experiment_name (str): Name of the experiment.
        mlflow_uri (str): URI of the MLFlow server.
        tags (Optional[Dict[str, str]]): Tags for the experiment.
        aws_access_key_id (Optional[str]): AWS access key ID. ``None``
            (default) leaves ``$AWS_ACCESS_KEY_ID`` from the host
            environment intact; supply a non-empty string to override it.
        aws_secret_access_key (Optional[str]): AWS secret access key.
            Same semantics as ``aws_access_key_id``.
        run_name_formatter (Optional[Callable[[BaseStrategyParams | Dict], str]]):
            Formatter for the run name.
    """

    experiment_name: str
    mlflow_uri: str
    tags: Optional[Dict[str, str]] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    run_name_formatter: Optional[Callable[[Union[BaseStrategyParams, Dict]], str]] = None


@dataclass
class ExperimentConfig:
    """
    Configuration for the experiment.

    Attributes:
        strategy_type (Type[BaseStrategy]): Strategy class to run.
        params_grid (ParameterGrid): Grid of parameters to run the strategy.
        backtest_observations (Optional[List[Observation]]): Observations for backtesting.
        backtest_trajectories (Optional[List[List[Observation]]): Trajectories for backtesting iterations.
        window_size (Optional[int]): Window size for scenarios.
        step_size (int): Sliding-window stride for scenarios — defaults
            to 24 (one day of hourly bars). Override for non-hourly
            data or coarser/finer sampling.
        debug (Optional[bool]): Enable strategy debug logging.
    """
    strategy_type: Type[BaseStrategy]
    params_grid: Union[Iterable[BaseStrategyParams], ParameterGrid]
    observations_storage_type: Optional[Type[ObservationsStorage]] = None
    backtest_observations: Optional[List[Observation]] = None
    backtest_trajectories: Optional[List[List[Observation]]] = None
    window_size: Optional[int] = None
    step_size: int = 24
    debug: Optional[bool] = False


def _params_to_dict(params: Union[BaseStrategyParams, Mapping, Any]) -> Dict[str, Any]:
    """Coerce ``params`` to a plain dict for ``mlflow.log_params``.

    Accepts:
    * a ``Mapping`` — copied verbatim;
    * a dataclass instance (incl. ``BaseStrategyParams`` subclasses) —
      converted via :func:`dataclasses.asdict`;
    * any object exposing ``__dict__`` — copied as-is.

    Raises:
        TypeError: if none of the above applies.
    """
    if isinstance(params, Mapping):
        return dict(params)
    if is_dataclass(params) and not isinstance(params, type):
        return asdict(params)
    if hasattr(params, "__dict__"):
        return dict(params.__dict__)
    raise TypeError(
        f"cannot coerce params of type {type(params).__name__} to dict"
    )


class Pipeline(ABC):
    """
    Pipeline for running experiments.
    """
    def __init__(self, mlflow_config: MLFlowConfig, experiment_config: ExperimentConfig) -> None:
        """
        Initialize the pipeline.

        Args:
            mlflow_config (MLFlowConfig): MLFlow configuration to store metrics and artifacts.
            experiment_config (ExperimentConfig): Experiment configuration where defining steps to run.
        """
        self._mlflow_config: MLFlowConfig = mlflow_config
        self._config: ExperimentConfig = experiment_config
        self._connected: bool = False

    def _set_aws_env(self) -> None:
        """Inject AWS credentials into the environment iff configured.

        Empty strings or ``None`` leave the existing env vars (or AWS
        profile chain) intact — fixes P1-5.1, where the previous
        unconditional assignment clobbered host credentials with ``""``.
        """
        if self._mlflow_config.aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self._mlflow_config.aws_access_key_id
        if self._mlflow_config.aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self._mlflow_config.aws_secret_access_key

    def _connect_mlflow(self) -> None:
        """Configure tracking URI and ensure the experiment exists."""
        self._set_aws_env()
        mlflow.set_tracking_uri(self._mlflow_config.mlflow_uri)
        experiment = mlflow.get_experiment_by_name(self._mlflow_config.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                name=self._mlflow_config.experiment_name,
                tags=self._mlflow_config.tags,
            )
        mlflow.set_experiment(self._mlflow_config.experiment_name)

    def _ensure_connected(self) -> None:
        """Connect on first invocation; subsequent calls are no-ops (P2-5.5)."""
        if not self._connected:
            self._connect_mlflow()
            self._connected = True

    @abstractmethod
    def grid_step(self, params: Union[BaseStrategyParams, Dict]) -> None:
        """
        Run a step of the pipeline. Each step runs a full experiment with a set of parameters.

        Args:
            params (BaseStrategyParams | Dict): Parameters for the strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        """
        Run steps through the grid of parameters.
        """
        raise NotImplementedError


class DefaultPipeline(Pipeline):
    """Standard pipeline implementation: backtest + trajectories + scenario."""

    def _log_secondary_metrics(self, metrics: List[StrategyMetrics], prefix: str) -> None:
        """Aggregate per-trajectory metrics into mean/quantile/cvar series."""
        sharpe = np.array([metric.sharpe for metric in metrics])
        apy = np.array([metric.apy for metric in metrics])
        max_dd = np.array([metric.max_drawdown for metric in metrics])
        acc_return = np.array([metric.accumulated_return for metric in metrics])
        mlflow.log_metrics(
            {
                f"{prefix}_mean_sharpe": sharpe.mean(),
                f"{prefix}_mean_apy": apy.mean(),
                f"{prefix}_mean_accumulated_return": acc_return.mean(),
                f"{prefix}_mean_max_drawdown": max_dd.mean(),
                f"{prefix}_q05_sharpe": np.quantile(sharpe, 0.05),
                f"{prefix}_q95_sharpe": np.quantile(sharpe, 0.95),
                f"{prefix}_q05_apy": np.quantile(apy, 0.05),
                f"{prefix}_q95_apy": np.quantile(apy, 0.95),
                f"{prefix}_q05_max_drawdown": np.quantile(max_dd, 0.05),
                f"{prefix}_q95_max_drawdown": np.quantile(max_dd, 0.95),
                f"{prefix}_cvar05_sharpe": sharpe[sharpe < np.quantile(sharpe, 0.05)].mean(),
                f"{prefix}_cvar05_apy": apy[apy < np.quantile(apy, 0.05)].mean(),
                f"{prefix}_cvar05_max_drawdown": max_dd[max_dd < np.quantile(max_dd, 0.05)].mean(),
            }
        )

    def _log_backtest(self, launcher: Launcher) -> None:
        strategy_data: StrategyResult = launcher.run_strategy(
            self._config.backtest_observations, debug=self._config.debug
        )
        strategy_data_df: pd.DataFrame = strategy_data.to_dataframe()
        metrics: StrategyMetrics = strategy_data.get_metrics(strategy_data_df)
        csv_buffer = StringIO()
        strategy_data_df.to_csv(csv_buffer, index=False)
        mlflow.log_text(csv_buffer.getvalue(), "strategy_backtest_data.csv")
        mlflow.log_metrics(metrics.__dict__)
        if launcher.last_created_instance.debug:
            mlflow.log_artifact(launcher.last_created_instance.logger.logs_path)

    def _log_trajectories(self, launcher: Launcher) -> None:
        strategy_data_list: List[StrategyResult] = launcher.run_multiple_trajectories(
            self._config.backtest_trajectories, debug=False
        )
        metrics: List[StrategyMetrics] = [
            sd.get_metrics(sd.to_dataframe()) for sd in strategy_data_list
        ]
        metrics_df = pd.DataFrame([m.__dict__ for m in metrics])
        csv_buffer = StringIO()
        metrics_df.to_csv(csv_buffer, index=False)
        mlflow.log_text(csv_buffer.getvalue(), "backtest_trajectories_metrics.csv")
        self._log_secondary_metrics(metrics, prefix="backtest_trajectories")

    def _log_scenario(self, launcher: Launcher) -> None:
        strategy_data_list: List[StrategyResult] = launcher.run_scenario(
            self._config.backtest_observations,
            window_size=self._config.window_size,
            step_size=self._config.step_size,
            debug=False,
        )
        metrics: List[StrategyMetrics] = [
            sd.get_metrics(sd.to_dataframe()) for sd in strategy_data_list
        ]
        metrics_df = pd.DataFrame([m.__dict__ for m in metrics])
        csv_buffer = StringIO()
        metrics_df.to_csv(csv_buffer, index=False)
        mlflow.log_text(csv_buffer.getvalue(), "window_trajectories_metrics.csv")
        self._log_secondary_metrics(metrics, prefix="window_trajectories")

    def grid_step(self, params: Union[BaseStrategyParams, Dict]) -> None:
        """
        Run a step of the pipeline. Each step runs a full experiment with a set of parameters.
        Check ExperimentConfig for the different types of experiments that can be run.

        Args:
            params (BaseStrategyParams | Dict): Parameters for the strategy.
        """
        self._ensure_connected()
        launcher = Launcher(
            strategy_type=self._config.strategy_type, params=params,
            observations_storage_type=self._config.observations_storage_type,
        )
        run_name: Optional[str] = None
        if self._mlflow_config.run_name_formatter:
            run_name = self._mlflow_config.run_name_formatter(params)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(_params_to_dict(params))
            if self._config.backtest_observations:
                self._log_backtest(launcher)
            if self._config.backtest_trajectories:
                self._log_trajectories(launcher)
            if self._config.window_size:
                self._log_scenario(launcher)
            mlflow.end_run()

    def run(self) -> None:
        self._ensure_connected()
        for params in self._config.params_grid:
            self.grid_step(params)
