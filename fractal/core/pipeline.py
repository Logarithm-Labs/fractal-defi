import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Callable, Dict, Iterable, List, Optional, Type

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from fractal.core.base.strategy import (BaseStrategy, BaseStrategyParams,
                                        Observation, StrategyMetrics,
                                        StrategyResult)
from fractal.core.launcher import Launcher


@dataclass
class MLFlowConfig:
    """
    MLFlow Configuration for the pipeline.

    Attributes:
        experiment_name (str): Name of the experiment.
        mlflow_uri (str): URI of the MLFlow server.
        tags (Optional[Dict[str, str]]): Tags for the experiment.
        aws_access_key_id (Optional[str]): AWS access key ID.
        aws_secret_access_key (Optional[str]): AWS secret access key.
        run_name_formatter (Optional[Callable[[BaseStrategyParams | Dict], str]]):
            Formatter for the run name.
    """

    experiment_name: str
    mlflow_uri: str
    tags: Optional[Dict[str, str]] = None
    aws_access_key_id: Optional[str] = ''
    aws_secret_access_key: Optional[str] = ''
    run_name_formatter: Optional[Callable[[BaseStrategyParams | Dict], str]] = None


@dataclass
class ExperimentConfig:
    """
    Configuration for the experiment.

    Attributes:
        strategy_type (Type[BaseStrategy]): Strategy class to run.
        params_grid (ParameterGrid): Grid of parameters to run the strategy.
        fractal_observations (Optional[List[Observation]]): Observations for fractaling.
        fractal_trajectories (Optional[List[List[Observation]]): Trajectories for fractaling.
        window_size (Optional[int]): Window size for scenarios.
    """
    strategy_type: Type[BaseStrategy]
    params_grid: Iterable[BaseStrategyParams] | ParameterGrid
    fractal_observations: Optional[List[Observation]] = None
    fractal_trajectories: Optional[List[List[Observation]]] = None
    window_size: Optional[int] = None
    debug: Optional[bool] = False


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
        self.__connect_mlflow()

    def __connect_mlflow(self) -> None:
        """
        Connect to MLFlow server.
        """
        os.environ["AWS_ACCESS_KEY_ID"] = self._mlflow_config.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = self._mlflow_config.aws_secret_access_key
        mlflow.set_tracking_uri(self._mlflow_config.mlflow_uri)
        # check if the experiment already exists
        experiment = mlflow.get_experiment_by_name(self._mlflow_config.experiment_name)
        if experiment is None:
            mlflow.create_experiment(name=self._mlflow_config.experiment_name, tags=self._mlflow_config.tags)
        mlflow.set_experiment(self._mlflow_config.experiment_name)

    @abstractmethod
    def grid_step(self, params: BaseStrategyParams | Dict) -> None:
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

    def __log_secondary_metrics(self, metrics: List[StrategyMetrics], prefix: str) -> None:
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

    def grid_step(self, params: BaseStrategyParams | Dict) -> None:
        """
        Run a step of the pipeline. Each step runs a full experiment with a set of parameters.
        Check ExperimentConfig for the different types of experiments that can be run.

        Args:
            params (BaseStrategyParams | Dict): Parameters for the strategy.
        """
        # set up the launcher
        launcher = Launcher(strategy_type=self._config.strategy_type, params=params)

        # set run name
        run_name = None
        if self._mlflow_config.run_name_formatter:
            run_name = self._mlflow_config.run_name_formatter(params)

        # start run
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            # check all levels of experiment
            if self._config.fractal_observations:
                strategy_data: StrategyResult = launcher.run_strategy(
                    self._config.fractal_observations, debug=self._config.debug
                )
                strategy_data_df: pd.DataFrame = strategy_data.to_dataframe()
                metrics: StrategyMetrics = strategy_data.get_metrics(strategy_data_df)
                # Save all artifacts and metrics
                csv_buffer = StringIO()
                strategy_data_df.to_csv(csv_buffer, index=False)
                mlflow.log_text(csv_buffer.getvalue(), "strategy_fractal_data.csv")
                mlflow.log_metrics(metrics.__dict__)
                mlflow.log_artifact(launcher.last_created_instance.logger.logs_path)
            if self._config.fractal_trajectories:
                strategy_data_list: List[StrategyResult] = launcher.run_multiple_trajectories(
                    self._config.fractal_trajectories, debug=False
                )
                metrics: List[StrategyMetrics] = [
                    strategy_data.get_metrics(strategy_data.to_dataframe()) for strategy_data in strategy_data_list
                ]
                # Save all artifacts and metrics
                metrics_df: pd.DataFrame = pd.DataFrame([metric.__dict__ for metric in metrics])
                csv_buffer = StringIO()
                metrics_df.to_csv(csv_buffer, index=False)
                mlflow.log_text(csv_buffer.getvalue(), "fractal_trajectories_metrics.csv")
                self.__log_secondary_metrics(metrics, prefix="fractal_trajectories")
            if self._config.window_size:
                strategy_data_list: List[StrategyResult] = launcher.run_scenario(
                    self._config.fractal_observations, self._config.window_size, debug=False
                )
                metrics: List[StrategyMetrics] = [
                    strategy_data.get_metrics(strategy_data.to_dataframe()) for strategy_data in strategy_data_list
                ]
                metrics_df: pd.DataFrame = pd.DataFrame([metric.__dict__ for metric in metrics])
                # show number of rows with all zeros
                csv_buffer = StringIO()
                metrics_df.to_csv(csv_buffer, index=False)
                mlflow.log_text(csv_buffer.getvalue(), "window_trajectories_metrics.csv")
                self.__log_secondary_metrics(metrics, prefix="window_trajectories")
            mlflow.end_run()

    def run(self) -> None:
        for params in self._config.params_grid:
            self.grid_step(params)
