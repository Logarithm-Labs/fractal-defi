from typing import List, Optional, Type

from fractal.core.base.observations import Observation, ObservationsStorage
from fractal.core.base.strategy import (BaseStrategy, BaseStrategyParams,
                                        StrategyResult)


class Launcher:
    """
    Launcher is used to run strategies with different scenarios:
    - Single trajectory
    - Multiple trajectories
    - Scenario (multiple fractals in sliding window across the observations)
    """
    def __init__(self, strategy_type: Type[BaseStrategy], params: BaseStrategyParams,
                 observations_storage_type: Optional[Type[ObservationsStorage]] = None):
        self._strategy_type: Type[BaseStrategy] = strategy_type
        self._observations_storage_type: Type[ObservationsStorage] = observations_storage_type
        self._params: BaseStrategyParams = params
        self._last_created_instance: BaseStrategy | None = None

    def strategy_instance(self, debug: bool) -> BaseStrategy:
        """
        Get the copy of the strategy to avoid storing outdated states.
        """
        if self._observations_storage_type is None:
            observations_storage_instance: ObservationsStorage = None
        else:
            observations_storage_instance: ObservationsStorage = self._observations_storage_type()

        instance: BaseStrategy = self._strategy_type(params=self._params, debug=debug,
                                                     observations_storage=observations_storage_instance)
        self._last_created_instance: BaseStrategy = instance
        return instance

    @property
    def last_created_instance(self) -> BaseStrategy | None:
        return self._last_created_instance

    def run_strategy(self, observations: List[Observation], debug: bool = False) -> StrategyResult:
        """
        Run strategy for a single trajectory.
        """
        strategy: BaseStrategy = self.strategy_instance(debug=debug)
        return strategy.run(observations)

    def run_multiple_trajectories(self, observations: List[List[Observation]],
                                  debug: bool = False) -> List[StrategyResult]:
        """
        Run the fractal for multiple trajectories.
        For simulation, we run the fractal for multiple Monte Carlo simulated trajectories.

        Args:
            observations: List of trajectories, where each trajectory is a list of observations.

        Returns:
            List of fractal results for each trajectory.
        """
        return [self.run_strategy(obs, debug=debug) for obs in observations]

    def run_scenario(self,
                     observations: List[Observation],
                     window_size: int = 24 * 30, step_size: int = 24,
                     debug: bool = False) -> List[StrategyResult]:
        """
        Run the scenario (multiple fractals in sliding window across the observations).
        """
        return [self.run_strategy(observations[i:i + window_size], debug=debug)
                for i in range(0, len(observations) - window_size + 1, step_size)]
