from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, NamedTuple, Optional, Type

from fractal.core.base.entity import (Action, BaseEntity, GlobalState,
                                      InternalState)
from fractal.core.base.strategy.logger import BaseLogger, DefaultLogger
from fractal.core.base.strategy.observation import Observation
from fractal.core.base.strategy.result import StrategyResult

NamedEntity = NamedTuple('NamedEntity', [('entity_name', str), ('entity', BaseEntity)])
ActionToTake = NamedTuple('ActionToTake', [('entity_name', str), ('action', Action)])


@dataclass
class BaseStrategyParams:
    """
    Base class for strategy parameters.
    It is used to store hyperparameters of the strategy.
    It can be initialized by a dictionary.
    """
    def __init__(self, data: Optional[Dict] = None):
        if data is not None:
            for key, value in data.items():
                setattr(self, key, value)


class BaseStrategy(ABC):
    """
    Base class for strategies.

    Strategies are responsible for predicting the next action to take
    based on the current state of the entities.

    Abstract Methods:
        `set_up(*args, **kwargs)`: Register entities, load models, execute some initial actions.
        Important Note: This method is called while initializing the strategy.
        In child classes, if implemented, also should be called parent's set_up.
        `predict(*args, **kwargs)` -> List[Action]: Predict the next action to take.
    """
    def __init__(self, *args, params: Optional[BaseStrategyParams | Dict] = None, debug: bool = False, **kwargs):
        """
        Initialize the strategy.

        Args:
            params (BaseStrategyParams | None): Parameters for the strategy. Can be None while using in Pipeline.
            debug (bool): Debug mode. Defaults to False. All debug messages are stored in runs/logs.
        """
        self.debug: bool = debug
        self.set_params(params)
        self._entities: Dict[str, BaseEntity] = {}
        self.set_up()
        self._logger: BaseLogger = self._create_logger()

    def _create_logger(self, base_artifacts_path: Optional[str] = None) -> BaseLogger:
        return DefaultLogger(base_artifacts_path=base_artifacts_path, class_name=self.__class__.__name__)

    @property
    def logger(self) -> BaseLogger:
        return self._logger

    def _debug(self, message: str):
        """
        Log a debug message if debug mode is enabled.
        It uses loguru logger.
        """
        if self.debug:
            self._logger.debug(message)

    @property
    def params(self) -> Dict:
        return self._params.__dict__

    def set_params(self, params: BaseStrategyParams | Dict) -> None:
        """
        Set parameters for the strategy.

        Args:
            params (BaseStrategyParams | Dict): Parameters for the strategy.
            If dict is passed, it will be converted to BaseStrategyParams
            with the fields of the dictionary keys and values.
        """
        if isinstance(params, dict):
            self._params = BaseStrategyParams(data=params)
        elif isinstance(params, BaseStrategyParams):
            self._params = params
        else:
            raise ValueError("Params should be dict or BaseStrategyParams")

    @abstractmethod
    def set_up(self, *args, **kwargs):
        """
        Set up the strategy.
        Register entities, load models, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, *args, **kwargs) -> List[ActionToTake]:
        """
        Predict the next action to take based on the current state of the entities.

        Returns:
            List[Action]: List of actions to take.
        """
        raise NotImplementedError

    def register_entity(self, entity: NamedEntity) -> None:
        """
        Register an entity for the strategy.
        It is used to keep track of the entities in the strategy.
        Each entity should have a unique name in entities registry of the strategy.
        """
        if entity.entity_name in self._entities:
            raise ValueError(f"Entity {entity.entity_name} already exists.")
        self._entities[entity.entity_name] = entity.entity

    def _remove_entity(self, entity_name: str) -> None:
        """
        Remove an entity from the strategy.
        """
        if entity_name not in self._entities:
            raise ValueError(f"Entity {entity_name} is not registered.")
        del self._entities[entity_name]

    def get_entity(self, entity_name: str) -> BaseEntity:
        """
        Get an entity by name.
        """
        if entity_name not in self._entities:
            raise ValueError(f"Entity {entity_name} is not registered.")
        return self._entities.get(entity_name)

    def get_all_available_entities(self) -> Dict[str, Type[BaseEntity]]:
        """
        Get all available entities.
        """
        return self._entities

    def __validate_observation(self, observation: Observation):
        """
        Validate the observation.
        There should not be any state in the observation
        that is not registered in the strategy.
        """
        for entity_name, _ in observation.states.items():
            if entity_name not in self._entities:
                raise ValueError(f"Entity {entity_name} is not registered.")

    def step(self, observation: Observation):
        """
        Take a step in the simulation by observations.
        """
        self._debug("=" * 30)
        self._debug("Running step...")
        self._debug(f"Observation: {observation.timestamp}")

        # validate observation
        self.__validate_observation(observation)

        # update states of the entities
        for entity_name, state in observation.states.items():
            entity = self.get_entity(entity_name)
            entity.update_state(state)

        # predict the next action to take
        actions: List[ActionToTake] = self.predict()
        self._debug(f"Actions to take: {actions}")

        # execute the actions
        for action in actions:
            self._debug(f"Action: {action}")
            entity = self.get_entity(action.entity_name)
            for arg_name, arg_value in action.action.args.items():
                # check if the argument is a callable function
                # it can be delegated function to get the state of the entity
                # between the time of the prediction and the execution of the action
                if callable(arg_value):
                    action.action.args[arg_name] = arg_value(self)
            self._debug(f"Before action {action.action}: {entity.internal_state}")
            # execute the action
            entity.execute(action.action)
            self._debug(f"After action: {entity.internal_state}")

    def run(self, observations: List[Observation]) -> StrategyResult:
        """
        Run the strategy on a sequence of observations.
        Execute self.step for each observation.
        """
        self._debug("=" * 30)
        self._debug(f"Running strategy on {len(observations)} observations.")
        self._debug(f"Strategy parameters: {self.params}")
        self._debug(f"Entities: {self.get_all_available_entities()}")
        self._debug(f"Entities states: {[entity.internal_state for entity in self._entities.values()]}")

        # collect all the states of the entities to build the StrategyResult
        timestamps: List[datetime] = []
        internal_states: List[Dict[str, InternalState]] = []
        balances: List[Dict[str, float]] = []
        global_states: List[Dict[str, GlobalState]] = []

        for observation in observations:
            self.step(observation)
            timestamps.append(observation.timestamp)
            balances.append({entity_name: entity.balance for entity_name, entity in self._entities.items()})
            # make copy of internal state to avoid changing it in the future
            internal_states.append({entity_name: copy(entity.internal_state)
                                    for entity_name, entity in self._entities.items()})
            global_states.append({entity_name: entity.global_state
                                  for entity_name, entity in self._entities.items()})
        return StrategyResult(
            timestamps=timestamps,
            internal_states=internal_states,
            global_states=global_states,
            balances=balances
        )
