import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
from types import MappingProxyType
from typing import (Callable, Dict, Generic, List, Mapping, NamedTuple, Optional,
                    Type, TypeVar, Union)

from fractal.core.base.entity import (Action, BaseEntity, GlobalState,
                                      InternalState)
from fractal.core.base.observations import Observation, ObservationsStorage
from fractal.core.base.strategy.logger import BaseLogger, DefaultLogger
from fractal.core.base.strategy.result import StrategyResult

NamedEntity = NamedTuple('NamedEntity', [('entity_name', str), ('entity', BaseEntity)])
ActionToTake = NamedTuple('ActionToTake', [('entity_name', str), ('action', Action)])


class BaseStrategyParams:
    """Base class for strategy hyperparameters.

    A simple namespace populated either by keyword (``cls(field=value)``
    once a subclass adds dataclass fields) or by a dict
    (``BaseStrategyParams(data={"FOO": 1})``).

    Subclasses are expected to add fields via :func:`@dataclasses.dataclass`
    decorator on their own definition; the dataclass-generated ``__init__``
    cleanly replaces the dict-form constructor below.
    """

    def __init__(self, data: Optional[Dict] = None):
        if data is not None:
            for key, value in data.items():
                setattr(self, key, value)


#: Type parameter for :class:`BaseStrategy`. Concrete strategies declare
#: their params class as ``MyStrategy(BaseStrategy[MyParams])`` and IDE
#: autocompletes ``self._params.<field>``. The class is also stored as
#: ``cls.PARAMS_CLS`` so :meth:`set_params` can construct a default
#: instance when ``params=None``.
PT = TypeVar("PT", bound=BaseStrategyParams)


class BaseStrategy(ABC, Generic[PT]):
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

    #: When ``True`` (default), :meth:`step` requires every registered
    #: entity to have a state in the incoming :class:`Observation`. Set
    #: to ``False`` on subclasses that intentionally use partial
    #: observations (e.g. mixed-frequency data feeds where some entities
    #: only update every N bars). Forward validation — observation states
    #: must reference registered entities — is always on.
    STRICT_OBSERVATIONS: bool = True

    #: Concrete params class for this strategy. Auto-derived from the
    #: generic argument: a subclass declaring ``MyStrategy(BaseStrategy[MyParams])``
    #: gets ``PARAMS_CLS = MyParams`` automatically (via :meth:`__init_subclass__`).
    #: Used by :meth:`set_params` to construct a default instance when
    #: ``params=None`` (only succeeds when every field on the params class
    #: has a default value). An explicit class attribute override wins.
    PARAMS_CLS: Optional[Type[BaseStrategyParams]] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("PARAMS_CLS") is not None:
            return  # explicit override wins
        for base in getattr(cls, "__orig_bases__", ()):
            origin = typing.get_origin(base)
            if isinstance(origin, type) and issubclass(origin, BaseStrategy):
                args = typing.get_args(base)
                if args and isinstance(args[0], type) and issubclass(args[0], BaseStrategyParams):
                    cls.PARAMS_CLS = args[0]
                    return

    def __init__(  # pylint: disable=unused-argument
        self,
        *args,
        params: Optional[Union[BaseStrategyParams, Dict]] = None,
        debug: bool = False,
        observations_storage: Optional[ObservationsStorage] = None,
        **kwargs,
    ):
        """Initialize the strategy.

        Args:
            params: Parameters for the strategy. Can be ``None`` —
                a default :class:`PARAMS_CLS` instance is constructed
                if all its fields have defaults.
            debug: Enable debug logging into ``runs/<class>/<id>/logs``.
            observations_storage: Optional persistence layer for observations.
        """
        self.debug: bool = debug
        self.set_params(params)
        # Logger is initialized BEFORE ``set_up`` so subclasses may use
        # ``self._debug`` from inside their ``set_up`` hook.
        self._logger: Optional[BaseLogger] = self._create_logger() if debug else None
        self._entities: Dict[str, BaseEntity] = {}
        self.set_up()
        self.observations_storage: Optional[ObservationsStorage] = observations_storage

    def _create_logger(self) -> BaseLogger:
        return DefaultLogger(class_name=self.__class__.__name__)

    @property
    def logger(self) -> BaseLogger:
        return self._logger

    def _debug(self, message: str):
        """Log a debug message when debug mode is on (no-op otherwise)."""
        if self._logger is not None:
            self._logger.debug(message)

    _params: PT

    @property
    def params(self) -> Dict:
        """Read-only snapshot of strategy hyperparameters.

        Returns a **copy** of the underlying namespace so callers cannot
        mutate the running strategy's parameters by accident.
        """
        return dict(self._params.__dict__)

    def set_params(self, params: Optional[Union[BaseStrategyParams, Dict]]) -> None:
        """Set parameters for the strategy.

        Args:
            params: Parameters for the strategy. Accepts:

                * ``None`` — construct ``cls.PARAMS_CLS()`` (the auto-derived
                  generic param class). Raises ``TypeError`` if any field
                  on that class lacks a default. Falls back to an empty
                  :class:`BaseStrategyParams` if ``PARAMS_CLS`` is unset.
                * a ``dict`` — when ``PARAMS_CLS`` is declared (typed
                  strategy), the dict is splatted into it
                  (``PARAMS_CLS(**params)``) so dataclass defaults,
                  ``__post_init__`` and unknown-key rejection apply.
                  Without ``PARAMS_CLS`` it is wrapped in a generic
                  :class:`BaseStrategyParams`.
                * a :class:`BaseStrategyParams` instance — used as-is.
        """
        if params is None:
            cls = self.PARAMS_CLS or BaseStrategyParams
            self._params = cls()
        elif isinstance(params, dict):
            if self.PARAMS_CLS is not None:
                # Typed strategy: instantiate the declared dataclass so
                # defaults, type hints and unknown-field detection apply.
                self._params = self.PARAMS_CLS(**params)
            else:
                self._params = BaseStrategyParams(data=params)
        elif isinstance(params, BaseStrategyParams):
            self._params = params
        else:
            raise ValueError("Params should be None, dict, or BaseStrategyParams")

    @abstractmethod
    def set_up(self) -> None:
        """Set up the strategy: register entities, load models, …

        Called once from ``__init__`` after ``set_params`` and after the
        logger is initialized — :meth:`_debug` is safe to use here.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self) -> List[ActionToTake]:
        """Predict the next actions to take from the current entity states."""
        raise NotImplementedError

    def register_entity(self, entity: NamedEntity) -> None:
        """Register an entity under a unique name.

        Args:
            entity: a :class:`NamedEntity` carrying the registry name and
                the :class:`BaseEntity` instance.

        Raises:
            TypeError: if the bound object is not a :class:`BaseEntity`.
            ValueError: if the entity name is empty / non-string, or if
                the name is already registered on this strategy.
        """
        if not isinstance(entity.entity_name, str) or not entity.entity_name:
            raise ValueError(
                f"entity_name must be a non-empty str, got {entity.entity_name!r}"
            )
        if not isinstance(entity.entity, BaseEntity):
            raise TypeError(
                f"register_entity expects a BaseEntity instance for "
                f"{entity.entity_name!r}, got {type(entity.entity).__name__}"
            )
        if entity.entity_name in self._entities:
            raise ValueError(f"Entity {entity.entity_name} already exists.")
        self._entities[entity.entity_name] = entity.entity

    def get_entity(self, entity_name: str) -> BaseEntity:
        """
        Get an entity by name.
        """
        if entity_name not in self._entities:
            raise ValueError(f"Entity {entity_name} is not registered.")
        return self._entities.get(entity_name)

    def get_all_available_entities(self) -> Mapping[str, BaseEntity]:
        """Read-only view over registered entities.

        Returns a :class:`MappingProxyType` so callers cannot bypass
        :meth:`register_entity` by mutating the registry directly. The
        annotation is ``Mapping[str, BaseEntity]`` (instance values, not
        types — the prior ``Dict[..., Type[BaseEntity]]`` was a typo).
        """
        return MappingProxyType(self._entities)

    @property
    def total_balance(self) -> float:
        """Sum of all registered entities' ``balance`` properties.

        Predicate: every entity reports its balance in the **same**
        accounting unit (typically USD). Fractal entities are
        unit-agnostic — they accept whatever prices you pass to
        ``update_state`` — so the strategy author is responsible for
        keeping units consistent across entities. See the README's
        "Pricing convention" section for the recommended pattern
        (USD-denominated by default; ETH/BTC as opt-in if every entity
        in the strategy is configured in that unit).

        Returns ``0.0`` when no entities are registered.
        """
        return sum(entity.balance for entity in self._entities.values())

    def transfer(
        self,
        from_entity: str,
        to_entity: str,
        amount_in_notional: Union[float, Callable[["BaseStrategy"], float]],
    ) -> List["ActionToTake"]:
        """Move notional cash from one registered entity to another.

        Returns a 2-action list ordered **deposit-first, withdraw-second**.
        The ordering matters when ``amount_in_notional`` is a delegate
        (callable) — both calls then evaluate against the same source
        state, since the deposit on the destination does not mutate the
        source.

        Args:
            from_entity: Name of the source entity (will be debited).
            to_entity: Name of the destination entity (will be credited).
            amount_in_notional: Either a float, or a callable
                ``(strategy) -> float`` evaluated at execute time.

        Raises:
            ValueError: if ``from_entity == to_entity``.
            ValueError: if either entity is not registered.
        """
        if from_entity == to_entity:
            raise ValueError(
                f"transfer: from_entity and to_entity must differ, got {from_entity!r}"
            )
        # ``get_entity`` raises ValueError if missing — this validates both names.
        self.get_entity(from_entity)
        self.get_entity(to_entity)
        return [
            ActionToTake(
                entity_name=to_entity,
                action=Action("deposit", {"amount_in_notional": amount_in_notional}),
            ),
            ActionToTake(
                entity_name=from_entity,
                action=Action("withdraw", {"amount_in_notional": amount_in_notional}),
            ),
        ]

    def _validate_observation(self, observation: Observation):
        """Validate that the observation matches the registered entities.

        * Forward (always): every entity name in ``observation.states``
          must be registered.
        * Reverse (when :attr:`STRICT_OBSERVATIONS`): every registered
          entity must have a state in the observation. Catches the typo
          "I forgot to put HEDGE in this Observation" early.

        Single-underscore so subclasses may override / augment validation
        (e.g. require monotonic timestamps).
        """
        for entity_name in observation.states:
            if entity_name not in self._entities:
                raise ValueError(
                    f"Entity {entity_name!r} appears in observation but is not registered."
                )
        if self.STRICT_OBSERVATIONS:
            missing = [n for n in self._entities if n not in observation.states]
            if missing:
                raise ValueError(
                    "Observation is missing states for registered entities "
                    f"{missing!r}; set STRICT_OBSERVATIONS = False on the "
                    "strategy class to allow partial observations."
                )

    def step(self, observation: Observation):
        """
        Take a step in the simulation by observations.
        """
        self._debug("=" * 30)
        self._debug("Running step...")
        self._debug(f"Observation: {observation.timestamp}")

        # Validate first (subclasses may override) so that storage never
        # contains snapshots that would have been rejected at the boundary.
        self._validate_observation(observation)

        if self.observations_storage is not None:
            self.observations_storage.write(observation)

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
            # Resolve callable (delegated) args at execution time without
            # mutating the original Action — strategies often build action
            # templates and reuse them across steps.
            resolved_args = {
                name: (val(self) if callable(val) else val)
                for name, val in action.action.args.items()
            }
            resolved_action = Action(action.action.action, resolved_args)
            self._debug(f"Before action {resolved_action}: {entity.internal_state}")
            entity.execute(resolved_action)
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

            # Deep-copy both states so later mutations on the entity (or
            # subsequent ``update_state`` calls) cannot leak into this
            # historical snapshot.
            internal_states.append({entity_name: deepcopy(entity.internal_state)
                                    for entity_name, entity in self._entities.items()})
            global_states.append({entity_name: deepcopy(entity.global_state)
                                  for entity_name, entity in self._entities.items()})
        return StrategyResult(
            timestamps=timestamps,
            internal_states=internal_states,
            global_states=global_states,
            balances=balances
        )
