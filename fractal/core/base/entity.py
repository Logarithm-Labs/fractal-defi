from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, TypeVar

from fractal.core.base.action import Action


class EntityException(Exception):
    """
    Exception raised for errors in the entity.
    """


@dataclass
class GlobalState:
    """
    Global state of the entity.
    It includes the state of the environment.
    For example, price, time, etc.
    """
    def __repr__(self) -> str:
        return f"{self.__dict__}"


@dataclass
class InternalState:
    """
    Internal state of the entity.
    It includes the internal state of the entity.
    For example, cash balance, positions, etc.
    """
    def __repr__(self) -> str:
        return f"{self.__dict__}"


# Type parameters for :class:`BaseEntity`. Concrete entities that wish
# to expose narrowly-typed ``internal_state`` / ``global_state`` to
# strategy code should declare themselves as
# ``MyEntity(BaseEntity[MyGlobalState, MyInternalState])``.
GS = TypeVar("GS", bound=GlobalState)
IS = TypeVar("IS", bound=InternalState)


class BaseEntity(ABC, Generic[GS, IS]):
    """
    Base class for entities.
    Entities are responsible for managing their internal state
    and executing actions. Each entity has a global state and
    internal state. The global state is the state of the environment,
    and the internal state is the state of the entity.

    Each entity is a representation of DeFi object.
    For instance, a pool, a landing, a vault, etc.

    Important Note: Each method that starts with 'action_' is considered
    as an action that can be executed by the simulation engine.

    Abstract Methods:
    - `update_state(state: GlobalState, *args, **kwargs) -> None` - Update the Global State of the entity.
    Here it can be calculated changes in the entity's internal state based on the global state.
    For example, while the price of the asset changes, the value of the entity's assets changes.
    - `action_deposit(amount_in_notional: float) -> None` - Deposit the specified amount in notional value
    into the entity. This is not mandatory for all entities but it can be useful for most of them.
    - `action_withdraw(amount_in_notional: float) -> None` - Withdraw the specified amount from the entity's account.
    - `balance` - Property that returns the balance of the entity.
    """
    _internal_state: IS
    _global_state: GS

    def __init__(self):
        # Concrete subclasses populate ``_internal_state`` / ``_global_state``
        # inside ``_initialize_states``; no need to pre-set to None.
        self._initialize_states()

    @abstractmethod
    def _initialize_states(self):
        """
        Initialize the `_global_state` and `_internal_state` attributes.
        This method must be overridden by all concrete subclasses to set these attributes.
        """
        raise NotImplementedError

    def get_available_actions(self) -> List[str]:
        """List actions exposed by ``action_*`` methods.

        Walks the **class**, not the instance, so we never accidentally
        invoke property getters during introspection.
        """
        cls = type(self)
        return [
            name.replace("action_", "")
            for name in dir(cls)
            if name.startswith("action_") and callable(getattr(cls, name, None))
        ]

    @property
    def global_state(self) -> GS:
        return self._global_state

    @property
    def internal_state(self) -> IS:
        return self._internal_state

    @abstractmethod
    def update_state(self, state: GS) -> None:
        """Apply the new ``state`` to the entity.

        Single-argument by design: all environment context flows through
        :class:`GlobalState`. If you need new context, extend the state
        dataclass — do **not** add side-channel parameters here.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def balance(self) -> float:
        raise NotImplementedError

    def action_deposit(self, amount_in_notional: float) -> None:
        """Deposit notional cash into the entity.

        The default raises ``NotImplementedError`` so that entities that
        do not handle cash (e.g. pure data feeds) fail loudly instead of
        silently dropping the amount. Concrete entities holding notional
        cash MUST override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support action_deposit; "
            "override it on the concrete entity if it holds notional cash."
        )

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Withdraw notional cash from the entity.

        Same default-raise semantics as :meth:`action_deposit`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support action_withdraw; "
            "override it on the concrete entity if it holds notional cash."
        )

    def execute(self, action: Action) -> None:
        """Execute an action on the entity.

        Action methods on entities only mutate internal state; their
        return value (if any) is discarded. The strategy-side flow only
        cares about side effects, so we hide the return contract.
        """
        available = self.get_available_actions()
        if action.action not in available:
            raise EntityException(
                f"Action {action.action!r} is not available for entity "
                f"{self.__class__.__name__}. Available actions: {available}"
            )
        getattr(self, "action_" + action.action)(**action.args)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"global_state={self.global_state}, "
            f"internal_state={self.internal_state})"
        )
