from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List

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


class BaseEntity(ABC):
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
    def __init__(self):
        self._internal_state: InternalState = None
        self._global_state: GlobalState = None
        self._initialize_states()

    @abstractmethod
    def _initialize_states(self):
        """
        Initialize the `_global_state` and `_internal_state` attributes.
        This method must be overridden by all concrete subclasses to set these attributes.
        """
        raise NotImplementedError

    def get_available_actions(self) -> List[str]:
        """
        Get available actions for the entity.

        Returns:
            List[str]: List of available actions.
        """
        return [
            func.replace('action_', '')
            for func in dir(self)
            if callable(getattr(self, func)) and func.startswith("action")
        ]

    @property
    def global_state(self) -> GlobalState:
        return self._global_state

    @property
    def internal_state(self) -> InternalState:
        return self._internal_state

    @abstractmethod
    def update_state(self, state: GlobalState, *args, **kwargs) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def balance(self) -> float:
        raise NotImplementedError

    def action_deposit(self, amount_in_notional: float) -> None:
        """
        Deposits the specified amount in notional value into the entity.
        Most entities can store the cash balance in notional value.

        Args:
            amount_in_notional (float): The amount to be
            deposited in notional value.
        """

    def action_withdraw(self, amount_in_notional: float) -> None:
        """
        Withdraws the specified amount from the entity's account.
        Most entities can store the cash balance in notional value.

        Args:
            amount_in_notional (float): The amount to withdraw
            in notional value.
        """

    def execute(self, action: Action) -> Any:
        """
        Execute action on the entity.
        """
        # check if action is available
        if action.action not in self.get_available_actions():
            raise EntityException(
                f"Action {action.action} is not available\
                for entity {self.__class__.__name__}.\
                Available actions: {self.get_available_actions()}"
            )
        return getattr(self, 'action_' + action.action)(**action.args)

    def __repr__(self) -> str:
        repr: str = f"{self.__class__.__name__}("
        repr += f"global_state={self.global_state}, "
        repr += f"internal_state={self.internal_state})"
        return repr
