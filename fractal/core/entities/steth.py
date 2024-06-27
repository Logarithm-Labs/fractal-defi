from dataclasses import dataclass

from fractal.core.base.entity import (EntityException, GlobalState,
                                      InternalState)
from fractal.core.entities.spot import BaseSpotEntity


class StakedETHEntityException(EntityException):
    """
    Represents an exception for the StakedETH entity.
    """


@dataclass
class StakedETHGlobalState(GlobalState):
    """
    Represents the global state of any LST ETH token.

    Attributes:
        price (float): The price of the stETH.
        rate (float): The rate of the .
    """
    price: float = 0.0
    rate: float = 0.0


@dataclass
class StakedETHInternalState(InternalState):
    """
    Represents the internal state of the StakedETH entity.

    Attributes:
        amount (float): The stored amount of stETH.
        cash (float): The amount of cash in notional.
    """
    amount: float = 0.0
    cash: float = 0.0


class StakedETHEntity(BaseSpotEntity):
    """
    Represents an entity for trading on the StakedETH token.
    This is a simple entity that can buy, sell, deposit, and withdraw stETH.
    Also this entity control staking rate.
    """
    def __init__(self, *args, trading_fee: float = 0.003, **kwargs):
        super().__init__(*args, **kwargs)
        self.TRADING_FEE: float = trading_fee

    def _initialize_states(self):
        self._internal_state: StakedETHInternalState = StakedETHInternalState()
        self._global_state: StakedETHGlobalState = StakedETHGlobalState()

    def action_buy(self, amount_in_notional: float):
        """
        Executes a buy action on the StakedETH protocol.

        Args:
            amount_in_notional (float, optional): The amount to buy in notional value.

        Raises:
            ValueError: If there is not enough cash to buy.
        """
        if amount_in_notional > self._internal_state.cash:
            raise StakedETHEntityException(
                f"Not enough cash to buy: {amount_in_notional} > {self._internal_state.cash}")
        self._internal_state.cash -= amount_in_notional
        self._internal_state.amount += amount_in_notional * (1 - self.TRADING_FEE) / self._global_state.price

    def action_sell(self, amount_in_product: float):
        """
        Executes a sell action on the StakedETH protocol.

        Args:
            amount_in_product (float, optional): The amount to sell in product value.

        Raises:
            ValueError: If there is not enough product to sell.
        """
        if amount_in_product > self._internal_state.amount:
            raise StakedETHEntityException(
                f"Not enough product to sell: {amount_in_product} > {self._internal_state.amount}")
        self._internal_state.amount -= amount_in_product
        self._internal_state.cash += amount_in_product * (1 - self.TRADING_FEE) * self._global_state.price

    def action_withdraw(self, amount_in_notional: float):
        """
        Executes a withdraw action on the StakedETH protocol.

        Args:
            amount_in_notional (float, optional): The amount to withdraw in notional value.

        Raises:
            ValueError: If there is not enough cash to withdraw.
        """
        if amount_in_notional > self._internal_state.cash:
            raise StakedETHEntityException(
                f"Not enough cash to withdraw: {amount_in_notional} > {self._internal_state.cash}")
        self._internal_state.cash -= amount_in_notional

    def action_deposit(self, amount_in_notional: float):
        """
        Executes a deposit action on the StakedETH protocol.

        Args:
            amount_in_notional (float): The amount to deposit in notional value.
        """
        if amount_in_notional <= 0:
            raise StakedETHEntityException(f"Invalid deposit amount: {amount_in_notional}")
        self._internal_state.cash += amount_in_notional

    def update_state(self, state: StakedETHGlobalState, *args, **kwargs) -> None:
        """
        Updates the global state of the StakedETH protocol.
        1. Updates the global state.
        2. Add staking rewards to the internal state.

        Args:
            state (StakedETHGlobalState): The new global state.
        """
        self._global_state: StakedETHGlobalState = state
        self._internal_state.amount *= (self._global_state.rate + 1)

    @property
    def balance(self) -> float:
        """
        Calculates the balance of the StakedETH entity.

        The balance is calculated as the sum of the amount of stETH and the cash balance.
        Returns:
            float: The balance of the entity.
        """
        return self._internal_state.amount * self._global_state.price + self._internal_state.cash
