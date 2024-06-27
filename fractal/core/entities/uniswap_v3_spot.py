from dataclasses import dataclass

from fractal.core.base.entity import (EntityException, GlobalState,
                                      InternalState)
from fractal.core.entities.spot import BaseSpotEntity


class UniswapV3SpotEntityException(EntityException):
    pass


@dataclass
class UniswapV3SpotGlobalState(GlobalState):
    price: float = 0.0


@dataclass
class UniswapV3SpotInternalState(InternalState):
    amount: float = 0.0
    cash: float = 0.0


class UniswapV3SpotEntity(BaseSpotEntity):
    """
    Represents an entity for trading on the Uniswap V3 Spot protocol.
    """
    def __init__(self, *args, trading_fee: float = 0.003, **kwargs):
        super().__init__(*args, **kwargs)
        self.TRADING_FEE: float = trading_fee

    def _initialize_states(self):
        self._internal_state: UniswapV3SpotInternalState = UniswapV3SpotInternalState()
        self._global_state: UniswapV3SpotGlobalState = UniswapV3SpotGlobalState()

    def action_buy(self, amount_in_notional: float):
        """
        Executes a buy action on the Uniswap V3 Spot protocol.

        Args:
            amount_in_notional (float, optional): The amount to buy in notional value.

        Raises:
            ValueError: If there is not enough cash to buy.
        """
        if amount_in_notional > self._internal_state.cash:
            raise UniswapV3SpotEntityException(
                f"Not enough cash to buy: {amount_in_notional} > {self._internal_state.cash}")
        self._internal_state.cash -= amount_in_notional
        self._internal_state.amount += amount_in_notional * (1 - self.TRADING_FEE) / self._global_state.price

    def action_sell(self, amount_in_product: float):
        """
        Executes a sell action on the Uniswap V3 Spot protocol.

        Args:
            amount_in_product (float, optional): The amount to sell in product value.

        Raises:
            ValueError: If there is not enough product to sell.
        """
        if amount_in_product > self._internal_state.amount:
            raise UniswapV3SpotEntityException(
                f"Not enough product to sell: {amount_in_product} > {self._internal_state.amount}")
        self._internal_state.amount -= amount_in_product
        self._internal_state.cash += amount_in_product * (1 - self.TRADING_FEE) * self._global_state.price

    def action_withdraw(self, amount_in_notional: float):
        """
        Executes a withdraw action on the Uniswap V3 Spot protocol.

        Args:
            amount_in_notional (float, optional): The amount to withdraw in notional value.

        Raises:
            ValueError: If there is not enough cash to withdraw.
        """
        if amount_in_notional > self._internal_state.cash:
            raise UniswapV3SpotEntityException(
                f"Not enough cash to withdraw: {amount_in_notional} > {self._internal_state.cash}")
        self._internal_state.cash -= amount_in_notional

    def action_deposit(self, amount_in_notional: float):
        """
        Executes a deposit action on the Uniswap V3 Spot protocol.

        Args:
            amount_in_notional (float): The amount to deposit in notional value.
        """
        if amount_in_notional <= 0:
            raise UniswapV3SpotEntityException(f"Invalid deposit amount: {amount_in_notional}")
        self._internal_state.cash += amount_in_notional

    def update_state(self, state: UniswapV3SpotGlobalState, *args, **kwargs) -> None:
        """
        Updates the global state of the Uniswap V3 Spot protocol.

        Args:
            state (UniswapV3SpotGlobalState): The new global state.
        """
        self._global_state: UniswapV3SpotGlobalState = state

    @property
    def balance(self) -> float:
        """
        Calculates the balance of the Uniswap V3 Spot entity.

        Returns:
            float: The balance of the entity.
        """
        return self._internal_state.amount * self._global_state.price + self._internal_state.cash
