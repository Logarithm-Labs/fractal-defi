from dataclasses import dataclass, field
from typing import List

import numpy as np

from fractal.core.base.entity import (EntityException, GlobalState,
                                      InternalState)
from fractal.core.entities.hedge import BaseHedgeEntity


class HyperliquidEntityException(EntityException):
    """
    Exception raised for errors in the HyperLiquid entity.
    """


class HyperLiquidPosition:
    """
    A position in the Hyperliquid Hedge.
    It includes the amount, the entry price and desired leverage of the position.
        """

    def __init__(self, amount: float, entry_price: float, pos_leverage: float):
        """
        Initializes the HyperLiquidPosition.

        Args:
            amount (float): The amount of the position in product (e.g. BTC, ETH, etc.)
            entry_price (float): The entry price of the position in notional (e.g. USD, USDC, etc.)
            pos_leverage (float): Position leverage to open with. Can be set by value from 1 to MAX_LEVERAGE
        """
        self.amount: float = amount
        self.entry_price: float = entry_price
        self.pos_leverage: float = pos_leverage

    def __repr__(self):
        return (f"HyperLiquidPosition(amount={self.amount}, "
                f"entry_price={self.entry_price}")

    def unrealised_pnl(self, price: float) -> float:
        """
        Calculates the profit and loss (PNL) of the position.

        PNL = -1 * Amount * (Price - Entry Price)
        """
        return -1 * (self.amount * (price - self.entry_price))


@dataclass
class HyperLiquidGlobalState(GlobalState):
    """
    Global state of the entity.
    It includes the state of the environment.
    For example, price, time, etc.
    """
    mark_price: float = 0.0
    funding_rate_short: float = 0.0
    funding_rate_long: float = 0.0
    longs_pay_shorts: bool = True


@dataclass
class HyperLiquidInternalState(InternalState):
    """
    Represents the internal state of the HyperLiquid entity.

    It includes the collateral and the positions of the entity.
    """
    collateral: float = 0.0
    positions: List[HyperLiquidPosition] = field(default_factory=list)


class HyperliquidEntity(BaseHedgeEntity):
    """
    Represents a Hyperliquid isolated market entity.
    """

    def __init__(self, *args, max_leverage: int = 20, trading_fee: float = 0.00025, **kwargs):
        """
        Initializes the Hyperliquid entity.

        Args:
            trading_fee (float, optional): Trading fee. Defaults to 0.00025.
            MAX_LEVERAGE (float, optional): Max leverage available for current pair (from 3x to 50x on HyperLiquid) default 20
        """
        super().__init__(*args, **kwargs)
        self.MAX_LEVERAGE = max_leverage
        self.TRADING_FEE = trading_fee

    def _initialize_states(self):
        self._internal_state: HyperLiquidInternalState = HyperLiquidInternalState()
        self._global_state: HyperLiquidGlobalState = HyperLiquidGlobalState()

    def action_deposit(self, amount_in_notional: float):
        """
        Deposits a specified amount of notional into the entity's collateral.

        Args:
            amount_in_notional (float): The amount of notional to deposit.
        """
        if amount_in_notional < 0:
            raise HyperliquidEntityException(f"Invalid deposit amount: {amount_in_notional} < 0.")

        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float):
        """
        Withdraws a specified amount of notional from the entity's collateral.

        Args:
            amount_in_notional (float): The amount of notional to withdraw.

        Returns:
            float: The remaining balance after the withdrawal.

        Raises:
            ValueError: If there is not enough balance to withdraw.
        """
        if self.balance < amount_in_notional:
            raise HyperliquidEntityException(f"Not enough balance to withdraw: {self.balance} < {amount_in_notional}.")

        self._internal_state.collateral -= amount_in_notional

    def action_open_position(self, portion_of_product):
        """
        Opens a position with a specified amount of product.

        Args:
            portion_of_product (float): The amount of product to open the position with.
        """
        self._internal_state.positions.append(
            HyperLiquidPosition(amount=portion_of_product,
                                entry_price=self._global_state.mark_price,
                                pos_leverage=self.leverage))

        self._internal_state.collateral -= np.abs(self._internal_state.collateral * portion_of_product * self.TRADING_FEE)

        self._clearing()  # consider only one position for simplicity

    @property
    def pnl(self) -> float:
        """
        Calculates the total profit and loss (PNL) of all positions.

        PNL is a sum of PNLs of all positions.
        Returns:
            float: The total PNL.
        """
        return sum(pos.unrealised_pnl(self._global_state.mark_price) for pos in self._internal_state.positions)

    @property
    def balance(self) -> float:
        """
        Calculates the current balance of the entity.

        Balance is a sum of the collateral and the PNL.
        Returns:
            float: The current balance.
        """
        return self._internal_state.collateral + self.pnl

    @property
    def size(self):
        """
        Calculates the total size of all positions.

        Size is a sum of all position amounts.
        Returns:
            float: The total size.
        """
        return sum(pos.amount for pos in self._internal_state.positions)

    @property
    def leverage(self):
        """
        Calculates the leverage of the entity.

        The leverage is the absolute ratio of the total position size to the balance.
        Returns:
            float: The leverage.
        """
        if self.balance == 0 and self.size == 0:
            return 0
        return np.abs(self.size * self._global_state.mark_price / self.balance)

    def _clearing(self):
        """
        Performs clearing of the entity's state.
        It helps to manage only one position for simplicity.
        """
        self._internal_state.collateral = self.balance
        size = self.size
        price = self._global_state.mark_price
        self._internal_state.positions = [
            HyperLiquidPosition(amount=size, entry_price=price, pos_leverage=size * price / self.balance)]

    def _check_liquidation(self) -> bool:
        """
        A liquidation event occurs when a trader's positions move against them to the point
        where the account equity falls below the maintenance margin.

        The maintenance margin is half of the initial margin at max leverage, which varies from 3-50x.

        In other words, the maintenance margin is between
        1% (for 50x max leverage assets)
        and 16.7% (for 3x max leverage assets) depending on the asset.

        Returns:
            bool: True if the entity is at risk of liquidation, False otherwise.
        """

        entry_price: float = sum(
            pos.entry_price for pos in self._internal_state.positions)  # Valid if only one position is opened

        if self.size == 0:
            return 0

        liquidation_price = entry_price + self.balance * (self.MAX_LEVERAGE - 2) / (self.MAX_LEVERAGE * self.size)

        print(f"liquidation_price is {liquidation_price},\n"
              f"mark_price is {self._global_state.mark_price}, \n"
              # f"entry_price is {sum(pos.entry_price for pos in self._internal_state.positions)}, \n"
              f"balance is {self.balance}, \n"
              f"max leverage is {self.MAX_LEVERAGE},\n"
              f"position_size is {self.size}")

        return self._global_state.mark_price >= liquidation_price

    def update_state(self, state: HyperLiquidGlobalState, *args, **kwargs) -> None:
        """
        Updates the entity's state with the given global state.

        1. Updates the global state.
        2. Check liquidation.
        3. Settle fundings.

        Args:
            state (HyperLiquidGlobalState): The global state to update with.
        """
        self._global_state = state
        if self._check_liquidation():
            self._internal_state.collateral = 0
            self._internal_state.positions = []
        # settle short position fundings
        global_price: float = self._global_state.mark_price
        current_size: float = self.size

        self._internal_state.collateral += current_size * global_price * self._global_state.funding_rate_short
