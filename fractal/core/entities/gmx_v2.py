from dataclasses import dataclass, field
from typing import List

import numpy as np

from fractal.core.base.entity import (EntityException, GlobalState,
                                      InternalState)
from fractal.core.entities.hedge import BaseHedgeEntity


class GMXV2EntityException(EntityException):
    """
    Exception raised for errors in the GMX V2 entity.
    """


class GMXV2Position:
    """
    A position in the GMXHedge.
    It includes the amount and the entry price of the position.
    """

    def __init__(self, amount: float, entry_price: float):
        """
        Initializes the GMXPosition.

        Args:
            amount (float): The amount of the position in product (e.g. BTC, ETH, etc.)
            entry_price (float): The entry price of the position in notional (e.g. USD, USDC, etc.)
        """
        self.amount: float = amount
        self.entry_price: float = entry_price

    def __repr__(self):
        return f"GMXPosition(amount={self.amount}, entry_price={self.entry_price})"

    def pnl(self, price: float) -> float:
        """
        Calculates the profit and loss (PNL) of the position.

        PNL = Amount * (Price - Entry Price)
        """
        return self.amount * (price - self.entry_price)


@dataclass
class GMXV2GlobalState(GlobalState):
    """
    Represents the global state of the GMX V2 entity.

    GMX V2 has two-sided funding rates and borrowing rates.

    price: float - The current price of the product.
    funding_rate_short: float - The short funding rate.
    funding_rate_long: float - The long funding rate.
    borrowing_rate_short: float - The short borrowing rate.
    borrowing_rate_long: float - The long borrowing rate.
    longs_pay_shorts: bool - Whether longs pay shorts.
    """
    price: float = 0.0
    funding_rate_short: float = 0.0
    funding_rate_long: float = 0.0
    borrowing_rate_short: float = 0.0
    borrowing_rate_long: float = 0.0
    longs_pay_shorts: bool = True


@dataclass
class GMXV2InternalState(InternalState):
    """
    Represents the internal state of the GMX V2 entity.

    It includes the collateral and the positions of the entity.
    """
    collateral: float = 0.0
    positions: List[GMXV2Position] = field(default_factory=list)


class GMXV2Entity(BaseHedgeEntity):
    """
    Represents a GMX isolated market entity.
    """
    def __init__(self, *args, trading_fee: float = 0.001,
                 liquidation_leverage: float = 100, **kwargs):
        """
        Initializes the GMX entity.

        Args:
            trading_fee (float, optional): Trading fee. Defaults to 0.001.
            liquidation_leverage (float, optional): Liquidation leverage in GMX. Defaults to 100.
        """
        super().__init__(*args, **kwargs)
        self.TRADING_FEE = trading_fee
        self.LIQUIDATION_LEVERAGE = liquidation_leverage

    def _initialize_states(self):
        self._internal_state: GMXV2InternalState = GMXV2InternalState()
        self._global_state: GMXV2GlobalState = GMXV2GlobalState()

    def action_deposit(self, amount_in_notional: float):
        """
        Deposits a specified amount of notional into the entity's collateral.

        Args:
            amount_in_notional (float): The amount of notional to deposit.
        """
        if amount_in_notional < 0:
            raise GMXV2EntityException(f"Invalid deposit amount: {amount_in_notional} < 0.")
        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> float:
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
            raise GMXV2EntityException(f"Not enough balance to withdraw: {self.balance} < {amount_in_notional}.")
        max_withdrawal: float = self.balance - np.abs(self.size * self._global_state.price) / self.LIQUIDATION_LEVERAGE
        if amount_in_notional > max_withdrawal:
            raise GMXV2EntityException("Exceeds maximum withdrawal limit: {amount_in_notional} > {max_withdrawal}.")
        self._internal_state.collateral -= amount_in_notional

    def action_open_position(self, amount_in_product: float):
        """
        Opens a position with a specified amount of product.

        Args:
            amount_in_product (float): The amount of product to open the position with.
        """
        self._internal_state.positions.append(
            GMXV2Position(amount=amount_in_product, entry_price=self._global_state.price))
        self._internal_state.collateral -= np.abs(amount_in_product * self.TRADING_FEE * self._global_state.price)
        self._clearing()  # consider only one position for simplicity

    @property
    def pnl(self) -> float:
        """
        Calculates the total profit and loss (PNL) of all positions.

        PNL is a sum of PNLs of all positions.
        Returns:
            float: The total PNL.
        """
        return sum(pos.pnl(self._global_state.price) for pos in self._internal_state.positions)

    @property
    def balance(self):
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
        return np.abs(self.size * self._global_state.price / self.balance)

    def _check_liquidation(self) -> bool:
        """
        Checks if the entity is at risk of liquidation.

        Liquidation occurs when the leverage exceeds the liquidation leverage.
        Returns:
            bool: True if the entity is at risk of liquidation, False otherwise.
        """
        return self.leverage >= self.LIQUIDATION_LEVERAGE

    def _clearing(self):
        """
        Performs clearing of the entity's state.
        It helps to manage only one position for simplicity.
        """
        self._internal_state.collateral = self.balance
        size = self.size
        price = self._global_state.price
        self._internal_state.positions = [GMXV2Position(amount=size, entry_price=price)]

    def update_state(self, state: GMXV2GlobalState, *args, **kwargs) -> None:
        """
        Updates the entity's state with the given global state.

        1. Updates the global state.
        2. Check liquidation.
        3. Settle fundings and borrowings.

        Args:
            state (GMXGlobalState): The global state to update with.
        """
        self._global_state = state
        if self._check_liquidation():
            self._internal_state.collateral = 0
            self._internal_state.positions = []
        # settle fundings and borrowings
        global_price: float = self._global_state.price
        current_size: float = self.size
        if current_size > 0:
            self._internal_state.collateral += current_size * global_price * self._global_state.funding_rate_long
            self._internal_state.collateral -= current_size * global_price * self._global_state.borrowing_rate_long
        elif current_size < 0:
            self._internal_state.collateral -= current_size * global_price * self._global_state.funding_rate_short
            self._internal_state.collateral += current_size * global_price * self._global_state.borrowing_rate_short
