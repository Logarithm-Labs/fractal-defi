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

    def __init__(self, amount: float, entry_price: float, max_leverage: float):
        """
        Initializes the HyperLiquidPosition.

        Args:
            amount (float): The amount of the position in product (e.g. BTC, ETH, etc.)
            entry_price (float): The entry price of the position in notional (e.g. USD, USDC, etc.)
            max_leverage (float): Pair's max leverage available to trade (from 3x to 50x on HyperLiquid)
        """
        self.amount: float = amount  # negative if short
        self.entry_price: float = entry_price
        self.max_leverage: float = max_leverage

    def __repr__(self):
        return (f"HyperLiquidPosition(amount={self.amount}, entry_price={self.entry_price})")

    def unrealised_pnl(self, price: float) -> float:
        """
        Calculates the profit and loss (PNL) of the position.

        PNL = Amount * (Price - Entry Price)
        """
        return self.amount * (price - self.entry_price)


@dataclass
class HyperLiquidGlobalState(GlobalState):
    """
    Global state of the entity.
    It includes the state of the environment.
    For example, price, time, etc.
    """
    mark_price: float = 0.0
    funding_rate: float = 0.0


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

    def __init__(self, *args, max_leverage: int = 50, trading_fee: float = 0.00035, **kwargs):
        """
        Initializes the Hyperliquid entity.

        Args:
            trading_fee (float, optional): Trading fee. Defaults to 0.00035.
        """
        super().__init__(*args, **kwargs)
        self.TRADING_FEE = trading_fee
        self.MAX_LEVERAGE = max_leverage

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
        if amount_in_notional < 0:
            raise HyperliquidEntityException(f"Invalid withdraw amount: {amount_in_notional} < 0.")
        if self.balance < amount_in_notional:
            raise HyperliquidEntityException(f"Not enough balance to withdraw: {self.balance} < {amount_in_notional}.")

        if self._internal_state.positions:
            position_size = sum(np.abs(pos.amount) for pos in self._internal_state.positions)
            entry_price: float = sum(
                (pos.entry_price * np.abs(pos.amount) / position_size) for pos in self._internal_state.positions)
            leverage = self._internal_state.positions[0].max_leverage

            maintenance_margin: float = (entry_price * position_size) / (2 * leverage)

            if self.balance - amount_in_notional < maintenance_margin:
                raise HyperliquidEntityException(
                    f"Not enough maintenance margin after withdraw: {maintenance_margin} < "
                    f"{self.balance - amount_in_notional}.")
        self._internal_state.collateral -= amount_in_notional

    def action_open_position(self, amount_in_product):
        """
        Opens a position with a specified amount of product.

        Args:
            amount_in_product (float): The amount of product to open the position with.
        """
        self._internal_state.positions.append(
            HyperLiquidPosition(amount=amount_in_product,
                                entry_price=self._global_state.mark_price,
                                max_leverage=self.MAX_LEVERAGE))

        self._internal_state.collateral -= np.abs(self._global_state.mark_price * amount_in_product * self.TRADING_FEE)
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
        if self.balance == 0 or self.size == 0:
            return 0
        return np.abs(self.size * self._global_state.mark_price / self.balance)

    def _clearing(self):
        """
        Aggregates the entity's state into a single position.
        
        This method handles cases where new trades partially close an existing position.
        For example, if there's an open short position of -10 and a long position of +5 is opened,
        the effective entry price should remain that of the original short for the remaining -5 position.
        
        Algorithm:
        1. Compute net_amount as the sum of all position amounts.
        2. If net_amount equals zero, it means the position is fully closed; clear the positions list.
        3. If net_amount is positive (a net long), compute the weighted average entry price using only the long trades.
        4. If net_amount is negative (a net short), compute the weighted average entry price using only the short trades.
        """
        net_amount = self.size
        if net_amount == 0:
            self._internal_state.positions = []
            return

        longs = [pos for pos in self._internal_state.positions if pos.amount > 0]
        shorts = [pos for pos in self._internal_state.positions if pos.amount < 0]

        if net_amount > 0:
            total_long = sum(pos.amount for pos in longs)
            effective_entry_price = sum(pos.entry_price * pos.amount for pos in longs) / total_long
        else:
            total_short = sum(abs(pos.amount) for pos in shorts)
            effective_entry_price = sum(pos.entry_price * abs(pos.amount) for pos in shorts) / total_short            

        new_position = HyperLiquidPosition(amount=net_amount, entry_price=effective_entry_price, max_leverage=self.MAX_LEVERAGE)
        self._internal_state.positions = [new_position]

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

        global_price: float = self._global_state.mark_price
        current_size: float = self.size
        self._internal_state.collateral -= current_size * global_price * self._global_state.funding_rate

    def _check_liquidation(self):
        """
        Determine if a position should be liquidated on Hyperliquid.

        Returns:
          bool : True if liquidation should be triggered, False otherwise.

        Liquidation is determined based on the liquidation price:
          For a long position (side = 1): if current_price <= liq_price --> liquidate.
          For a short position (side = -1): if current_price >= liq_price --> liquidate.

        The formula used:
          maintenance_margin_required = (entry_price * position_size) / (2 * leverage)
          margin_available = margin_balance - maintenance_margin_required
          liq_price = entry_price - side * (margin_available) / (position_size * (1 - (1/leverage)*side))

        This formula follows the Hyperliquid documentation on computing liquidation price.
        """

        if not self._internal_state.positions:
            return False

        position_size = sum(np.abs(pos.amount) for pos in self._internal_state.positions)
        entry_price: float = sum(
            (pos.entry_price * np.abs(pos.amount) / position_size) for pos in self._internal_state.positions)
        current_price = self._global_state.mark_price
        margin_balance = self._internal_state.collateral
        leverage = self._internal_state.positions[0].max_leverage  # One position has one max leverage value
        side = self._internal_state.positions[0].amount / np.abs(self._internal_state.positions[0].amount)

        maintenance_margin = (entry_price * position_size) / (2 * leverage)

        margin_available = margin_balance - maintenance_margin

        # If margin_available is negative, the position should be liquidated immediately.
        if margin_available < 0:
            return True

        # Note: (1 - (1/leverage)*side) is:
        #   For a long (side=1): 1 - 1/leverage.
        #   For a short (side=-1): 1 + 1/leverage.
        liq_price = entry_price - side * (margin_available) / (position_size * (1 - (1 / leverage) * side))

        if liq_price <= 0:
            liq_price = 0

        if side < 0:
            return current_price >= liq_price
        else:
            return current_price <= liq_price
