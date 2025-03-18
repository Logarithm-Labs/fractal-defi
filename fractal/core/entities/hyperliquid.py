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
    def __init__(self, *args, trading_fee: float = 0.00035, max_leverage: float = 50, **kwargs):
        """
        Initializes the Hyperliquid entity.

        Args:
            trading_fee (float, optional): Trading fee. Defaults to 0.00035.
            max_leverage (float, optional): Maximum leverage. Defaults to 50.
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
        return abs(self.size) * self._global_state.mark_price / self.balance

    def _clearing(self):
        """
        Aggregates all positions into a single position.

        If positions of opposite sides are present (i.e., a closing trade occurs),
        the entry price of the remaining open position does not change, and the
        realized PnL for the closed quantity is added to the collateral.
        For positions on the same side (i.e., opening trades), the entry price
        is updated to the weighted average.
        """
        positions = self._internal_state.positions
        if not positions or len(positions) == 1:
            return

        # Use the first position as the aggregated base position.
        base = positions[0]
        realized_pnl = 0.0

        # Process all subsequent positions.
        for pos in positions[1:]:
            # If both positions are in the same direction, aggregate them.
            if base.amount * pos.amount > 0:
                new_amount = base.amount + pos.amount
                base.entry_price = (base.amount * base.entry_price + pos.amount * pos.entry_price) / new_amount
                base.amount = new_amount
            else:
                # Opposite direction: a closing trade.
                closed_qty = min(abs(base.amount), abs(pos.amount))
                # Determine the side of the base position: 1 for long, -1 for short.
                sign = 1 if base.amount > 0 else -1
                # For a long base (sign = 1): realized pnl = closed_qty * (pos.entry_price - base.entry_price)
                # For a short base (sign = -1): realized pnl = closed_qty * (base.entry_price - pos.entry_price)
                realized_pnl += closed_qty * (pos.entry_price - base.entry_price) * sign
                # Reduce the base position by the closed quantity.
                base.amount -= sign * closed_qty
                # Calculate the remaining quantity in the incoming position.
                remaining = abs(pos.amount) - closed_qty
                if remaining > 0:
                    # If the base position is fully closed, adopt the remainder as the new base position.
                    if abs(base.amount) < 1e-9:
                        base = HyperLiquidPosition(
                            amount=remaining if pos.amount > 0 else -remaining,
                            entry_price=pos.entry_price,
                            max_leverage=self.MAX_LEVERAGE
                        )
                    # If the base position is not fully closed, the remaining incoming position is effectively absorbed.

        # Add the realized pnl from the closed quantity to the collateral.
        self._internal_state.collateral += realized_pnl

        # If the aggregated position is effectively closed (net amount nearly zero), clear the positions.
        if abs(base.amount) < 1e-9:
            self._internal_state.positions = []
        else:
            self._internal_state.positions = [base]

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

        position_size = abs(self.size)
        entry_price: float = self._internal_state.positions[0].entry_price
        current_price = self._global_state.mark_price
        margin_balance = self.balance
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
        liq_price = current_price - side * (margin_available) / (position_size * (1 - (1 / leverage) * side))
        if liq_price <= 0:
            liq_price = 0

        if side < 0:
            return current_price >= liq_price
        else:
            return current_price <= liq_price
