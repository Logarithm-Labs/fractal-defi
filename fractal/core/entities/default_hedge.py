from dataclasses import dataclass

from fractal.core.base.entity import (EntityException, GlobalState,
                                      InternalState)
from fractal.core.entities.hedge import BaseHedgeEntity


class DefaultHedgeEntityException(EntityException):
    """
    Exception raised for errors in the Default Hedge entity.
    """


@dataclass
class DefaultHedgeGlobalState(GlobalState):
    """
    Represents the global state of the Default Hedge entity.

    Attributes:
        price (float): The current price of the asset.
        funding_rate (float): The funding rate (same for both long and short positions).
    """
    price: float = 0.0
    funding_rate: float = 0.0


@dataclass
class DefaultHedgeInternalState(InternalState):
    """
    Represents the internal state of the Default Hedge entity.

    Attributes:
        collateral (float): The collateral held.
        position_size (float): The size of the open position.
        is_long (bool): Whether the position is long.
    """
    collateral: float = 0.0
    position_size: float = 0.0
    is_long: bool = True


class DefaultHedgeEntity(BaseHedgeEntity):
    """
    Represents a general hedge entity without borrowing fees, only funding fees.

    This entity is suitable for general exchanges with perpetual futures where funding
    fees exist but borrowing fees do not.

    Attributes:
        TRADING_FEE (float): The trading fee applied to each transaction.
    """

    def __init__(self, *args, trading_fee: float = 0.001, **kwargs):
        """
        Initializes the DefaultHedge entity.

        Args:
            trading_fee (float, optional): Trading fee. Defaults to 0.001.
        """
        super().__init__(*args, **kwargs)
        self.TRADING_FEE = trading_fee

    def _initialize_states(self):
        """
        Initializes the internal and global states for the hedge entity.
        """
        self._internal_state: DefaultHedgeInternalState = DefaultHedgeInternalState()
        self._global_state: DefaultHedgeGlobalState = DefaultHedgeGlobalState()

    def action_open_position(self, amount_in_product: float, is_long: bool):
        """
        Opens a position in the hedge entity.

        Args:
            amount_in_product (float): The size of the position in product units.
            is_long (bool): Whether the position is long or short.

        Raises:
            DefaultHedgeEntityException: If insufficient collateral to open position.
        """
        if self._internal_state.collateral <= 0:
            raise DefaultHedgeEntityException("No collateral available to open a position.")

        # Set the position size and direction
        self._internal_state.position_size = amount_in_product
        self._internal_state.is_long = is_long

    def action_close_position(self):
        """
        Closes the open position.

        Sets position size to zero.
        """
        self._internal_state.position_size = 0.0

    def action_deposit(self, amount_in_notional: float):
        """
        Deposits collateral into the hedge entity.

        Args:
            amount_in_notional (float): The amount of collateral to deposit.
        """
        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float):
        """
        Withdraws collateral from the hedge entity.

        Args:
            amount_in_notional (float): The amount of collateral to withdraw.

        Raises:
            DefaultHedgeEntityException: If insufficient collateral to withdraw.
        """
        if amount_in_notional > self._internal_state.collateral:
            raise DefaultHedgeEntityException("Insufficient collateral to withdraw.")
        self._internal_state.collateral -= amount_in_notional

    def calculate_funding_fee(self, time_period: float) -> float:
        """
        Calculates the funding fee over a given time period.

        Args:
            time_period (float): The period over which to calculate the funding fee.

        Returns:
            float: The funding fee to be applied.
        """
        funding_rate = self._global_state.funding_rate
        funding_fee = self._internal_state.position_size * funding_rate * time_period
        return funding_fee

    @property
    def leverage(self) -> float:
        """
        Returns the leverage of the position.

        Returns:
            float: The leverage ratio.
        """
        if self._internal_state.collateral == 0:
            return 0.0
        return self._internal_state.position_size / self._internal_state.collateral

    @property
    def size(self) -> float:
        """
        Returns the size of the open position.

        Returns:
            float: The position size.
        """
        return self._internal_state.position_size
