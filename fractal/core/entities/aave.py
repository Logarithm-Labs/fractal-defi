from dataclasses import dataclass

from fractal.core.base.entity import EntityException
from fractal.core.entities.lending import BaseLendingEntity


@dataclass
class AaveGlobalState:
    """
    Represents the global state of the Aave protocol.

    Attributes:
        notional_price (float): The notional price.
        product_price (float): The product price.
        lending_rate (float): The lending rate.
        borrowing_rate (float): The borrowing rate.
    """
    notional_price: float = 0.0
    product_price: float = 0.0
    lending_rate: float = 0.0
    borrowing_rate: float = 0.0


@dataclass
class AaveInternalState:
    """
    Represents the internal state of an Aave entity.

    Attributes:
        collateral (float): The amount of collateral in notional.
        borrowed (float): The amount borrowed in product.
    """
    collateral: float = 0.0
    borrowed: float = 0.0


class AaveEntity(BaseLendingEntity):
    """
    Represents an Aave isolated market entity.
    """
    def __init__(self, *args, max_ltv: float = 0.8, liq_threshold: float = 0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_ltv: float = max_ltv
        self.liq_threshold: float = liq_threshold

    def _initialize_states(self):
        self._internal_state: AaveInternalState = AaveInternalState()
        self._global_state: AaveGlobalState = AaveGlobalState()

    def action_redeem(self, amount_in_product: float):
        """
        Redeems an amount on the Aave protocol.

        Args:
            amount_in_product (float, optional): The amount to redeem in product value.
        """
        if amount_in_product > self._internal_state.borrowed:
            raise EntityException(
                f"Repay amount exceeds borrowed amount. Borrowed:\
                {self._internal_state.borrowed}, Repay: {amount_in_product}"
            )
        self._internal_state.borrowed -= amount_in_product

    def action_borrow(self, amount_in_product: float):
        """
        Borrows an amount on the Aave protocol.

        Args:
            amount_in_product (float, optional): The amount to borrow in product value.
        """
        if (
            amount_in_product * self._global_state.product_price
            > self._internal_state.collateral * self._global_state.notional_price * self.max_ltv
        ):
            raise EntityException("Exceeds maximum loan-to-value ratio.")
        self._internal_state.borrowed += amount_in_product

    def action_deposit(self, amount_in_notional: float) -> None:
        """
        Deposits the specified amount in notional value into the entity.
        Each entity stores the cash balance in notional value.

        Args:
            amount_in_notional (float): The amount to be deposited in notional value.
        """
        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        """
        Withdraws the specified amount from the entity's account.
        Each entity stores the cash balance in notional value.

        Args:
            amount_in_notional (float): The amount to withdraw in notional value.
        """
        if (
            self._internal_state.borrowed * self._global_state.product_price
            / ((self._internal_state.collateral - amount_in_notional) * self._global_state.notional_price)
        ) > self.max_ltv:
            raise EntityException("Exceeds maximum loan-to-value ratio.")
        self._internal_state.collateral -= amount_in_notional

    @property
    def balance(self) -> float:
        """
        Calculates the balance of the Aave entity.

        Aave entity balance is calculated as the difference between the collateral
        and the borrowed amount.

        Returns:
            float: The balance of the Aave entity.
        """
        return (
            self._internal_state.collateral * self._global_state.notional_price
            - self._internal_state.borrowed * self._global_state.product_price
        )

    @property
    def ltv(self) -> float:
        """
        Calculates the loan-to-value (LTV) ratio of the Aave entity.

        LTV ratio is calculated as the ratio of the borrowed amount to the collateral.
        Returns:
            float: The LTV ratio.
        """
        return (
            self._internal_state.borrowed * self._global_state.product_price
            / (self._internal_state.collateral * self._global_state.notional_price)
        )

    def check_liquidation(self):
        """
        Checks if the entity is eligible for liquidation.

        Liquidation occurs when the LTV ratio exceeds the liquidation threshold.
        """
        if self.ltv >= self.liq_threshold:
            self._internal_state.collateral = 0
            self._internal_state.borrowed = 0

    def update_state(self, state: AaveGlobalState):
        self._global_state = state
        self._internal_state.collateral *= (state.lending_rate + 1)
        self._internal_state.borrowed *= (state.borrowing_rate + 1)
        self.check_liquidation()
