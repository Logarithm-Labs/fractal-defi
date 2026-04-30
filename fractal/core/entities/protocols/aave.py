from dataclasses import dataclass

from fractal.core.base.entity import EntityException, GlobalState, InternalState
from fractal.core.entities.base.lending import BaseLendingEntity


@dataclass
class AaveGlobalState(GlobalState):
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
class AaveInternalState(InternalState):
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

    _internal_state: AaveInternalState
    _global_state: AaveGlobalState

    def __init__(self, *args, max_ltv: float = 0.8, liq_thr: float = 0.85, **kwargs):
        # Set config BEFORE super so any subclass override of
        # ``_initialize_states`` can rely on ``self.max_ltv`` / ``self.liq_thr``.
        self.max_ltv: float = max_ltv
        self.liq_thr: float = liq_thr
        super().__init__(*args, **kwargs)

    def _initialize_states(self):
        self._internal_state = AaveInternalState()
        self._global_state = AaveGlobalState()

    @property
    def internal_state(self) -> AaveInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> AaveGlobalState:  # type: ignore[override]
        return self._global_state

    def action_repay(self, amount_in_product: float):
        """Repay borrowed product debt on Aave.

        Args:
            amount_in_product: Amount of borrowed product to repay.
        """
        if amount_in_product < 0:
            raise EntityException(
                f"repay amount must be >= 0, got {amount_in_product}"
            )
        if amount_in_product > self._internal_state.borrowed:
            raise EntityException("Repay amount exceeds borrowed amount")
        self._internal_state.borrowed -= amount_in_product

    def action_redeem(self, amount_in_product: float):
        """Deprecated alias of :meth:`action_repay`.

        Kept for back-compat: the method was originally named ``redeem``
        even though semantically it repays debt (Aave V3 uses
        ``repay``). New code should use :meth:`action_repay`. Calls
        through ``execute(Action('redeem', ...))`` continue to work.
        """
        import warnings
        warnings.warn(
            "AaveEntity.action_redeem is deprecated; use action_repay instead "
            "(it repays borrowed debt — 'redeem' was a misnomer).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.action_repay(amount_in_product)

    def action_borrow(self, amount_in_product: float):
        """
        Borrows an amount on the Aave protocol.

        Args:
            amount_in_product (float, optional): The amount to borrow in product value.
        """
        if amount_in_product < 0:
            raise EntityException(
                f"borrow amount must be >= 0, got {amount_in_product}"
            )
        if self._internal_state.collateral == 0:
            raise EntityException("No collateral available.")
        if self._global_state.notional_price <= 0:
            raise EntityException(
                f"notional_price must be > 0, got {self._global_state.notional_price}"
            )
        if self._global_state.product_price <= 0:
            raise EntityException(
                f"product_price must be > 0, got {self._global_state.product_price}"
            )
        if (
            amount_in_product
            * self._global_state.product_price
            / (self._internal_state.collateral * self._global_state.notional_price)
            > self.max_ltv
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
        if amount_in_notional < 0:
            raise EntityException(
                f"deposit amount must be >= 0, got {amount_in_notional}"
            )
        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        """
        Withdraws the specified amount from the entity's account.
        Each entity stores the cash balance in notional value.

        Args:
            amount_in_notional (float): The amount to withdraw in notional value.
        """
        if amount_in_notional < 0:
            raise EntityException(
                f"withdraw amount must be >= 0, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.collateral:
            raise EntityException("Withdrawal amount exceeds collateral.")
        post_collateral = self._internal_state.collateral - amount_in_notional
        if self._internal_state.borrowed > 0:
            if post_collateral == 0:
                raise EntityException(
                    "cannot withdraw all collateral while debt remains"
                )
            if self._global_state.notional_price <= 0:
                raise EntityException(
                    f"notional_price must be > 0, got {self._global_state.notional_price}"
                )
            if self._global_state.product_price <= 0:
                raise EntityException(
                    f"product_price must be > 0, got {self._global_state.product_price}"
                )
            if (
                self._internal_state.borrowed
                * self._global_state.product_price
                / (post_collateral * self._global_state.notional_price)
                > self.max_ltv
            ):
                raise EntityException("Exceeds maximum loan-to-value ratio.")
        self._internal_state.collateral -= amount_in_notional

    def calculate_repay(self, target_ltv: float) -> float:
        """
        Calculates the amount to repay in order to reach the target loan-to-value ratio.

        Args:
            target_ltv (float): The target loan-to-value ratio.

        Returns:
            float: The amount to repay in product value.

        Raises:
            EntityException: when current LTV is non-finite (no collateral
                with outstanding debt), when ``target_ltv`` is outside
                ``[0, current_ltv]``, or when ``product_price <= 0``.
        """
        import math
        current = self.ltv
        if not math.isfinite(current):
            raise EntityException(
                "calculate_repay is undefined when current LTV is non-finite "
                "(no collateral against outstanding debt); fully repay first."
            )
        if target_ltv < 0 or target_ltv > current:
            raise EntityException(
                f"target_ltv {target_ltv} must be in [0, current_ltv={current}]"
            )
        if self._global_state.product_price <= 0:
            raise EntityException(
                f"product_price must be > 0, got {self._global_state.product_price}"
            )
        return (
            self._internal_state.collateral
            * self._global_state.notional_price
            * (current - target_ltv)
            / self._global_state.product_price
        )

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
        Loan-to-value ratio.

        Returns ``0`` when no debt is outstanding, ``+inf`` when there is
        debt against zero collateral or zero notional price (so callers
        like :meth:`calculate_repay` can detect the undefined case loudly).
        """
        if self._internal_state.borrowed == 0:
            return 0.0
        denom = self._internal_state.collateral * self._global_state.notional_price
        if denom <= 0:
            return float("inf")
        return self._internal_state.borrowed * self._global_state.product_price / denom

    def check_liquidation(self):
        """
        Checks if the entity is eligible for liquidation.

        Liquidation occurs when the LTV ratio exceeds the liquidation threshold.
        """
        if self._internal_state.collateral == 0 or self.ltv >= self.liq_thr:
            self._internal_state.collateral = 0
            self._internal_state.borrowed = 0

    def update_state(self, state: AaveGlobalState):
        self._global_state = state
        self._internal_state.collateral *= state.lending_rate + 1
        self._internal_state.borrowed *= state.borrowing_rate + 1
        self.check_liquidation()
