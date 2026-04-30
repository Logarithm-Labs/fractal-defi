import math
import warnings
from dataclasses import dataclass

from fractal.core.base.entity import EntityException, GlobalState, InternalState
from fractal.core.entities.base.lending import BaseLendingEntity


@dataclass
class AaveGlobalState(GlobalState):
    """Aave market state.

    Field names are direction-neutral (don't assume which side is stable):
    they describe the **role** of each asset in the loan, not its character.

    Attributes:
        collateral_price: Price of the collateral asset in the strategy's
            accounting unit (e.g. USD). For default stable-collateral mode
            with USDC: ``1.0``. For volatile-collateral mode with ETH: ``ETH/USD``.
        debt_price: Price of the borrowed asset in the same accounting unit.
        lending_rate: Per-step interest credited to collateral.
        borrowing_rate: Per-step interest charged on debt.
    """
    collateral_price: float = 0.0
    debt_price: float = 0.0
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
    """Aave isolated market entity. Direction-agnostic.

    Aave on-chain supports any (collateral, borrow) pair. Our model is
    symmetric: ``collateral`` is whatever you deposited; ``borrowed`` is
    whatever you took out. The two natural configurations:

    * **Stable collateral, volatile debt** (``collateral_is_volatile=False``,
      default) — e.g. deposit USDC, borrow ETH. Synthetic short ETH.
      ``collateral_price = 1.0``, ``debt_price = ETH price``.
    * **Volatile collateral, stable debt** (``collateral_is_volatile=True``)
      — e.g. deposit ETH, borrow USDC. Setup for leveraged-long ETH (when
      composed with a spot entity that re-acquires the borrowed stable).
      ``collateral_price = ETH price``, ``debt_price = 1.0``.

    The ``collateral_is_volatile`` flag is informational — the math
    doesn't depend on it (LTV / balance / liquidation are computed from
    prices, regardless of which side is stable). Strategies that need to
    know the direction can branch on the flag for clarity.

    Helper properties that work regardless of direction:

    * :attr:`collateral_value` — collateral in the accounting unit.
    * :attr:`debt_value` — debt in the accounting unit.
    * :attr:`health_factor` — distance to liquidation (``liq_thr / ltv``).
    """

    _internal_state: AaveInternalState
    _global_state: AaveGlobalState

    def __init__(
        self,
        *args,
        max_ltv: float = 0.8,
        liq_thr: float = 0.85,
        collateral_is_volatile: bool = False,
        **kwargs,
    ):
        if not 0 < max_ltv <= 1:
            raise EntityException(f"max_ltv must be in (0, 1], got {max_ltv}")
        if not 0 < liq_thr <= 1:
            raise EntityException(f"liq_thr must be in (0, 1], got {liq_thr}")
        if liq_thr < max_ltv:
            raise EntityException(
                f"liq_thr ({liq_thr}) must be >= max_ltv ({max_ltv})"
            )
        # Set config BEFORE super so any subclass override of
        # ``_initialize_states`` can rely on ``self.max_ltv`` / ``self.liq_thr``.
        self.max_ltv: float = max_ltv
        self.liq_thr: float = liq_thr
        self.collateral_is_volatile: bool = collateral_is_volatile
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
        if self._global_state.collateral_price <= 0:
            raise EntityException(
                f"collateral_price must be > 0, got {self._global_state.collateral_price}"
            )
        if self._global_state.debt_price <= 0:
            raise EntityException(
                f"debt_price must be > 0, got {self._global_state.debt_price}"
            )
        if (
            amount_in_product
            * self._global_state.debt_price
            / (self._internal_state.collateral * self._global_state.collateral_price)
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
            if self._global_state.collateral_price <= 0:
                raise EntityException(
                    f"collateral_price must be > 0, got {self._global_state.collateral_price}"
                )
            if self._global_state.debt_price <= 0:
                raise EntityException(
                    f"debt_price must be > 0, got {self._global_state.debt_price}"
                )
            if (
                self._internal_state.borrowed
                * self._global_state.debt_price
                / (post_collateral * self._global_state.collateral_price)
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
                ``[0, current_ltv]``, or when ``debt_price <= 0``.
        """
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
        if self._global_state.debt_price <= 0:
            raise EntityException(
                f"debt_price must be > 0, got {self._global_state.debt_price}"
            )
        return (
            self._internal_state.collateral
            * self._global_state.collateral_price
            * (current - target_ltv)
            / self._global_state.debt_price
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
            self._internal_state.collateral * self._global_state.collateral_price
            - self._internal_state.borrowed * self._global_state.debt_price
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
        denom = self._internal_state.collateral * self._global_state.collateral_price
        if denom <= 0:
            return float("inf")
        return self._internal_state.borrowed * self._global_state.debt_price / denom

    @property
    def collateral_value(self) -> float:
        """Collateral value in the entity's accounting unit (``collateral × collateral_price``)."""
        return self._internal_state.collateral * self._global_state.collateral_price

    @property
    def debt_value(self) -> float:
        """Debt value in the entity's accounting unit (``borrowed × debt_price``)."""
        return self._internal_state.borrowed * self._global_state.debt_price

    @property
    def health_factor(self) -> float:
        """Margin against liquidation: ``liq_thr / ltv``.

        Returns ``+inf`` when no debt is outstanding, ``0`` when LTV is
        non-finite (no collateral but debt remains — already liquidatable).
        Below 1.0 means liquidation should have triggered.
        """
        ltv = self.ltv
        if ltv == 0:
            return float("inf")
        if not math.isfinite(ltv):
            return 0.0
        return self.liq_thr / ltv

    def _check_liquidation(self) -> None:
        """Wipe position if LTV crosses liquidation threshold.

        Liquidation when ``ltv >= liq_thr`` OR ``collateral == 0`` while
        debt remains. Wipes both legs (loud / irreversible — model the
        worst case rather than partial liquidations).
        """
        if self._internal_state.collateral == 0 or self.ltv >= self.liq_thr:
            self._internal_state.collateral = 0
            self._internal_state.borrowed = 0

    def check_liquidation(self) -> None:
        """Deprecated public alias of :meth:`_check_liquidation`.

        Internal lifecycle hook — strategies should not call this directly,
        it is invoked from ``update_state``. Will be removed in a future
        release.
        """
        warnings.warn(
            "AaveEntity.check_liquidation is deprecated; the liquidation "
            "check is invoked internally from update_state. If you need "
            "to call it manually, use AaveEntity._check_liquidation.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._check_liquidation()

    def update_state(self, state: AaveGlobalState):
        """Apply Aave market state, accrue interest on both legs, check liquidation.

        Rates below ``-1`` would flip the balances negative; rejected loudly.
        """
        if state.lending_rate < -1:
            raise EntityException(
                f"lending_rate must be >= -1, got {state.lending_rate}"
            )
        if state.borrowing_rate < -1:
            raise EntityException(
                f"borrowing_rate must be >= -1, got {state.borrowing_rate}"
            )
        self._global_state = state
        self._internal_state.collateral *= state.lending_rate + 1
        self._internal_state.borrowed *= state.borrowing_rate + 1
        self._check_liquidation()
