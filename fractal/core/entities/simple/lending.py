"""Minimal generic lending-protocol entity.

Models the canonical pattern of every collateralized-borrow protocol:
deposit notional collateral, borrow product against it up to a max LTV,
accrue per-period interest on both sides, get wiped if LTV crosses the
liquidation threshold. Use it for tests, examples and as the simplest
concrete lending entity. For Aave-specific behaviour use
:class:`fractal.core.entities.protocols.aave.AaveEntity`.
"""
import math
from dataclasses import dataclass

from fractal.core.base.entity import EntityException, GlobalState, InternalState
from fractal.core.entities.base.lending import BaseLendingEntity


class SimpleLendingException(EntityException):
    """Errors raised by :class:`SimpleLendingEntity`."""


@dataclass
class SimpleLendingGlobalState(GlobalState):
    """Market state.

    Attributes:
        collateral_price: USD price of the collateral asset.
        debt_price: USD price of the borrowed asset.
        lending_rate: Per-step interest credited to collateral.
        borrowing_rate: Per-step interest charged on borrowed amount.
    """
    collateral_price: float = 0.0
    debt_price: float = 0.0
    lending_rate: float = 0.0
    borrowing_rate: float = 0.0


@dataclass
class SimpleLendingInternalState(InternalState):
    """Position state.

    Attributes:
        collateral: Notional units deposited.
        borrowed: Product units borrowed.
    """
    collateral: float = 0.0
    borrowed: float = 0.0


class SimpleLendingEntity(BaseLendingEntity):
    """Generic lending entity with LTV-based liquidation.

    LTV is computed as ``borrowed · debt_price / (collateral · collateral_price)``.
    When LTV crosses ``liq_thr`` after interest accrual, both balances are wiped
    (loud and irreversible — model the worst case rather than partial liquidations).
    """

    def __init__(
        self,
        *,
        max_ltv: float = 0.8,
        liq_thr: float = 0.85,
        collateral_is_volatile: bool = False,
    ) -> None:
        if not 0 < max_ltv <= 1:
            raise SimpleLendingException(f"max_ltv must be in (0, 1], got {max_ltv}")
        if not 0 < liq_thr <= 1:
            raise SimpleLendingException(f"liq_thr must be in (0, 1], got {liq_thr}")
        if liq_thr < max_ltv:
            raise SimpleLendingException(
                f"liq_thr ({liq_thr}) must be >= max_ltv ({max_ltv})"
            )
        self.max_ltv: float = float(max_ltv)
        self.liq_thr: float = float(liq_thr)
        self.collateral_is_volatile: bool = collateral_is_volatile
        super().__init__()

    _internal_state: SimpleLendingInternalState
    _global_state: SimpleLendingGlobalState

    def _initialize_states(self) -> None:
        self._internal_state = SimpleLendingInternalState()
        self._global_state = SimpleLendingGlobalState()

    @property
    def internal_state(self) -> SimpleLendingInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> SimpleLendingGlobalState:  # type: ignore[override]
        return self._global_state

    # --------------------------------------------------------- collateral
    def action_deposit(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimpleLendingException(
                f"deposit amount must be >= 0, got {amount_in_notional}"
            )
        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimpleLendingException(
                f"withdraw amount must be >= 0, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.collateral:
            raise SimpleLendingException(
                f"withdraw exceeds collateral: {amount_in_notional} > "
                f"{self._internal_state.collateral}"
            )
        post = self._internal_state.collateral - amount_in_notional
        if post == 0 and self._internal_state.borrowed > 0:
            raise SimpleLendingException(
                "cannot withdraw all collateral while debt remains"
            )
        if post > 0:
            new_ltv = (
                self._internal_state.borrowed * self._global_state.debt_price
                / (post * self._global_state.collateral_price)
            )
            if new_ltv > self.max_ltv:
                raise SimpleLendingException(
                    f"withdraw would push LTV to {new_ltv} > max {self.max_ltv}"
                )
        self._internal_state.collateral -= amount_in_notional

    # --------------------------------------------------------- debt
    def action_borrow(self, amount_in_product: float) -> None:
        if amount_in_product < 0:
            raise SimpleLendingException(
                f"borrow amount must be >= 0, got {amount_in_product}"
            )
        if self._internal_state.collateral == 0:
            raise SimpleLendingException("no collateral available to borrow against")
        new_debt = self._internal_state.borrowed + amount_in_product
        new_ltv = (
            new_debt * self._global_state.debt_price
            / (self._internal_state.collateral * self._global_state.collateral_price)
        )
        if new_ltv > self.max_ltv:
            raise SimpleLendingException(
                f"borrow would push LTV to {new_ltv} > max {self.max_ltv}"
            )
        self._internal_state.borrowed += amount_in_product

    def action_repay(self, amount_in_product: float) -> None:
        if amount_in_product < 0:
            raise SimpleLendingException(
                f"repay amount must be >= 0, got {amount_in_product}"
            )
        if amount_in_product > self._internal_state.borrowed:
            raise SimpleLendingException(
                f"repay exceeds borrowed: {amount_in_product} > "
                f"{self._internal_state.borrowed}"
            )
        self._internal_state.borrowed -= amount_in_product

    # --------------------------------------------------------- readouts
    @property
    def balance(self) -> float:
        """Equity in notional units."""
        return (
            self._internal_state.collateral * self._global_state.collateral_price
            - self._internal_state.borrowed * self._global_state.debt_price
        )

    @property
    def ltv(self) -> float:
        """Current loan-to-value ratio (0 when no debt or no collateral)."""
        if self._internal_state.borrowed == 0:
            return 0.0
        if self._internal_state.collateral == 0:
            return float("inf")
        return (
            self._internal_state.borrowed * self._global_state.debt_price
            / (self._internal_state.collateral * self._global_state.collateral_price)
        )

    @property
    def collateral_value(self) -> float:
        """Collateral value in the accounting unit (``collateral × collateral_price``)."""
        return self._internal_state.collateral * self._global_state.collateral_price

    @property
    def debt_value(self) -> float:
        """Debt value in the accounting unit (``borrowed × debt_price``)."""
        return self._internal_state.borrowed * self._global_state.debt_price

    @property
    def health_factor(self) -> float:
        """Margin against liquidation: ``liq_thr / ltv``.

        ``+inf`` when no debt; ``0`` when LTV is non-finite (already liquidatable).
        """
        ltv = self.ltv
        if ltv == 0:
            return float("inf")
        if not math.isfinite(ltv):
            return 0.0
        return self.liq_thr / ltv

    def calculate_repay(self, target_ltv: float) -> float:
        """Amount of product to repay to drive LTV down to ``target_ltv``.

        Raises:
            SimpleLendingException: when current LTV is undefined (inf —
                no collateral but outstanding debt) or when ``target_ltv``
                is outside ``[0, self.ltv]``.
        """
        current = self.ltv
        import math
        if not math.isfinite(current):
            raise SimpleLendingException(
                "calculate_repay is undefined when current LTV is non-finite "
                "(no collateral against outstanding debt); fully repay first."
            )
        if target_ltv < 0 or target_ltv > current:
            raise SimpleLendingException(
                f"target_ltv {target_ltv} must be in [0, current_ltv={current}]"
            )
        return (
            self._internal_state.collateral
            * self._global_state.collateral_price
            * (current - target_ltv)
            / self._global_state.debt_price
        )

    # --------------------------------------------------------- lifecycle
    def _check_liquidation(self) -> None:
        if self._internal_state.collateral == 0 or self.ltv >= self.liq_thr:
            self._internal_state.collateral = 0.0
            self._internal_state.borrowed = 0.0

    def update_state(self, state: SimpleLendingGlobalState) -> None:
        """Apply prices and accrue per-step interest, then check liquidation.

        Rates below ``-1`` would flip the balances negative; rejected loudly.
        """
        if state.lending_rate < -1:
            raise SimpleLendingException(
                f"lending_rate must be >= -1, got {state.lending_rate}"
            )
        if state.borrowing_rate < -1:
            raise SimpleLendingException(
                f"borrowing_rate must be >= -1, got {state.borrowing_rate}"
            )
        self._global_state = state
        self._internal_state.collateral *= 1.0 + state.lending_rate
        self._internal_state.borrowed *= 1.0 + state.borrowing_rate
        self._check_liquidation()
