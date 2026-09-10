"""Morpho Blue isolated-lending market.

One market = one collateral token, one loan token, one oracle, one
LLTV. Sibling of :class:`SimpleLendingEntity` / :class:`AaveEntity`
with the same action signatures and readouts, so strategies can swap
lending backends. Differences that follow the protocol:

* **One threshold.** Morpho has no separate ``max_ltv``: you may borrow
  up to LLTV and are liquidatable strictly above it. ``max_ltv`` is
  kept as an optional *strategy-side* cushion (defaults to LLTV).
* **Oracle vs market price.** ``collateral_price`` is the *oracle*
  price (what health and ``maxBorrow`` use — for PT collateral usually
  a linear-discount feed). ``collateral_market_price`` optionally
  carries the market price for PnL; ``balance`` uses it when given.
* **Liquidation** closes the whole debt at the liquidation incentive
  factor ``LIF = min(1.15, 1/(1 − 0.3(1 − LLTV)))``: collateral worth
  ``debt × LIF`` is seized, the remainder stays with the borrower, and
  the position remains usable (no latch). If collateral runs out the
  rest is bad debt (socialised to suppliers — nothing further to pay).
* **Accrual** uses Morpho's third-order Taylor expansion of ``e^x − 1``
  on the per-bar exponent ``x`` carried in ``borrowing_rate``
  (``ln(1 + borrowApy) · Δt / YEAR``, computed by the loader).
"""
import math
from dataclasses import dataclass
from typing import Optional

from fractal.core.base.entity import EntityException, GlobalState, InternalState
from fractal.core.entities.base.lending import BaseLendingEntity
from fractal.core.entities.models.morpho_math import liquidation_incentive_factor, taylor_compounded


class MorphoException(EntityException):
    """Errors raised by :class:`MorphoEntity`."""


@dataclass
class MorphoGlobalState(GlobalState):
    """Market state.

    Attributes:
        collateral_price: **Oracle** price of one collateral unit in
            notional — drives ``ltv``, ``max_borrow_amount`` and liquidation.
        debt_price: Notional price of one loan unit (``1.0`` for a USD
            loan token in a USD backtest).
        lending_rate: Per-bar rate credited to collateral (``0`` for PT
            collateral — it does not earn supplier yield).
        borrowing_rate: Per-bar exponent ``x = ln(1 + borrowApy) · Δt / YEAR``;
            debt grows by ``taylor_compounded(x)``.
        collateral_market_price: Optional market price of one collateral
            unit for PnL; ``None`` ⇒ the oracle price is used.
    """
    collateral_price: float = 0.0
    debt_price: float = 0.0
    lending_rate: float = 0.0
    borrowing_rate: float = 0.0
    collateral_market_price: Optional[float] = None


@dataclass
class MorphoInternalState(InternalState):
    """Position state.

    Attributes:
        collateral: Collateral units deposited.
        borrowed: Loan units borrowed (grows with accrual).
        liquidation_count: Liquidations suffered so far.
    """
    collateral: float = 0.0
    borrowed: float = 0.0
    liquidation_count: int = 0


class MorphoEntity(BaseLendingEntity):
    """Morpho Blue market position with LLTV health and LIF liquidation."""

    def __init__(
        self,
        *,
        lltv: float = 0.86,
        max_ltv: Optional[float] = None,
        liquidation_incentive_factor: Optional[float] = None,  # pylint: disable=redefined-outer-name
        collateral_is_volatile: bool = True,
    ) -> None:
        """
        Args:
            lltv: Market liquidation LTV (``0.86`` / ``0.915`` on live PT markets).
            max_ltv: Strategy-side borrow cap; defaults to ``lltv``.
            liquidation_incentive_factor: Override of the protocol formula.
            collateral_is_volatile: Informational flag for ``liquidation_price``.
        """
        if not 0 < lltv <= 1:
            raise MorphoException(f"lltv must be in (0, 1], got {lltv}")
        max_ltv = lltv if max_ltv is None else max_ltv
        if not 0 < max_ltv <= 1:
            raise MorphoException(f"max_ltv must be in (0, 1], got {max_ltv}")
        if lltv < max_ltv:
            raise MorphoException(
                f"liq_thr (lltv={lltv}) must be >= max_ltv ({max_ltv})"
            )
        lif = (
            self._default_lif(lltv) if liquidation_incentive_factor is None
            else float(liquidation_incentive_factor)
        )
        if lif < 1:
            raise MorphoException(f"liquidation_incentive_factor must be >= 1, got {lif}")
        # Set config BEFORE super so subclass overrides of
        # ``_initialize_states`` can rely on these.
        self.lltv: float = float(lltv)
        self.liq_thr: float = float(lltv)
        self.max_ltv: float = float(max_ltv)
        self.lif: float = lif
        self.collateral_is_volatile: bool = collateral_is_volatile
        super().__init__()

    @staticmethod
    def _default_lif(lltv: float) -> float:
        return liquidation_incentive_factor(lltv)

    _internal_state: MorphoInternalState
    _global_state: MorphoGlobalState

    def _initialize_states(self) -> None:
        self._internal_state = MorphoInternalState()
        self._global_state = MorphoGlobalState()

    @property
    def internal_state(self) -> MorphoInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> MorphoGlobalState:  # type: ignore[override]
        return self._global_state

    # --------------------------------------------------------- collateral
    def action_deposit(self, amount_in_notional: float) -> None:
        """Supply ``amount_in_notional`` collateral units."""
        if amount_in_notional < 0:
            raise MorphoException(f"deposit amount must be >= 0, got {amount_in_notional}")
        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Withdraw collateral units; the remaining position must stay within ``max_ltv``."""
        if amount_in_notional < 0:
            raise MorphoException(f"withdraw amount must be >= 0, got {amount_in_notional}")
        if amount_in_notional > self._internal_state.collateral:
            raise MorphoException(
                f"withdraw exceeds collateral: {amount_in_notional} > "
                f"{self._internal_state.collateral}"
            )
        post = self._internal_state.collateral - amount_in_notional
        if post == 0 and self._internal_state.borrowed > 0:
            raise MorphoException("cannot withdraw all collateral while debt remains")
        if post > 0 and self._internal_state.borrowed > 0:
            self._require_prices()
            new_ltv = self.debt_value / (post * self._global_state.collateral_price)
            if new_ltv > self.max_ltv:
                raise MorphoException(f"withdraw would push LTV to {new_ltv} > max {self.max_ltv}")
        self._internal_state.collateral -= amount_in_notional

    # --------------------------------------------------------- debt
    def action_borrow(self, amount_in_product: float) -> None:
        """Borrow loan units; cumulative LTV must stay within ``max_ltv``."""
        if amount_in_product < 0:
            raise MorphoException(f"borrow amount must be >= 0, got {amount_in_product}")
        if self._internal_state.collateral == 0:
            raise MorphoException("no collateral available to borrow against")
        self._require_prices()
        new_debt = self._internal_state.borrowed + amount_in_product
        new_ltv = new_debt * self._global_state.debt_price / self.collateral_value
        if new_ltv > self.max_ltv:
            raise MorphoException(f"borrow would push LTV to {new_ltv} > max {self.max_ltv}")
        self._internal_state.borrowed = new_debt

    def action_repay(self, amount_in_product: float) -> None:
        """Repay loan units."""
        if amount_in_product < 0:
            raise MorphoException(f"repay amount must be >= 0, got {amount_in_product}")
        if amount_in_product > self._internal_state.borrowed:
            raise MorphoException(
                f"repay exceeds borrowed: {amount_in_product} > {self._internal_state.borrowed}"
            )
        self._internal_state.borrowed -= amount_in_product

    def _require_prices(self) -> None:
        if self._global_state.collateral_price <= 0:
            raise MorphoException(
                f"collateral_price must be > 0, got {self._global_state.collateral_price}"
            )
        if self._global_state.debt_price <= 0:
            raise MorphoException(f"debt_price must be > 0, got {self._global_state.debt_price}")

    # --------------------------------------------------------- readouts
    @property
    def collateral_value(self) -> float:
        """Collateral at the **oracle** price (health basis)."""
        return self._internal_state.collateral * self._global_state.collateral_price

    @property
    def collateral_market_value(self) -> float:
        """Collateral at the market price when given, else at the oracle price (PnL basis)."""
        price = self._global_state.collateral_market_price
        if price is None:
            price = self._global_state.collateral_price
        return self._internal_state.collateral * price

    @property
    def debt_value(self) -> float:
        """``borrowed × debt_price``."""
        return self._internal_state.borrowed * self._global_state.debt_price

    @property
    def balance(self) -> float:
        """Equity in notional: ``collateral_market_value − debt_value``."""
        return self.collateral_market_value - self.debt_value

    @property
    def ltv(self) -> float:
        """``debt_value / collateral_value``; ``0`` without debt, ``inf`` without collateral."""
        if self._internal_state.borrowed == 0:
            return 0.0
        if self.collateral_value <= 0:
            return float("inf")
        return self.debt_value / self.collateral_value

    @property
    def health_factor(self) -> float:
        """``lltv / ltv``; ``inf`` without debt, ``0`` when LTV is non-finite."""
        ltv = self.ltv
        if ltv == 0:
            return float("inf")
        if not math.isfinite(ltv):
            return 0.0
        return self.lltv / ltv

    @property
    def max_borrow_amount(self) -> float:
        """Extra loan units borrowable before breaching ``max_ltv``."""
        headroom_value = self.collateral_value * self.max_ltv - self.debt_value
        if headroom_value <= 0 or self._global_state.debt_price <= 0:
            return 0.0
        return headroom_value / self._global_state.debt_price

    @property
    def liquidation_price(self) -> float:
        """Volatile-asset price at which LTV reaches ``lltv`` (see :attr:`AaveEntity.liquidation_price`)."""
        if self._internal_state.borrowed == 0 or self._internal_state.collateral == 0:
            return float("nan")
        if self.collateral_is_volatile:
            denom = self._internal_state.collateral * self.lltv
            return self.debt_value / denom if denom else float("inf")
        denom = self._internal_state.borrowed
        return (self.lltv * self.collateral_value) / denom if denom else float("inf")

    def calculate_repay(self, target_ltv: float) -> float:
        """Loan units to repay to bring LTV down to ``target_ltv``."""
        current = self.ltv
        if not math.isfinite(current):
            raise MorphoException(
                "calculate_repay is undefined when current LTV is non-finite "
                "(no collateral against outstanding debt); fully repay first."
            )
        if target_ltv < 0 or target_ltv > current:
            raise MorphoException(f"target_ltv {target_ltv} must be in [0, current_ltv={current}]")
        return self.collateral_value * (current - target_ltv) / self._global_state.debt_price

    # --------------------------------------------------------- lifecycle
    def _check_liquidation(self) -> None:
        """Morpho ``liquidate``: liquidatable when ``ltv > lltv``; close the whole debt at LIF."""
        if self._internal_state.borrowed == 0:
            return
        if self._internal_state.collateral > 0 and self.ltv <= self.lltv:
            return
        price = self._global_state.collateral_price
        seized = self._internal_state.collateral
        if price > 0:
            seized = min(seized, self.debt_value * self.lif / price)
        self._internal_state.collateral = max(0.0, self._internal_state.collateral - seized)
        self._internal_state.borrowed = 0.0
        self._internal_state.liquidation_count += 1

    def update_state(self, state: MorphoGlobalState) -> None:
        """Validate rates, apply the snapshot, accrue, then check liquidation."""
        if state.lending_rate < -1:
            raise MorphoException(f"lending_rate must be >= -1, got {state.lending_rate}")
        if state.borrowing_rate < -1:
            raise MorphoException(f"borrowing_rate must be >= -1, got {state.borrowing_rate}")
        if state.collateral_market_price is not None and state.collateral_market_price < 0:
            raise MorphoException(
                f"collateral_market_price must be >= 0, got {state.collateral_market_price}"
            )
        self._global_state = state
        self._internal_state.collateral *= 1.0 + state.lending_rate
        self._internal_state.borrowed *= 1.0 + taylor_compounded(state.borrowing_rate)
        self._check_liquidation()
