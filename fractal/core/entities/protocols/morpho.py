"""Morpho isolated-market lending entity — Session 2.

Morpho Blue exposes one isolated market per (collateral, loan) pair.
For our strategy the relevant market is **PT-sUSDe → USDC**:
deposit PT as collateral, borrow USDC against it, with a fixed
liquidation loan-to-value (LLTV) parameter set per market (typically
0.86 or 0.915 for PT collateral).

This entity is intentionally close in shape to ``fractal.core.entities.protocols.aave.AaveEntity``
(borrowed style, dropped the ``collateral_is_volatile`` knob since our
collateral is PT and direction is fixed: collateral = PT, debt = USDC).

* ``GlobalState`` — collateral price (PT mark in USDC), debt price (USDC = 1),
  borrowing rate, utilization, observation timestamp.
* ``InternalState`` — collateral amount (PT face), debt amount (USDC),
  last accrual timestamp, liquidation flag.
* Actions:
  - ``action_deposit`` — add PT collateral (in face units).
  - ``action_withdraw`` — remove PT collateral.
  - ``action_borrow`` — draw USDC against collateral.
  - ``action_repay`` — pay back USDC debt.

Session 2 mechanics:
    * ``update_state`` accrues debt at ``borrowing_rate * dt`` between
      successive observations (continuous-compounding approximated by a
      single per-step multiplier — close enough at our 1h/1d granularity).
    * ``action_withdraw`` and ``action_borrow`` reject mutations that
      would push ``ltv`` above the market LLTV. Failed actions revert
      the partial mutation so the entity never holds half-applied state.
    * Once an observation pushes ``ltv > lltv`` (e.g. via a collateral
      price drop), the entity sets ``is_liquidated = True`` and every
      subsequent action raises. Realistic seizure/penalty modeling is
      deferred to Session 5; the flag is enough for strategy logic.

We deliberately do NOT accrue a "supplier" yield on the collateral side:
the collateral is PT face amount, not a yield-bearing deposit. PT pays
off via its price climbing to 1 at expiry — that effect lives on
``MorphoGlobalState.collateral_price``, not on the internal balance.
"""

from __future__ import annotations

from dataclasses import dataclass

from fractal.core.base.entity import (
    EntityException,
    GlobalState,
    InternalState,
)
from fractal.core.entities.base.lending import BaseLendingEntity


# Annualization base for rate accrual. Matches the convention used by
# Morpho IRMs and most DeFi rate feeds (Julian year, 365.25 days).
SECONDS_PER_YEAR: float = 365.25 * 24 * 3600


@dataclass
class MorphoGlobalState(GlobalState):
    """Market context for a single Morpho isolated market.

    Field naming mirrors AaveGlobalState so a strategy can swap one
    lending backend for another without rewriting predict logic.

    Attributes:
        collateral_price: USDC value of one unit of collateral (= PT mark price).
        debt_price: USDC value of one unit of debt asset. For USDC debt: 1.0.
        lending_rate: Per-step interest credited to collateral. Zero for
            Morpho's PT markets — PT does not earn supplier yield.
        borrowing_rate: Annualized interest charged on debt. Derived from
            Morpho IRM (interest-rate model) on each step; accrued inside
            :meth:`MorphoEntity.update_state` against ``dt``.
        utilization: Fraction of supplied debt asset currently borrowed.
            Drives ``borrowing_rate`` through the IRM; we expose it for
            risk monitoring (rate spikes occur at high utilization).
        timestamp_seconds: Unix epoch seconds of the observation. The
            strategy populates this from ``Observation.timestamp`` so the
            entity can compute ``dt`` between successive updates.
    """

    collateral_price: float = 0.0
    debt_price: float = 1.0
    lending_rate: float = 0.0
    borrowing_rate: float = 0.0
    utilization: float = 0.0
    timestamp_seconds: float = 0.0


@dataclass
class MorphoInternalState(InternalState):
    """Position state inside the Morpho market.

    Attributes:
        collateral: PT collateral amount in face units.
        debt: USDC debt amount (compounds via ``borrowing_rate`` accrual).
        last_timestamp: Unix epoch seconds of the last observation. ``None``
            until the first ``update_state`` call; thereafter the entity
            accrues debt over ``timestamp_seconds - last_timestamp``.
        is_liquidated: Flag set when an observation pushes ``ltv > lltv``.
            Once true, every action raises. Cleared only by re-instantiation.
    """

    collateral: float = 0.0
    debt: float = 0.0
    last_timestamp: float | None = None
    is_liquidated: bool = False


@dataclass
class MorphoConfig:
    """Configuration for a Morpho lending market.

    Attributes:
        market_id: Morpho's 32-byte market identifier (hex string).
            Informational at backtest level.
        lltv: Liquidation loan-to-value threshold for this market.
            When ``ltv > lltv`` the position is liquidatable.
            Typical Morpho PT markets: 0.86 or 0.915.
        liquidation_penalty: Bonus paid to liquidators on seized collateral.
            Typical Morpho: 0.05–0.075. Used in Session 5 for realistic
            liquidation modeling (seizure at penalty); Session 2 only
            flags the position.
    """

    market_id: str = "0x" + "00" * 32
    lltv: float = 0.86
    liquidation_penalty: float = 0.05


class MorphoEntity(BaseLendingEntity):
    """Morpho isolated market: PT-collateral → USDC-debt.

    Notional unit: USDC. The entity holds collateral on the protocol
    side and tracks accrued debt; the user-visible "equity" is
    ``collateral_value - debt`` in USDC.
    """

    _internal_state: MorphoInternalState
    _global_state: MorphoGlobalState

    def __init__(self, config: MorphoConfig | None = None) -> None:
        self._config = config or MorphoConfig()
        super().__init__()

    def _initialize_states(self) -> None:
        self._global_state = MorphoGlobalState()
        self._internal_state = MorphoInternalState()

    def update_state(self, state: MorphoGlobalState) -> None:
        """Apply new market context and accrue debt.

        Flow:
            1. On the first call (``last_timestamp is None``) we cannot
               compute ``dt``, so we simply store the new state and the
               timestamp — no accrual happens yet.
            2. On subsequent calls we accrue debt by
               ``debt *= 1 + borrowing_rate * dt_years`` using the
               *previously stored* borrowing rate-equivalent — here we
               use ``state.borrowing_rate`` directly because the IRM
               provides a forward-looking rate for the next interval at
               the start of that interval. (For backtest granularity
               1h/1d this is indistinguishable from end-of-step accrual.)
            3. After accrual, if ``ltv > lltv`` we flag the position as
               liquidated. The flag latches: subsequent actions raise.
        """
        if self._internal_state.last_timestamp is not None:
            dt_years = (
                state.timestamp_seconds - self._internal_state.last_timestamp
            ) / SECONDS_PER_YEAR
            self._internal_state.debt *= 1.0 + state.borrowing_rate * dt_years
        self._internal_state.last_timestamp = state.timestamp_seconds
        self._global_state = state
        if self.ltv > self._config.lltv:
            self._internal_state.is_liquidated = True

    # ------------------------------------------------------------------
    # Derived quantities. Available immediately (no Session 2 dependency).
    # ------------------------------------------------------------------

    @property
    def collateral_value(self) -> float:
        """Collateral mark-to-market in USDC."""
        return (
            self._internal_state.collateral * self._global_state.collateral_price
        )

    @property
    def debt_value(self) -> float:
        """Debt mark-to-market in USDC. For USDC debt this equals ``debt``."""
        return self._internal_state.debt * self._global_state.debt_price

    @property
    def ltv(self) -> float:
        """Current loan-to-value ratio.

        Returns 0.0 when collateral is zero (debt should also be zero by
        invariant; if not, the position is already insolvent — Session 2
        will treat that case as ``ltv = +inf`` via a separate check).
        """
        cv = self.collateral_value
        if cv <= 0:
            return 0.0
        return self.debt_value / cv

    @property
    def health_factor(self) -> float:
        """Distance to liquidation. >1 = safe, ≤1 = liquidatable.

        Defined as ``lltv / ltv`` to match the standard DeFi convention
        (Aave health factor formula, modulo small differences in how
        each protocol expresses the liquidation threshold).
        """
        cur_ltv = self.ltv
        if cur_ltv == 0:
            return float("inf")
        return self._config.lltv / cur_ltv

    @property
    def balance(self) -> float:
        """Equity in USDC = collateral_value - debt_value."""
        return self.collateral_value - self.debt_value

    @property
    def is_liquidated(self) -> bool:
        """Whether the position has been flagged as liquidatable.

        Set by :meth:`update_state` whenever the post-update LTV exceeds
        the market LLTV. Latching: once true, no action can clear it.
        """
        return self._internal_state.is_liquidated

    # ------------------------------------------------------------------
    # Action methods. All reject when ``is_liquidated`` is set.
    # ``withdraw`` and ``borrow`` additionally enforce post-mutation LLTV.
    # ------------------------------------------------------------------

    def _require_active(self, action_name: str) -> None:
        """Raise if the position has already been liquidated."""
        if self._internal_state.is_liquidated:
            raise EntityException(
                "entity is liquidated, no further actions allowed "
                f"(rejected {action_name})"
            )

    def action_deposit(self, amount_in_notional: float) -> None:
        """Add PT collateral (notional amount is in PT face units).

        Always reduces LTV (or leaves it unchanged at zero debt), so no
        LLTV check is needed.
        """
        self._require_active("action_deposit")
        if amount_in_notional < 0:
            raise EntityException(
                f"action_deposit: amount must be non-negative, got {amount_in_notional}"
            )
        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Remove PT collateral (in face units).

        After the mutation we verify ``ltv <= lltv``. If the post-state
        is unsafe the collateral is restored and the call raises — the
        entity never holds half-applied state.
        """
        self._require_active("action_withdraw")
        if amount_in_notional < 0:
            raise EntityException(
                f"action_withdraw: amount must be non-negative, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.collateral:
            raise EntityException(
                f"action_withdraw: requested {amount_in_notional} but only "
                f"{self._internal_state.collateral} collateral held"
            )
        self._internal_state.collateral -= amount_in_notional
        if self.ltv > self._config.lltv:
            self._internal_state.collateral += amount_in_notional
            raise EntityException(
                f"action_withdraw: would push ltv={self.ltv:.6f} above "
                f"lltv={self._config.lltv:.6f}"
            )

    def action_borrow(self, amount_in_notional: float) -> None:
        """Draw USDC against collateral.

        After the mutation we verify ``ltv <= lltv``. If the post-state
        is unsafe the debt is rolled back and the call raises.
        """
        self._require_active("action_borrow")
        if amount_in_notional < 0:
            raise EntityException(
                f"action_borrow: amount must be non-negative, got {amount_in_notional}"
            )
        self._internal_state.debt += amount_in_notional
        if self.ltv > self._config.lltv:
            self._internal_state.debt -= amount_in_notional
            raise EntityException(
                f"action_borrow: would push ltv={self.ltv:.6f} above "
                f"lltv={self._config.lltv:.6f}"
            )

    def action_repay(self, amount_in_notional: float) -> None:
        """Pay back USDC debt.

        Always reduces LTV (or leaves it unchanged), so no LLTV check
        is needed.
        """
        self._require_active("action_repay")
        if amount_in_notional < 0:
            raise EntityException(
                f"action_repay: amount must be non-negative, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.debt:
            raise EntityException(
                f"action_repay: requested {amount_in_notional} but only "
                f"{self._internal_state.debt} debt outstanding"
            )
        self._internal_state.debt -= amount_in_notional
