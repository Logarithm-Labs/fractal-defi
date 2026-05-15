"""Pendle Principal Token (PT) entity — Session 2 math.

A PT is a transferable claim on one unit of the underlying SY
(standardized yield token, e.g. sUSDe) redeemable 1:1 at expiry.
Before expiry, PT trades at a discount on Pendle's bespoke AMM; the
discount is the *implied yield* the market is currently pricing.

This entity tracks one PT position on a single (market, expiry) pair.

* ``GlobalState`` — current PT mark price, implied yield, seconds-to-expiry.
* ``InternalState`` — PT face amount held, USDC cash leftover.
* Actions:
  - ``action_deposit`` — accept USDC into cash.
  - ``action_buy_pt`` — swap USDC -> PT through the Pendle AMM.
  - ``action_sell_pt`` — swap PT -> USDC (used for early exit / unwind).
  - ``action_redeem`` — redeem PT 1:1 for the underlying at/after expiry.
  - ``action_withdraw`` — return USDC cash to the caller.

Session 2 modelling choices
---------------------------
PT pricing.
    Pendle's actual AMM uses a logarithmic SY/PT curve parameterised by
    a scalar root and an anchor implied rate; replicating that in a
    research backtest gives us false precision (we lack the per-block
    pool reserves). Instead we treat ``pt_price`` as a function of the
    quoted implied yield and time-to-expiry. Two parametric forms are
    exposed via ``pricing_mode``:

    * ``"linear"`` (default) — :math:`P_{PT} = 1 - r_{impl}\\,\\tau`,
      clamped to :math:`[0, 1]`. Cheap, monotone, exact at expiry.
    * ``"exponential"`` — :math:`P_{PT} = e^{-r_{impl}\\,\\tau}`,
      continuous-compounding analogue. Closer to a zero-coupon bond
      and what Pendle's own UI displays for "fixed yield".

    For first-order analysis the two agree to within
    :math:`O((r\\tau)^2)` (Taylor): a 14% APY one-year PT differs by
    ~1.0 cent between the two forms, well below sampling noise.

Real-data path (``derive_pt_price=False``, default).
    When backtesting against historical Pendle quotes the caller passes
    the observed ``pt_price`` straight into ``update_state``; we trust
    that price and only cache ``implied_yield`` for reporting. This is
    the production path.

Stress / scenario path (``derive_pt_price=True``).
    When projecting forward without quote data (e.g. "what if APY drops
    to 8% by week 3?") we let the entity recompute ``pt_price`` from
    the passed implied yield and time-to-expiry under ``pricing_mode``.

AMM swaps.
    Pendle pools quote a price plus per-pool fee and incur slippage in
    the size of the swap relative to the pool. We model this as:

    .. math::

        \\mathit{slip} = \\sigma \\cdot \\frac{\\text{trade size}}
                                              {\\text{pool liquidity}}

    where :math:`\\sigma` is :attr:`PendlePTConfig.slippage_factor`.
    A swap consuming the full pool incurs :math:`\\sigma` (50% by
    default) of mid-price impact — a deliberately conservative sanity
    scaling, since real Pendle slippage is concave but we only need
    monotone-in-size for backtest realism. Fees are charged on the
    notional input side and slippage moves the effective execution
    price against the trader.

Storage / inheritance note.
    We use ``fractal.core.base.entity.BaseEntity`` directly because no
    existing fractal-defi base captures "fixed-yield discount bond" —
    the existing spot / LP / lending / perp bases assume different
    mechanics. If the YT entity (variant 3) later materialises with
    enough overlap we'll extract a ``BaseDiscountBondEntity``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from fractal.core.base.entity import (
    BaseEntity,
    EntityException,
    GlobalState,
    InternalState,
)


SECONDS_PER_YEAR: float = 365.25 * 24 * 3600


def compute_pt_price(
    implied_yield: float,
    seconds_to_expiry: float,
    mode: str = "linear",
) -> float:
    """Closed-form PT price from implied yield and time-to-expiry.

    Args:
        implied_yield: Annualised fixed yield in decimal (0.14 = 14% APY).
        seconds_to_expiry: Wall-clock seconds remaining until redeem.
        mode: ``"linear"`` for :math:`1 - r\\,\\tau` (clamped to
            :math:`[0, 1]`) or ``"exponential"`` for :math:`e^{-r\\,\\tau}`.

    Returns:
        PT mid-price in USDC. Always 1.0 at/after expiry regardless
        of ``mode`` or ``implied_yield``.

    Raises:
        EntityException: If ``mode`` is not one of the supported forms.
    """
    if seconds_to_expiry <= 0:
        return 1.0
    tau = seconds_to_expiry / SECONDS_PER_YEAR
    if mode == "linear":
        price = 1.0 - implied_yield * tau
        # Clamp: negative implied yields (premium) or absurdly long tau
        # could push outside [0, 1]; the bond can't.
        if price < 0.0:
            return 0.0
        if price > 1.0:
            return 1.0
        return price
    if mode == "exponential":
        return math.exp(-implied_yield * tau)
    raise EntityException(
        f"compute_pt_price: unsupported mode {mode!r}; "
        "expected 'linear' or 'exponential'"
    )


@dataclass
class PendlePTGlobalState(GlobalState):
    """Market context for a single PT/SY market on a given epoch.

    Attributes:
        pt_price: Current market price of 1 PT in USDC.
            Pre-expiry: in :math:`(0, 1]`, monotonically approaching 1.
            At/after expiry: exactly 1 (constant) in *SY units*; the
            redeemed USDC value still depends on ``sy_price_in_usdc``.
        implied_yield: Annualised fixed yield baked into ``pt_price``
            at the moment of observation, in decimal (0.14 = 14% APY).
            Convenience only — derivable from ``pt_price`` and
            ``seconds_to_expiry``; we cache it to avoid recomputing on
            every step.
        seconds_to_expiry: Wall-clock seconds remaining until PT redeem
            unlocks. Drops to 0 at expiry, then stays 0.
        pool_liquidity: Total liquidity in the PT/SY Pendle pool, in
            USDC equivalent. Used to estimate slippage on swaps.
        sy_price_in_usdc: Price of 1 unit of the underlying SY token
            (sUSDe, weETH, USDe, …) in USDC. Used by ``action_redeem``
            to convert PT face → USDC at expiry. Defaults to 1.0 so
            existing tests behave identically (SY = stablecoin at peg).
            When the underlying depegs, this captures the realised loss
            on the SY-side at expiry — addresses the linear-pricing
            limitation flagged in the original whitepaper.
    """

    pt_price: float = 1.0
    implied_yield: float = 0.0
    seconds_to_expiry: float = 0.0
    pool_liquidity: float = 0.0
    sy_price_in_usdc: float = 1.0


@dataclass
class PendlePTInternalState(InternalState):
    """Position state inside the PT entity.

    Attributes:
        pt_face_amount: Quantity of PT held, in face units
            (1 face = 1 USDC at expiry).
        cash: Free USDC sitting in the entity (not yet deployed to PT).
    """

    pt_face_amount: float = 0.0
    cash: float = 0.0


@dataclass
class PendlePTConfig:
    """Configuration for a Pendle PT entity.

    Attributes:
        market_address: The 20-byte address of the Pendle market.
            Informational at backtest level; used by loaders to scope
            historical data.
        amm_fee_rate: Fee charged by Pendle AMM on PT/SY swaps.
            Default 0.001 (10 basis points) — Pendle's typical pool fee.
        slippage_factor: Dimensionless scaling of size-impact: a swap
            equal to the full pool liquidity moves the effective price
            by ``slippage_factor`` (default 0.5 = 50%). Lower for deep
            pools, higher for thin ones.
        pricing_mode: Parametric form for ``compute_pt_price`` — either
            ``"linear"`` or ``"exponential"``. Only consulted when
            ``derive_pt_price`` is true.
        derive_pt_price: If false (default), ``update_state`` accepts
            the caller's ``pt_price`` as ground truth — the real-data
            path. If true, ``update_state`` overwrites ``pt_price``
            with ``compute_pt_price`` for forward-projection scenarios.
    """

    market_address: str = "0x0000000000000000000000000000000000000000"
    amm_fee_rate: float = 0.001
    slippage_factor: float = 0.5
    pricing_mode: Literal["linear", "exponential"] = "linear"
    derive_pt_price: bool = False


class PendlePTEntity(BaseEntity[PendlePTGlobalState, PendlePTInternalState]):
    """Pendle Principal Token position.

    Notional unit: USDC. A PT with face amount ``N`` is currently worth
    ``N * pt_price`` USDC and will redeem for ``N`` USDC at expiry.
    """

    _internal_state: PendlePTInternalState
    _global_state: PendlePTGlobalState

    def __init__(self, config: PendlePTConfig | None = None) -> None:
        self._config = config or PendlePTConfig()
        super().__init__()

    def _initialize_states(self) -> None:
        self._global_state = PendlePTGlobalState()
        self._internal_state = PendlePTInternalState()

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    def update_state(self, state: PendlePTGlobalState) -> None:
        """Apply new market context.

        Three branches:

        1. ``seconds_to_expiry <= 0`` — snap ``pt_price`` to 1.0 and
           ``seconds_to_expiry`` to 0.0 regardless of caller input.
           PT face-value is constant after expiry by construction.
        2. ``derive_pt_price=True`` — overwrite ``pt_price`` with the
           closed-form from :func:`compute_pt_price`. Used in scenario
           projection.
        3. Otherwise — accept ``pt_price`` as passed (real-data path).
        """
        if state.seconds_to_expiry <= 0:
            state.pt_price = 1.0
            state.seconds_to_expiry = 0.0
        elif self._config.derive_pt_price:
            state.pt_price = compute_pt_price(
                state.implied_yield,
                state.seconds_to_expiry,
                self._config.pricing_mode,
            )
        self._global_state = state

    @property
    def balance(self) -> float:
        """Mark-to-market equity of the entity in USDC."""
        return (
            self._internal_state.cash
            + self._internal_state.pt_face_amount * self._global_state.pt_price
        )

    # ------------------------------------------------------------------
    # Action methods
    # ------------------------------------------------------------------

    def action_deposit(self, amount_in_notional: float) -> None:
        """Add USDC cash to the entity."""
        if amount_in_notional < 0:
            raise EntityException(
                f"action_deposit: amount must be non-negative, got {amount_in_notional}"
            )
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Remove USDC cash from the entity."""
        if amount_in_notional < 0:
            raise EntityException(
                f"action_withdraw: amount must be non-negative, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise EntityException(
                f"action_withdraw: requested {amount_in_notional} but only "
                f"{self._internal_state.cash} cash available"
            )
        self._internal_state.cash -= amount_in_notional

    def action_buy_pt(self, amount_in_notional: float) -> None:
        """Swap ``amount_in_notional`` USDC for PT through the Pendle AMM.

        Effective price logic:

        * ``effective_in = amount * (1 - amm_fee_rate)`` — fee taken on
          input USDC.
        * ``slip = slippage_factor * amount / pool_liquidity`` —
          size-impact; positive on a buy (price walks up).
        * ``effective_price = pt_price * (1 + slip)``.
        * ``pt_received = effective_in / effective_price``.
        """
        if amount_in_notional < 0:
            raise EntityException(
                f"action_buy_pt: amount must be non-negative, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise EntityException(
                f"action_buy_pt: requested {amount_in_notional} but only "
                f"{self._internal_state.cash} cash available"
            )
        pt_price = self._global_state.pt_price
        if pt_price <= 0:
            raise EntityException(f"action_buy_pt: invalid pt_price {pt_price}")
        pool_liquidity = self._global_state.pool_liquidity
        if pool_liquidity <= 0:
            raise EntityException(
                f"action_buy_pt: pool_liquidity must be positive, got {pool_liquidity}"
            )

        effective_in = amount_in_notional * (1.0 - self._config.amm_fee_rate)
        slip = self._config.slippage_factor * amount_in_notional / pool_liquidity
        effective_price = pt_price * (1.0 + slip)
        pt_received = effective_in / effective_price

        self._internal_state.cash -= amount_in_notional
        self._internal_state.pt_face_amount += pt_received

    def action_sell_pt(self, amount_in_face: float) -> None:
        """Swap ``amount_in_face`` PT (face units) for USDC.

        Symmetric to :meth:`action_buy_pt`: slippage now drags the
        effective price *down* (you sell into the book), and the fee
        is taken on the USDC leg coming back.
        """
        if amount_in_face < 0:
            raise EntityException(
                f"action_sell_pt: amount must be non-negative, got {amount_in_face}"
            )
        if amount_in_face > self._internal_state.pt_face_amount:
            raise EntityException(
                f"action_sell_pt: requested {amount_in_face} but only "
                f"{self._internal_state.pt_face_amount} PT held"
            )
        pt_price = self._global_state.pt_price
        if pt_price <= 0:
            raise EntityException(f"action_sell_pt: invalid pt_price {pt_price}")
        pool_liquidity = self._global_state.pool_liquidity
        if pool_liquidity <= 0:
            raise EntityException(
                f"action_sell_pt: pool_liquidity must be positive, got {pool_liquidity}"
            )

        slip = self._config.slippage_factor * amount_in_face / pool_liquidity
        effective_price = pt_price * (1.0 - slip)
        if effective_price < 0:
            effective_price = 0.0
        gross_usdc = amount_in_face * effective_price
        usdc_received = gross_usdc * (1.0 - self._config.amm_fee_rate)

        self._internal_state.pt_face_amount -= amount_in_face
        self._internal_state.cash += usdc_received

    def action_redeem(self, amount_in_face: float) -> None:
        """Redeem PT for the underlying SY at/after expiry.

        Requires ``seconds_to_expiry == 0`` (enforced by
        :meth:`update_state`). PT redeems 1:1 for SY at expiry; the
        SY → USDC conversion happens at the prevailing SY price
        carried on ``global_state.sy_price_in_usdc`` (defaults to 1.0,
        which gives the old "stablecoin always at peg" behaviour). When
        the underlying SY has depegged (e.g. USDe in December 2025 or
        eETH in April 2024), redeem realises that loss correctly.
        """
        if amount_in_face < 0:
            raise EntityException(
                f"action_redeem: amount must be non-negative, got {amount_in_face}"
            )
        if self._global_state.seconds_to_expiry > 0:
            raise EntityException(
                f"action_redeem: PT not at expiry yet "
                f"(seconds_to_expiry={self._global_state.seconds_to_expiry})"
            )
        if amount_in_face > self._internal_state.pt_face_amount:
            raise EntityException(
                f"action_redeem: requested {amount_in_face} but only "
                f"{self._internal_state.pt_face_amount} PT held"
            )

        # Redeem PT → SY 1:1 (Pendle invariant), then SY → USDC at the
        # prevailing SY market price.
        sy_price = self._global_state.sy_price_in_usdc
        self._internal_state.pt_face_amount -= amount_in_face
        self._internal_state.cash += amount_in_face * sy_price
