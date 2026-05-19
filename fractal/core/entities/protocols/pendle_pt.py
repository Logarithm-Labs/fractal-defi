import math
from dataclasses import dataclass
from typing import Literal, Optional

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
            to convert PT face → USDC at expiry. Defaults to 1.0
            (SY = stablecoin at peg). When the underlying depegs, this
            captures the realised loss on the SY-side at expiry. Also
            consulted by :meth:`PendlePTEntity.balance` after expiry to
            mark PT face at the prevailing SY/USDC quote, so the NAV
            curve reflects the depeg before redemption is called.
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
    ``N * pt_price`` USDC and will redeem for ``N * sy_price_in_usdc``
    USDC at expiry (``sy_price_in_usdc`` defaults to 1.0 = no depeg).
    """

    _internal_state: PendlePTInternalState
    _global_state: PendlePTGlobalState

    def __init__(self, config: Optional[PendlePTConfig] = None) -> None:
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
           PT face-value is constant in SY units after expiry. The
           USDC-side mark uses ``sy_price_in_usdc`` from :attr:`balance`.
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
        """Mark-to-market equity of the entity in USDC.

        Pre-expiry: ``cash + pt_face_amount * pt_price``. The PT mark
        already embeds the implied yield, so SY-side depeg is implicit
        in the AMM quote and we don't multiply by ``sy_price_in_usdc``.

        Post-expiry: ``cash + pt_face_amount * sy_price_in_usdc``. PT
        face is locked to 1 SY per face, so the USDC mark IS the SY/USDC
        quote — the depeg flows straight into NAV even before
        ``action_redeem`` is called.
        """
        cash = self._internal_state.cash
        face = self._internal_state.pt_face_amount
        if self._global_state.seconds_to_expiry <= 0:
            return cash + face * self._global_state.sy_price_in_usdc
        return cash + face * self._global_state.pt_price

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
        which gives the "stablecoin always at peg" behaviour). When
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
