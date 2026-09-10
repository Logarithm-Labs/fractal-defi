"""Pendle V2 Principal Token position.

A PT is a zero-coupon claim on **one unit of the market's accounting
asset** (USDe for sUSDe markets, stETH for wstETH markets, …) at
expiry. Before expiry it trades at a discount set by the market's
implied APY; after expiry it earns nothing and redeems 1:1 (paid in SY
at the current index, so a falling SY exchange rate is borne by the PT
holder — Pendle's ``pyIndex`` ratchet).

Numeraire: ``cash`` is the strategy's notional (USD by default, like
every entity). PT is priced in the accounting asset and bridged to
notional by ``asset_price`` (notional per accounting asset, ``1.0`` for
stable markets). Conversions notional↔asset inside the actions are
fee-free; front a spot entity if the quote asset differs from the
accounting asset.

Swap cost comes from :mod:`fractal.core.entities.models.pendle_math`:
``impact_model="amm"`` (default) replays ``MarketMathCore`` from the
pool reserves carried on the global state; ``"rate_spread"`` applies a
fee and a size-impact as spreads on the implied rate. Own trades do
not persist in the pool — the next observation reloads it from data.
"""
from dataclasses import dataclass
from typing import Literal

from fractal.core.base.entity import EntityException
from fractal.core.entities.base.fixed_term import BaseFixedTermEntity, BaseFixedTermGlobalState
from fractal.core.entities.base.spot import BaseSpotEntity, BaseSpotInternalState
from fractal.core.entities.models.pendle_math import (
    MAX_MARKET_PROPORTION,
    amm_swap_exact_asset_in,
    amm_swap_exact_pt,
    pt_price_from_apy,
    rate_spread_buy,
    rate_spread_sell,
)

_IMPACT_MODELS = ("amm", "rate_spread")


class PendlePTException(EntityException):
    """Raised on invalid PT trades, states or configuration."""


@dataclass
class PendlePTGlobalState(BaseFixedTermGlobalState):
    """Market snapshot for one Pendle PT market.

    Attributes:
        seconds_to_expiry (float): ``expiry - now`` in seconds (inherited;
            ``<= 0`` = matured).
        implied_apy (float): market implied APY, compounded ACT/365,
            decimal (``0.049`` = 4.9%). The PT price is derived from it.
        asset_price (float): notional per accounting asset (``1.0`` for
            stable markets). Underlying depegs enter here.
        sy_exchange_rate (float): accounting asset per SY
            (``SY.exchangeRate``); converts ``total_sy`` to asset terms
            and feeds the redemption ratchet.
        total_pt (float): pool PT reserve (AMM replay only).
        total_sy (float): pool SY reserve (AMM replay only).
        scalar_root (float): market immutable ``scalarRoot`` (AMM replay only).
        ln_fee_rate_root (float): market ``lnFeeRateRoot`` (AMM replay only).
    """
    implied_apy: float = 0.0
    asset_price: float = 1.0
    sy_exchange_rate: float = 1.0
    total_pt: float = 0.0
    total_sy: float = 0.0
    scalar_root: float = 0.0
    ln_fee_rate_root: float = 0.0


@dataclass
class PendlePTInternalState(BaseSpotInternalState):
    """Position state.

    Attributes:
        amount (float): PT held (face units; ``1`` redeems to one accounting asset).
        cash (float): free notional.
        py_index_high (float): highest ``sy_exchange_rate`` observed —
            Pendle's ``pyIndex`` ratchet; ``0.0`` until the first state.
    """
    py_index_high: float = 0.0


@dataclass
class PendlePTConfig:
    """Swap-cost configuration.

    Attributes:
        impact_model (str): ``"amm"`` (replay ``MarketMathCore`` from the
            pool state) or ``"rate_spread"`` (fee + impact as spreads on
            the implied rate).
        fee_ln_rate (float): ``rate_spread`` only — fee as a spread on
            ``ln(1 + apy)``; cost ≈ ``fee_ln_rate * years_to_expiry`` of
            notional. Live Pendle markets: 5e-4 … 1e-3 through the router.
        impact_ln_rate_per_share (float): ``rate_spread`` only — rate
            impact per unit of ``trade / pool_asset`` (0.075 ≈ 10% of the
            pool moves the implied rate by 0.75 pp).
        max_pool_share (float): ``amm`` only — post-trade PT share cap
            (``0.96`` on-chain).
    """
    impact_model: Literal["amm", "rate_spread"] = "amm"
    fee_ln_rate: float = 0.001
    impact_ln_rate_per_share: float = 0.075
    max_pool_share: float = MAX_MARKET_PROPORTION


class PendlePTEntity(BaseFixedTermEntity, BaseSpotEntity):
    """Pendle PT as a spot product with a maturity.

    Deriving from :class:`BaseSpotEntity` keeps the leverage-loop
    primitives (``action_inject_product`` / ``action_remove_product``)
    and the ``buy(amount_in_notional)`` / ``sell(amount_in_product)``
    sizing convention every other spot entity uses.
    """
    _exception_cls = PendlePTException
    _internal_state: PendlePTInternalState
    _global_state: PendlePTGlobalState

    def __init__(self, config: PendlePTConfig = None) -> None:
        config = config or PendlePTConfig()
        if config.impact_model not in _IMPACT_MODELS:
            raise PendlePTException(
                f"impact_model must be one of {_IMPACT_MODELS}, got {config.impact_model!r}"
            )
        if config.fee_ln_rate < 0:
            raise PendlePTException(f"fee_ln_rate must be >= 0, got {config.fee_ln_rate}")
        if config.impact_ln_rate_per_share < 0:
            raise PendlePTException(
                f"impact_ln_rate_per_share must be >= 0, got {config.impact_ln_rate_per_share}"
            )
        if not 0 < config.max_pool_share <= 1:
            raise PendlePTException(f"max_pool_share must be in (0, 1], got {config.max_pool_share}")
        # Set config BEFORE super so subclass overrides of
        # ``_initialize_states`` can rely on these.
        self.impact_model: str = config.impact_model
        self.fee_ln_rate: float = config.fee_ln_rate
        self.impact_ln_rate_per_share: float = config.impact_ln_rate_per_share
        self.max_pool_share: float = config.max_pool_share
        super().__init__()

    def _initialize_states(self) -> None:
        self._internal_state = PendlePTInternalState()
        self._global_state = PendlePTGlobalState()

    @property
    def internal_state(self) -> PendlePTInternalState:  # type: ignore[override]
        return self._internal_state

    # ------------------------------------------------------------ readouts
    @property
    def implied_apy(self) -> float:
        """Market implied APY from the last applied state."""
        return self._global_state.implied_apy

    @property
    def pt_price_asset(self) -> float:
        """Accounting asset per PT from the implied APY; ``1.0`` once matured."""
        return pt_price_from_apy(self._global_state.implied_apy, self.years_to_expiry)

    @property
    def redeem_haircut(self) -> float:
        """``min(1, sy_exchange_rate / py_index_high)`` — Pendle's ``syIndex / pyIndex`` haircut."""
        high = self._internal_state.py_index_high
        if high <= 0.0:
            return 1.0
        return min(1.0, self._global_state.sy_exchange_rate / high)

    @property
    def current_price(self) -> float:
        """Notional per PT: ``pt_price_asset * redeem_haircut * asset_price``."""
        return self.pt_price_asset * self.redeem_haircut * self._global_state.asset_price

    @property
    def balance(self) -> float:
        """``cash + amount * current_price`` — continuous across expiry."""
        return self._internal_state.cash + self._internal_state.amount * self.current_price

    # ------------------------------------------------------------ actions
    def action_deposit(self, amount_in_notional: float) -> None:
        """Add notional cash."""
        if amount_in_notional < 0:
            raise PendlePTException(f"deposit amount must be >= 0, got {amount_in_notional}")
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Remove notional cash."""
        if amount_in_notional < 0:
            raise PendlePTException(f"withdraw amount must be >= 0, got {amount_in_notional}")
        if amount_in_notional > self._internal_state.cash:
            raise PendlePTException(
                f"withdraw exceeds cash: {amount_in_notional} > {self._internal_state.cash}"
            )
        self._internal_state.cash -= amount_in_notional

    def action_buy(self, amount_in_notional: float) -> None:
        """Spend ``amount_in_notional`` on PT through the market (fee + impact per ``impact_model``)."""
        self._require_not_matured("buy")
        if amount_in_notional < 0:
            raise PendlePTException(f"buy amount must be >= 0, got {amount_in_notional}")
        if amount_in_notional > self._internal_state.cash:
            raise PendlePTException(
                f"buy exceeds cash: {amount_in_notional} > {self._internal_state.cash}"
            )
        if amount_in_notional == 0:
            return
        asset_in = amount_in_notional / self._global_state.asset_price
        pt_out = self._quote_buy(asset_in)
        self._internal_state.cash -= amount_in_notional
        self._internal_state.amount += pt_out

    def action_sell(self, amount_in_product: float) -> None:
        """Sell ``amount_in_product`` PT through the market for notional cash."""
        self._require_not_matured("sell")
        if amount_in_product < 0:
            raise PendlePTException(f"sell amount must be >= 0, got {amount_in_product}")
        if amount_in_product > self._internal_state.amount:
            raise PendlePTException(
                f"sell exceeds holding: {amount_in_product} > {self._internal_state.amount}"
            )
        if amount_in_product == 0:
            return
        asset_out = self._quote_sell(amount_in_product)
        self._internal_state.amount -= amount_in_product
        self._internal_state.cash += asset_out * self._global_state.asset_price

    def action_redeem(self, amount_in_product: float) -> None:
        """Redeem matured PT: ``amount * redeem_haircut * asset_price`` of cash, no fee."""
        if amount_in_product < 0:
            raise PendlePTException(f"redeem amount must be >= 0, got {amount_in_product}")
        if not self.is_matured:
            raise PendlePTException(
                f"redeem is only available after expiry (seconds_to_expiry={self.seconds_to_expiry})"
            )
        if amount_in_product > self._internal_state.amount:
            raise PendlePTException(
                f"redeem exceeds holding: {amount_in_product} > {self._internal_state.amount}"
            )
        self._internal_state.amount -= amount_in_product
        self._internal_state.cash += amount_in_product * self.redeem_haircut * self._global_state.asset_price

    # ------------------------------------------------------------- quotes
    def _amm_kwargs(self) -> dict:
        state = self._global_state
        if state.total_pt <= 0 or state.total_sy <= 0 or state.scalar_root <= 0:
            raise PendlePTException(
                "impact_model='amm' needs total_pt, total_sy and scalar_root > 0 on the "
                "global state; load them from the market history or use 'rate_spread'"
            )
        return {
            "total_pt": state.total_pt,
            "total_asset": state.total_sy * state.sy_exchange_rate,
            "scalar_root": state.scalar_root,
            "ln_fee_rate_root": state.ln_fee_rate_root,
            "implied_apy": state.implied_apy,
            "years": self.years_to_expiry,
        }

    def _quote_buy(self, asset_in: float) -> float:
        state = self._global_state
        try:
            if self.impact_model == "amm":
                quote = amm_swap_exact_asset_in(asset_in, **self._amm_kwargs())
                return quote.net_pt_to_account
            return rate_spread_buy(
                asset_in, state.implied_apy, self.years_to_expiry,
                self.fee_ln_rate, self.impact_ln_rate_per_share,
                state.total_sy * state.sy_exchange_rate,
            )
        except ValueError as exc:
            raise PendlePTException(f"buy not executable: {exc}") from exc

    def _quote_sell(self, pt_in: float) -> float:
        state = self._global_state
        try:
            if self.impact_model == "amm":
                quote = amm_swap_exact_pt(-pt_in, **self._amm_kwargs())
                return quote.net_asset_to_account
            return rate_spread_sell(
                pt_in, state.implied_apy, self.years_to_expiry,
                self.fee_ln_rate, self.impact_ln_rate_per_share,
                state.total_sy * state.sy_exchange_rate,
            )
        except ValueError as exc:
            raise PendlePTException(f"sell not executable: {exc}") from exc

    # ------------------------------------------------------- state update
    def update_state(self, state: PendlePTGlobalState) -> None:
        """Validate and apply a snapshot; never mutates ``state``.

        Order: term validation → field validation → apply → ratchet the
        redemption index → maturity hook (no-op: Pendle has no auto-redeem).
        """
        self._validate_term(state)
        if state.implied_apy <= -1:
            raise PendlePTException(f"implied_apy must be > -1, got {state.implied_apy}")
        if state.asset_price <= 0:
            raise PendlePTException(f"asset_price must be > 0, got {state.asset_price}")
        if state.sy_exchange_rate <= 0:
            raise PendlePTException(f"sy_exchange_rate must be > 0, got {state.sy_exchange_rate}")
        for name in ("total_pt", "total_sy", "scalar_root", "ln_fee_rate_root"):
            if getattr(state, name) < 0:
                raise PendlePTException(f"{name} must be >= 0, got {getattr(state, name)}")
        self._global_state = state
        self._internal_state.py_index_high = max(self._internal_state.py_index_high, state.sy_exchange_rate)
        self._check_maturity()
