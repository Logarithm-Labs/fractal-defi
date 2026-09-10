"""Pendle Boros yield-unit position (fixed-vs-floating funding swap).

One yield unit (YU) is the funding on one coin of the underlying on a
named venue (``YU-ETHUSDT-Binance``). Long = pay fixed / receive the
venue's floating funding; short = receive fixed / pay floating — the
leg a cash-and-carry desk sells to lock its funding income.

The entity is perp-shaped (:class:`BasePerpEntity`: signed ``size``,
collateral, IM/MM, netting) plus a maturity
(:class:`BaseFixedTermEntity`). What differs from a price perp:

* ``size`` is in **coins**; the "price" is an annualised rate.
* Cash flows happen at the venue's funding cadence: the observation
  builder puts the raw per-period ``funding_rate`` and its
  ``funding_period_seconds`` on the bars where a settlement lands
  (``0`` elsewhere ⇒ no settlement that bar).
* Value between settlements is ``N·(mark − entry)·TTM``; margin uses
  rate and time floors, so it never decays to zero near maturity.
* At maturity the position is worth exactly zero and is closed.
* Collateral is the market's base coin (``coin_margined=True``) or a
  stable (USDT0 markets); the entity keeps ``collateral`` in that
  margin asset and reports ``balance`` in notional via ``underlying_price``.

Funding sign follows the library's perps: ``funding_rate > 0`` ⇒ perp
longs pay ⇒ a **long** YU receives ``N·f`` — a short perp hedged with
a long YU of the same size has no net funding exposure.
"""
import math
from dataclasses import dataclass

from fractal.core.base.entity import EntityException
from fractal.core.base.time import SECONDS_PER_DAY, SECONDS_PER_YEAR
from fractal.core.entities.base.fixed_term import BaseFixedTermEntity, BaseFixedTermGlobalState
from fractal.core.entities.base.perp import BasePerpEntity, BasePerpInternalState
from fractal.core.entities.models.boros_math import (
    LIQUIDATION_PROTOCOL_FEE_RATE,
    liquidation_penalty_fraction,
    margin_coin,
    mark_to_maturity_coin,
    open_fee_coin,
    settlement_pnl_coin,
)


class BorosException(EntityException):
    """Errors raised by :class:`BorosEntity`."""


@dataclass
class BorosGlobalState(BaseFixedTermGlobalState):
    """Market state.

    Attributes:
        seconds_to_expiry (float): time to the YU maturity (inherited).
        mark_rate (float): annualised mark implied APR (TWAP of traded rates).
        funding_rate (float): the venue's **raw per-period** funding landing
            on this bar (positive = perp longs pay); ignored when
            ``funding_period_seconds == 0``.
        funding_period_seconds (float): the settlement interval that
            landed on this bar (``28800`` Binance, ``3600`` Hyperliquid);
            ``0`` ⇒ no settlement this bar.
        underlying_price (float): notional per coin (index price).
    """
    mark_rate: float = 0.0
    funding_rate: float = 0.0
    funding_period_seconds: float = 0.0
    underlying_price: float = 1.0


@dataclass
class BorosInternalState(BasePerpInternalState):
    """Position state.

    Inherits ``collateral`` (margin-asset units: coins if coin-margined,
    else notional) and adds:
        size: signed yield units in coins (``> 0`` long = pay fixed).
        entry_rate: size-weighted fixed rate of the open position.
        realized_settlements: cumulative settlement cash flows (margin units).
        liquidation_count: liquidations suffered so far.
    """
    size: float = 0.0
    entry_rate: float = 0.0
    realized_settlements: float = 0.0
    liquidation_count: int = 0


class BorosEntity(BaseFixedTermEntity, BasePerpEntity):
    """Boros fixed-vs-floating yield-unit position with Boros margining."""

    _exception_cls = BorosException
    _EPS = 1e-12

    def __init__(
        self,
        *,
        max_leverage: float = 1.55,
        mm_to_im_ratio: float = 0.5,
        rate_floor: float = 0.06,
        time_threshold_seconds: float = 10 * SECONDS_PER_DAY,
        taker_fee_rate: float = 0.0005,
        settle_fee_rate: float = 0.001,
        coin_margined: bool = False,
        slippage_rate: float = 0.0,
    ) -> None:
        """
        Args:
            max_leverage: ``1 / kIM`` (1.55× on live BTC/ETH markets).
            mm_to_im_ratio: ``kMM / kIM`` (read from the market config; 0.34–0.69 live).
            rate_floor: margin rate floor (6% Binance venues, 8% Hyperliquid).
            time_threshold_seconds: margin time floor ``tThresh`` (10 days on 8h venues).
            taker_fee_rate: annualised taker fee on fills (0.05%); makers pay 0.
            settle_fee_rate: annualised settlement fee (0.1%).
            coin_margined: collateral held in coins of the underlying (else notional).
            slippage_rate: rate spread paid on fills (``0`` = fill at mark).
        """
        if max_leverage <= 0:
            raise BorosException(f"max_leverage must be > 0, got {max_leverage}")
        if not 0 < mm_to_im_ratio <= 1:
            raise BorosException(f"mm_to_im_ratio must be in (0, 1], got {mm_to_im_ratio}")
        for name, value in (
            ("rate_floor", rate_floor), ("time_threshold_seconds", time_threshold_seconds),
            ("taker_fee_rate", taker_fee_rate), ("settle_fee_rate", settle_fee_rate),
            ("slippage_rate", slippage_rate),
        ):
            if value < 0:
                raise BorosException(f"{name} must be >= 0, got {value}")
        # Set config BEFORE super so subclass overrides of
        # ``_initialize_states`` can rely on these.
        self.max_leverage: float = float(max_leverage)
        self.k_im: float = 1.0 / float(max_leverage)
        self.k_mm: float = self.k_im * float(mm_to_im_ratio)
        self.rate_floor: float = float(rate_floor)
        self.time_floor_years: float = float(time_threshold_seconds) / SECONDS_PER_YEAR
        self.taker_fee_rate: float = float(taker_fee_rate)
        self.settle_fee_rate: float = float(settle_fee_rate)
        self.coin_margined: bool = coin_margined
        self.slippage_rate: float = float(slippage_rate)
        super().__init__()

    _internal_state: BorosInternalState
    _global_state: BorosGlobalState

    def _initialize_states(self) -> None:
        self._internal_state = BorosInternalState()
        self._global_state = BorosGlobalState()

    @property
    def internal_state(self) -> BorosInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> BorosGlobalState:  # type: ignore[override]
        return self._global_state

    @property
    def trading_fee(self) -> float:
        """Perp-API alias of ``taker_fee_rate``."""
        return self.taker_fee_rate

    # ------------------------------------------------------------ units
    def _coin_to_margin(self, coins: float) -> float:
        return coins if self.coin_margined else coins * self._global_state.underlying_price

    def _margin_to_notional(self, margin: float) -> float:
        return margin * self._global_state.underlying_price if self.coin_margined else margin

    def _notional_to_margin(self, notional: float) -> float:
        return notional / self._global_state.underlying_price if self.coin_margined else notional

    # ---------------------------------------------------------- account
    def action_deposit(self, amount_in_notional: float) -> None:
        """Add margin (converted to coins when coin-margined)."""
        if amount_in_notional < 0:
            raise BorosException(f"deposit amount must be >= 0, got {amount_in_notional}")
        self._internal_state.collateral += self._notional_to_margin(amount_in_notional)

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Remove margin; the position must stay above initial margin."""
        if amount_in_notional < 0:
            raise BorosException(f"withdraw amount must be >= 0, got {amount_in_notional}")
        if amount_in_notional > self.balance:
            raise BorosException(f"insufficient balance: {self.balance} < {amount_in_notional}")
        if self.balance - amount_in_notional < self.initial_margin:
            raise BorosException("withdrawal would drop balance below initial margin")
        self._internal_state.collateral -= self._notional_to_margin(amount_in_notional)

    # --------------------------------------------------------- position
    def action_open_position(self, amount_in_product: float) -> None:
        """Trade ``amount_in_product`` yield units at the mark (± ``slippage_rate``).

        Same-sign fills average the fixed rate; opposite-sign fills
        realise ``closed · (fill − entry) · TTM`` into collateral; a flip
        carries the fill rate as the new entry. Risk-increasing fills
        are rolled back when the post-trade balance is below initial
        margin, so a margin-bound position can always be reduced.
        """
        if amount_in_product == 0:
            return
        self._require_not_matured("open_position")
        mark = self._global_state.mark_rate
        if not math.isfinite(mark):
            raise BorosException(f"cannot trade at non-finite mark rate {mark}")
        years = self.years_to_expiry
        sign = 1.0 if amount_in_product > 0 else -1.0
        fill = mark + sign * self.slippage_rate

        snap = (self._internal_state.size, self._internal_state.entry_rate, self._internal_state.collateral)
        self._internal_state.collateral -= self._coin_to_margin(
            open_fee_coin(amount_in_product, self.taker_fee_rate, years)
        )
        old_size, old_entry = snap[0], snap[1]
        if old_size == 0:
            self._internal_state.size = amount_in_product
            self._internal_state.entry_rate = fill
        elif old_size * amount_in_product > 0:
            new_size = old_size + amount_in_product
            self._internal_state.entry_rate = (old_entry * old_size + fill * amount_in_product) / new_size
            self._internal_state.size = new_size
        else:
            closed = min(abs(old_size), abs(amount_in_product))
            side = 1.0 if old_size > 0 else -1.0
            self._internal_state.collateral += self._coin_to_margin(closed * (fill - old_entry) * years * side)
            new_size = old_size + amount_in_product
            if abs(new_size) < self._EPS:
                self._internal_state.size = 0.0
                self._internal_state.entry_rate = 0.0
            elif abs(amount_in_product) > abs(old_size):
                self._internal_state.size = new_size
                self._internal_state.entry_rate = fill
            else:
                self._internal_state.size = new_size

        if abs(self._internal_state.size) > abs(snap[0]) and self.balance < self.initial_margin:
            balance, im = self.balance, self.initial_margin
            self._internal_state.size, self._internal_state.entry_rate, self._internal_state.collateral = snap
            raise BorosException(
                f"open_position would leave balance {balance} below initial margin {im}"
            )

    # --------------------------------------------------------- readouts
    @property
    def size(self) -> float:
        """Signed yield units (coins)."""
        return self._internal_state.size

    @property
    def entry_rate(self) -> float:
        """Fixed rate of the open position (``0`` when flat)."""
        return self._internal_state.entry_rate

    def _mtm_coin(self) -> float:
        return mark_to_maturity_coin(
            self._internal_state.size, self._global_state.mark_rate,
            self._internal_state.entry_rate, self.years_to_expiry,
        )

    @property
    def pnl(self) -> float:
        """Unrealised mark-to-maturity value in notional."""
        return self._margin_to_notional(self._coin_to_margin(self._mtm_coin()))

    @property
    def balance(self) -> float:
        """``collateral + unrealised`` in notional."""
        return self._margin_to_notional(self._internal_state.collateral + self._coin_to_margin(self._mtm_coin()))

    def _margin_notional(self, k: float) -> float:
        coins = margin_coin(
            self._internal_state.size, self._global_state.mark_rate, self.years_to_expiry,
            k, self.rate_floor, self.time_floor_years,
        )
        return self._margin_to_notional(self._coin_to_margin(coins))

    @property
    def initial_margin(self) -> float:
        """``kIM · |N| · max(TTM, floor) · max(|mark|, floor)`` in notional."""
        return self._margin_notional(self.k_im)

    @property
    def maintenance_margin(self) -> float:
        """``kMM · |N| · max(TTM, floor) · max(|mark|, floor)`` in notional."""
        return self._margin_notional(self.k_mm)

    @property
    def health_ratio(self) -> float:
        """``balance / maintenance_margin``; ``inf`` when flat."""
        mm = self.maintenance_margin
        if mm <= 0:
            return float("inf")
        return self.balance / mm

    @property
    def leverage(self) -> float:
        """Rate-exposure leverage ``IM / (kIM · balance)``; ``0`` when flat or wiped."""
        if self.balance <= 0 or self._internal_state.size == 0:
            return 0.0
        return self.initial_margin / (self.k_im * self.balance)

    # -------------------------------------------------------- lifecycle
    def _settle(self, state: BorosGlobalState) -> None:
        if state.funding_period_seconds <= 0 or self._internal_state.size == 0:
            return
        flow = self._coin_to_margin(settlement_pnl_coin(
            self._internal_state.size, state.funding_rate, self._internal_state.entry_rate,
            state.funding_period_seconds, self.settle_fee_rate,
        ))
        self._internal_state.collateral += flow
        self._internal_state.realized_settlements += flow

    def _check_maturity(self) -> None:
        """A yield unit is worth zero at maturity: close it without cost."""
        if self.is_matured and self._internal_state.size != 0:
            self._internal_state.size = 0.0
            self._internal_state.entry_rate = 0.0

    def _check_liquidation(self) -> None:
        """``HR <= 1``: close at the mark, pay the liquidator and the protocol."""
        if self._internal_state.size == 0:
            return
        hr = self.health_ratio
        if hr > 1.0:
            return
        penalty = liquidation_penalty_fraction(hr) * self.maintenance_margin + self._margin_to_notional(
            self._coin_to_margin(abs(self._internal_state.size) * LIQUIDATION_PROTOCOL_FEE_RATE * self.years_to_expiry)
        )
        self._internal_state.collateral += self._coin_to_margin(self._mtm_coin())
        self._internal_state.collateral = max(0.0, self._internal_state.collateral - self._notional_to_margin(penalty))
        self._internal_state.size = 0.0
        self._internal_state.entry_rate = 0.0
        self._internal_state.liquidation_count += 1

    def update_state(self, state: BorosGlobalState) -> None:
        """Validate → apply → settle the period (if any) → maturity → liquidation."""
        self._validate_term(state)
        if state.underlying_price <= 0:
            raise BorosException(f"underlying_price must be > 0, got {state.underlying_price}")
        if state.funding_period_seconds < 0:
            raise BorosException(f"funding_period_seconds must be >= 0, got {state.funding_period_seconds}")
        if not math.isfinite(state.mark_rate) or not math.isfinite(state.funding_rate):
            raise BorosException("mark_rate and funding_rate must be finite")
        self._global_state = state
        self._settle(state)
        self._check_maturity()
        self._check_liquidation()
