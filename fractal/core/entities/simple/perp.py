"""Simple single-position perpetual-futures entity.

A minimal generic perp model: one aggregated position with linear PnL,
single-sided funding (longs pay shorts when ``funding_rate > 0``), and
margin-based liquidation. Use it for tests, examples and quick
prototypes; for protocol-specific quirks (two-sided funding/borrowing,
oracle deviations, isolated-vs-cross modes) reach for
:class:`HyperliquidEntity` or :class:`GMXV2Entity`.

Position model:

* ``internal_state.size`` is **signed** (positive = long, negative = short).
* ``internal_state.entry_price`` is the weighted-average entry across
  all aggregated trades on the same side.
* Opening a trade with the **same** sign as the existing position adjusts
  ``entry_price`` to a quantity-weighted average and adds the size.
* Opening a trade with the **opposite** sign nets the position. The
  closed leg is realized into ``collateral`` at the trade price; if the
  remainder flips the direction, the new position carries the **trade**
  price as the new entry.

Update sequence on each ``update_state``:

1. Apply the new global state (mark price + funding rate).
2. Settle funding: ``collateral -= size · mark_price · funding_rate``.
3. Liquidation check: if ``balance < |size| · mark_price / MAX_LEVERAGE``,
   wipe collateral and the position.

Funding settles **before** liquidation so a positive funding tick can
genuinely save a position that just barely crossed maintenance.
"""
from dataclasses import dataclass

from fractal.core.base.entity import EntityException, GlobalState
from fractal.core.entities.base.perp import BasePerpEntity, BasePerpInternalState


class SimplePerpEntityException(EntityException):
    """Errors raised by :class:`SimplePerpEntity`."""


@dataclass
class SimplePerpGlobalState(GlobalState):
    """Market state seen by the entity.

    Attributes:
        mark_price: Current mark price used for PnL, fees and funding.
        funding_rate: Per-step funding rate. Positive ⇒ longs pay shorts.
    """
    mark_price: float = 0.0
    funding_rate: float = 0.0


@dataclass
class SimplePerpInternalState(BasePerpInternalState):
    """Aggregated single-position state.

    Inherits ``collateral`` and adds:
        size: Signed position size in product units (positive = long).
        entry_price: Weighted-average entry price; ``0`` when flat.
    """
    size: float = 0.0
    entry_price: float = 0.0


class SimplePerpEntity(BasePerpEntity):
    """Minimal single-position perpetual-futures entity."""

    _EPS = 1e-12

    def __init__(
        self,
        *args,
        trading_fee: float = 0.0005,
        max_leverage: float = 50.0,
        **kwargs,
    ) -> None:
        """
        Args:
            trading_fee: Taker fee charged on notional traded.
            max_leverage: Maximum leverage; also defines maintenance margin
                as ``|size| · mark_price / max_leverage``.
        """
        # Validate and store config BEFORE super().__init__ so that any
        # subclass override of ``_initialize_states`` can rely on it.
        if trading_fee < 0:
            raise SimplePerpEntityException(f"trading_fee must be >= 0, got {trading_fee}")
        if max_leverage <= 0:
            raise SimplePerpEntityException(f"max_leverage must be > 0, got {max_leverage}")
        self.TRADING_FEE: float = float(trading_fee)
        self.MAX_LEVERAGE: float = float(max_leverage)
        super().__init__(*args, **kwargs)

    _internal_state: SimplePerpInternalState
    _global_state: SimplePerpGlobalState

    def _initialize_states(self) -> None:
        self._internal_state = SimplePerpInternalState()
        self._global_state = SimplePerpGlobalState()

    @property
    def internal_state(self) -> SimplePerpInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> SimplePerpGlobalState:  # type: ignore[override]
        return self._global_state

    # ------------------------------------------------------------- account
    def action_deposit(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimplePerpEntityException(
                f"deposit amount must be >= 0, got {amount_in_notional}"
            )
        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimplePerpEntityException(
                f"withdraw amount must be >= 0, got {amount_in_notional}"
            )
        if amount_in_notional > self.balance:
            raise SimplePerpEntityException(
                f"insufficient balance: {self.balance} < {amount_in_notional}"
            )
        post_balance = self.balance - amount_in_notional
        if post_balance < self._maintenance_margin():
            raise SimplePerpEntityException(
                "withdrawal would drop balance below maintenance margin"
            )
        self._internal_state.collateral -= amount_in_notional

    # ------------------------------------------------------------ position
    def action_open_position(self, amount_in_product: float) -> None:
        """Add ``amount_in_product`` to the aggregated position.

        Same-sign trades update the weighted-average entry; opposite-sign
        trades realize PnL on the closed leg into collateral.
        """
        if amount_in_product == 0:
            return

        mark_price = self._global_state.mark_price
        if mark_price <= 0:
            raise SimplePerpEntityException(
                f"cannot trade at non-positive mark price {mark_price}"
            )

        # Charge taker fee on traded notional, regardless of net direction.
        self._internal_state.collateral -= (
            abs(amount_in_product) * mark_price * self.TRADING_FEE
        )

        old_size = self._internal_state.size
        old_entry = self._internal_state.entry_price

        if old_size == 0:
            self._internal_state.size = amount_in_product
            self._internal_state.entry_price = mark_price
            return

        if old_size * amount_in_product > 0:
            # Same direction → weighted-average entry, sum sizes.
            new_size = old_size + amount_in_product
            self._internal_state.entry_price = (
                (old_entry * old_size + mark_price * amount_in_product) / new_size
            )
            self._internal_state.size = new_size
            return

        # Opposite direction → close part (or flip).
        closed_qty = min(abs(old_size), abs(amount_in_product))
        sign = 1.0 if old_size > 0 else -1.0
        realized_pnl = closed_qty * (mark_price - old_entry) * sign
        self._internal_state.collateral += realized_pnl

        new_size = old_size + amount_in_product
        if abs(new_size) < self._EPS:
            # Fully closed.
            self._internal_state.size = 0.0
            self._internal_state.entry_price = 0.0
        elif abs(amount_in_product) > abs(old_size):
            # Flipped: remaining quantity inherits the trade price.
            self._internal_state.size = new_size
            self._internal_state.entry_price = mark_price
        else:
            # Partial close; existing entry stays.
            self._internal_state.size = new_size

    # ``action_close_position`` is inherited from :class:`BasePerpEntity`
    # (default impl: ``action_open_position(amount_in_product=-self.size)``).

    # ------------------------------------------------------------ readouts
    @property
    def pnl(self) -> float:
        """Unrealized PnL: ``size · (mark_price - entry_price)``."""
        if self._internal_state.size == 0:
            return 0.0
        return self._internal_state.size * (
            self._global_state.mark_price - self._internal_state.entry_price
        )

    @property
    def balance(self) -> float:
        """``collateral + unrealized_pnl``."""
        return self._internal_state.collateral + self.pnl

    @property
    def size(self) -> float:
        """Signed position size in product units."""
        return self._internal_state.size

    @property
    def leverage(self) -> float:
        """``|size| · mark_price / balance``. Returns 0 when flat or wiped out."""
        if self.balance <= 0 or self._internal_state.size == 0:
            return 0.0
        return (
            abs(self._internal_state.size) * self._global_state.mark_price / self.balance
        )

    # ----------------------------------------------------------- internals
    def _maintenance_margin(self) -> float:
        if self._internal_state.size == 0:
            return 0.0
        return (
            abs(self._internal_state.size)
            * self._global_state.mark_price
            / self.MAX_LEVERAGE
        )

    def _check_liquidation(self) -> bool:
        if self._internal_state.size == 0:
            return False
        return self.balance < self._maintenance_margin()

    def _wipe(self) -> None:
        self._internal_state.collateral = 0.0
        self._internal_state.size = 0.0
        self._internal_state.entry_price = 0.0

    def update_state(self, state: SimplePerpGlobalState) -> None:
        """Step the entity forward to ``state``.

        Order: apply state → settle funding → check liquidation. Funding
        settles **before** liquidation so a saving funding tick is honored.
        """
        self._global_state = state
        if self._internal_state.size != 0:
            self._internal_state.collateral -= (
                self._internal_state.size * state.mark_price * state.funding_rate
            )
        if self._check_liquidation():
            self._wipe()
