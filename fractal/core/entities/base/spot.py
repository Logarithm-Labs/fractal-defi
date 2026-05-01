"""Base classes for spot-trading entities.

Spot entities trade a *product* (e.g. BTC, ETH) against a *notional*
quote currency (e.g. USDC, USD). All spot entities hold the same two
state fields — ``amount`` (product) and ``cash`` (notional) — so we
formalize that as :class:`BaseSpotInternalState`. Concrete spot
internal-state classes inherit from it; this gives strategies that type
``spot: BaseSpotEntity`` proper IDE autocomplete on ``amount`` and
``cash``.

Convention — sizing is intentionally **asymmetric**:

* ``action_buy(amount_in_notional)`` — spend a fixed amount of notional
  cash; the entity decides how much product comes out (after fees and
  slippage).
* ``action_sell(amount_in_product)`` — sell a fixed amount of product;
  the entity decides how much notional comes back (after fees).

This matches the most common DeFi-strategy patterns:
* "buy with all my cash" → ``buy(amount_in_notional=cash)``
* "sell all my position" → ``sell(amount_in_product=amount)``

If you need the reverse (buy a fixed product amount or sell for a fixed
notional value), compute the inverse using
:attr:`BaseSpotEntity.current_price` at the strategy level.
"""
from abc import abstractmethod
from dataclasses import dataclass

from fractal.core.base.entity import BaseEntity, EntityException, InternalState


@dataclass
class BaseSpotInternalState(InternalState):
    """Common bookkeeping for any spot entity: product + notional cash.

    Concrete spot internal-state classes subclass this so that
    polymorphic strategy code (typing ``spot: BaseSpotEntity``) gets
    proper field-name autocomplete on ``amount`` and ``cash``.
    """
    amount: float = 0.0
    cash: float = 0.0


class BaseSpotEntity(BaseEntity):
    """Spot entity that buys *product* with *notional* and sells back."""

    # Narrow the annotation so Pylance/PyCharm see ``amount`` / ``cash``
    # when accessing ``self.internal_state`` on any spot entity.
    _internal_state: BaseSpotInternalState

    @property
    def internal_state(self) -> BaseSpotInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    @abstractmethod
    def current_price(self) -> float:
        """Current spot price used for trades and balance computation.

        Concrete impls usually return ``self._global_state.price`` or
        ``self._global_state.close``. Lifted to the base so polymorphic
        strategy code does not have to know the underlying field name.
        """
        raise NotImplementedError

    @abstractmethod
    def action_buy(self, amount_in_notional: float):
        """Spend ``amount_in_notional`` of notional cash to acquire product.

        Args:
            amount_in_notional: Amount of notional currency to spend.

        Raises:
            EntityException: If there is not enough cash.
        """
        raise NotImplementedError

    @abstractmethod
    def action_sell(self, amount_in_product: float):
        """Sell ``amount_in_product`` of product for notional cash.

        Args:
            amount_in_product: Amount of product to sell.

        Raises:
            EntityException: If there is not enough product.
        """
        raise NotImplementedError

    # --------------------------------------------------------- cross-entity transfer
    # The two primitives below are **strategy-level** helpers for moving
    # the product token between entities WITHOUT a swap (no fee, no price
    # impact). Useful when the strategy holds product that arrived from
    # outside the spot entity — for example, when Aave returns collateral
    # to the user, or when one spot entity transfers product to another.
    #
    # They are NOT a substitute for buy/sell — those go through the market;
    # these bypass it entirely. Strategies are responsible for ensuring
    # the product accounting is consistent across entities (the
    # framework can't enforce it because product token identity isn't
    # represented at the entity level).

    def action_inject_product(self, amount: float) -> None:
        """Add ``amount`` of product directly to ``internal_state.amount``.

        No swap, no trading fee. Used when product arrives from outside —
        e.g. as a withdrawal of collateral from a lending protocol that
        the strategy is now holding in this spot entity.

        Subclasses can override to layer protocol-specific behaviour.

        Raises:
            EntityException: If ``amount < 0``.
        """
        if amount < 0:
            raise EntityException(
                f"inject_product amount must be >= 0, got {amount}"
            )
        self._internal_state.amount += amount

    def action_remove_product(self, amount: float) -> None:
        """Remove ``amount`` of product directly from ``internal_state.amount``.

        No swap, no trading fee. Used when product leaves the entity —
        e.g. being deposited into a lending protocol as collateral, or
        transferred to another spot entity.

        Subclasses can override to layer protocol-specific behaviour.

        Raises:
            EntityException: If ``amount < 0`` or exceeds the current holding.
        """
        if amount < 0:
            raise EntityException(
                f"remove_product amount must be >= 0, got {amount}"
            )
        if amount > self._internal_state.amount:
            raise EntityException(
                f"remove_product exceeds holding: {amount} > {self._internal_state.amount}"
            )
        self._internal_state.amount -= amount
