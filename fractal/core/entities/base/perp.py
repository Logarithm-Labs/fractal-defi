"""Base class for perpetual-futures entities.

A perp entity holds collateral and an aggregated position. It exposes a
uniform contract so strategies can talk to any concrete perp
(:class:`HyperliquidEntity`, :class:`SimplePerpEntity`, …) polymorphically.

Closing convention: a perp entity is closed via an **opposite-sign
``action_open_position``** call. Concrete impls realize the close
through their own clearing routine (``HyperliquidEntity._clearing`` or
scalar netting in :class:`SimplePerpEntity`). The default
:meth:`BasePerpEntity.action_close_position` is a thin sugar wrapper —
override only if your impl needs special bookkeeping at close time.

Common state contract: every perp keeps free notional ``collateral``,
formalized via :class:`BasePerpInternalState`. Position layout differs
between impls (Hyperliquid keeps a ``positions: List``, SimplePerp
keeps a scalar ``size``) and is therefore exposed via the
:attr:`size` and :attr:`pnl` *properties* on the entity rather than via
the internal-state dataclass.
"""
from abc import abstractmethod
from dataclasses import dataclass

from fractal.core.base.entity import BaseEntity, InternalState


@dataclass
class BasePerpInternalState(InternalState):
    """Common bookkeeping for any perp entity: free notional collateral.

    Position layout varies (positions list vs scalar size); concrete
    perp internal-state classes subclass this and add their own
    position fields.
    """
    collateral: float = 0.0


class BasePerpEntity(BaseEntity):
    """Common interface for perpetual-futures entities."""

    # Narrow the annotation so polymorphic strategy code sees
    # ``collateral`` on ``self.internal_state`` for any perp entity.
    _internal_state: BasePerpInternalState

    @property
    def internal_state(self) -> BasePerpInternalState:  # type: ignore[override]
        return self._internal_state

    @abstractmethod
    def action_open_position(self, amount_in_product: float, *args, **kwargs):
        """Open or extend a position. Sign of ``amount_in_product``
        encodes direction (positive = long, negative = short).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def leverage(self) -> float:
        """``|size| · mark_price / balance``. Returns ``0`` when flat or wiped out."""
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> float:
        """Signed position size in product units."""
        raise NotImplementedError

    @property
    @abstractmethod
    def pnl(self) -> float:
        """Unrealized PnL at the current mark price."""
        raise NotImplementedError

    def action_close_position(self) -> None:
        """Flatten the position via an opposite-sign open. No-op when flat.

        Default implementation is sufficient for any perp whose
        ``action_open_position`` aggregates trades and nets opposite
        signs (which all current impls do). Override only for impls
        that need bespoke close bookkeeping.
        """
        if self.size != 0:
            self.action_open_position(amount_in_product=-self.size)
