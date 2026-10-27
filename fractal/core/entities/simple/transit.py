"""Accounting entities for capital that moves between venues.

Two small entities a multi-venue strategy registers next to its positions:

:class:`VenueCashEntity`
    Idle cash at one venue (USD notional, earns nothing). It is a zero-fee
    :class:`~fractal.core.entities.simple.spot.SimpleSpotExchange`, so it plugs
    into :meth:`BaseStrategy.transfer` like any cash entity, plus
    :meth:`~VenueCashEntity.action_pay`, which takes a route cost out of the book
    and records it as ``cost_<component>`` with the amount as ``fee_paid`` (so it
    reaches ``StrategyResult.fees_paid`` without inflating turnover).

:class:`InTransitEntity`
    Capital in flight between venues. It counts in NAV through ``balance``,
    earns nothing, and holds one :class:`TransitTicket` per transfer until the
    strategy releases it at arrival. Bars are the strategy's own bar counter:
    the entity never advances time by itself, so an arrival is never missed or
    early because of sparse observations.

A latency-bearing route is three actions on the send bar (pay the cash costs,
withdraw the net amount from the source, ``send`` it into transit) and two on
the arrival bar (``release`` the ticket, deposit the amount at the
destination). NAV is conserved across the whole route except for the costs
paid::

    actions = [ActionToTake("CASH_HOME", Action("pay", {"amount_in_notional": fee, "component": "bridge"})),
               ActionToTake("CASH_HOME", Action("withdraw", {"amount_in_notional": net})),
               ActionToTake("TRANSIT", Action("send", {"amount_in_notional": net,
                                                       "destination": "CASH_L2", "arrival_bar": now + 3}))]
    # ... at a later bar
    for ticket in transit.due(now):
        actions += [ActionToTake("TRANSIT", Action("release", {"ticket_id": ticket.ticket_id})),
                    ActionToTake(ticket.destination, Action("deposit", {"amount_in_notional": ticket.amount}))]

Route pricing lives in :mod:`fractal.core.entities.models.route_costs`.
"""
import math
from dataclasses import dataclass, field

from fractal.core.base.entity import BaseEntity, EntityException, GlobalState, InternalState
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeException

__all__ = [
    "InTransitEntity",
    "InTransitException",
    "InTransitGlobalState",
    "InTransitInternalState",
    "TransitTicket",
    "VenueCashEntity",
]


class InTransitException(EntityException):
    """Errors raised by :class:`InTransitEntity`."""


@dataclass(frozen=True)
class TransitTicket:
    """One transfer in flight: ``amount`` USD reaching ``destination`` at ``arrival_bar``."""
    ticket_id: int
    destination: str
    amount: float
    arrival_bar: int


@dataclass
class InTransitGlobalState(GlobalState):
    """Transit has no market context; observations may omit it."""


@dataclass
class InTransitInternalState(InternalState):
    """Tickets in flight by id, and the next id to issue."""
    tickets: dict[int, TransitTicket] = field(default_factory=dict)
    next_id: int = 0


class InTransitEntity(BaseEntity[InTransitGlobalState, InTransitInternalState]):
    """Capital in transit: counts in NAV, earns nothing, released to its destination at arrival."""

    def _initialize_states(self) -> None:
        self._global_state = InTransitGlobalState()
        self._internal_state = InTransitInternalState()

    def update_state(self, state: GlobalState) -> None:
        """Store the state; nothing accrues in transit."""
        self._global_state = state

    @property
    def balance(self) -> float:
        """USD currently in flight (the sum of all open tickets)."""
        return math.fsum(ticket.amount for ticket in self._internal_state.tickets.values())

    @property
    def tickets(self) -> tuple[TransitTicket, ...]:
        """Open tickets, oldest first."""
        return tuple(self._internal_state.tickets[key] for key in sorted(self._internal_state.tickets))

    def due(self, current_bar: int) -> list[TransitTicket]:
        """Tickets that have arrived by ``current_bar`` (``arrival_bar <= current_bar``), oldest first."""
        return [ticket for ticket in self.tickets if ticket.arrival_bar <= current_bar]

    def in_flight_to(self, destination: str) -> float:
        """USD in flight to ``destination``."""
        return math.fsum(ticket.amount for ticket in self._internal_state.tickets.values()
                         if ticket.destination == destination)

    def action_send(self, amount_in_notional: float, destination: str, arrival_bar: int) -> None:
        """Put ``amount_in_notional`` in flight to ``destination``, arriving at ``arrival_bar``.

        The caller withdraws the same amount from the source first; an empty
        transfer opens no ticket.
        """
        if not math.isfinite(amount_in_notional) or amount_in_notional < 0.0:
            raise InTransitException(f"transit amount must be finite and >= 0, got {amount_in_notional!r}")
        if not isinstance(destination, str) or not destination:
            raise InTransitException(f"destination must be a non-empty entity name, got {destination!r}")
        if isinstance(arrival_bar, bool) or not isinstance(arrival_bar, int) or arrival_bar < 0:
            raise InTransitException(f"arrival_bar must be an integer >= 0, got {arrival_bar!r}")
        if amount_in_notional == 0.0:
            return
        state = self._internal_state
        state.tickets[state.next_id] = TransitTicket(
            ticket_id=state.next_id, destination=destination, amount=float(amount_in_notional),
            arrival_bar=arrival_bar,
        )
        state.next_id += 1

    def action_release(self, ticket_id: int) -> None:
        """Close an arrived ticket; the caller deposits its amount at the destination."""
        if ticket_id not in self._internal_state.tickets:
            raise InTransitException(f"no open transit ticket {ticket_id!r}")
        del self._internal_state.tickets[ticket_id]


class VenueCashEntity(SimpleSpotExchange):
    """Idle cash at one venue: USD notional, zero fee, earns nothing.

    Observations may omit it: with no product held, ``balance`` is the cash
    whatever the price.
    """

    def __init__(self) -> None:
        super().__init__(trading_fee=0.0)

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Withdraw cash; a request above the balance by float residue only takes the whole balance."""
        cash = self._internal_state.cash
        if cash < amount_in_notional <= cash * (1.0 + 1e-12) + 1e-9:
            amount_in_notional = cash
        super().action_withdraw(amount_in_notional)

    def action_pay(self, amount_in_notional: float, component: str) -> None:
        """Pay a route cost: the amount leaves the book and is recorded as ``cost_<component>``."""
        if not isinstance(component, str) or not component:
            raise SimpleSpotExchangeException(f"component must be a non-empty name, got {component!r}")
        if not math.isfinite(amount_in_notional) or amount_in_notional < 0.0:
            raise SimpleSpotExchangeException(f"pay amount must be finite and >= 0, got {amount_in_notional!r}")
        if amount_in_notional == 0.0:
            return
        self.action_withdraw(amount_in_notional)
        self.record_execution(f"cost_{component}", 0.0, amount_in_notional)
