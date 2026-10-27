"""Synthetic tests for :class:`InTransitEntity` and :class:`VenueCashEntity`."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams, NamedEntity, Observation
from fractal.core.entities import InTransitEntity, VenueCashEntity
from fractal.core.entities.models.route_costs import RouteLeg
from fractal.core.entities.simple.spot import SimpleSpotExchangeException, SimpleSpotExchangeGlobalState
from fractal.core.entities.simple.transit import InTransitException, InTransitGlobalState, TransitTicket

pytestmark = pytest.mark.core

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ------------------------------------------------------------------ transit
def test_send_due_release_track_tickets_and_balance():
    transit = InTransitEntity()
    transit.action_send(100.0, "CASH_A", arrival_bar=3)
    transit.action_send(50.0, "CASH_B", arrival_bar=1)
    transit.action_send(25.0, "CASH_A", arrival_bar=2)
    assert transit.balance == 175.0
    assert transit.in_flight_to("CASH_A") == 125.0 and transit.in_flight_to("CASH_C") == 0.0
    assert [t.ticket_id for t in transit.tickets] == [0, 1, 2]
    assert transit.due(0) == []
    assert transit.due(2) == [TransitTicket(1, "CASH_B", 50.0, 1), TransitTicket(2, "CASH_A", 25.0, 2)]
    transit.action_release(1)
    assert transit.balance == 125.0 and [t.ticket_id for t in transit.due(10)] == [0, 2]
    transit.action_send(10.0, "CASH_B", arrival_bar=5)
    assert transit.tickets[-1].ticket_id == 3               # ids are never reused


def test_transit_earns_nothing_and_ignores_market_state():
    transit = InTransitEntity()
    transit.action_send(1_000.0, "CASH", arrival_bar=1)
    for _ in range(5):
        transit.update_state(transit.global_state)
    assert transit.balance == 1_000.0


def test_empty_send_opens_no_ticket():
    transit = InTransitEntity()
    transit.action_send(0.0, "CASH", arrival_bar=1)
    assert not transit.tickets and transit.internal_state.next_id == 0


@pytest.mark.parametrize("kwargs", [
    {"amount_in_notional": -1.0, "destination": "CASH", "arrival_bar": 1},
    {"amount_in_notional": float("nan"), "destination": "CASH", "arrival_bar": 1},
    {"amount_in_notional": 1.0, "destination": "", "arrival_bar": 1},
    {"amount_in_notional": 1.0, "destination": "CASH", "arrival_bar": -1},
    {"amount_in_notional": 1.0, "destination": "CASH", "arrival_bar": 1.5},
    {"amount_in_notional": 1.0, "destination": "CASH", "arrival_bar": True},
])
def test_send_rejects_invalid_arguments(kwargs):
    with pytest.raises(InTransitException):
        InTransitEntity().action_send(**kwargs)


def test_release_of_an_unknown_ticket_raises():
    transit = InTransitEntity()
    transit.action_send(1.0, "CASH", arrival_bar=1)
    transit.action_release(0)
    with pytest.raises(InTransitException, match="no open transit ticket"):
        transit.action_release(0)


def test_transit_has_no_deposit_or_withdraw():
    transit = InTransitEntity()
    assert set(transit.get_available_actions()) >= {"send", "release"}
    with pytest.raises(NotImplementedError):
        transit.action_deposit(1.0)


def test_state_snapshot_is_isolated():
    transit = InTransitEntity()
    transit.action_send(1.0, "CASH", arrival_bar=1)
    snapshot = deepcopy(transit.internal_state)
    transit.action_release(0)
    assert len(snapshot.tickets) == 1 and not transit.tickets


# --------------------------------------------------------------- venue cash
def test_pay_records_the_cost_as_a_fee_without_turnover():
    cash = VenueCashEntity()
    records = []
    cash.attach_execution_recorder(lambda *row: records.append(row))
    cash.action_deposit(1_000.0)
    cash.action_pay(12.5, "bridge")
    cash.action_pay(0.0, "gas")                              # empty: nothing recorded
    assert cash.balance == 987.5
    assert records == [("cost_bridge", 0.0, 12.5)]
    with pytest.raises(SimpleSpotExchangeException):
        cash.action_pay(5_000.0, "gas")
    with pytest.raises(SimpleSpotExchangeException):
        cash.action_pay(1.0, "")
    with pytest.raises(SimpleSpotExchangeException):
        cash.action_pay(-1.0, "gas")


def test_withdraw_absorbs_float_residue_but_not_real_overdrafts():
    cash = VenueCashEntity()
    cash.action_deposit(0.1 + 0.2)
    cash.action_withdraw(0.30000000000000016)                 # a planned full withdrawal, off by float residue
    assert cash.balance == 0.0
    cash.action_deposit(100.0)
    with pytest.raises(SimpleSpotExchangeException):
        cash.action_withdraw(100.01)


# ------------------------------------------------------------ in a strategy
@dataclass
class _RouteParams(BaseStrategyParams):
    AMOUNT: float = 10_000.0


class _OneRoute(BaseStrategy[_RouteParams]):
    """Sends AMOUNT home -> L2 on bar 1 over a latency leg, then deposits it on arrival."""

    LEG = RouteLeg(bridge_fixed_usd=3.0, bridge_bps=2.0, gas_usd=1.0, latency_bars=3)

    def set_up(self):
        self.register_entity(NamedEntity("HOME", VenueCashEntity()))
        self.register_entity(NamedEntity("L2", VenueCashEntity()))
        self.register_entity(NamedEntity("TRANSIT", InTransitEntity()))
        self.bar_index = 0

    def predict(self):
        self.bar_index += 1
        transit = self.get_entity("TRANSIT")
        actions = []
        for ticket in transit.due(self.bar_index):
            actions += [ActionToTake("TRANSIT", Action("release", {"ticket_id": ticket.ticket_id})),
                        ActionToTake(ticket.destination,
                                     Action("deposit", {"amount_in_notional": ticket.amount}))]
        if self.bar_index == 1:
            gross = self._params.AMOUNT
            costs = self.LEG.costs(gross, cross_asset=False)
            net = gross - sum(costs.values())
            actions += [ActionToTake("HOME", Action("pay", {"amount_in_notional": c, "component": k}))
                        for k, c in costs.items() if c > 0.0]
            actions += [ActionToTake("HOME", Action("withdraw", {"amount_in_notional": net})),
                        ActionToTake("TRANSIT", Action("send", {"amount_in_notional": net, "destination": "L2",
                                                                "arrival_bar": self.bar_index + self.LEG.latency()}))]
        return actions


def test_a_latency_route_conserves_nav_net_of_costs_and_lands_on_time():
    strategy = _OneRoute()
    strategy.get_entity("HOME").action_deposit(10_000.0)
    states = {"HOME": SimpleSpotExchangeGlobalState(close=1.0), "L2": SimpleSpotExchangeGlobalState(close=1.0),
              "TRANSIT": InTransitGlobalState()}
    observations = [Observation(timestamp=START + timedelta(hours=hour), states=states) for hour in range(6)]
    result = strategy.run(observations)
    fee = 3.0 + 2.0 + 1.0                                     # fixed bridge + 2 bps of 10k + gas
    frame = result.to_dataframe()
    assert frame["net_balance"].tolist() == pytest.approx([10_000.0 - fee] * 6, rel=1e-12)
    in_flight = [row["TRANSIT"] for row in result.balances]
    landed = [row["L2"] for row in result.balances]
    assert in_flight == pytest.approx([10_000.0 - fee] * 3 + [0.0] * 3)
    assert landed == pytest.approx([0.0] * 3 + [10_000.0 - fee] * 3)  # bar 4 = 1 + 3 bars of latency
    assert result.get_default_metrics().fees_paid == pytest.approx(fee, rel=1e-12)
    assert {r.action for r in result.execution_records} == {"cost_bridge", "cost_gas"}
