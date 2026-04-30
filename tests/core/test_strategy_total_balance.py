"""Tests for :attr:`BaseStrategy.total_balance` — portfolio aggregator.

``total_balance`` simply sums ``entity.balance`` across all registered
entities. It assumes the strategy author has kept units consistent
across entities (typically USD). See README's "Pricing convention".
"""
import pytest

from fractal.core.base import (BaseStrategy, BaseStrategyParams,
                               NamedEntity)
from fractal.core.entities import (SimpleSpotExchange,
                                   SimpleSpotExchangeGlobalState)
from fractal.core.entities.simple.lending import (SimpleLendingEntity,
                                                  SimpleLendingGlobalState)


class _SingleEntityStrategy(BaseStrategy):
    def set_up(self):
        self.register_entity(NamedEntity("SPOT", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


class _MultiEntityStrategy(BaseStrategy):
    def set_up(self):
        self.register_entity(NamedEntity("A", SimpleSpotExchange(trading_fee=0.0)))
        self.register_entity(NamedEntity("B", SimpleSpotExchange(trading_fee=0.0)))
        self.register_entity(NamedEntity("LEND", SimpleLendingEntity()))

    def predict(self):
        return []


@pytest.mark.core
def test_total_balance_zero_when_no_entities():
    """Strategy with empty entity registry → total = 0."""
    class _Empty(BaseStrategy):
        def set_up(self):
            pass
        def predict(self):
            return []
    s = _Empty(debug=False, params=BaseStrategyParams())
    assert s.total_balance == 0.0


@pytest.mark.core
def test_total_balance_single_entity_equals_entity_balance():
    s = _SingleEntityStrategy(debug=False, params=BaseStrategyParams())
    spot = s.get_entity("SPOT")
    spot.update_state(SimpleSpotExchangeGlobalState(close=2000, high=2000, low=2000, open=2000, volume=0))
    spot.action_deposit(1000)
    assert s.total_balance == spot.balance == 1000


@pytest.mark.core
def test_total_balance_aggregates_across_entities():
    s = _MultiEntityStrategy(debug=False, params=BaseStrategyParams())
    a = s.get_entity("A")
    b = s.get_entity("B")
    lend = s.get_entity("LEND")

    a.update_state(SimpleSpotExchangeGlobalState(close=2000, high=2000, low=2000, open=2000, volume=0))
    b.update_state(SimpleSpotExchangeGlobalState(close=2000, high=2000, low=2000, open=2000, volume=0))
    lend.update_state(SimpleLendingGlobalState(collateral_price=1.0, debt_price=1.0))

    a.action_deposit(500)
    b.action_deposit(300)
    lend.action_deposit(200)
    # All in USD: total = 500 + 300 + 200 = 1000
    assert s.total_balance == 1000


@pytest.mark.core
def test_total_balance_reflects_position_value_changes():
    """When an entity's balance changes (e.g. price move), total_balance reflects it."""
    s = _SingleEntityStrategy(debug=False, params=BaseStrategyParams())
    spot = s.get_entity("SPOT")
    spot.update_state(SimpleSpotExchangeGlobalState(close=2000, high=2000, low=2000, open=2000, volume=0))
    spot.action_deposit(1000)
    spot.action_buy(1000)  # all-in: 0.5 ETH
    assert s.total_balance == pytest.approx(1000)

    # Price up 50% → balance up 50%
    spot.update_state(SimpleSpotExchangeGlobalState(close=3000, high=3000, low=3000, open=3000, volume=0))
    assert s.total_balance == pytest.approx(1500)


@pytest.mark.core
def test_total_balance_with_borrowed_and_collateral():
    """Lending: collateral_value − debt_value contributes to total."""
    s = _MultiEntityStrategy(debug=False, params=BaseStrategyParams())
    lend = s.get_entity("LEND")
    lend.update_state(SimpleLendingGlobalState(collateral_price=1.0, debt_price=1.0))
    lend.action_deposit(1000)
    lend.action_borrow(400)
    # Equity = 1000 − 400 = 600
    assert lend.balance == 600
    assert s.total_balance == 600  # other entities have 0 balance
