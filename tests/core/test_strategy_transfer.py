"""Tests for :meth:`BaseStrategy.transfer` — the canonical
withdraw+deposit pair helper for cross-entity cash moves.
"""
from datetime import datetime

import pytest

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity, Observation)
from fractal.core.entities import (SimpleSpotExchange,
                                   SimpleSpotExchangeGlobalState)


class _TransferStrategy(BaseStrategy):
    """Two SimpleSpotExchange entities; predict() emits a single transfer."""

    def set_up(self):
        self.register_entity(NamedEntity("A", SimpleSpotExchange(trading_fee=0.0)))
        self.register_entity(NamedEntity("B", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        # Subclasses replace this in tests.
        return []


def _make_strategy() -> _TransferStrategy:
    return _TransferStrategy(debug=False, params=BaseStrategyParams())


def _state(close: float) -> SimpleSpotExchangeGlobalState:
    return SimpleSpotExchangeGlobalState(open=close, high=close, low=close, close=close)


# --------------------------------------------------------- pure helper shape
@pytest.mark.core
def test_transfer_returns_deposit_first_withdraw_second():
    strat = _make_strategy()
    actions = strat.transfer("A", "B", amount_in_notional=100.0)
    assert len(actions) == 2
    # Deposit on the destination, withdraw on the source — in that order.
    assert actions[0].entity_name == "B"
    assert actions[0].action.action == "deposit"
    assert actions[1].entity_name == "A"
    assert actions[1].action.action == "withdraw"
    assert actions[0].action.args["amount_in_notional"] == 100.0
    assert actions[1].action.args["amount_in_notional"] == 100.0


@pytest.mark.core
def test_transfer_rejects_self_loop():
    strat = _make_strategy()
    with pytest.raises(ValueError):
        strat.transfer("A", "A", 100.0)


@pytest.mark.core
def test_transfer_rejects_unknown_entity():
    strat = _make_strategy()
    with pytest.raises(ValueError):
        strat.transfer("A", "DOES_NOT_EXIST", 100.0)
    with pytest.raises(ValueError):
        strat.transfer("DOES_NOT_EXIST", "B", 100.0)


# ------------------------------------------------------------- end-to-end
class _LiteralTransferStrategy(_TransferStrategy):
    """Seed A with cash on first step, then transfer half to B."""

    def predict(self):
        a = self.get_entity("A")
        if a.internal_state.cash >= 1000 and self.get_entity("B").internal_state.cash == 0:
            return self.transfer("A", "B", 500.0)
        return []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_entity("A").action_deposit(1000.0)


@pytest.mark.core
def test_transfer_with_literal_amount_runs_through_step():
    strat = _LiteralTransferStrategy(debug=False, params=BaseStrategyParams())
    strat.step(Observation(timestamp=datetime(2024, 1, 1), states={
        "A": _state(100), "B": _state(100),
    }))
    assert strat.get_entity("A").internal_state.cash == pytest.approx(500)
    assert strat.get_entity("B").internal_state.cash == pytest.approx(500)


# --------------------------------------------- delegate amount conservation
class _DelegateTransferStrategy(_TransferStrategy):
    """Move ALL of A's cash to B in a single step using a delegate."""

    def predict(self):
        if (self.get_entity("A").internal_state.cash > 0
                and self.get_entity("B").internal_state.cash == 0):
            return self.transfer(
                "A", "B",
                amount_in_notional=lambda obj: obj.get_entity("A").internal_state.cash,
            )
        return []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_entity("A").action_deposit(750.0)


@pytest.mark.core
def test_transfer_with_delegate_conserves_value_across_step():
    """The deposit-first / withdraw-second order matters: both delegate
    evaluations read the same source state, so all 750 ends up on B."""
    strat = _DelegateTransferStrategy(debug=False, params=BaseStrategyParams())
    strat.step(Observation(timestamp=datetime(2024, 1, 1), states={
        "A": _state(100), "B": _state(100),
    }))
    assert strat.get_entity("A").internal_state.cash == pytest.approx(0)
    assert strat.get_entity("B").internal_state.cash == pytest.approx(750)
