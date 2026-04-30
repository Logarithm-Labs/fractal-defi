"""Lock-in tests for two recent fixes in :class:`BaseStrategy`:

* :meth:`BaseStrategy.step` no longer mutates ``Action.args`` when
  resolving callable (delegate) arguments.
* :meth:`BaseStrategy.register_entity` validates entity name + instance.
"""
from datetime import datetime

import pytest

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity, Observation)
from fractal.core.entities import (SimpleSpotExchange,
                                   SimpleSpotExchangeGlobalState)


def _state(close: float) -> SimpleSpotExchangeGlobalState:
    return SimpleSpotExchangeGlobalState(open=close, high=close, low=close, close=close)


# ---------------------------------------------------- Action.args immutability
class _DelegateUserStrategy(BaseStrategy):
    """Stores a single Action template with a callable arg and reuses it
    on every step. After the second step the lambda must still be there
    (i.e. step() must NOT have replaced it with a constant)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.template = Action(
            "deposit",
            {"amount_in_notional": lambda obj: 100.0},
        )

    def set_up(self):
        self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return [ActionToTake(entity_name="X", action=self.template)]


@pytest.mark.core
def test_step_does_not_mutate_action_args():
    s = _DelegateUserStrategy(debug=False, params=BaseStrategyParams())
    obs = Observation(timestamp=datetime(2024, 1, 1), states={"X": _state(100)})
    s.step(obs)
    s.step(obs)
    # template's lambda must still be a callable, not the resolved float.
    assert callable(s.template.args["amount_in_notional"])
    # And entity actually got the deposits (200 total: two 100-USD deposits).
    assert s.get_entity("X").internal_state.cash == pytest.approx(200.0)


# ---------------------------------------------------- register_entity validation
class _Strat(BaseStrategy):
    def set_up(self):
        pass

    def predict(self):
        return []


@pytest.mark.core
def test_register_entity_rejects_non_base_entity():
    s = _Strat(debug=False, params=BaseStrategyParams())
    with pytest.raises(TypeError, match="BaseEntity"):
        s.register_entity(NamedEntity("X", "not_an_entity"))  # type: ignore[arg-type]


@pytest.mark.core
def test_register_entity_rejects_empty_name():
    s = _Strat(debug=False, params=BaseStrategyParams())
    with pytest.raises(ValueError):
        s.register_entity(NamedEntity("", SimpleSpotExchange(trading_fee=0.0)))


@pytest.mark.core
def test_register_entity_rejects_non_string_name():
    s = _Strat(debug=False, params=BaseStrategyParams())
    with pytest.raises(ValueError):
        s.register_entity(NamedEntity(42, SimpleSpotExchange(trading_fee=0.0)))  # type: ignore[arg-type]


@pytest.mark.core
def test_register_entity_rejects_duplicate_name():
    s = _Strat(debug=False, params=BaseStrategyParams())
    s.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))
    with pytest.raises(ValueError, match="already exists"):
        s.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))
