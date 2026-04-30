"""Validation behavior of :class:`Observation` and :class:`BaseStrategy`."""
from dataclasses import dataclass
from datetime import datetime

import pytest

from fractal.core.base import (BaseStrategy, BaseStrategyParams, GlobalState,
                               NamedEntity, Observation)
from fractal.core.entities import (SimpleSpotExchange,
                                   SimpleSpotExchangeGlobalState)


# -------------------------------------------------- Observation shape check
@dataclass
class _SomeState(GlobalState):
    price: float = 0.0


@pytest.mark.core
def test_observation_rejects_empty_states():
    with pytest.raises(ValueError):
        Observation(timestamp=datetime(2024, 1, 1), states={})


@pytest.mark.core
def test_observation_rejects_non_string_key():
    with pytest.raises(ValueError):
        Observation(timestamp=datetime(2024, 1, 1), states={42: _SomeState(price=1)})  # type: ignore[dict-item]


@pytest.mark.core
def test_observation_rejects_empty_string_key():
    with pytest.raises(ValueError):
        Observation(timestamp=datetime(2024, 1, 1), states={"": _SomeState(price=1)})


@pytest.mark.core
def test_observation_rejects_non_global_state_value():
    with pytest.raises(ValueError):
        Observation(
            timestamp=datetime(2024, 1, 1),
            states={"X": "this is not a GlobalState"},  # type: ignore[dict-item]
        )


@pytest.mark.core
def test_observation_accepts_valid_construction():
    obs = Observation(
        timestamp=datetime(2024, 1, 1),
        states={"X": _SomeState(price=100)},
    )
    assert obs.states["X"].price == 100


# -------------------------------------------------- BaseStrategy strict
class _TwoEntityStrategy(BaseStrategy):
    def set_up(self):
        self.register_entity(NamedEntity("A", SimpleSpotExchange(trading_fee=0.0)))
        self.register_entity(NamedEntity("B", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


class _LooseTwoEntityStrategy(_TwoEntityStrategy):
    STRICT_OBSERVATIONS = False


def _state(close: float) -> SimpleSpotExchangeGlobalState:
    return SimpleSpotExchangeGlobalState(open=close, high=close, low=close, close=close)


@pytest.mark.core
def test_strict_observation_raises_on_missing_registered_entity():
    """Default STRICT_OBSERVATIONS = True: observation missing 'B' must raise."""
    s = _TwoEntityStrategy(debug=False, params=BaseStrategyParams())
    obs = Observation(timestamp=datetime(2024, 1, 1), states={"A": _state(100)})
    with pytest.raises(ValueError, match="missing states for registered entities"):
        s.step(obs)


@pytest.mark.core
def test_loose_observation_allows_missing_registered_entity():
    """Subclass with STRICT_OBSERVATIONS = False: missing 'B' is OK."""
    s = _LooseTwoEntityStrategy(debug=False, params=BaseStrategyParams())
    obs = Observation(timestamp=datetime(2024, 1, 1), states={"A": _state(100)})
    s.step(obs)  # must not raise
    assert s.get_entity("A").global_state.close == 100


@pytest.mark.core
def test_observation_with_unknown_entity_always_raises():
    """Forward check is always on, regardless of strict mode."""
    s = _LooseTwoEntityStrategy(debug=False, params=BaseStrategyParams())
    obs = Observation(timestamp=datetime(2024, 1, 1), states={
        "A": _state(100), "B": _state(100), "GHOST": _state(100),
    })
    with pytest.raises(ValueError, match="not registered"):
        s.step(obs)


@pytest.mark.core
def test_full_observation_passes_strict_validation():
    s = _TwoEntityStrategy(debug=False, params=BaseStrategyParams())
    obs = Observation(timestamp=datetime(2024, 1, 1), states={
        "A": _state(100), "B": _state(200),
    })
    s.step(obs)  # must not raise
    assert s.get_entity("A").global_state.close == 100
    assert s.get_entity("B").global_state.close == 200
