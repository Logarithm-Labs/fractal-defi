"""L1 tests for :class:`BaseFixedTermEntity` — the maturity contract shared
by Pendle PT and Boros entities."""
import math
from dataclasses import dataclass

import pytest

from fractal.core.base.entity import EntityException, InternalState
from fractal.core.base.time import SECONDS_PER_DAY, SECONDS_PER_YEAR
from fractal.core.entities.base import BaseFixedTermEntity, BaseFixedTermGlobalState


class _TermException(EntityException):
    pass


@dataclass
class _State(BaseFixedTermGlobalState):
    price: float = 1.0


@dataclass
class _Internal(InternalState):
    cash: float = 0.0


class _TermEntity(BaseFixedTermEntity):
    """Minimal concrete fixed-term entity: one guarded action, one hook counter."""
    _exception_cls = _TermException

    def __init__(self):
        self.matured_hook_calls = 0
        super().__init__()

    def _initialize_states(self):
        self._internal_state = _Internal()
        self._global_state = _State()

    def update_state(self, state: _State) -> None:
        self._validate_term(state)
        self._global_state = state
        self._check_maturity()

    def _check_maturity(self) -> None:
        if self.is_matured:
            self.matured_hook_calls += 1

    def action_trade(self, amount_in_notional: float) -> None:
        self._require_not_matured("trade")
        self._internal_state.cash += amount_in_notional

    @property
    def balance(self) -> float:
        return self._internal_state.cash


@pytest.mark.core
def test_year_constant_is_act_365():
    assert SECONDS_PER_DAY == 86_400
    assert SECONDS_PER_YEAR == 31_536_000


@pytest.mark.core
def test_default_state_is_matured_and_trading_raises():
    """A state that forgot ``seconds_to_expiry`` reads as matured; trades fail loudly."""
    entity = _TermEntity()
    assert entity.is_matured
    assert entity.seconds_to_expiry == 0.0
    with pytest.raises(_TermException, match="after expiry"):
        entity.action_trade(1.0)


@pytest.mark.core
def test_readouts_follow_the_applied_state():
    entity = _TermEntity()
    entity.update_state(_State(seconds_to_expiry=SECONDS_PER_YEAR / 2))
    assert not entity.is_matured
    assert entity.years_to_expiry == pytest.approx(0.5)
    entity.action_trade(3.0)
    assert entity.balance == 3.0


@pytest.mark.core
def test_expiry_must_be_non_increasing():
    """Time to expiry can shrink or repeat, never grow: unsorted feeds are rejected."""
    entity = _TermEntity()
    entity.update_state(_State(seconds_to_expiry=100.0))
    entity.update_state(_State(seconds_to_expiry=100.0))  # repeated bar ok
    entity.update_state(_State(seconds_to_expiry=50.0))
    with pytest.raises(_TermException, match="increased"):
        entity.update_state(_State(seconds_to_expiry=60.0))


@pytest.mark.core
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_expiry_rejected(bad):
    entity = _TermEntity()
    with pytest.raises(_TermException, match="finite"):
        entity.update_state(_State(seconds_to_expiry=bad))


@pytest.mark.core
def test_maturity_hook_fires_at_and_after_zero_without_mutating_input():
    """Negative values are matured too; the entity leaves the caller's state untouched."""
    entity = _TermEntity()
    entity.update_state(_State(seconds_to_expiry=10.0))
    assert entity.matured_hook_calls == 0
    state = _State(seconds_to_expiry=-5.0)
    entity.update_state(state)
    assert entity.matured_hook_calls == 1
    assert entity.years_to_expiry == 0.0
    assert state.seconds_to_expiry == -5.0
    with pytest.raises(_TermException):
        entity.action_trade(1.0)
