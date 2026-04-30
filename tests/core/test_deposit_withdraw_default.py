"""``BaseEntity.action_deposit/withdraw`` default-raise contract.

The default implementation raises ``NotImplementedError`` so that an
entity that does not handle notional cash fails loudly when the user
tries to deposit/withdraw against it. Concrete entities holding cash
override these methods (almost all of them do).
"""
from dataclasses import dataclass

import pytest

from fractal.core.base import (Action, BaseEntity, GlobalState,
                               InternalState)


@dataclass
class _CashlessGlobalState(GlobalState):
    price: float = 0.0


@dataclass
class _CashlessInternalState(InternalState):
    units: float = 0.0


class _Cashless(BaseEntity):
    """Toy entity with no cash semantics — does not override deposit/withdraw."""
    def _initialize_states(self):
        self._global_state = _CashlessGlobalState()
        self._internal_state = _CashlessInternalState()

    def update_state(self, state):
        self._global_state = state

    @property
    def balance(self):
        return self._internal_state.units * self._global_state.price


@pytest.mark.core
def test_default_action_deposit_raises_not_implemented():
    e = _Cashless()
    with pytest.raises(NotImplementedError) as excinfo:
        e.action_deposit(1000)
    assert "_Cashless" in str(excinfo.value)


@pytest.mark.core
def test_default_action_withdraw_raises_not_implemented():
    e = _Cashless()
    with pytest.raises(NotImplementedError) as excinfo:
        e.action_withdraw(500)
    assert "_Cashless" in str(excinfo.value)


@pytest.mark.core
def test_execute_deposit_on_cashless_entity_propagates_not_implemented():
    """Routing through ``execute`` (the path strategies actually use)
    surfaces the same NotImplementedError, not a silent no-op."""
    e = _Cashless()
    with pytest.raises(NotImplementedError):
        e.execute(Action("deposit", {"amount_in_notional": 100}))
