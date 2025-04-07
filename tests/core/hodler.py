from dataclasses import dataclass

import pytest

from fractal.core.base import (Action, ActionToTake, BaseEntity, BaseStrategy,
                               BaseStrategyParams, GlobalState, InternalState,
                               NamedEntity, Observation)


@dataclass
class HodlerGlobalState(GlobalState):
    """
    Testing global state.
    """
    price: float = 0.0


@dataclass
class HodlerInternalState(InternalState):
    """
    Testing internal state.
    """
    amount: float = 0.0


class Hodler(BaseEntity):
    """
    Testing entity. Simple asset that holds an amount of money.
    """    
    def _initialize_states(self):
        self._global_state = HodlerGlobalState()
        self._internal_state = HodlerInternalState()

    def update_state(self, state: HodlerGlobalState, *args, **kwargs) -> None:
        self._global_state = state

    @property
    def balance(self) -> float:
        return self._internal_state.amount * self._global_state.price

    def action_buy(self, amount: float) -> None:
        self._internal_state.amount += amount

    def action_sell(self, amount: float) -> None:
        self._internal_state.amount -= amount



@dataclass
class HodlerParams(BaseStrategyParams):
    """
    Testing strategy parameters.
    """
    BUY_THRESHOLD: float = 3000.0


class HodlerStrategy(BaseStrategy):
    """
    Testing strategy. Simple strategy that buys and sells based on the price.
    """
    def __init__(self, debug: bool, params: HodlerParams):
        super().__init__(debug=debug, params=params)

    def set_up(self, *args, **kwargs):
        self.register_entity(
            NamedEntity('stupid_hodler', Hodler())
        )

    def predict(self, *args, **kwargs):
        if self._entities['stupid_hodler'].global_state.price > self._params.BUY_THRESHOLD:
            return [
                ActionToTake(
                    entity_name='stupid_hodler',
                    action=Action(action='buy', args={'amount': 500})
                )
            ]
        else:
            return [
                ActionToTake(
                    entity_name='stupid_hodler',
                    action=Action(action='sell', args={'amount': 500})
                )
            ]

    def estimate_predict(self, next_observation: Observation):
        if next_observation.states['stupid_hodler'].price > self._params.BUY_THRESHOLD:
            return [
                ActionToTake(
                    entity_name='stupid_hodler',
                    action=Action(action='buy', args={'amount': 500})
                )
            ]
        else:
            return [
                ActionToTake(
                    entity_name='stupid_hodler',
                    action=Action(action='sell', args={'amount': 500})
                )
            ]
