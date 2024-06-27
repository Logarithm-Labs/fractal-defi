from dataclasses import dataclass

from fractal.core.base import (
    GlobalState, InternalState,
)
from fractal.core.entities import BaseSpotEntity


@dataclass
class BinanceGlobalState(GlobalState):
    price: float = 0.0 # global state is BTC price


@dataclass
class BinanceInternalState(InternalState):
    amount: float = 0.0 # internally we manage current BTC amount
    cash: float = 0.0   # internally we manage current cash amount


class BinanceSpot(BaseSpotEntity):

    def _initialize_states(self):
        self._global_state = BinanceGlobalState()
        self._internal_state = BinanceInternalState()

    # we can update state of the binance
    def update_state(self, state: BinanceGlobalState, *args, **kwargs) -> None:
        self._global_state = state

    @property
    def balance(self) -> float:
        return self._internal_state.amount * self._global_state.price + self._internal_state.cash

	# we can buy BTC
    def action_buy(self, amount: float) -> None:
        amount_in_notional = amount * self._global_state.price
        if amount_in_notional > self._internal_state.cash:
            raise ValueError(f'Not enough cash to buy {amount} BTC')
        self._internal_state.amount += amount
        self._internal_state.cash -= amount_in_notional

	# we can sell BTC
    def action_sell(self, amount: float) -> None:
        if amount > self._internal_state.amount:
            raise ValueError(f'Not enough BTC to sell {amount}')
        self._internal_state.amount -= amount
        self._internal_state.cash += amount * self._global_state.price

    def action_deposit(self, amount_in_notional: float) -> None:
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        self._internal_state.cash -= amount_in_notional
