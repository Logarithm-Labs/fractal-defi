from dataclasses import dataclass, field

from fractal.core.base import (
    GlobalState, InternalState,
)
from fractal.core.entities import BaseSpotEntity

# Глобальное состояние используется для хранения данных, которые являются общими для всех сущностей.
# Например, цена BTC.
# Эти данные задаются заранее и не меняются в течение всего времени работы программы.
@dataclass
class BinanceGlobalState(GlobalState):
    price: float = 0.0 # global state is BTC price


@dataclass
class BinanceInternalState(InternalState):
    amount: float = 0.0 # internally we manage current BTC amount
    cash: float = 0.0   # internally we manage current cash amount
    sma_buffer: list[float] = field(default_factory=list)


class BinanceSpot(BaseSpotEntity):

    def _initialize_states(self):
        self._global_state = BinanceGlobalState()
        self._internal_state = BinanceInternalState()

    # we can update staВоte of the binance
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
    
    def calculate_sma(self, period: int) -> float:
        self._internal_state.sma_buffer.append(self._global_state.price)
        self._internal_state.sma_buffer = self._internal_state.sma_buffer[-period:]

        if len(self._internal_state.sma_buffer) < period:
            return 0.0
        return sum(self._internal_state.sma_buffer) / period
