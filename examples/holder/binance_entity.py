from dataclasses import dataclass

from fractal.core.base import GlobalState
from fractal.core.entities import BaseSpotEntity, BaseSpotInternalState


@dataclass
class BinanceGlobalState(GlobalState):
    price: float = 0.0  # global state is BTC price


@dataclass
class BinanceInternalState(BaseSpotInternalState):
    """Inherits ``amount`` (BTC tokens) and ``cash`` (USD notional)."""
    pass


class BinanceSpot(BaseSpotEntity):
    """Toy spot exchange used by the holder example.

    Conforms to :class:`BaseSpotEntity`: ``buy`` takes ``amount_in_notional``,
    ``sell`` takes ``amount_in_product``.
    """

    def _initialize_states(self):
        self._global_state = BinanceGlobalState()
        self._internal_state = BinanceInternalState()

    def update_state(self, state: BinanceGlobalState) -> None:
        self._global_state = state

    @property
    def current_price(self) -> float:
        return self._global_state.price

    @property
    def balance(self) -> float:
        return self._internal_state.amount * self._global_state.price + self._internal_state.cash

    def action_buy(self, amount_in_notional: float) -> None:
        if amount_in_notional > self._internal_state.cash:
            raise ValueError(
                f"Not enough cash to buy: {amount_in_notional} > {self._internal_state.cash}"
            )
        self._internal_state.amount += amount_in_notional / self._global_state.price
        self._internal_state.cash -= amount_in_notional

    def action_sell(self, amount_in_product: float) -> None:
        if amount_in_product > self._internal_state.amount:
            raise ValueError(
                f"Not enough product to sell: {amount_in_product} > {self._internal_state.amount}"
            )
        self._internal_state.amount -= amount_in_product
        self._internal_state.cash += amount_in_product * self._global_state.price

    def action_deposit(self, amount_in_notional: float) -> None:
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        self._internal_state.cash -= amount_in_notional
