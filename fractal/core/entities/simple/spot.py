"""Minimal generic spot exchange with OHLCV market state.

Counterpart to :class:`SimplePerpEntity` on the spot side. The entity
holds product tokens and notional cash, executes buys/sells against the
last *close* price from a kline-style global state, and charges a
two-sided trading fee.

Sizing follows the :class:`BaseSpotEntity` convention:

* ``action_buy(amount_in_notional)`` — spend X of notional cash.
* ``action_sell(amount_in_product)`` — sell X of product.

Trading fee is applied to the **received** asset on each side: buying
pays the fee out of the product received; selling pays the fee out of
the notional received. ``balance`` is computed at the close price.
"""
from dataclasses import dataclass
from typing import Optional

from fractal.core.base.entity import EntityException, GlobalState
from fractal.core.entities.base.spot import BaseSpotEntity, BaseSpotInternalState


class SimpleSpotExchangeException(EntityException):
    """Errors raised by :class:`SimpleSpotExchange`."""


@dataclass
class SimpleSpotExchangeGlobalState(GlobalState):
    """OHLCV bar from the price feed. Trades execute at ``close``."""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0


@dataclass
class SimpleSpotExchangeInternalState(BaseSpotInternalState):
    """Spot account: inherits ``amount`` (product) and ``cash`` (notional)."""
    pass


class SimpleSpotExchange(BaseSpotEntity):
    """Single-asset spot exchange driven by OHLCV bars."""

    def __init__(self, trading_fee: Optional[float] = 0.005) -> None:
        """
        Args:
            trading_fee: Fee charged on the received asset on each trade.
        """
        if trading_fee is None or trading_fee < 0:
            raise SimpleSpotExchangeException(
                f"trading_fee must be >= 0, got {trading_fee}"
            )
        self._trading_fee: float = float(trading_fee)
        super().__init__()

    _internal_state: SimpleSpotExchangeInternalState
    _global_state: SimpleSpotExchangeGlobalState

    def _initialize_states(self) -> None:
        self._global_state = SimpleSpotExchangeGlobalState()
        self._internal_state = SimpleSpotExchangeInternalState()

    def update_state(self, state: SimpleSpotExchangeGlobalState) -> None:
        self._global_state = state

    @property
    def current_price(self) -> float:
        return self._global_state.close

    @property
    def balance(self) -> float:
        """Total in notional: ``amount · close + cash``."""
        return self._internal_state.amount * self._global_state.close + self._internal_state.cash

    def action_buy(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimpleSpotExchangeException(
                f"buy amount_in_notional must be >= 0, got {amount_in_notional}"
            )
        if self._global_state.close <= 0:
            raise SimpleSpotExchangeException(
                f"cannot trade at non-positive close price {self._global_state.close}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise SimpleSpotExchangeException(
                f"not enough cash to buy {amount_in_notional}: "
                f"available {self._internal_state.cash}"
            )
        product_received = amount_in_notional * (1 - self._trading_fee) / self._global_state.close
        self._internal_state.amount += product_received
        self._internal_state.cash -= amount_in_notional

    def action_sell(self, amount_in_product: float) -> None:
        if amount_in_product < 0:
            raise SimpleSpotExchangeException(
                f"sell amount_in_product must be >= 0, got {amount_in_product}"
            )
        if amount_in_product > self._internal_state.amount:
            raise SimpleSpotExchangeException(
                f"not enough product to sell {amount_in_product}: "
                f"available {self._internal_state.amount}"
            )
        notional_received = amount_in_product * self._global_state.close * (1 - self._trading_fee)
        self._internal_state.amount -= amount_in_product
        self._internal_state.cash += notional_received

    def action_deposit(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimpleSpotExchangeException(
                f"deposit amount must be >= 0, got {amount_in_notional}"
            )
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimpleSpotExchangeException(
                f"withdraw amount must be >= 0, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise SimpleSpotExchangeException(
                f"not enough cash to withdraw {amount_in_notional}: "
                f"available {self._internal_state.cash}"
            )
        self._internal_state.cash -= amount_in_notional
