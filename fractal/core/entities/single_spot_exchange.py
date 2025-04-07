from dataclasses import dataclass
from typing import Optional

from fractal.core.base import GlobalState, InternalState
from fractal.core.entities import BaseSpotEntity


@dataclass
class SingleSpotExchangeGlobalState(GlobalState):
    """
    Klines Data
    """
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0


@dataclass
class SingleSpotExchangeInternalState(InternalState):
    amount: float = 0.0  # hold amount of tokens
    cash: float = 0.0   # balance in notional (USD)


class SingleSpotExchange(BaseSpotEntity):

    def __init__(self, trading_fee: Optional[float] = 0.005):
        """
        Single Spot Exchange Entity

        Args:
            trading_fee (float, optional): Fee for the trade that is collected
            in side (in tokens for buy and in notional for sell).
            Defaults to 0.005.
        """
        self._trading_fee: float = trading_fee
        super().__init__()

    def _initialize_states(self):
        self._global_state: SingleSpotExchangeGlobalState = SingleSpotExchangeGlobalState()
        self._internal_state: SingleSpotExchangeInternalState = SingleSpotExchangeInternalState()

    def update_state(self, state: SingleSpotExchangeGlobalState, *args, **kwargs) -> None:
        self._global_state = state

    @property
    def balance(self) -> float:
        """
        Returns the balance in notional.
        balance = amount * close + cash
        """
        return self._internal_state.amount * self._global_state.close + self._internal_state.cash

    def action_buy(self, amount: float) -> None:
        """
        Buy tokens.

        Args:
            amount (float): amount of tokens to buy
        """
        amount_in_notional = amount * self._global_state.close
        if amount_in_notional > self._internal_state.cash:
            raise ValueError(f'Not enough cash to buy {amount}')
        self._internal_state.amount += amount * (1 - self._trading_fee)
        self._internal_state.cash -= amount_in_notional

    def action_sell(self, amount: float) -> None:
        """
        Sell tokens.

        Args:
            amount (float): amount of tokens to sell
        """
        if amount > self._internal_state.amount:
            raise ValueError(f'Not enough balance to sell {amount}')
        self._internal_state.amount -= amount
        self._internal_state.cash += amount * self._global_state.close * (1 - self._trading_fee)

    def action_deposit(self, amount_in_notional: float) -> None:
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        self._internal_state.cash -= amount_in_notional
