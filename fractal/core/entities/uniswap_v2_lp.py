from dataclasses import dataclass

import numpy as np

from fractal.core.base.entity import EntityException
from fractal.core.entities.pool import BasePoolEntity


@dataclass
class UniswapV2LPGlobalState:
    """
    Represents the global state of the UniswapV2 LP entity.

    Attributes:
        tvl (float): The total value locked.
        volume (float): The trading volume.
        fees (float): The trading fees.
        liquidity (float): The pool liquidity.
        price (float): The pool price [token1 / token0].
    """

    tvl: float = 0.0
    volume: float = 0.0
    fees: float = 0.0
    liquidity: float = 0.0
    price: float = 0.0


@dataclass
class UniswapV2LPInternalState:
    """
    Represents the internal state of an UniswapV2 LP entity.

    Attributes:
        token0_amount (float): The amount of token0.
        token1_amount (float): The amount of token1.
        price_init (float): The position initial price.
        liquidity (float): The position liquidity.
        cash (float): The cash balance.
    """
    token0_amount: float = 0.0
    token1_amount: float = 0.0
    price_init: float = 0.0
    liquidity: float = 0.0
    cash: float = 0.0


@dataclass
class UniswapV2LPConfig:
    """
    Represents the configuration of an UniswapV2 LP entity.

    Attributes:
        fees_rate (float): The fees rate.
        token0_decimals (int): The token0 decimals.
        token1_decimals (int): The token1 decimals.
        trading_fee (float): The trading fee.
    """
    fees_rate: float = 0.005
    token0_decimals: int = 18
    token1_decimals: int = 18
    trading_fee: float = 0.003


class UniswapV2LPEntity(BasePoolEntity):
    """
    Represents an Uniswap V2-like LP entity.

    It maintains exact 50-50 position of token0 and token1 in the pool.
    """
    def __init__(self, config: UniswapV2LPConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_position = False
        self.fees_rate = config.fees_rate
        self.token0_decimals = config.token0_decimals
        self.token1_decimals = config.token1_decimals
        self.trading_fee = config.trading_fee

    def _initialize_states(self):
        self._internal_state: UniswapV2LPInternalState = UniswapV2LPInternalState()
        self._global_state: UniswapV2LPGlobalState = UniswapV2LPGlobalState()

    def action_deposit(self, amount_in_notional: float) -> None:
        """
        Deposit funds into the LP entity.

        Args:
            amount_in_notional (float): The amount to deposit.
        """
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        """
        Withdraw funds from the LP entity.

        Args:
            amount_in_notional (float): The amount to withdraw.
        """
        if amount_in_notional > self._internal_state.cash:
            raise EntityException("Insufficient funds to withdraw.")
        self._internal_state.cash -= amount_in_notional

    def action_open_position(self, amount_in_notional: float) -> None:
        """
        Open a position in the LP entity.

        Args:
            amount_in_notional (float): The amount to invest.
        """
        if self.is_position:
            raise EntityException("Position already open.")
        if amount_in_notional > self._internal_state.cash:
            raise EntityException("Insufficient funds to open position.")
        self.is_position = True
        self._internal_state.cash -= amount_in_notional
        amount_in_position = amount_in_notional * (1 - self.trading_fee)
        x = amount_in_position / 2
        y = amount_in_position / 2 / self._global_state.price
        self._internal_state.token0_amount = x
        self._internal_state.token1_amount = y
        self._internal_state.price_init = self._global_state.price
        self._internal_state.liquidity = (amount_in_position ** 2) *\
                                         (10 ** (self.token0_decimals / self.token1_decimals))

    def action_close_position(self):
        """
        Close the position in the LP entity.
        """
        if not self.is_position:
            raise EntityException("No position to close.")
        cash = self.balance * (1 - self.trading_fee)
        self.is_position = False
        self._internal_state = UniswapV2LPInternalState(cash=cash)

    def update_state(self, state: UniswapV2LPGlobalState) -> None:
        """
        Update the state of the LP entity.

        1. Update the global state.
        2. Update token0 and token1 amounts following AMM formula.
        Args:
            state (UniswapV2LPGlobalState): The state of the pool.
        """
        self._global_state = state
        p = state.price
        if self.is_position:
            self._internal_state.token0_amount = np.sqrt(self._internal_state.liquidity * p)
            self._internal_state.token1_amount = np.sqrt(self._internal_state.liquidity / p)
        self._internal_state.cash += self.calculate_fees()

    @property
    def balance(self) -> float:
        """
        Returns the balance of the LP entity.

        Returns:
            float: The balance of the LP entity.
        """
        if not self.is_position:
            return self._internal_state.cash
        return (
            self._internal_state.token0_amount
            + self._internal_state.token1_amount * self._global_state.price
            + self._internal_state.cash
        )

    def calculate_fees(self) -> float:
        """
        Calculate fees for position
        Returns:
            float: acc fees for position
        """
        if not self.is_position:
            return 0

        return (self._internal_state.liquidity / self._global_state.liquidity) * self._global_state.fees
