"""Minimal generic constant-product (Uniswap-V2-style) LP entity.

Models the universal pattern of a 50/50 AMM LP position: the holder
mints LP tokens proportional to their share of the pool, accrues a
proportional share of pool fees on every step, and can close the
position to receive the current pro-rata pool value.

This entity intentionally does NOT model concentrated-liquidity (V3)
mechanics — for that use
:class:`fractal.core.entities.protocols.uniswap_v3_lp.UniswapV3LPEntity`.
"""
from dataclasses import dataclass

from fractal.core.base.entity import EntityException, GlobalState, InternalState
from fractal.core.entities.base.pool import BasePoolEntity


class SimplePoolException(EntityException):
    """Errors raised by :class:`SimplePoolEntity`."""


@dataclass
class SimplePoolGlobalState(GlobalState):
    """V2-style pool snapshot.

    Attributes:
        price: Spot price as ``token1 / token0``.
        tvl: Total value locked in the pool, in notional units.
        volume: Trading volume during the previous bar.
        fees: Fees collected by the pool during the previous bar.
        liquidity: Total LP tokens outstanding.
    """
    price: float = 0.0
    tvl: float = 0.0
    volume: float = 0.0
    fees: float = 0.0
    liquidity: float = 0.0


@dataclass
class SimplePoolInternalState(InternalState):
    """LP position state.

    Attributes:
        liquidity: LP tokens held by this entity.
        cash: Free cash (unallocated notional + accrued fees).
    """
    liquidity: float = 0.0
    cash: float = 0.0


class SimplePoolEntity(BasePoolEntity):
    """Constant-product LP entity with proportional fee accrual.

    Position lifecycle:

    * :meth:`action_open_position` deposits ``amount_in_notional`` of cash
      into the pool. After a trading fee, the deployed amount mints LP
      tokens proportionally to the holder's share of the pool's TVL.
    * Every :meth:`update_state` accrues
      ``share · global_state.fees`` into ``cash`` (LP yield).
    * :meth:`action_close_position` burns all LP tokens for the current
      pro-rata pool value (less the trading fee).

    Notation: balance is ``cash + share · tvl`` while a position is open
    (the pool's TVL already absorbs impermanent-loss because reserve
    composition shifts with price).
    """

    def __init__(self, *, trading_fee: float = 0.003) -> None:
        if trading_fee < 0:
            raise SimplePoolException(f"trading_fee must be >= 0, got {trading_fee}")
        self._trading_fee: float = float(trading_fee)
        super().__init__()

    _internal_state: SimplePoolInternalState
    _global_state: SimplePoolGlobalState

    def _initialize_states(self) -> None:
        self._internal_state = SimplePoolInternalState()
        self._global_state = SimplePoolGlobalState()

    @property
    def internal_state(self) -> SimplePoolInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> SimplePoolGlobalState:  # type: ignore[override]
        return self._global_state

    # --------------------------------------------------------- account
    def action_deposit(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimplePoolException(
                f"deposit amount must be >= 0, got {amount_in_notional}"
            )
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimplePoolException(
                f"withdraw amount must be >= 0, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise SimplePoolException(
                f"withdraw exceeds free cash: {amount_in_notional} > "
                f"{self._internal_state.cash}"
            )
        self._internal_state.cash -= amount_in_notional

    # --------------------------------------------------------- position
    @property
    def is_position(self) -> bool:
        return self._internal_state.liquidity > 0

    def action_open_position(self, amount_in_notional: float) -> None:
        """Mint LP tokens proportional to ``amount_in_notional``.

        Raises:
            SimplePoolException: If a position already exists, if there
                is not enough free cash, or if the pool's TVL/liquidity
                are non-positive.
        """
        if self.is_position:
            raise SimplePoolException("position already open; close it first")
        if amount_in_notional <= 0:
            raise SimplePoolException(
                f"open amount must be > 0, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise SimplePoolException(
                f"insufficient cash to open: {amount_in_notional} > "
                f"{self._internal_state.cash}"
            )
        if self._global_state.tvl <= 0 or self._global_state.liquidity <= 0:
            raise SimplePoolException(
                f"pool tvl/liquidity must be positive, got tvl={self._global_state.tvl}, "
                f"liquidity={self._global_state.liquidity}"
            )
        self._internal_state.cash -= amount_in_notional
        deployed = amount_in_notional * (1.0 - self._trading_fee)
        share = deployed / self._global_state.tvl
        self._internal_state.liquidity = share * self._global_state.liquidity

    def action_close_position(self) -> None:
        """Burn all LP tokens for the current pro-rata pool value."""
        if not self.is_position:
            return
        share = self._internal_state.liquidity / self._global_state.liquidity
        proceeds = share * self._global_state.tvl * (1.0 - self._trading_fee)
        self._internal_state.cash += proceeds
        self._internal_state.liquidity = 0.0

    # --------------------------------------------------------- readouts
    @property
    def share(self) -> float:
        """Fraction of total pool LP tokens held by this entity."""
        if self._global_state.liquidity <= 0:
            return 0.0
        return self._internal_state.liquidity / self._global_state.liquidity

    @property
    def balance(self) -> float:
        """Total equity in notional: ``cash + share · tvl``."""
        return self._internal_state.cash + self.share * self._global_state.tvl

    # --------------------------------------------------------- lifecycle
    def update_state(self, state: SimplePoolGlobalState) -> None:
        """Apply pool snapshot and accrue proportional fees from the prior bar.

        Fees credited: ``share_at_new_state · state.fees``. Approximation —
        the share at the start of the bar may have been slightly different,
        but on a hourly/daily resolution the drift is negligible.
        """
        self._global_state = state
        if self._internal_state.liquidity > 0 and state.liquidity > 0:
            share = self._internal_state.liquidity / state.liquidity
            self._internal_state.cash += share * state.fees
