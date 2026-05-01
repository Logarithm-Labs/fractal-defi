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

from fractal.core.base.entity import EntityException, InternalState
from fractal.core.entities.base.pool import BasePoolEntity, BasePoolGlobalState


class SimplePoolException(EntityException):
    """Errors raised by :class:`SimplePoolEntity`."""


@dataclass
class SimplePoolGlobalState(BasePoolGlobalState):
    """V2-style pool snapshot.

    Inherits ``tvl, volume, fees, liquidity, price`` from
    :class:`BasePoolGlobalState`. ``price`` is interpreted as
    ``token1 / token0``.
    """


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

    * :meth:`action_open_position` zaps in: half of ``amount_in_notional``
      stays as the stable side, the other half is swapped to the volatile
      side, paying ``effective_fee_rate`` on the **swapped half only**.
      LP tokens minted proportionally to the post-fee deployed value.
    * Every :meth:`update_state` accrues
      ``share · global_state.fees`` into ``cash`` (LP yield).
    * :meth:`action_close_position` burns LP tokens; stable half returns at
      full value, volatile half swaps back to notional at
      ``effective_fee_rate`` on that half.

    Notation: balance is ``cash + share · tvl`` while a position is open
    (the pool's TVL already absorbs impermanent-loss because reserve
    composition shifts with price).
    """

    def __init__(self, *, pool_fee_rate: float = 0.003, slippage_pct: float = 0.0) -> None:
        if pool_fee_rate < 0:
            raise SimplePoolException(f"pool_fee_rate must be >= 0, got {pool_fee_rate}")
        if slippage_pct < 0:
            raise SimplePoolException(f"slippage_pct must be >= 0, got {slippage_pct}")
        self._pool_fee_rate: float = float(pool_fee_rate)
        self._slippage_pct: float = float(slippage_pct)
        super().__init__()

    @property
    def effective_fee_rate(self) -> float:
        """Combined ``pool_fee_rate + slippage_pct``."""
        return self._pool_fee_rate + self._slippage_pct

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
        # Only the swapped half pays fee. ``deployed`` value entering LP =
        # half (no fee, stable side) + half × (1-fee) (post-fee volatile side)
        # = amount × (1 - fee/2).
        deployed = amount_in_notional * (1.0 - self.effective_fee_rate / 2)
        share = deployed / self._global_state.tvl
        self._internal_state.liquidity = share * self._global_state.liquidity

    def action_close_position(self) -> None:
        """Burn LP tokens. Stable half returns at full value, volatile half swaps with fee."""
        if not self.is_position:
            return
        share = self._internal_state.liquidity / self._global_state.liquidity
        # Position value at close = share × tvl. Half is stable (returns at full
        # value), half is volatile (swap → notional at effective_fee_rate).
        proceeds = share * self._global_state.tvl * (1.0 - self.effective_fee_rate / 2)
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
