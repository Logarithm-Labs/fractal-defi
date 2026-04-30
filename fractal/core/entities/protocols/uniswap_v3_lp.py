from dataclasses import dataclass
from typing import Tuple

import numpy as np

from fractal.core.base.entity import EntityException, InternalState
from fractal.core.entities.models.uniswap_v3_fees import (estimate_fee,
                                                          get_liquidity_delta)
from fractal.core.entities.base.pool import BasePoolEntity, BasePoolGlobalState


@dataclass
class UniswapV3LPGlobalState(BasePoolGlobalState):
    """V3-style pool snapshot.

    Inherits ``tvl, volume, fees, liquidity, price`` from
    :class:`BasePoolGlobalState`. ``liquidity`` here is the pool's
    aggregate ``L`` (active-tick liquidity), not LP-token count.
    ``price`` is **notional per non-notional unit** (see
    :class:`UniswapV3LPConfig`).
    """


@dataclass
class UniswapV3LPInternalState(InternalState):
    """Internal state of a UniswapV3 LP position.

    Attributes:
        token0_amount (float): Current on-chain token0 amount in position.
        token1_amount (float): Current on-chain token1 amount in position.
        entry_token0_amount (float): Token0 in position at open — for hodl/IL.
        entry_token1_amount (float): Token1 in position at open — for hodl/IL.
        price_init (float): Pool price at position open.
        price_lower (float): Position lower price bound.
        price_upper (float): Position upper price bound.
        liquidity (float): Position ``L`` parameter.
        cash (float): Free notional cash held by entity.
    """
    token0_amount: float = 0.0
    token1_amount: float = 0.0
    entry_token0_amount: float = 0.0
    entry_token1_amount: float = 0.0
    price_init: float = 0.0
    price_lower: float = 0.0
    price_upper: float = 0.0
    liquidity: float = 0.0
    cash: float = 0.0


@dataclass
class UniswapV3LPConfig:
    """V3 LP entity configuration.

    Attributes:
        pool_fee_rate (float): Pool's swap-fee tier (e.g. ``0.003`` = 30bps).
            Charged on the volatile-side swap during zap-in
            (``action_open_position``) and zap-out (``action_close_position``).
            Only the swapped portion incurs the fee — for a V3 position, the
            swap portion depends on the range relative to current price.
        slippage_pct (float): Additional execution-cost on top of pool fee.
            Captures slippage / MEV. Default ``0.0``.
        token0_decimals (int): Token0 decimals (used by V3 fees model).
        token1_decimals (int): Token1 decimals.
        notional_side (str): Which on-chain slot — ``"token0"`` or
            ``"token1"`` — is the notional/stable side. Default ``"token0"``.
            Pass ``GlobalState.price`` and ``price_lower`` / ``price_upper``
            in the same convention: **notional per non-notional unit**.
    """
    pool_fee_rate: float = 0.003
    slippage_pct: float = 0.0
    token0_decimals: int = 18
    token1_decimals: int = 18
    notional_side: str = "token0"


class UniswapV3LPEntity(BasePoolEntity):
    """Uniswap V3 LP entity (concentrated liquidity).

    Position lifecycle:

    * :meth:`action_open_position` (zap-in) deposits ``amount_in_notional``
      into a position over ``[price_lower, price_upper]``. The optimal
      stable/volatile split for the range is computed; the volatile portion
      is swapped from notional, paying ``effective_fee_rate`` on the swap.
      Edge cases:

      * ``price ≤ price_lower`` (range above current price) — full
        notional is swapped to volatile, fee on full amount.
      * ``price ≥ price_upper`` (range below current price) — no swap
        needed, no fee, position holds only stable.

      Any leftover stable that couldn't be deposited at the V3 ratio
      (typically ``stable_pre × fee``) returns to cash.
    * :meth:`update_state` rebalances ``token0_amount`` / ``token1_amount``
      following the V3 in-range / out-of-range formulas, and accrues
      pro-rata pool swap fees into cash.
    * :meth:`action_close_position` (zap-out) burns the position. Stable
      leg returns at full value; volatile leg swaps back at
      ``effective_fee_rate``.

    Lower-level pair-based mint / burn are exposed as ``_open_from_pair``
    and ``_close_to_pair``.
    """

    _internal_state: UniswapV3LPInternalState
    _global_state: UniswapV3LPGlobalState

    def __init__(self, config: UniswapV3LPConfig, *args, **kwargs):
        if config.notional_side not in ("token0", "token1"):
            raise EntityException(
                f"notional_side must be 'token0' or 'token1', got {config.notional_side!r}"
            )
        if config.pool_fee_rate < 0:
            raise EntityException(
                f"pool_fee_rate must be >= 0, got {config.pool_fee_rate}"
            )
        if config.slippage_pct < 0:
            raise EntityException(
                f"slippage_pct must be >= 0, got {config.slippage_pct}"
            )
        # Set config BEFORE super so any subclass override of
        # ``_initialize_states`` can rely on these.
        self.pool_fee_rate: float = config.pool_fee_rate
        self.slippage_pct: float = config.slippage_pct
        self.token0_decimals: int = config.token0_decimals
        self.token1_decimals: int = config.token1_decimals
        self.notional_side: str = config.notional_side
        super().__init__(*args, **kwargs)

    def _initialize_states(self):
        self._internal_state = UniswapV3LPInternalState()
        self._global_state = UniswapV3LPGlobalState()

    @property
    def internal_state(self) -> UniswapV3LPInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> UniswapV3LPGlobalState:  # type: ignore[override]
        return self._global_state

    @property
    def is_position(self) -> bool:
        """Whether an LP position is currently open (derived from ``liquidity > 0``)."""
        return self._internal_state.liquidity > 0

    @property
    def effective_fee_rate(self) -> float:
        """Combined ``pool_fee_rate + slippage_pct`` — applied on the swapped portion."""
        return self.pool_fee_rate + self.slippage_pct

    @property
    def is_in_range(self) -> bool:
        """Whether the current pool price sits inside the position's range.

        Returns ``False`` when no position is open.
        """
        if not self.is_position:
            return False
        return (
            self._internal_state.price_lower
            < self._global_state.price
            < self._internal_state.price_upper
        )

    # --- Slot-aware accessors (hide which on-chain slot holds the stable side).
    @property
    def stable_amount(self) -> float:
        if self.notional_side == "token0":
            return self._internal_state.token0_amount
        return self._internal_state.token1_amount

    @property
    def volatile_amount(self) -> float:
        if self.notional_side == "token0":
            return self._internal_state.token1_amount
        return self._internal_state.token0_amount

    @property
    def entry_stable_amount(self) -> float:
        if self.notional_side == "token0":
            return self._internal_state.entry_token0_amount
        return self._internal_state.entry_token1_amount

    @property
    def entry_volatile_amount(self) -> float:
        if self.notional_side == "token0":
            return self._internal_state.entry_token1_amount
        return self._internal_state.entry_token0_amount

    def _set_position_amounts(self, stable: float, volatile: float) -> None:
        if self.notional_side == "token0":
            self._internal_state.token0_amount = stable
            self._internal_state.token1_amount = volatile
        else:
            self._internal_state.token1_amount = stable
            self._internal_state.token0_amount = volatile

    def _set_entry_amounts(self, stable: float, volatile: float) -> None:
        if self.notional_side == "token0":
            self._internal_state.entry_token0_amount = stable
            self._internal_state.entry_token1_amount = volatile
        else:
            self._internal_state.entry_token1_amount = stable
            self._internal_state.entry_token0_amount = volatile

    # --- account-level
    def action_deposit(self, amount_in_notional: float) -> None:
        """Deposit notional cash into the entity."""
        if amount_in_notional < 0:
            raise EntityException(
                f"deposit amount must be >= 0, got {amount_in_notional}"
            )
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Withdraw notional cash from the entity."""
        if amount_in_notional < 0:
            raise EntityException(
                f"withdraw amount must be >= 0, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise EntityException("Insufficient funds to withdraw.")
        self._internal_state.cash -= amount_in_notional

    # --- pair-level helpers (no swap, no fee)
    def _open_from_pair(
        self,
        token0_amount: float,
        token1_amount: float,
        price_lower: float,
        price_upper: float,
    ) -> Tuple[float, float]:
        """Mint V3 position from on-chain ``(token0, token1)`` over ``[pl, pu]``.

        No swap, no fee. Internal/advanced — use ``action_open_position`` for
        zap-in semantics. Returns ``(token0_leftover, token1_leftover)``.
        """
        if self.is_position:
            raise EntityException("Position already open.")
        if price_lower >= price_upper:
            raise EntityException(
                f"price_lower must be less than price_upper - {price_lower} >= {price_upper}"
            )
        if price_lower <= 0 or price_upper <= 0:
            raise EntityException("price bounds must be positive")
        if self._global_state.price <= 0:
            raise EntityException(
                f"price must be > 0, got {self._global_state.price}"
            )
        if token0_amount < 0 or token1_amount < 0:
            raise EntityException(
                f"token amounts must be >= 0, got token0={token0_amount}, token1={token1_amount}"
            )

        if self.notional_side == "token0":
            stable, volatile = token0_amount, token1_amount
        else:
            stable, volatile = token1_amount, token0_amount

        p = self._global_state.price
        sqp = p**0.5
        sqpl = price_lower**0.5
        sqpu = price_upper**0.5

        if p <= price_lower:
            # Above range — only volatile leg counts; stable returned as leftover.
            stable_factor = 0.0
            volatile_factor = 1 / sqpl - 1 / sqpu
            L = volatile / volatile_factor if volatile > 0 else 0.0
        elif p >= price_upper:
            # Below range — only stable leg counts; volatile returned as leftover.
            stable_factor = sqpu - sqpl
            volatile_factor = 0.0
            L = stable / stable_factor if stable > 0 else 0.0
        else:
            # In range — limited by smaller side.
            stable_factor = sqp - sqpl
            volatile_factor = 1 / sqp - 1 / sqpu
            L_stable = stable / stable_factor if stable > 0 else float('inf')
            L_volatile = volatile / volatile_factor if volatile > 0 else float('inf')
            L = min(L_stable, L_volatile)

        if L <= 0 or not np.isfinite(L):
            raise EntityException("Insufficient amounts to open position.")

        stable_used = L * stable_factor
        volatile_used = L * volatile_factor

        self._set_position_amounts(stable_used, volatile_used)
        self._set_entry_amounts(stable_used, volatile_used)
        self._internal_state.price_init = p
        self._internal_state.price_lower = price_lower
        self._internal_state.price_upper = price_upper
        self._internal_state.liquidity = L

        stable_leftover = stable - stable_used
        volatile_leftover = volatile - volatile_used

        if self.notional_side == "token0":
            return stable_leftover, volatile_leftover
        return volatile_leftover, stable_leftover

    def _close_to_pair(self) -> Tuple[float, float]:
        """Burn V3 LP, return current on-chain ``(token0, token1)`` amounts. No swap."""
        if not self.is_position:
            raise EntityException("No position to close.")

        token0_back = self._internal_state.token0_amount
        token1_back = self._internal_state.token1_amount

        self._internal_state.token0_amount = 0.0
        self._internal_state.token1_amount = 0.0
        self._internal_state.entry_token0_amount = 0.0
        self._internal_state.entry_token1_amount = 0.0
        self._internal_state.price_init = 0.0
        self._internal_state.price_lower = 0.0
        self._internal_state.price_upper = 0.0
        self._internal_state.liquidity = 0.0

        return token0_back, token1_back

    # --- high-level zap-in / zap-out
    def action_open_position(
        self,
        amount_in_notional: float,
        price_lower: float,
        price_upper: float,
    ) -> None:
        """Zap-in: open a V3 position over ``[price_lower, price_upper]`` from notional cash.

        Computes the optimal stable/volatile split for the range, swaps the
        volatile portion (paying ``effective_fee_rate`` on the swap), mints
        the position, and returns any leftover stable to cash.
        """
        if amount_in_notional < 0:
            raise EntityException(
                f"open_position amount must be >= 0, got {amount_in_notional}"
            )
        if self.is_position:
            raise EntityException("Position already open.")
        if amount_in_notional > self._internal_state.cash:
            raise EntityException("Insufficient funds to open position.")
        if self._global_state.price <= 0:
            raise EntityException(
                f"price must be > 0, got {self._global_state.price}"
            )
        if price_lower >= price_upper:
            raise EntityException(
                f"price_lower must be less than price_upper - {price_lower} >= {price_upper}"
            )
        if price_lower <= 0 or price_upper <= 0:
            raise EntityException("price bounds must be positive")

        self._internal_state.cash -= amount_in_notional

        p = self._global_state.price
        sqp = p**0.5
        sqpl = price_lower**0.5
        sqpu = price_upper**0.5
        fee = self.effective_fee_rate

        if p <= price_lower:
            # Range entirely above current price — swap full notional → volatile.
            stable_at_mint = 0.0
            volatile_at_mint = (amount_in_notional / p) * (1 - fee)
        elif p >= price_upper:
            # Range entirely below — no swap, position is 100% stable.
            stable_at_mint = amount_in_notional
            volatile_at_mint = 0.0
        else:
            # In range — split by V3 ratio in notional terms.
            stable_factor = sqp - sqpl                     # stable per L
            volatile_value_factor = (1 / sqp - 1 / sqpu) * p  # volatile_value per L (in notional)
            total_factor = stable_factor + volatile_value_factor
            stable_pre = amount_in_notional * stable_factor / total_factor
            volatile_value_pre = amount_in_notional * volatile_value_factor / total_factor
            stable_at_mint = stable_pre
            volatile_at_mint = (volatile_value_pre / p) * (1 - fee)

        if self.notional_side == "token0":
            token0_amt, token1_amt = stable_at_mint, volatile_at_mint
        else:
            token0_amt, token1_amt = volatile_at_mint, stable_at_mint

        token0_leftover, token1_leftover = self._open_from_pair(
            token0_amt, token1_amt, price_lower, price_upper
        )

        if self.notional_side == "token0":
            cash_leftover = token0_leftover
            if token1_leftover > 0:
                cash_leftover += token1_leftover * p * (1 - fee)
        else:
            cash_leftover = token1_leftover
            if token0_leftover > 0:
                cash_leftover += token0_leftover * p * (1 - fee)

        self._internal_state.cash += cash_leftover

    def action_close_position(self) -> None:
        """Zap-out: burn V3 LP, swap volatile leg back to notional (with fee).

        Stable leg returns at full value. Existing free cash is **not**
        haircut by the fee.
        """
        if not self.is_position:
            raise EntityException("No position to close.")

        token0_back, token1_back = self._close_to_pair()

        if self.notional_side == "token0":
            stable_back, volatile_back = token0_back, token1_back
        else:
            stable_back, volatile_back = token1_back, token0_back

        p = self._global_state.price
        fee = self.effective_fee_rate
        volatile_proceeds = volatile_back * p * (1 - fee) if volatile_back > 0 else 0.0

        self._internal_state.cash += stable_back + volatile_proceeds

    def update_state(self, state: UniswapV3LPGlobalState) -> None:
        """Apply pool snapshot, rebalance position by V3 formula, accrue fees."""
        self._global_state = state
        if not self.is_position:
            return
        p = state.price
        pl = self._internal_state.price_lower
        pu = self._internal_state.price_upper
        liq = self._internal_state.liquidity
        if p <= pl:
            stable = 0.0
            volatile = liq * (1 / (pl**0.5) - 1 / (pu**0.5))
        elif pl < p < pu:
            stable = liq * (p**0.5 - pl**0.5)
            volatile = liq * (1 / (p**0.5) - 1 / (pu**0.5))
        else:
            stable = liq * (pu**0.5 - pl**0.5)
            volatile = 0.0
        self._set_position_amounts(stable, volatile)
        self._internal_state.cash += self.calculate_fees()

    @property
    def balance(self) -> float:
        """Total entity equity in notional units."""
        if not self.is_position:
            return self._internal_state.cash
        return (
            self.stable_amount
            + self.volatile_amount * self._global_state.price
            + self._internal_state.cash
        )

    @property
    def hodl_value(self) -> float:
        """Notional balance had the entry composition been held instead of LPed."""
        if not self.is_position:
            return self._internal_state.cash
        return (
            self.entry_stable_amount
            + self.entry_volatile_amount * self._global_state.price
            + self._internal_state.cash
        )

    @property
    def impermanent_loss(self) -> float:
        """Hodl-vs-LP gap, in notional. Positive when LP underperforms hodl."""
        if not self.is_position:
            return 0.0
        return self.hodl_value - self.balance

    def calculate_fees(self) -> float:
        """Pro-rata share of pool swap-fees over the previous bar.

        Returns 0 when out of range or when any of ``p``, ``pl``, ``pu`` is
        non-positive (degenerate state — e.g. fresh entity before first
        ``update_state``). Handles ``notional_side`` inversion correctly.
        """
        p = self._global_state.price
        pl = self._internal_state.price_lower
        pu = self._internal_state.price_upper
        if p <= 0 or pl <= 0 or pu <= 0:
            return 0
        if p <= pl or p >= pu:
            return 0

        # The underlying V3 fees model expects standard convention
        # P = on-chain-token1 / on-chain-token0. Our entity's ``price`` is
        # always ``notional / non-notional``. Map accordingly:
        # * notional_side="token0" → on-chain token0 = stable,
        #   so standard P = volatile/stable = 1/entity_p
        # * notional_side="token1" → on-chain token1 = stable,
        #   so standard P = stable/volatile = entity_p
        if self.notional_side == "token0":
            P_std = 1 / p
            lower_std = 1 / pu
            upper_std = 1 / pl
        else:
            P_std = p
            lower_std = pl
            upper_std = pu

        delta_liquidity = get_liquidity_delta(
            P=P_std,
            lower_price=lower_std,
            upper_price=upper_std,
            amount0=self._internal_state.token0_amount,
            amount1=self._internal_state.token1_amount,
            token0_decimal=self.token0_decimals,
            token1_decimal=self.token1_decimals,
        )
        fees = estimate_fee(
            liquidity_delta=delta_liquidity,
            liquidity=self._global_state.liquidity,
            fees=self._global_state.fees,
        )
        return min(fees, self._global_state.fees)

    def price_to_tick(self, price: float) -> float:
        return np.floor(np.log(price) / np.log(1.0001))

    def tick_to_price(self, tick: float) -> float:
        return 1.0001**tick
