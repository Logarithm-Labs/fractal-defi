from dataclasses import dataclass
from typing import Literal, Tuple

from fractal.core.base.entity import EntityException, InternalState
from fractal.core.entities.base.pool import BasePoolEntity, BasePoolGlobalState


@dataclass
class UniswapV2LPGlobalState(BasePoolGlobalState):
    """V2-style pool snapshot.

    Inherits ``tvl, volume, fees, liquidity, price`` from
    :class:`BasePoolGlobalState`. ``price`` is interpreted as
    **notional per non-notional unit** (e.g. USDC/ETH ≈ 3000 for an
    ETH/USDC pool with USDC as notional, regardless of which slot
    USDC occupies on-chain). Use the ``notional_side`` config field to
    say which slot holds the notional.
    """


@dataclass
class UniswapV2LPInternalState(InternalState):
    """
    Represents the internal state of an UniswapV2 LP entity.

    Attributes:
        token0_amount (float): Current on-chain token0 amount in the position.
        token1_amount (float): Current on-chain token1 amount in the position.
        entry_token0_amount (float): Token0 in position at open — used for hodl/IL.
        entry_token1_amount (float): Token1 in position at open — used for hodl/IL.
        price_init (float): Pool price at position open.
        liquidity (float): Position LP-token count.
        cash (float): Free notional cash held by the entity.
        compounded_token0_amount (float): Cumulative fee-derived growth of
            token0 in the position. Stays at 0 in ``"cash"`` mode; grows
            monotonically in ``"compound"`` mode and is added on top of
            the share-based formula every ``update_state``.
        compounded_token1_amount (float): Same for token1.
    """

    token0_amount: float = 0.0
    token1_amount: float = 0.0
    entry_token0_amount: float = 0.0
    entry_token1_amount: float = 0.0
    price_init: float = 0.0
    liquidity: float = 0.0
    cash: float = 0.0
    compounded_token0_amount: float = 0.0
    compounded_token1_amount: float = 0.0


@dataclass
class UniswapV2LPConfig:
    """V2 LP entity configuration.

    Attributes:
        pool_fee_rate (float): The pool's swap-fee tier (e.g. ``0.003`` for the
            30-bps tier). Charged on the volatile-side of zap-in
            (``action_open_position``) and zap-out (``action_close_position``)
            — i.e. only on the swapped portion of the deposit/withdrawal,
            **not** on the full notional.
        slippage_pct (float): Additional execution-cost handwave on top of
            the pool fee — captures slippage / MEV. Applied identically to
            ``pool_fee_rate`` on the swapped portion. Default ``0.0``.
        token0_decimals (int): Token0 decimals (used by the V3 fees model;
            kept here for symmetry with V3 config).
        token1_decimals (int): Token1 decimals.
        notional_side (str): Which on-chain slot — ``"token0"`` or
            ``"token1"`` — is the notional/stable side. Default
            ``"token0"``. Together with ``GlobalState.price`` (notional
            per non-notional unit) tells the entity how to map the on-chain
            pair onto the (stable, volatile) abstraction internally.
        fees_compounding_model (str): How per-bar pool fees flow through
            the position. Default ``"cash"`` (backward-compatible).

            * ``"cash"`` — fees are added to ``InternalState.cash`` each
              bar; ``liquidity`` and token amounts reflect pure
              price-divergence. ``impermanent_loss`` is the textbook
              hodl-vs-LP gap and useful for IL modelling in isolation.
            * ``"compound"`` — fees are implicitly reinvested into the
              position by growing ``token0/token1`` amounts (no swap
              fee, no growth in ``liquidity`` LP-token count — mirrors
              on-chain V2 where reserves grow but LP supply does not).
              ``balance`` matches the ``"cash"`` total but
              ``impermanent_loss`` becomes a fee-adjusted gap (pool
              yield offsets price-divergence cost).

            Both modes assume the loader contract: ``GlobalState.tvl``
            is the **pre-fee** pool TVL for the current bar (= reserves
            BEFORE the bar's fees), and ``GlobalState.fees`` is the
            absolute USD fees accrued during the bar.
    """

    pool_fee_rate: float = 0.003
    slippage_pct: float = 0.0
    token0_decimals: int = 18
    token1_decimals: int = 18
    notional_side: str = "token0"
    fees_compounding_model: Literal["cash", "compound"] = "cash"


class UniswapV2LPEntity(BasePoolEntity):
    """Uniswap V2-style LP entity (50/50 by value).

    Position lifecycle:

    * :meth:`action_open_position` (zap-in) deposits ``amount_in_notional`` of
      cash. Half stays as stable; the other half is swapped to volatile,
      paying ``pool_fee_rate + slippage_pct`` on the swap. The minted LP is
      limited by the volatile leg (smaller after fee); the unbalanced sliver
      of stable returns to cash.
    * :meth:`update_state` rebalances ``token0_amount`` / ``token1_amount``
      by the pool's new TVL/price, and accrues pro-rata pool swap-fees
      (``calculate_fees``) into cash.
    * :meth:`action_close_position` (zap-out) burns LP tokens. Stable side
      returns at full notional value; volatile side is swapped back to
      notional at ``pool_fee_rate + slippage_pct``.

    Lower-level pair-based mint / burn are exposed as ``_open_from_pair``
    and ``_close_to_pair`` for advanced strategies that already hold both
    on-chain tokens (no swap, no fee).
    """

    _internal_state: UniswapV2LPInternalState
    _global_state: UniswapV2LPGlobalState

    def __init__(self, config: UniswapV2LPConfig, *args, **kwargs):
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
        if config.fees_compounding_model not in ("cash", "compound"):
            raise EntityException(
                f"fees_compounding_model must be 'cash' or 'compound', "
                f"got {config.fees_compounding_model!r}"
            )
        # Set config BEFORE super so any subclass override of
        # ``_initialize_states`` can rely on these.
        self.pool_fee_rate: float = config.pool_fee_rate
        self.slippage_pct: float = config.slippage_pct
        self.token0_decimals: int = config.token0_decimals
        self.token1_decimals: int = config.token1_decimals
        self.notional_side: str = config.notional_side
        self.fees_compounding_model: str = config.fees_compounding_model
        super().__init__(*args, **kwargs)

    def _initialize_states(self):
        self._internal_state = UniswapV2LPInternalState()
        self._global_state = UniswapV2LPGlobalState()

    @property
    def internal_state(self) -> UniswapV2LPInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> UniswapV2LPGlobalState:  # type: ignore[override]
        return self._global_state

    @property
    def is_position(self) -> bool:
        """Whether an LP position is currently open (derived from ``liquidity > 0``)."""
        return self._internal_state.liquidity > 0

    @property
    def effective_fee_rate(self) -> float:
        """Combined ``pool_fee_rate + slippage_pct`` — applied on the swapped portion."""
        return self.pool_fee_rate + self.slippage_pct

    # --- Slot-aware accessors (hide which on-chain slot holds the stable side).
    @property
    def stable_amount(self) -> float:
        """Current stable-leg amount, in notional units."""
        if self.notional_side == "token0":
            return self._internal_state.token0_amount
        return self._internal_state.token1_amount

    @property
    def volatile_amount(self) -> float:
        """Current volatile-leg amount, in volatile-token units."""
        if self.notional_side == "token0":
            return self._internal_state.token1_amount
        return self._internal_state.token0_amount

    @property
    def entry_stable_amount(self) -> float:
        """Stable leg at position open, in notional units."""
        if self.notional_side == "token0":
            return self._internal_state.entry_token0_amount
        return self._internal_state.entry_token1_amount

    @property
    def entry_volatile_amount(self) -> float:
        """Volatile leg at position open, in volatile-token units."""
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

    # --- account-level actions
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

    # --- pair-level helpers (no swap, no fee) — internal/advanced API
    def _open_from_pair(self, token0_amount: float, token1_amount: float) -> Tuple[float, float]:
        """Mint LP from on-chain (token0, token1) amounts. No swap, no fee.

        The pool ratio (50/50 by value) is the limit: only the proportional
        amount of each leg is consumed; the rest is returned as leftover.

        Returns:
            Tuple ``(token0_leftover, token1_leftover)`` — the unbalanced
            part the caller can do with as it sees fit (typically returned
            to cash by ``action_open_position``).
        """
        if self.is_position:
            raise EntityException("Position already open.")
        if self._global_state.price <= 0:
            raise EntityException(
                f"price must be > 0, got {self._global_state.price}"
            )
        if self._global_state.tvl <= 0 or self._global_state.liquidity <= 0:
            raise EntityException(
                f"pool tvl/liquidity must be positive, got tvl={self._global_state.tvl}, "
                f"liquidity={self._global_state.liquidity}"
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
        stable_reserve = self._global_state.tvl / 2
        volatile_reserve = stable_reserve / p

        # Limiting side determines share (Uniswap V2 mint logic).
        share_stable = stable / stable_reserve if stable_reserve > 0 else 0
        share_volatile = volatile / volatile_reserve if volatile_reserve > 0 else 0
        share = min(share_stable, share_volatile)

        if share <= 0:
            raise EntityException("Insufficient amounts to open position.")

        liquidity = share * self._global_state.liquidity
        stable_used = share * stable_reserve
        volatile_used = share * volatile_reserve

        self._set_position_amounts(stable_used, volatile_used)
        self._set_entry_amounts(stable_used, volatile_used)
        self._internal_state.price_init = p
        self._internal_state.liquidity = liquidity

        stable_leftover = stable - stable_used
        volatile_leftover = volatile - volatile_used

        if self.notional_side == "token0":
            return stable_leftover, volatile_leftover
        return volatile_leftover, stable_leftover

    def _close_to_pair(self) -> Tuple[float, float]:
        """Burn LP, return current on-chain ``(token0, token1)`` amounts. No swap, no fee.

        Resets all position fields. Caller is responsible for what to do
        with the returned token amounts (typically swap volatile back to
        notional in ``action_close_position``).
        """
        if not self.is_position:
            raise EntityException("No position to close.")

        token0_back = self._internal_state.token0_amount
        token1_back = self._internal_state.token1_amount

        self._internal_state.token0_amount = 0.0
        self._internal_state.token1_amount = 0.0
        self._internal_state.entry_token0_amount = 0.0
        self._internal_state.entry_token1_amount = 0.0
        self._internal_state.price_init = 0.0
        self._internal_state.liquidity = 0.0
        self._internal_state.compounded_token0_amount = 0.0
        self._internal_state.compounded_token1_amount = 0.0

        return token0_back, token1_back

    # --- high-level zap-in / zap-out
    def action_open_position(self, amount_in_notional: float) -> None:
        """Zap-in: open a 50/50 LP position from a notional cash deposit.

        Half of the notional stays as the stable leg (no swap, no fee).
        The other half is swapped into the volatile token, paying
        ``effective_fee_rate`` on the swap. The minted LP is limited by
        the (post-fee) volatile leg; any unbalanced sliver of stable that
        couldn't be deposited at pool ratio returns to cash.
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

        self._internal_state.cash -= amount_in_notional

        p = self._global_state.price
        half = amount_in_notional / 2
        fee = self.effective_fee_rate

        stable_at_mint = half                       # not swapped, no fee
        volatile_at_mint = (half / p) * (1 - fee)   # post-fee on swapped half

        if self.notional_side == "token0":
            token0_amt, token1_amt = stable_at_mint, volatile_at_mint
        else:
            token0_amt, token1_amt = volatile_at_mint, stable_at_mint

        token0_leftover, token1_leftover = self._open_from_pair(token0_amt, token1_amt)

        # Convert leftovers back to notional cash.
        # In standard zap-in the volatile leftover is 0 (engineered as the
        # limiting side after fee); the stable leftover is ``half * fee``.
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
        """Zap-out: burn LP and consolidate to notional cash.

        Stable leg returns at full notional value (no swap). Volatile leg
        is swapped back to notional, paying ``effective_fee_rate`` on the
        swap. Existing free cash is **not** touched by the fee.
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

    def update_state(self, state: UniswapV2LPGlobalState) -> None:
        """Apply pool snapshot, rebalance position amounts, route fees per
        :attr:`fees_compounding_model`.

        Loader contract: ``state.tvl`` is the pre-fee pool TVL for the bar
        (= on-chain reserves BEFORE this bar's fees) and ``state.fees`` is
        the absolute USD fees accrued during the bar. The share-based
        position formula ``share * tvl`` therefore underestimates by
        ``share * fees``, which we add either to ``cash`` (default) or
        directly to the on-chain token amounts (compound mode). Total
        ``balance`` is identical between the two modes; only IL semantics
        and where the value lives differ.
        """
        self._global_state = state

        if not self.is_position:
            return

        if self._global_state.price == 0:
            raise EntityException("Price is 0.")
        if self._global_state.liquidity == 0:
            raise EntityException("Pool liquidity is 0.")

        share = self._internal_state.liquidity / self._global_state.liquidity
        stable_reserve = self._global_state.tvl / 2
        volatile_reserve = stable_reserve / self._global_state.price
        bar_fees = self.calculate_fees()

        if self.fees_compounding_model == "compound":
            # Split this-bar fees 50/50 by VALUE at current price (mirrors
            # V2's pool-ratio invariant) and accumulate into the buffer.
            stable_delta = bar_fees / 2
            volatile_delta = (bar_fees / 2) / self._global_state.price
            if self.notional_side == "token0":
                self._internal_state.compounded_token0_amount += stable_delta
                self._internal_state.compounded_token1_amount += volatile_delta
            else:
                self._internal_state.compounded_token1_amount += stable_delta
                self._internal_state.compounded_token0_amount += volatile_delta

        # Position amounts: share-based (pre-fee tvl) + cumulative compound
        # buffer. In ``"cash"`` mode the buffer stays at 0, so this matches
        # the prior ``share * reserve`` semantics exactly.
        if self.notional_side == "token0":
            extra_t0 = self._internal_state.compounded_token0_amount
            extra_t1 = self._internal_state.compounded_token1_amount
            self._set_position_amounts(
                share * stable_reserve + extra_t0,
                share * volatile_reserve + extra_t1,
            )
        else:
            extra_t1 = self._internal_state.compounded_token1_amount
            extra_t0 = self._internal_state.compounded_token0_amount
            self._set_position_amounts(
                share * stable_reserve + extra_t1,
                share * volatile_reserve + extra_t0,
            )

        if self.fees_compounding_model == "cash":
            self._internal_state.cash += bar_fees

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
        """Notional balance had the entry composition been held instead of LPed.

        ``entry_stable + entry_volatile · current_price + cash``. Returns
        plain cash when no position is open. Useful for IL decomposition:
        ``impermanent_loss = hodl_value - balance``.
        """
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
        """Pro-rata share of the pool's swap fees over the previous bar."""
        if not self.is_position:
            return 0
        return (self._internal_state.liquidity / self._global_state.liquidity) * self._global_state.fees
