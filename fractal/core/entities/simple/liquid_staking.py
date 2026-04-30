"""Minimal generic liquid-staking-token entity.

Counterpart to :class:`SimpleSpotExchange` for the LST category. Buys
and sells the token at ``global_state.price`` (with a flat trading fee
on the received side), and rebases ``amount`` by
``global_state.staking_rate`` on every :meth:`update_state`.

For protocol-specific LSTs (Lido stETH, Rocket rETH, Frax sfrxETH,
Coinbase cbETH …) reach for the corresponding class in
:mod:`fractal.core.entities.protocols`.
"""
from dataclasses import dataclass

from fractal.core.base.entity import EntityException, GlobalState
from fractal.core.entities.base.liquid_staking import BaseLiquidStakingToken
from fractal.core.entities.base.spot import BaseSpotInternalState


class SimpleLiquidStakingTokenException(EntityException):
    """Errors raised by :class:`SimpleLiquidStakingToken`."""


@dataclass
class SimpleLiquidStakingTokenGlobalState(GlobalState):
    """Market state.

    Attributes:
        price: Spot price of the LST in notional units (USDC/USD).
        staking_rate: Per-step underlying-balance accrual rate
            (e.g. ``0.0001`` per hour for ~88% APR).
    """
    price: float = 0.0
    staking_rate: float = 0.0


@dataclass
class SimpleLiquidStakingTokenInternalState(BaseSpotInternalState):
    """Inherits ``amount`` (LST tokens held) and ``cash`` (notional)."""
    pass


class SimpleLiquidStakingToken(BaseLiquidStakingToken):
    """Generic LST: spot trades + per-step rebasing."""

    def __init__(self, *, trading_fee: float = 0.003) -> None:
        if trading_fee < 0:
            raise SimpleLiquidStakingTokenException(
                f"trading_fee must be >= 0, got {trading_fee}"
            )
        self._trading_fee: float = float(trading_fee)
        super().__init__()

    _internal_state: SimpleLiquidStakingTokenInternalState
    _global_state: SimpleLiquidStakingTokenGlobalState

    def _initialize_states(self) -> None:
        self._internal_state = SimpleLiquidStakingTokenInternalState()
        self._global_state = SimpleLiquidStakingTokenGlobalState()

    @property
    def internal_state(self) -> SimpleLiquidStakingTokenInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> SimpleLiquidStakingTokenGlobalState:  # type: ignore[override]
        return self._global_state

    # ------------------------------------------------------------ readouts
    @property
    def trading_fee(self) -> float:
        """Public read-only accessor for the trading-fee config."""
        return self._trading_fee

    @property
    def effective_fee_rate(self) -> float:
        """Combined execution-cost rate. LST has no slippage component;
        aliases :attr:`trading_fee` for polymorphic use."""
        return self._trading_fee

    @property
    def current_price(self) -> float:
        return self._global_state.price

    @property
    def staking_rate(self) -> float:
        return self._global_state.staking_rate

    @property
    def balance(self) -> float:
        """``amount · price + cash``."""
        return (
            self._internal_state.amount * self._global_state.price
            + self._internal_state.cash
        )

    # ------------------------------------------------------------ trading
    def action_buy(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimpleLiquidStakingTokenException(
                f"buy amount_in_notional must be >= 0, got {amount_in_notional}"
            )
        if self._global_state.price <= 0:
            raise SimpleLiquidStakingTokenException(
                f"cannot trade at non-positive price {self._global_state.price}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise SimpleLiquidStakingTokenException(
                f"not enough cash to buy {amount_in_notional}: "
                f"available {self._internal_state.cash}"
            )
        product_received = (
            amount_in_notional * (1.0 - self._trading_fee) / self._global_state.price
        )
        self._internal_state.amount += product_received
        self._internal_state.cash -= amount_in_notional

    def action_sell(self, amount_in_product: float) -> None:
        if amount_in_product < 0:
            raise SimpleLiquidStakingTokenException(
                f"sell amount_in_product must be >= 0, got {amount_in_product}"
            )
        if amount_in_product > self._internal_state.amount:
            raise SimpleLiquidStakingTokenException(
                f"not enough product to sell {amount_in_product}: "
                f"available {self._internal_state.amount}"
            )
        notional_received = (
            amount_in_product * self._global_state.price * (1.0 - self._trading_fee)
        )
        self._internal_state.amount -= amount_in_product
        self._internal_state.cash += notional_received

    # ------------------------------------------------------------ account
    def action_deposit(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimpleLiquidStakingTokenException(
                f"deposit amount must be >= 0, got {amount_in_notional}"
            )
        self._internal_state.cash += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        if amount_in_notional < 0:
            raise SimpleLiquidStakingTokenException(
                f"withdraw amount must be >= 0, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise SimpleLiquidStakingTokenException(
                f"not enough cash to withdraw {amount_in_notional}: "
                f"available {self._internal_state.cash}"
            )
        self._internal_state.cash -= amount_in_notional

    # ------------------------------------------------------------ lifecycle
    def update_state(self, state: SimpleLiquidStakingTokenGlobalState) -> None:
        """Apply market state and rebase the held balance.

        ``amount`` grows by ``(1 + staking_rate)`` each step. A negative
        ``staking_rate`` (modelling slashing or downtime) shrinks it,
        but values below ``-1`` would flip ``amount`` negative and are
        rejected loudly.
        """
        if state.staking_rate < -1:
            raise SimpleLiquidStakingTokenException(
                f"staking_rate must be >= -1, got {state.staking_rate}"
            )
        self._global_state = state
        self._internal_state.amount *= 1.0 + state.staking_rate
