import warnings
from dataclasses import dataclass

from fractal.core.base.entity import EntityException, GlobalState
from fractal.core.entities.base.liquid_staking import BaseLiquidStakingToken
from fractal.core.entities.base.spot import BaseSpotInternalState


class StakedETHEntityException(EntityException):
    """
    Represents an exception for the StakedETH entity.
    """


@dataclass
class StakedETHGlobalState(GlobalState):
    """Market state for a stETH-like LST.

    Attributes:
        price: Spot price of the LST in notional units (USD).
        staking_rate: Per-step underlying-balance accrual rate. Renamed
            from legacy ``rate`` for parity with
            :class:`SimpleLiquidStakingTokenGlobalState` and the
            polymorphic :attr:`BaseLiquidStakingToken.staking_rate`.
    """
    price: float = 0.0
    staking_rate: float = 0.0


@dataclass
class StakedETHInternalState(BaseSpotInternalState):
    """Internal state of the StakedETH entity (inherits ``amount``, ``cash``)."""
    pass


class StakedETHEntity(BaseLiquidStakingToken):
    """Lido stETH (or any LST-ETH-like) entity.

    Categorically a :class:`BaseLiquidStakingToken`: spot-traded with
    rebasing balance accruing at ``global_state.rate`` per step.
    The historical ``rate`` field name is preserved for back-compat;
    polymorphic strategy code should use the
    :attr:`staking_rate` property instead.
    """

    _internal_state: StakedETHInternalState
    _global_state: StakedETHGlobalState

    def __init__(self, *args, trading_fee: float = 0.003, **kwargs):
        if trading_fee < 0:
            raise StakedETHEntityException(
                f"trading_fee must be >= 0, got {trading_fee}"
            )
        # Set config BEFORE super so any subclass override of
        # ``_initialize_states`` can rely on ``self.trading_fee``.
        self.trading_fee: float = trading_fee
        super().__init__(*args, **kwargs)

    @property
    def effective_fee_rate(self) -> float:
        """Combined execution-cost rate. LST entities have no slippage
        component, so this aliases :attr:`trading_fee` for polymorphic use."""
        return self.trading_fee

    @property
    def TRADING_FEE(self) -> float:  # noqa: N802  (deprecated UPPERCASE alias)
        """Deprecated alias for :attr:`trading_fee`.

        Python convention reserves UPPERCASE for module/class constants;
        instance attributes should be lowercase. Use ``trading_fee`` instead.
        """
        warnings.warn(
            "StakedETHEntity.TRADING_FEE is deprecated; use trading_fee (lowercase).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.trading_fee

    def _initialize_states(self):
        self._internal_state = StakedETHInternalState()
        self._global_state = StakedETHGlobalState()

    @property
    def internal_state(self) -> StakedETHInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> StakedETHGlobalState:  # type: ignore[override]
        return self._global_state

    @property
    def staking_rate(self) -> float:
        """Polymorphic LST contract — delegates to ``global_state.staking_rate``."""
        return self._global_state.staking_rate

    def action_buy(self, amount_in_notional: float):
        """
        Executes a buy action on the StakedETH protocol.

        Args:
            amount_in_notional (float, optional): The amount to buy in notional value.

        Raises:
            StakedETHEntityException: If amount is negative or exceeds cash.
        """
        if amount_in_notional < 0:
            raise StakedETHEntityException(
                f"buy amount must be >= 0, got {amount_in_notional}"
            )
        if self._global_state.price <= 0:
            raise StakedETHEntityException(
                f"price must be > 0, got {self._global_state.price}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise StakedETHEntityException(
                f"Not enough cash to buy: {amount_in_notional} > {self._internal_state.cash}")
        self._internal_state.cash -= amount_in_notional
        self._internal_state.amount += amount_in_notional * (1 - self.trading_fee) / self._global_state.price

    def action_sell(self, amount_in_product: float):
        """
        Executes a sell action on the StakedETH protocol.

        Args:
            amount_in_product (float, optional): The amount to sell in product value.

        Raises:
            StakedETHEntityException: If amount is negative or exceeds holdings.
        """
        if amount_in_product < 0:
            raise StakedETHEntityException(
                f"sell amount must be >= 0, got {amount_in_product}"
            )
        if self._global_state.price <= 0:
            raise StakedETHEntityException(
                f"price must be > 0, got {self._global_state.price}"
            )
        if amount_in_product > self._internal_state.amount:
            raise StakedETHEntityException(
                f"Not enough product to sell: {amount_in_product} > {self._internal_state.amount}")
        self._internal_state.amount -= amount_in_product
        self._internal_state.cash += amount_in_product * (1 - self.trading_fee) * self._global_state.price

    def action_withdraw(self, amount_in_notional: float):
        """
        Executes a withdraw action on the StakedETH protocol.

        Args:
            amount_in_notional (float, optional): The amount to withdraw in notional value.

        Raises:
            StakedETHEntityException: If amount is negative or exceeds cash.
        """
        if amount_in_notional < 0:
            raise StakedETHEntityException(
                f"withdraw amount must be >= 0, got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.cash:
            raise StakedETHEntityException(
                f"Not enough cash to withdraw: {amount_in_notional} > {self._internal_state.cash}")
        self._internal_state.cash -= amount_in_notional

    def action_deposit(self, amount_in_notional: float):
        """
        Executes a deposit action on the StakedETH protocol.

        Args:
            amount_in_notional (float): The amount to deposit in notional value.
        """
        if amount_in_notional < 0:
            raise StakedETHEntityException(
                f"deposit amount must be >= 0, got {amount_in_notional}"
            )
        self._internal_state.cash += amount_in_notional

    def update_state(self, state: StakedETHGlobalState) -> None:
        """Apply market state and rebase the held balance.

        ``amount`` grows by ``(1 + staking_rate)`` per step.
        ``staking_rate < -1`` would flip ``amount`` negative —
        physically meaningless and likely corrupt input — so rejected loudly.

        Args:
            state: The new global state.
        """
        if state.staking_rate < -1:
            raise StakedETHEntityException(
                f"staking_rate must be >= -1 (cannot rebase to negative balance), got {state.staking_rate}"
            )
        self._global_state = state
        self._internal_state.amount *= (self._global_state.staking_rate + 1)

    @property
    def current_price(self) -> float:
        return self._global_state.price

    @property
    def balance(self) -> float:
        """
        Calculates the balance of the StakedETH entity.

        The balance is calculated as the sum of the amount of stETH and the cash balance.
        Returns:
            float: The balance of the entity.
        """
        return self._internal_state.amount * self._global_state.price + self._internal_state.cash
