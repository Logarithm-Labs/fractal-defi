from dataclasses import dataclass, field
from typing import List

from fractal.core.base.entity import EntityException, GlobalState
from fractal.core.entities.base.perp import BasePerpEntity, BasePerpInternalState


class HyperliquidEntityException(EntityException):
    """
    Exception raised for errors in the HyperLiquid entity.
    """


@dataclass
class HyperLiquidPosition:
    """A single Hyperliquid perp position.

    Attributes:
        amount: Signed product quantity. Positive = long, negative = short.
        entry_price: Mark price at which the position was opened (in notional).
        max_leverage: Per-position max leverage (3x–50x on Hyperliquid mainnet).
            Together with ``MMR = 1 / (2 × max_leverage)`` drives the
            maintenance-margin and liquidation-price computations.
    """
    amount: float
    entry_price: float
    max_leverage: float

    def unrealised_pnl(self, price: float) -> float:
        """Mark-to-market PnL: ``amount × (price − entry_price)``."""
        return self.amount * (price - self.entry_price)


@dataclass
class HyperLiquidGlobalState(GlobalState):
    """
    Global state of the entity.
    It includes the state of the environment.
    For example, price, time, etc.
    """
    mark_price: float = 0.0
    funding_rate: float = 0.0


@dataclass
class HyperLiquidInternalState(BasePerpInternalState):
    """Internal state of the HyperLiquid entity.

    Inherits ``collateral`` and adds an aggregated ``positions`` list.
    """
    positions: List[HyperLiquidPosition] = field(default_factory=list)


class HyperliquidEntity(BasePerpEntity):
    """
    Represents a Hyperliquid isolated market entity.
    """

    _internal_state: HyperLiquidInternalState
    _global_state: HyperLiquidGlobalState

    def __init__(self, *args, trading_fee: float = 0.00035, max_leverage: float = 50, **kwargs):
        if trading_fee < 0:
            raise HyperliquidEntityException(
                f"trading_fee must be >= 0, got {trading_fee}"
            )
        if max_leverage <= 0:
            raise HyperliquidEntityException(
                f"max_leverage must be > 0, got {max_leverage}"
            )
        # Set config BEFORE super so any subclass override of
        # ``_initialize_states`` can rely on ``self.TRADING_FEE`` / ``self.MAX_LEVERAGE``.
        self.TRADING_FEE: float = trading_fee
        self.MAX_LEVERAGE: float = max_leverage
        super().__init__(*args, **kwargs)

    def _initialize_states(self):
        self._internal_state = HyperLiquidInternalState()
        self._global_state = HyperLiquidGlobalState()

    @property
    def internal_state(self) -> HyperLiquidInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> HyperLiquidGlobalState:  # type: ignore[override]
        return self._global_state

    def action_deposit(self, amount_in_notional: float):
        """
        Deposits a specified amount of notional into the entity's collateral.

        Args:
            amount_in_notional (float): The amount of notional to deposit.
        """
        if amount_in_notional < 0:
            raise HyperliquidEntityException(f"Invalid deposit amount: {amount_in_notional} < 0.")

        self._internal_state.collateral += amount_in_notional

    def action_withdraw(self, amount_in_notional: float):
        """Withdraws notional collateral, blocking if it would push the
        position below maintenance margin at the **current** mark price.

        Args:
            amount_in_notional: Amount of notional to withdraw.

        Raises:
            HyperliquidEntityException: If amount is negative, exceeds balance,
                or would push post-withdrawal balance strictly below the
                maintenance margin requirement.
        """
        if amount_in_notional < 0:
            raise HyperliquidEntityException(f"Invalid withdraw amount: {amount_in_notional} < 0.")
        if self.balance < amount_in_notional:
            raise HyperliquidEntityException(f"Not enough balance to withdraw: {self.balance} < {amount_in_notional}.")

        if self._internal_state.positions:
            mm = self.maintenance_margin
            if self.balance - amount_in_notional < mm:
                raise HyperliquidEntityException(
                    f"Not enough maintenance margin after withdraw: post-withdraw balance "
                    f"{self.balance - amount_in_notional} < maintenance_margin {mm}."
                )
        self._internal_state.collateral -= amount_in_notional

    def action_open_position(self, amount_in_product):
        """
        Opens a position with a specified amount of product.

        Args:
            amount_in_product (float): The amount of product to open the position with.
                Negative values open a short. Zero is a no-op.
        """
        if self._global_state.mark_price <= 0:
            raise HyperliquidEntityException(
                f"mark_price must be > 0, got {self._global_state.mark_price}"
            )
        # Zero-amount short-circuit: no fee, no state mutation.
        if amount_in_product == 0:
            return
        self._internal_state.positions.append(
            HyperLiquidPosition(amount=amount_in_product,
                                entry_price=self._global_state.mark_price,
                                max_leverage=self.MAX_LEVERAGE))

        self._internal_state.collateral -= abs(self._global_state.mark_price * amount_in_product * self.TRADING_FEE)
        self._clearing()  # consider only one position for simplicity

    @property
    def pnl(self) -> float:
        """
        Calculates the total profit and loss (PNL) of all positions.

        PNL is a sum of PNLs of all positions.
        Returns:
            float: The total PNL.
        """
        return sum(pos.unrealised_pnl(self._global_state.mark_price) for pos in self._internal_state.positions)

    @property
    def balance(self) -> float:
        """
        Calculates the current balance of the entity.

        Balance is a sum of the collateral and the PNL.
        Returns:
            float: The current balance.
        """
        return self._internal_state.collateral + self.pnl

    @property
    def size(self):
        """
        Calculates the total size of all positions.

        Size is a sum of all position amounts.
        Returns:
            float: The total size.
        """
        return sum(pos.amount for pos in self._internal_state.positions)

    @property
    def leverage(self) -> float:
        """Effective leverage: ``|size| × mark_price / balance``.

        Edge cases:
        * No position (``size == 0``): returns ``0``.
        * Position exists but ``balance ≤ 0`` (already underwater /
          past-liquidation): returns ``+inf`` rather than a meaningless
          negative ratio.
        """
        if self.size == 0:
            return 0
        if self.balance <= 0:
            return float("inf")
        return abs(self.size) * self._global_state.mark_price / self.balance

    @property
    def maintenance_margin(self) -> float:
        """Maintenance margin required at the **current** mark price.

        Per Hyperliquid: ``MM = |size| × mark_price × MMR``, where
        ``MMR = 1 / (2 × max_leverage)`` (single-tier simplification).

        Returns ``0.0`` when no position is open.
        """
        if not self._internal_state.positions:
            return 0.0
        pos = self._internal_state.positions[0]
        mmr = 1.0 / (2.0 * pos.max_leverage)
        return abs(self.size) * self._global_state.mark_price * mmr

    @property
    def liquidation_price(self) -> float:
        """Closed-form liquidation price for the current position.

        Solves ``balance(p) = MM(p)`` where:

            balance(p) = collateral + size × (p − entry)
            MM(p)      = |size| × p × MMR

        Result (with signed ``size`` and ``side ∈ {+1, −1}``):

            liq = (size × entry − collateral) / (size × (1 − MMR × side))

        For a long (side = +1): ``liq < entry``; the position is liquidated
        on a price drop. For a short (side = −1): ``liq > entry``; liquidated
        on a price rise.

        Returns ``NaN`` when no position is open. May return a value ``<= 0``
        for an over-collateralized long (no liquidation possible at any
        positive price) — strategies should treat ``<= 0`` as "safe forever".
        """
        if not self._internal_state.positions:
            return float("nan")
        pos = self._internal_state.positions[0]
        size = pos.amount  # signed
        if size == 0:
            return float("nan")
        entry = pos.entry_price
        mmr = 1.0 / (2.0 * pos.max_leverage)
        side = 1.0 if size > 0 else -1.0
        return (size * entry - self._internal_state.collateral) / (size * (1.0 - mmr * side))

    def _clearing(self):
        """
        Aggregates all positions into a single position.

        If positions of opposite sides are present (i.e., a closing trade occurs),
        the entry price of the remaining open position does not change, and the
        realized PnL for the closed quantity is added to the collateral.
        For positions on the same side (i.e., opening trades), the entry price
        is updated to the weighted average.
        """
        positions = self._internal_state.positions
        if not positions or len(positions) == 1:
            return

        # Use the first position as the aggregated base position.
        base = positions[0]
        realized_pnl = 0.0

        # Process all subsequent positions.
        for pos in positions[1:]:
            # If both positions are in the same direction, aggregate them.
            if base.amount * pos.amount > 0:
                new_amount = base.amount + pos.amount
                base.entry_price = (base.amount * base.entry_price + pos.amount * pos.entry_price) / new_amount
                base.amount = new_amount
            else:
                # Opposite direction: a closing trade.
                closed_qty = min(abs(base.amount), abs(pos.amount))
                # Determine the side of the base position: 1 for long, -1 for short.
                sign = 1 if base.amount > 0 else -1
                # For a long base (sign = 1): realized pnl = closed_qty * (pos.entry_price - base.entry_price)
                # For a short base (sign = -1): realized pnl = closed_qty * (base.entry_price - pos.entry_price)
                realized_pnl += closed_qty * (pos.entry_price - base.entry_price) * sign
                # Reduce the base position by the closed quantity.
                base.amount -= sign * closed_qty
                # Calculate the remaining quantity in the incoming position.
                remaining = abs(pos.amount) - closed_qty
                if remaining > 0:
                    # If the base position is fully closed, adopt the remainder as the new base position.
                    # Carry over the **incoming position's** ``max_leverage`` —
                    # the remainder belongs to that position, not to the entity default.
                    if abs(base.amount) < 1e-9:
                        base = HyperLiquidPosition(
                            amount=remaining if pos.amount > 0 else -remaining,
                            entry_price=pos.entry_price,
                            max_leverage=pos.max_leverage,
                        )
                    # ``remaining > 0 AND abs(base.amount) > 0`` is unreachable
                    # because ``closed_qty = min(|base|, |pos|)`` so at least one
                    # side fully closes per iteration.

        # Add the realized pnl from the closed quantity to the collateral.
        self._internal_state.collateral += realized_pnl

        # If the aggregated position is effectively closed (net amount nearly zero), clear the positions.
        if abs(base.amount) < 1e-9:
            self._internal_state.positions = []
        else:
            self._internal_state.positions = [base]

    def update_state(self, state: HyperLiquidGlobalState) -> None:
        """Step the entity forward to ``state``.

        Order: apply state → **settle funding** → **check liquidation**.
        Funding settles BEFORE the liquidation check so a saving (or
        damning) funding tick is honored in the same bar — matching the
        ``SimplePerpEntity`` convention and Hyperliquid's continuous
        funding accrual.

        Funding sign convention: with ``funding_rate > 0`` the long pays
        the short. For a long (``size > 0``): ``collateral`` decreases.
        For a short (``size < 0``): ``collateral`` increases (received).
        Symmetrically with negative rate.

        Args:
            state: The global state to update with.
        """
        self._global_state = state

        # 1. Settle funding on the alive position.
        if self._internal_state.positions:
            funding_payment = self.size * state.mark_price * state.funding_rate
            self._internal_state.collateral -= funding_payment

        # 2. Check liquidation on the POST-funding balance.
        if self._check_liquidation():
            self._internal_state.collateral = 0
            self._internal_state.positions = []

    def _check_liquidation(self) -> bool:
        """Whether the position should be liquidated at the current mark price.

        Per Hyperliquid: liquidate when ``account_value ≤ maintenance_margin``,
        where both sides are evaluated **at the current mark price**:

            account_value(p) = collateral + size × (p − entry)
            maintenance_margin(p) = |size| × p × MMR

        Equivalent to: ``mark_price <= liquidation_price`` for a long,
        ``mark_price >= liquidation_price`` for a short.

        The original implementation computed maintenance margin from
        ``entry_price`` instead of ``mark_price`` — this mixed reference
        points (``balance`` is current, ``MM(entry)`` is not) and produced
        a premature liquidation by ``(collateral / |size|) × MMR`` in price
        terms. Catastrophically wrong for low-leverage positions; small
        but real for high-leverage ones.
        """
        if not self._internal_state.positions:
            return False
        return bool(self.balance <= self.maintenance_margin)
