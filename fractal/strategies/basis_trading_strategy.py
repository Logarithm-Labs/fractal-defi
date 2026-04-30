import math
from dataclasses import dataclass
from typing import List

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams)
from fractal.core.entities import BasePerpEntity, BaseSpotEntity


class BasisTradingStrategyException(Exception):
    pass


# ----------------------------------------------------------- delegate helpers
# Module-level so they show up in tracebacks with a meaningful name. Each
# takes the strategy instance and resolves the value at execute time —
# the framework recognises ``callable(arg)`` and invokes it inside
# ``BaseStrategy.step`` right before the corresponding entity action.

def _delegate_spot_cash(strategy: BaseStrategy) -> float:
    """Resolve ``SPOT.internal_state.cash`` lazily for deposit/withdraw delegates."""
    return strategy.get_entity('SPOT').internal_state.cash


def _delegate_negative_spot_amount(strategy: BaseStrategy) -> float:
    """Resolve ``-SPOT.internal_state.amount`` for opening the matching short."""
    return -strategy.get_entity('SPOT').internal_state.amount


@dataclass
class BasisTradingStrategyHyperparams(BaseStrategyParams):
    """
    Hyperparameters for the BasisTradingStrategy.

    MIN_LEVERAGE: float - Minimum leverage ratio to maintain.
    TARGET_LEVERAGE: float - Target leverage ratio to maintain.
    MAX_LEVERAGE: float - Maximum leverage ratio to maintain.
    INITIAL_BALANCE: float - Initial balance to deposit into the strategy.
    """
    MIN_LEVERAGE: float
    TARGET_LEVERAGE: float
    MAX_LEVERAGE: float
    INITIAL_BALANCE: float


class BasisTradingStrategy(BaseStrategy[BasisTradingStrategyHyperparams]):
    """
    A basis trading strategy that implements the BaseStrategy interface.
    This strategy aims to maintain a target leverage ratio between a hedge entity and a spot entity.
    """

    #: Set to ``True`` after the initial deposit fires. Guards against
    #: silently re-depositing ``INITIAL_BALANCE`` if the strategy is
    #: ever fully wiped (both HEDGE and SPOT balances at zero) — that
    #: would print money out of thin air (B-3).
    _deposited: bool = False

    def set_up(self):
        """
        Set up the strategy by registering the hedge and spot entities.
        """
        hedge = self.get_entity('HEDGE')
        if not isinstance(hedge, BasePerpEntity):
            raise BasisTradingStrategyException(
                f"HEDGE must be a BasePerpEntity, got {type(hedge).__name__}"
            )
        spot = self.get_entity('SPOT')
        if not isinstance(spot, BaseSpotEntity):
            raise BasisTradingStrategyException(
                f"SPOT must be a BaseSpotEntity, got {type(spot).__name__}"
            )

    def predict(self) -> List[ActionToTake]:
        """
        Predict the actions to take based on the current state of the entities.
        Returns a list of ActionToTake objects representing the actions to be executed.
        """
        hedge: BasePerpEntity = self.get_entity('HEDGE')
        spot: BaseSpotEntity = self.get_entity('SPOT')
        if hedge.balance == 0 and spot.balance == 0:
            if self._deposited:
                raise BasisTradingStrategyException(
                    "strategy fully wiped (HEDGE.balance=0 AND SPOT.balance=0) "
                    "after initial deposit — refusing to re-fund from "
                    "INITIAL_BALANCE (would print money out of thin air). "
                    "Widen MAX_LEVERAGE or reduce position sizing."
                )
            self._debug("Depositing initial funds into the strategy...")
            self._deposited = True
            return self._deposit_into_strategy()
        if hedge.balance == 0 and spot.balance > 0:
            self._debug(f"HEDGE balance is 0, but SPOT balance is {spot.balance}")
            return self._rebalance()
        if hedge.leverage > self._params.MAX_LEVERAGE or hedge.leverage < self._params.MIN_LEVERAGE:
            self._debug(f"HEDGE leverage is {hedge.leverage}, rebalancing...")
            return self._rebalance()
        return []

    def _rebalance(self) -> List[ActionToTake]:
        """
        Rebalance the entities to maintain the target leverage ratio.
        Returns a list of ActionToTake objects representing the rebalancing actions to be executed.
        """
        hedge: BasePerpEntity = self.get_entity('HEDGE')
        spot: BaseSpotEntity = self.get_entity('SPOT')
        hedge_balance = hedge.balance
        spot_balance = spot.balance
        spot_amount = spot.internal_state.amount

        equity = hedge_balance + spot_balance
        target_hedge = equity / (1 + self._params.TARGET_LEVERAGE)
        target_spot = equity - target_hedge

        delta_spot = target_spot - spot_balance
        delta_hedge = target_hedge - hedge_balance

        if hedge_balance == 0:  # hedge is liquidated
            if hedge.size != 0:
                raise BasisTradingStrategyException(
                    f"liquidated rebalance entered with hedge.size={hedge.size}, "
                    "expected 0 after liquidation wipe"
                )
            if spot_amount <= 0:
                raise BasisTradingStrategyException(
                    f"liquidated rebalance entered with spot_amount={spot_amount}, "
                    "expected > 0 (otherwise total wipe — predict() should have "
                    "routed to _deposit_into_strategy)"
                )
            # Same delegate is reused on HEDGE.deposit and SPOT.withdraw —
            # deposit must come first, otherwise the withdraw zeroes
            # SPOT.cash before the delegate resolves on deposit (B-1).
            return [
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('sell', {'amount_in_product': -delta_spot / spot.global_state.price})
                ),
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('deposit', {'amount_in_notional': _delegate_spot_cash})
                ),
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('withdraw', {'amount_in_notional': _delegate_spot_cash})
                ),
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('open_position', {'amount_in_product': (self._params.TARGET_LEVERAGE * delta_spot /
                                                    spot.global_state.price)})
                ),
            ]

        # Basis hedge invariant: hedge.size ≈ -spot_amount (perfect short hedge).
        # ``math.isclose`` handles the 0/0 corner cleanly and uses both relative
        # and absolute tolerances — superior to the prior ratio expression that
        # produced ``nan`` when ``hedge.size == spot_amount`` (e.g. both zero).
        if not math.isclose(hedge.size, -spot_amount, rel_tol=1e-6, abs_tol=1e-9):
            raise BasisTradingStrategyException(
                f"basis hedge mismatch: hedge.size={hedge.size}, "
                f"spot.amount={spot_amount} (expected hedge.size ≈ -spot.amount)"
            )

        if delta_spot > 0:  # price_now < price_0, we need to buy spot
            if delta_hedge >= 0:
                raise BasisTradingStrategyException(
                    f"delta_spot>0 branch invariant violated: delta_hedge={delta_hedge}, "
                    "expected < 0 (delta_hedge = -delta_spot by construction)"
                )
            # B-4: pre-check free margin on HEDGE before issuing the withdraw.
            # Without this guard, the underlying perp entity raises a generic
            # "withdrawal would drop balance below maintenance margin" — wrap
            # it in a strategy-level message that points at the configuration.
            free_margin = hedge.balance - hedge.maintenance_margin
            if -delta_hedge > free_margin:
                raise BasisTradingStrategyException(
                    f"rebalance requires withdrawing {-delta_hedge:.2f} from HEDGE, "
                    f"but free margin is only {free_margin:.2f} "
                    f"(balance={hedge.balance:.2f}, MM={hedge.maintenance_margin:.2f}). "
                    "Tighten MIN_LEVERAGE or widen MAX_LEVERAGE so rebalances "
                    "fire before the hedge becomes margin-bound."
                )
            self._debug(f'delta_spot: {delta_spot} | delta_hedge: {delta_hedge}')

            # Resolved at execute time — captures the pre-buy ``spot_amount``
            # so the open-position size matches exactly what the buy added.
            def _delegate_bought_product(strategy: BaseStrategy,
                                         _initial: float = spot_amount) -> float:
                return _initial - strategy.get_entity('SPOT').internal_state.amount

            return [
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('withdraw', {'amount_in_notional': -delta_hedge})
                ),
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('deposit', {'amount_in_notional': -delta_hedge})
                ),
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('buy', {'amount_in_notional': -delta_hedge})
                ),
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('open_position', {'amount_in_product': _delegate_bought_product})
                ),
            ]
        if delta_spot < 0:  # price_now > price_0, we need to sell spot
            if delta_hedge <= 0:
                raise BasisTradingStrategyException(
                    f"delta_spot<0 branch invariant violated: delta_hedge={delta_hedge}, "
                    "expected > 0 (delta_hedge = -delta_spot by construction)"
                )
            self._debug(f'delta_spot: {delta_spot} | delta_hedge: {delta_hedge}')
            return [
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('sell', {'amount_in_product': -delta_spot / spot.global_state.price})
                ),
                # Deposit must precede withdraw because both share
                # ``_delegate_spot_cash``, which reads the current value
                # of ``SPOT.cash`` at execute time.
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('deposit', {'amount_in_notional': _delegate_spot_cash})
                ),
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('withdraw', {'amount_in_notional': _delegate_spot_cash})
                ),
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('open_position', {'amount_in_product': -delta_spot / spot.global_state.price})
                ),
            ]
        return []

    def _deposit_into_strategy(self) -> List[ActionToTake]:
        """
        Deposit initial funds into the strategy and open a position.
        Returns a list of ActionToTake objects representing the deposit actions to be executed.
        """
        spot_share = self._params.INITIAL_BALANCE - self._params.INITIAL_BALANCE / (
            1 + self._params.TARGET_LEVERAGE
        )
        hedge_share = self._params.INITIAL_BALANCE / (1 + self._params.TARGET_LEVERAGE)
        return [
            ActionToTake(
                entity_name='SPOT',
                action=Action('deposit', {'amount_in_notional': spot_share}),
            ),
            ActionToTake(
                entity_name='HEDGE',
                action=Action('deposit', {'amount_in_notional': hedge_share}),
            ),
            ActionToTake(
                entity_name='SPOT',
                action=Action('buy', {'amount_in_notional': spot_share}),
            ),
            ActionToTake(
                entity_name='HEDGE',
                action=Action('open_position',
                              {'amount_in_product': _delegate_negative_spot_amount}),
            ),
        ]
