from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams)
from fractal.core.entities import BaseHedgeEntity, BaseSpotEntity


class BasisTradingStrategyException(Exception):
    pass


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


class BasisTradingStrategy(BaseStrategy):
    """
    A basis trading strategy that implements the BaseStrategy interface.
    This strategy aims to maintain a target leverage ratio between a hedge entity and a spot entity.
    """
    def __init__(self, *args, params: Optional[BasisTradingStrategyHyperparams] = None,
                 debug: bool = False, **kwargs):
        self._params: BasisTradingStrategyHyperparams = None  # set for type hinting
        super().__init__(params=params, debug=debug, *args, **kwargs)

    def set_up(self, *args, **kwargs):
        """
        Set up the strategy by registering the hedge and spot entities.
        """
        # Check if the SPOT and HEDGE entities are already registered
        assert isinstance(self.get_entity('HEDGE'), BaseHedgeEntity)
        assert isinstance(self.get_entity('SPOT'), BaseSpotEntity)

    def predict(self, *args, **kwargs) -> List[ActionToTake]:
        """
        Predict the actions to take based on the current state of the entities.
        Returns a list of ActionToTake objects representing the actions to be executed.
        """
        hedge: BaseHedgeEntity = self.get_entity('HEDGE')
        spot: BaseSpotEntity = self.get_entity('SPOT')
        if hedge.balance == 0 and spot.balance == 0:
            self._debug("Depositing initial funds into the strategy...")
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
        hedge: BaseHedgeEntity = self.get_entity('HEDGE')
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
            assert hedge.size == 0
            assert spot_amount > 0
            delegate_get_cash = lambda obj: obj.get_entity('SPOT').internal_state.cash  # in notional
            return [
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('sell', {'amount_in_product': -delta_spot / spot.global_state.price})
                ),
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('deposit', {'amount_in_notional': delegate_get_cash})
                ),
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('open_position', {'amount_in_product': (self._params.TARGET_LEVERAGE * delta_spot /
                                                    spot.global_state.price)})
                ),
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('withdraw', {'amount_in_notional': delegate_get_cash})
                ),
            ]

        assert np.abs(hedge.size + spot_amount) / np.abs(hedge.size - spot_amount) <= 1e-6  # hedge.size ~= -spot_amount

        if delta_spot > 0:  # price_now < price_0, we need to buy spot
            assert delta_hedge < 0
            self._debug(f'delta_spot: {delta_spot} | delta_hedge: {delta_hedge}')

            # in product
            spot_bought_product_lambda = lambda obj: (spot_amount - obj.get_entity('SPOT').internal_state.amount)

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
                    action=Action('open_position', {'amount_in_product': spot_bought_product_lambda})
                ),
            ]
        if delta_spot < 0:  # price_now > price_0, we need to sell spot
            assert delta_hedge > 0
            self._debug(f'delta_spot: {delta_spot} | delta_hedge: {delta_hedge}')
            delegate_get_cash = lambda obj: obj.get_entity('SPOT').internal_state.cash  # in notional
            return [
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('sell', {'amount_in_product': -delta_spot / spot.global_state.price})
                ),
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('deposit', {'amount_in_notional': delegate_get_cash})
                ),
                ActionToTake(
                    entity_name='HEDGE',
                    action=Action('open_position', {'amount_in_product': -delta_spot / spot.global_state.price})
                ),
                ActionToTake(
                    entity_name='SPOT',
                    action=Action('withdraw', {'amount_in_notional': delegate_get_cash})
                ),
            ]
        return []

    def _deposit_into_strategy(self) -> List[ActionToTake]:
        """
        Deposit initial funds into the strategy and open a position.
        Returns a list of ActionToTake objects representing the deposit actions to be executed.
        """
        product_to_hedge_lambda = lambda obj: -obj.get_entity('SPOT').internal_state.amount
        return [
            ActionToTake(
                entity_name='SPOT',
                action=Action(
                    'deposit',
                    {'amount_in_notional': (self._params.INITIAL_BALANCE - self._params.INITIAL_BALANCE /
                                            (1 + self._params.TARGET_LEVERAGE))})
            ),
            ActionToTake(
                entity_name='HEDGE',
                action=Action(
                    'deposit',
                    {'amount_in_notional': self._params.INITIAL_BALANCE /
                     (1 + self._params.TARGET_LEVERAGE)})
            ),
            ActionToTake(
                entity_name='SPOT',
                action=Action(
                    'buy',
                    {'amount_in_notional': (self._params.INITIAL_BALANCE - self._params.INITIAL_BALANCE /
                                            (1 + self._params.TARGET_LEVERAGE))})
            ),
            ActionToTake(
                entity_name='HEDGE',
                action=Action('open_position', {'amount_in_product': product_to_hedge_lambda})
            ),
        ]
