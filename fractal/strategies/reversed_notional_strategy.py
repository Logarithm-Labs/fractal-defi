from dataclasses import dataclass
from typing import List, Optional

from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams
from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import BaseHedgeEntity, BaseSpotEntity, GMXV2Entity, StakedETHEntity
from fractal.core.entities.aave import AaveEntity
from fractal.core.entities.lending import BaseLendingEntity
from fractal.core.entities.uniswap_v3_spot import UniswapV3SpotEntity


@dataclass
class ReversedNotionalHyperparams(BaseStrategyParams):
    TARGET_LEVERAGE: float
    MAX_LEVERAGE: float
    TARGET_LTV: float
    MAX_LTV: float
    INITIAL_BALANCE: float


class ReversedNotionalStrategy(BaseStrategy):
    def __init__(
        self,
        *args,
        params: Optional[ReversedNotionalHyperparams] = None,
        debug: bool = False,
        stake: bool = True,
        **kwargs,
    ):
        self._params: ReversedNotionalHyperparams = None  # set for type hinting
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self._stake = stake

    def set_up(self, *args, **kwargs):
        self.register_entity(NamedEntity(entity_name="HEDGE", entity=GMXV2Entity()))
        self.register_entity(NamedEntity(entity_name="LENDING", entity=AaveEntity()))
        if self._stake:
            self.register_entity(NamedEntity(entity_name="SPOT", entity=StakedETHEntity()))
        else:
            self.register_entity(NamedEntity(entity_name="SPOT", entity=UniswapV3SpotEntity()))

    def predict(self, *args, **kwargs) -> List[ActionToTake]:
        hedge: BaseHedgeEntity = self.get_entity("HEDGE")
        spot: BaseSpotEntity = self.get_entity("SPOT")
        lending: BaseLendingEntity = self.get_entity("LENDING")

        if hedge.balance == 0 and spot.balance == 0 and lending.balance == 0:
            self._debug("Depositing initial funds into the strategy...")
            return self._deposit_into_strategy()
        if hedge.balance == 0 and spot.balance > 0 and lending.balance > 0:
            self._debug("Hedge is liquidated. Re-hedging...")
            return self._rehedge(liquidated=True)
        if lending.balance == 0 and spot.balance > 0 and hedge.balance > 0:
            self._debug("Lending is liquidated. Re-lending...")
            return self._relend(liquidated=True)
        if lending.ltv > self._params.MAX_LTV:
            self._debug("LTV exceeds maximum. Repaying...")
            return self._relend()
        if hedge.leverage > self._params.MAX_LEVERAGE:
            self._debug("Leverage exceeds maximum. Re-hedging...")
            return self._rehedge()
        return []

    def _relend(self, liquidated: bool = False) -> List[ActionToTake]:
        hedge: BaseHedgeEntity = self.get_entity("HEDGE")
        spot: BaseSpotEntity = self.get_entity("SPOT")
        lending: BaseLendingEntity = self.get_entity("LENDING")

        repay_amount = lending.calculate_repay(self._params.TARGET_LTV)

        if liquidated:
            self._params.INITIAL_BALANCE = spot.balance + hedge.balance
            return [
                ActionToTake(
                    entity_name="SPOT",
                    action=Action("sell", {"amount_in_product": spot.internal_state.amount}),
                ),
                ActionToTake(
                    entity_name="SPOT",
                    action=Action("withdraw", {"amount_in_notional": spot.internal_state.cash}),
                ),
                ActionToTake(
                    entity_name="HEDGE",
                    action=Action("open_position", {"amount_in_product": -hedge.internal_state.size}),
                ),
                ActionToTake(
                    entity_name="HEDGE",
                    action=Action("withdraw", {"amount_in_notional": hedge.internal_state.collateral}),
                ),
            ]

        if repay_amount > 0:
            spot_cash = lambda obj: obj.get_entity("SPOT").internal_state.cash

            x = (hedge.internal_state.collateral * repay_amount) / (
                hedge.internal_state.collateral - hedge.size * hedge.global_state.price
            )
            y = (-hedge.size * repay_amount) / (
                hedge.internal_state.collateral - hedge.size * hedge.global_state.price
            )
            repay_amount_real = lambda obj: obj.get_entity("SPOT").internal_state.cash + x
            return [
                ActionToTake(
                    entity_name="SPOT",
                    action=Action("sell", {"amount_in_product": y}),
                ),
                ActionToTake(
                    entity_name="HEDGE",
                    action=Action("open_position", {"amount_in_product": y}),
                ),
                ActionToTake(
                    entity_name="HEDGE",
                    action=Action("withdraw", {"amount_in_notional": x}),
                ),
                ActionToTake(
                    entity_name="LENDING",
                    action=Action("redeem", {"amount_in_product": repay_amount_real}),
                ),
                ActionToTake(
                    entity_name="SPOT",
                    action=Action("withdraw", {"amount_in_notional": spot_cash}),
                ),
            ]

    def _rehedge(self, liquidated: bool = False) -> List[ActionToTake]:
        """
        Rebalance the entities to maintain the target leverage ratio.
        Returns a list of ActionToTake objects representing the rebalancing actions to be executed.
        """
        hedge: BaseHedgeEntity = self.get_entity("HEDGE")
        spot: BaseSpotEntity = self.get_entity("SPOT")
        hedge_balance = hedge.balance
        spot_balance = spot.balance
        spot_amount = spot.internal_state.amount

        equity = hedge_balance + spot_balance
        target_hedge = equity / (1 + self._params.TARGET_LEVERAGE)
        target_spot = equity - target_hedge

        delta_spot = target_spot - spot_balance
        delta_hedge = target_hedge - hedge_balance

        if liquidated:  # hedge is liquidated
            assert hedge.size == 0
            assert spot_amount > 0
            delegate_get_cash = lambda obj: obj.get_entity("SPOT").internal_state.cash  # in notional
            return [
                ActionToTake(
                    entity_name="SPOT",
                    action=Action("sell", {"amount_in_product": -delta_spot / spot.global_state.price}),
                ),
                ActionToTake(entity_name="HEDGE", action=Action("deposit", {"amount_in_notional": delegate_get_cash})),
                ActionToTake(
                    entity_name="HEDGE",
                    action=Action(
                        "open_position",
                        {"amount_in_product": (self._params.TARGET_LEVERAGE * delta_spot / spot.global_state.price)},
                    ),
                ),
                ActionToTake(entity_name="SPOT", action=Action("withdraw", {"amount_in_notional": delegate_get_cash})),
            ]

        if delta_spot > 0:  # price_now < price_0, we need to buy spot
            assert delta_hedge < 0
            self._debug(f"delta_spot: {delta_spot} | delta_hedge: {delta_hedge}")

            # in product
            spot_bought_product_lambda = lambda obj: (spot_amount - obj.get_entity("SPOT").internal_state.amount)

            return [
                ActionToTake(entity_name="HEDGE", action=Action("withdraw", {"amount_in_notional": -delta_hedge})),
                ActionToTake(entity_name="SPOT", action=Action("deposit", {"amount_in_notional": -delta_hedge})),
                ActionToTake(entity_name="SPOT", action=Action("buy", {"amount_in_notional": -delta_hedge})),
                ActionToTake(
                    entity_name="HEDGE",
                    action=Action("open_position", {"amount_in_product": spot_bought_product_lambda}),
                ),
            ]
        if delta_spot < 0:  # price_now > price_0, we need to sell spot
            assert delta_hedge > 0
            self._debug(f"delta_spot: {delta_spot} | delta_hedge: {delta_hedge}")
            delegate_get_cash = lambda obj: obj.get_entity("SPOT").internal_state.cash  # in notional
            return [
                ActionToTake(
                    entity_name="SPOT",
                    action=Action("sell", {"amount_in_product": -delta_spot / spot.global_state.price}),
                ),
                ActionToTake(entity_name="HEDGE", action=Action("deposit", {"amount_in_notional": delegate_get_cash})),
                ActionToTake(
                    entity_name="HEDGE",
                    action=Action("open_position", {"amount_in_product": -delta_spot / spot.global_state.price}),
                ),
                ActionToTake(entity_name="SPOT", action=Action("withdraw", {"amount_in_notional": delegate_get_cash})),
            ]
        return []

    def _deposit_into_strategy(self) -> List[ActionToTake]:

        lending: BaseLendingEntity = self.get_entity("LENDING")

        product_to_hedge_lambda = lambda obj: -obj.get_entity("SPOT").internal_state.amount

        borrowed_balance = self._params.INITIAL_BALANCE * self._params.TARGET_LTV * lending.global_state.notional_price

        return [
            ActionToTake(
                entity_name="LENDING",
                action=Action("deposit", {"amount_in_notional": (self._params.INITIAL_BALANCE)}),
            ),
            ActionToTake(
                entity_name="LENDING",
                action=Action(
                    "borrow",
                    {"amount_in_product": (borrowed_balance)},
                ),
            ),
            ActionToTake(
                entity_name="SPOT",
                action=Action(
                    "deposit",
                    {"amount_in_notional": (borrowed_balance - borrowed_balance / (1 + self._params.TARGET_LEVERAGE))},
                ),
            ),
            ActionToTake(
                entity_name="HEDGE",
                action=Action(
                    "deposit", {"amount_in_notional": borrowed_balance / (1 + self._params.TARGET_LEVERAGE)}
                ),
            ),
            ActionToTake(
                entity_name="SPOT",
                action=Action(
                    "buy",
                    {"amount_in_notional": (borrowed_balance - borrowed_balance / (1 + self._params.TARGET_LEVERAGE))},
                ),
            ),
            ActionToTake(
                entity_name="HEDGE",
                action=Action("open_position", {"amount_in_product": product_to_hedge_lambda}),
            ),
        ]
