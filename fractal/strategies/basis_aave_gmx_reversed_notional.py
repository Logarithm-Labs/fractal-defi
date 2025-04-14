from dataclasses import dataclass
from typing import List, Optional

from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams
from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import BaseHedgeEntity, BaseSpotEntity, GMXV2Entity, StakedETHEntity
from fractal.core.entities.aave import AaveEntity
from fractal.core.entities.lending import BaseLendingEntity
from fractal.core.entities.uniswap_v3_spot import UniswapV3SpotEntity


@dataclass
class BasisAaveGmxReversedNotionalParams(BaseStrategyParams):
    """
    Hyperparameters for the BasisAaveGmxReversedNotional strategy.

    Attributes:
        TARGET_LEVERAGE (float): The target leverage ratio.
        MAX_LEVERAGE (float): The maximum leverage ratio.
        TARGET_LTV (float): The target loan-to-value ratio.
        MAX_LTV (float): The maximum loan-to-value ratio.
        INITIAL_BALANCE (float): The initial balance to deposit into the strategy.
        STAKE (bool): Indicates whether to stake spot.
    """
    TARGET_LEVERAGE: float
    MAX_LEVERAGE: float
    TARGET_LTV: float
    MAX_LTV: float
    INITIAL_BALANCE: float
    STAKE: bool


class BasisAaveGmxReversedNotional(BaseStrategy):
    """
    A basis strategy with non usd notional.
    Strategy is based on the following steps:
        1. Lend notional (for example, ETH) to Aave.
        2. Borrow stablecoin (for example, USDC) from Aave.
        3. Buy spot (for example, ETH) with borrowed stablecoin.
        4. Open a short position on GMX with spot.
        5. Rebalance the entities to maintain the target leverage ratio.
        6. Re-lend the assets to maintain the target loan-to-value ratio.

    Example strategy using WETH notional, AAVE V3 lending, and WETH-USDC GMXV2-UNIV3 underlying Basis strategy:
    1. Starting with Y in WETH:
        - Deposit Y WETH on AAVE as collateral.
        - Borrow X = Y * LTV * P USDC from AAVE.
        - Buy (X * LVG) / (P * (LVG + 1)) WETH on UNIV3 for (X * LVG) / (LVG + 1) USDC.
        - Deposit X / (LVG + 1) USDC as collateral on GMXV2.
        - Open a short position on GMXV2 with the same size in WETH as on UNIV3.

    2. If leverage on GMXV2 exceeds the maximum leverage:
        - Sell part of WETH on UNIV3.
        - Transfer cash to GMXV2.
        - Partially close the GMXV2 short to maintain a delta-neutral strategy.

    3. If LTV on AAVE exceeds the maximum LTV:
        - Calculate the needed repay amount as:
            N = L_A * LTV_0 * P - B_A,
            where L_A is the amount lent on AAVE, B_A is the amount borrowed on AAVE, 
            P is the current price, and LTV_0 is the target LTV.
        - Calculate:
            alpha = (C_H * N) / (S_H * P + C_H),
            beta = (S_H * N) / (S_H * P + C_H),
            where C_H is the GMXV2 collateral, and S_H is the size of the short.
        - Sell beta amount of WETH on UNIV3.
        - Partially close the position on GMXV2 for the same size.
        - Withdraw alpha USDC from GMXV2 to AAVE collateral.
        - Withdraw beta * P USDC from UNIV3 to AAVE collateral.

    Attributes:
        _params (BasisAaveGmxReversedNotionalParams): Hyperparameters for the strategy.
        _debug (bool): Flag to enable debug mode.

    Examples:
        For correct calculation of metrics, the notional price should be provided.

        >>> result = BasisAaveGmxReversedNotional(params).run()
        >>> metrics = result.get_metrics(result.to_dataframe(), notional_price='SPOT_price')
    """

    def __init__(
        self,
        *args,
        params: Optional[BasisAaveGmxReversedNotionalParams] = None,
        debug: bool = False,
        **kwargs,
    ):
        """
        Initializes the BasisAaveGmxReversedNotional.

        Args:
            *args: Variable length argument list.
            params (Optional[BasisAaveGmxReversedNotionalParams]): Hyperparameters for the strategy.
            debug (bool): Flag to enable debug mode.
            **kwargs: Arbitrary keyword arguments.
        """
        self._params: BasisAaveGmxReversedNotionalParams = None  # set for type hinting
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self._stake = params.STAKE

    def set_up(self, *args, **kwargs):
        """
        Sets up the strategy by registering entities.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.register_entity(NamedEntity(entity_name="HEDGE", entity=GMXV2Entity()))
        self.register_entity(NamedEntity(entity_name="LENDING", entity=AaveEntity()))
        if self._stake:
            self.register_entity(NamedEntity(entity_name="SPOT", entity=StakedETHEntity()))
        else:
            self.register_entity(NamedEntity(entity_name="SPOT", entity=UniswapV3SpotEntity()))

    def predict(self, *args, **kwargs) -> List[ActionToTake]:
        """
        Predicts the actions to take based on the current state of the entities.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[ActionToTake]: A list of actions to take.
        """
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
        """
        Re-lends the assets to maintain the target loan-to-value ratio.

        Args:
            liquidated (bool): Flag to indicate if the lending entity is liquidated.

        Returns:
            List[ActionToTake]: A list of actions to take.
        """
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
        Rebalances the entities to maintain the target leverage ratio.

        Args:
            liquidated (bool): Flag to indicate if the hedge entity is liquidated.

        Returns:
            List[ActionToTake]: A list of actions to take.
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
        """
        Deposits initial funds into the strategy.

        Returns:
            List[ActionToTake]: A list of actions to take.
        """
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
