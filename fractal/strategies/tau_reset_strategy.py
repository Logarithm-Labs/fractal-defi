from dataclasses import dataclass
from typing import List

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity)
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity


@dataclass
class TauResetParams(BaseStrategyParams):
    """
    Parameters for the τ-reset strategy:
    - TAU: The width of the price range (bucket) around the current price.
    - INITIAL_BALANCE: The initial balance for liquidity allocation.
    """
    TAU: float
    INITIAL_BALANCE: float


class TauResetStrategy(BaseStrategy):
    """
    The τ-reset strategy manages liquidity in Uniswap v3 by concentrating it
    within a price range around the current market price. If the price exits this range,
    the liquidity is reallocated. If no position is open, it deposits funds first.

    Based on
    https://drops.dagstuhl.de/storage/00lipics/lipics-vol282-aft2023/LIPIcs.AFT.2023.25/LIPIcs.AFT.2023.25.pdf
    """

    # Decimals for token0 and token1 for Uniswap V3 LP Config
    # This is pool-specific and should be set before running the strategy
    # They are not in the BaseStrategyParams because they are not hyperparameters
    # and are specific to the pool being traded consts.
    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1

    def __init__(self, params: TauResetParams, debug: bool = False, *args, **kwargs):
        self._params: TauResetParams = None  # set for type hinting
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial_funds = False

    def set_up(self):
        """
        Register the Uniswap V3 LP entity to manage liquidity in the pool.
        """
        self.register_entity(NamedEntity(
            entity_name='UNISWAP_V3',
            entity=UniswapV3LPEntity(
                UniswapV3LPConfig(
                    token0_decimals=self.token0_decimals,
                    token1_decimals=self.token1_decimals
                )
            )
        ))
        assert isinstance(self.get_entity('UNISWAP_V3'), UniswapV3LPEntity)

    def predict(self) -> List[ActionToTake]:
        """
        Main logic of the strategy. Checks if the price has moved outside
        the predefined range and takes actions if necessary.
        """
        # Retrieve the pool state from the registered entity
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        global_state = uniswap_entity.global_state
        current_price = global_state.price  # Get the current market price

        # Check if we need to deposit funds into the LP before proceeding
        if not uniswap_entity.is_position and not self.deposited_initial_funds:
            self._debug("No active position. Depositing initial funds...")
            self.deposited_initial_funds = True
            return self._deposit_to_lp()

        if not uniswap_entity.is_position:
            self._debug("No active position. Run first rebalance")
            return self._rebalance()

        # Calculate the boundaries of the price range (bucket)
        lower_bound, upper_bound = uniswap_entity.internal_state.price_lower, uniswap_entity.internal_state.price_upper

        # If the price moves outside the range, reallocate liquidity
        if current_price < lower_bound or current_price > upper_bound:
            self._debug(f"Rebalance {current_price} moved outside range [{lower_bound}, {upper_bound}].")
            return self._rebalance()
        return []

    def _deposit_to_lp(self) -> List[ActionToTake]:
        """
        Deposit funds into the Uniswap LP if no position is currently open.
        """
        return [ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        )]

    def _rebalance(self) -> List[ActionToTake]:
        """
        Reallocate liquidity to a new range centered around the new price.
        """
        actions = []
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')

        # Step 1: Withdraw liquidity from the current range
        if entity.internal_state.liquidity > 0:
            actions.append(
                ActionToTake(entity_name='UNISWAP_V3', action=Action(action='close_position', args={}))
            )
            self._debug("Liquidity withdrawn from the current range.")

        # Step 2: Calculate new range boundaries
        tau = self._params.TAU
        reference_price: float = entity.global_state.price
        tick_spacing = self.tick_spacing
        price_lower = reference_price * 1.0001 ** (-tau * tick_spacing)
        price_upper = reference_price * 1.0001 ** (tau * tick_spacing)

        # Step 3: Open a new position centered around the new price
        delegate_get_cash = lambda obj: obj.get_entity('UNISWAP_V3').internal_state.cash
        actions.append(ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(
                action='open_position',
                args={
                    'amount_in_notional': delegate_get_cash,  # Allocate all available cash
                    'price_lower': price_lower,
                    'price_upper': price_upper
                }
            )
        ))
        self._debug(f"New position opened with range [{price_lower}, {price_upper}].")
        return actions
