from dataclasses import dataclass
from typing import List, Optional

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity)
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity


class TauResetException(Exception):
    pass


def _delegate_uniswap_cash(strategy: BaseStrategy) -> float:
    """Resolve UniswapV3 LP cash lazily — used as a delegate so the
    open-position action sees the cash that ``close_position`` released
    in the same step."""
    return strategy.get_entity('UNISWAP_V3').internal_state.cash


@dataclass
class TauResetParams(BaseStrategyParams):
    """
    Parameters for the τ-reset strategy.

    TAU: float — width of the price range expressed in **buckets**, where
        one bucket equals ``tick_spacing`` ticks (Uniswap V3 convention).
        The active range becomes ``[P · 1.0001^(-TAU·tick_spacing),
        P · 1.0001^(TAU·tick_spacing)]`` around the reference price ``P``.
        Example: ``TAU=1, tick_spacing=60`` ⇒ range ≈ ±0.6% around ``P``.
        Larger TAU = wider range = lower fee yield density but rarer
        rebalances.
    INITIAL_BALANCE: float — Initial balance for liquidity allocation.
    """
    TAU: float
    INITIAL_BALANCE: float


class TauResetStrategy(BaseStrategy[TauResetParams]):
    """
    The τ-reset strategy manages liquidity in Uniswap v3 by concentrating it
    within a price range around the current market price. If the price exits this range,
    the liquidity is reallocated. If no position is open, it deposits funds first.

    Based on
    https://drops.dagstuhl.de/storage/00lipics/lipics-vol282-aft2023/LIPIcs.AFT.2023.25/LIPIcs.AFT.2023.25.pdf

    Pool config (``token0_decimals``, ``token1_decimals``, ``tick_spacing``)
    can be supplied either via constructor kwargs (preferred — instance
    state, safe under parallel runs) or via class attributes (legacy
    pattern). Constructor kwargs take precedence; class-level fallback
    is preserved for backwards compatibility.
    """

    # Pool-specific constants. Class-level defaults used as a fallback when
    # constructor kwargs are not provided. ``-1`` is the unset sentinel.
    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1

    def __init__(
        self,
        params: TauResetParams,
        *args,
        debug: bool = False,
        token0_decimals: Optional[int] = None,
        token1_decimals: Optional[int] = None,
        tick_spacing: Optional[int] = None,
        **kwargs,
    ):
        # Resolve precedence: constructor kwargs > class-level > sentinel.
        self._token0_decimals = (
            token0_decimals if token0_decimals is not None
            else self.__class__.token0_decimals
        )
        self._token1_decimals = (
            token1_decimals if token1_decimals is not None
            else self.__class__.token1_decimals
        )
        self._tick_spacing = (
            tick_spacing if tick_spacing is not None
            else self.__class__.tick_spacing
        )
        if self._token0_decimals == -1 or self._token1_decimals == -1 or self._tick_spacing == -1:
            raise TauResetException(
                "TauResetStrategy needs token0_decimals, token1_decimals, "
                "and tick_spacing — pass them as constructor kwargs "
                "(preferred) or set them on the class attribute."
            )
        super().__init__(params=params, debug=debug, *args, **kwargs)
        # Instance attribute (NOT class-level) so independent strategy
        # instances run side-by-side without sharing the deposit flag.
        self.deposited_initial_funds = False

    def set_up(self):
        """
        Register the Uniswap V3 LP entity to manage liquidity in the pool.
        """
        self.register_entity(NamedEntity(
            entity_name='UNISWAP_V3',
            entity=UniswapV3LPEntity(
                UniswapV3LPConfig(
                    token0_decimals=self._token0_decimals,
                    token1_decimals=self._token1_decimals,
                )
            )
        ))

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

        # Step 2: Calculate new range boundaries.
        # TAU is in **buckets**, each bucket = ``tick_spacing`` ticks.
        # Range covers ±(TAU·tick_spacing) ticks around ``reference_price``
        # using the Uniswap V3 ``1.0001^tick`` price-from-tick formula.
        tau = self._params.TAU
        reference_price: float = entity.global_state.price
        tick_spacing = self._tick_spacing
        price_lower = reference_price * 1.0001 ** (-tau * tick_spacing)
        price_upper = reference_price * 1.0001 ** (tau * tick_spacing)

        # Step 3: Open a new position centered around the new price.
        # ``_delegate_uniswap_cash`` resolves at execute time so the
        # open-position step sees the cash freed by close_position.
        actions.append(ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(
                action='open_position',
                args={
                    'amount_in_notional': _delegate_uniswap_cash,
                    'price_lower': price_lower,
                    'price_upper': price_upper,
                }
            )
        ))
        self._debug(f"New position opened with range [{price_lower}, {price_upper}].")
        return actions
