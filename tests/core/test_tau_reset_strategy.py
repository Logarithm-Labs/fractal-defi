from datetime import datetime

import pytest

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState
from fractal.strategies import TauResetStrategy, TauResetParams


@pytest.fixture
def strategy():
    """
    Initialize the τ-reset strategy with test parameters.
    """
    return TauResetStrategy(
        debug=True,
        params=TauResetParams(
            TAU=100,  # Price range width
            INITIAL_BALANCE=100_000  # Initial liquidity balance
        )
    )

def test_initial_deposit(strategy: TauResetStrategy):
    """
    Test that the strategy correctly deposits funds into the Uniswap LP when no position exists.
    """
    strategy.run(
        [
            Observation(
                timestamp=datetime(2022, 1, 1),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(
                        price=3000,  # Initial price
                        tvl=0,       # Total value locked
                        volume=0,
                        fees=0,
                        liquidity=0
                    )
                }
            )
        ]
    )

    uniswap_lp = strategy.get_entity("UNISWAP_V3")
    assert uniswap_lp.internal_state.cash == 100_000  # Ensure the deposit was made
    assert uniswap_lp.is_position is False  # No position yet opened

def test_rebalance_on_price_outside_range(strategy: TauResetStrategy):
    """
    Test that the strategy rebalances when the price moves outside the [P - τ, P + τ] range.
    """
    strategy.run(
        [
            Observation(
                timestamp=datetime(2022, 1, 1),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3000)  # Initial price
                }
            ),
            Observation(
                timestamp=datetime(2022, 1, 2),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3050)  # Price within range
                }
            ),
            Observation(
                timestamp=datetime(2022, 1, 3),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3200)  # Price outside range
                }
            ),
        ]
    )

    uniswap_lp = strategy.get_entity("UNISWAP_V3")
    assert uniswap_lp.internal_state.cash == 0
    assert uniswap_lp.internal_state.price_lower == 3100  # New lower bound
    assert uniswap_lp.internal_state.price_upper == 3300  # New upper bound

def test_rebalance_within_bounds(strategy: TauResetStrategy):
    """
    Test that the strategy does not rebalance when the price remains within [P - τ, P + τ].
    """
    strategy.run(
        [
            Observation(
                timestamp=datetime(2022, 1, 1),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3000)  # Initial price (deposit here)
                }
            ),
            Observation(
                timestamp=datetime(2022, 1, 2),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3000)  # Initial price
                }
            ),
            Observation(
                timestamp=datetime(2022, 1, 3),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3050)  # Price within range
                }
            ),
        ]
    )

    uniswap_lp = strategy.get_entity("UNISWAP_V3")
    assert uniswap_lp.internal_state.price_lower == 2900  # Original lower bound
    assert uniswap_lp.internal_state.price_upper == 3100  # Original upper bound

def test_multiple_rebalances(strategy: TauResetStrategy):
    """
    Test multiple rebalances as the price fluctuates beyond the defined range.
    """
    strategy.run(
        [
            Observation(
                timestamp=datetime(2022, 1, 1),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3000)
                }
            ),
            Observation(
                timestamp=datetime(2022, 1, 1, 12),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3000)
                }
            ),
            Observation(
                timestamp=datetime(2022, 1, 2),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3200)
                }
            ),
            Observation(
                timestamp=datetime(2022, 1, 3),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3250)
                }
            ),
            Observation(
                timestamp=datetime(2022, 1, 4),
                states={
                    "UNISWAP_V3": UniswapV3LPGlobalState(price=3350)
                }
            ),
        ]
    )

    uniswap_lp = strategy.get_entity("UNISWAP_V3")
    # Ensure multiple reallocations were triggered
    assert uniswap_lp.internal_state.price_init == 3350
    assert uniswap_lp.internal_state.price_lower == 3250
    assert uniswap_lp.internal_state.price_upper == 3450
