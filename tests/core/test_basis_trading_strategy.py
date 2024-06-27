from datetime import datetime

import numpy as np
import pytest

from fractal.core.base import Observation
from fractal.core.entities import GMXV2GlobalState, UniswapV3SpotGlobalState
from fractal.strategies import (BasisTradingStrategyHyperparams,
                                GMXV2UniswapV3Basis)


@pytest.fixture
def strategy():
    return GMXV2UniswapV3Basis(
        debug=True,
        params=BasisTradingStrategyHyperparams(
            TARGET_LEVERAGE=3, MAX_LEVERAGE=5, MIN_LEVERAGE=1.5, INITIAL_BALANCE=1000000
        )
    )

def test_run(strategy: GMXV2UniswapV3Basis):
    strategy.run(
        [
            Observation(
                timestamp=datetime(2022, 1, 1),
                states={
                    "HEDGE": GMXV2GlobalState(
                        price=3000, funding_rate_short=0, borrowing_rate_short=0
                    ),
                    "SPOT": UniswapV3SpotGlobalState(price=3000),
                },
            ),
            Observation(
                timestamp=datetime(2022, 1, 2),
                states={
                    "HEDGE": GMXV2GlobalState(
                        price=3100, funding_rate_short=0, borrowing_rate_short=0
                    ),
                    "SPOT": UniswapV3SpotGlobalState(price=3100),
                },
            ),
        ]
    )

    hedge = strategy.get_entity("HEDGE")
    spot = strategy.get_entity("SPOT")
    assert hedge.size == -spot.internal_state.amount
    assert spot.internal_state.amount == 250 * (1 - spot.TRADING_FEE)
    assert hedge.balance == (1e6 / 4) - spot.internal_state.amount * hedge.TRADING_FEE * 3000 - spot.internal_state.amount * (3100 - 3000)
    assert spot.balance == spot.internal_state.amount * 3100

def test_run_with_min_rebalance(strategy: GMXV2UniswapV3Basis):
    strategy.run(
        [
            Observation(
                timestamp=datetime(2022, 1, 1),
                states={
                    "HEDGE": GMXV2GlobalState(
                        price=3000, funding_rate_short=0, borrowing_rate_short=0
                    ),
                    "SPOT": UniswapV3SpotGlobalState(price=3000),
                },
            ),
            Observation(
                timestamp=datetime(2022, 1, 2),
                states={
                    "HEDGE": GMXV2GlobalState(
                        price=2000, funding_rate_short=0, borrowing_rate_short=0
                    ),
                    "SPOT": UniswapV3SpotGlobalState(price=2000),
                },
            ),
        ]
    )

    hedge = strategy.get_entity("HEDGE")
    spot = strategy.get_entity("SPOT")
    assert hedge.size == -spot.internal_state.amount

    # 0.5% tolerance
    assert np.abs(hedge.leverage / 3 - 1) < 5e-3
    assert np.abs(spot.internal_state.cash) < 5e-3
    assert np.abs(spot.internal_state.amount * 2000 / 750000 - 1) < 5e-3
    assert np.abs(hedge.balance / 250000 - 1) < 5e-3
    assert np.abs(spot.balance / 750000 - 1) < 5e-3

def test_run_with_max_rebalance(strategy: GMXV2UniswapV3Basis):
    strategy.run(
        [
            Observation(
                timestamp=datetime(2022, 1, 1),
                states={
                    "HEDGE": GMXV2GlobalState(
                        price=3000, funding_rate_short=0, borrowing_rate_short=0
                    ),
                    "SPOT": UniswapV3SpotGlobalState(price=3000),
                },
            ),
            Observation(
                timestamp=datetime(2022, 1, 2),
                states={
                    "HEDGE": GMXV2GlobalState(
                        price=3500, funding_rate_short=0, borrowing_rate_short=0
                    ),
                    "SPOT": UniswapV3SpotGlobalState(price=3500),
                },
            ),
        ]
    )

    hedge = strategy.get_entity("HEDGE")
    spot = strategy.get_entity("SPOT")
    assert hedge.size == -spot.internal_state.amount
    
    # 0.5% tolerance
    assert np.abs(hedge.leverage / 3 - 1) < 5e-3
    assert np.abs(spot.internal_state.cash) < 5e-3
    assert np.abs(spot.internal_state.amount * 3500 / 750000 - 1) < 5e-3
    assert np.abs(hedge.balance / 250000 - 1) < 5e-3
    assert np.abs(spot.balance / 750000 - 1) < 5e-3
