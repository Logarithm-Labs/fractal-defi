"""Strategy-configuration lock-ins for HyperliquidBasis and TauResetStrategy."""
from dataclasses import dataclass
from datetime import datetime

import pytest

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState
from fractal.strategies.basis_trading_strategy import \
    BasisTradingStrategyHyperparams
from fractal.strategies.hyperliquid_basis import HyperliquidBasis
from fractal.strategies.tau_reset_strategy import (TauResetException,
                                                   TauResetParams,
                                                   TauResetStrategy)


@dataclass
class _BasisP(BasisTradingStrategyHyperparams):
    EXECUTION_COST: float = 0.0


@pytest.mark.core
def test_hyperliquid_basis_max_leverage_has_safe_default():
    assert HyperliquidBasis.MAX_LEVERAGE == pytest.approx(10.0)


@pytest.mark.core
def test_hyperliquid_basis_constructs_without_overriding_max_leverage():
    s = HyperliquidBasis(params=_BasisP(
        MIN_LEVERAGE=1.0, TARGET_LEVERAGE=3.0, MAX_LEVERAGE=5.0,
        INITIAL_BALANCE=100_000.0, EXECUTION_COST=0.0,
    ))
    assert s.get_entity('HEDGE').max_leverage == pytest.approx(10.0)


@pytest.mark.core
def test_hyperliquid_basis_class_level_override_still_takes_effect():
    class _Probe(HyperliquidBasis):
        MAX_LEVERAGE = 25.0

    s = _Probe(params=_BasisP(
        MIN_LEVERAGE=1.0, TARGET_LEVERAGE=3.0, MAX_LEVERAGE=5.0,
        INITIAL_BALANCE=100_000.0, EXECUTION_COST=0.0,
    ))
    assert s.get_entity('HEDGE').max_leverage == 25.0


@pytest.mark.core
def test_tau_reset_accepts_constructor_kwargs():
    s = TauResetStrategy(
        params=TauResetParams(TAU=15, INITIAL_BALANCE=100_000.0),
        token0_decimals=6, token1_decimals=18, tick_spacing=60,
    )
    assert s._token0_decimals == 6
    assert s._token1_decimals == 18
    assert s._tick_spacing == 60


@pytest.mark.core
def test_tau_reset_constructor_kwargs_take_precedence_over_class_level():
    class _Probe(TauResetStrategy):
        token0_decimals = 18
        token1_decimals = 18
        tick_spacing = 200

    s = _Probe(
        params=TauResetParams(TAU=15, INITIAL_BALANCE=100_000.0),
        token0_decimals=6, tick_spacing=60,
    )
    assert s._token0_decimals == 6
    assert s._token1_decimals == 18
    assert s._tick_spacing == 60


@pytest.mark.core
def test_tau_reset_raises_clear_error_when_pool_config_missing():
    class _Probe(TauResetStrategy):
        pass

    with pytest.raises(TauResetException, match="token0_decimals"):
        _Probe(params=TauResetParams(TAU=15, INITIAL_BALANCE=100_000.0))


@pytest.mark.core
def test_tau_reset_class_level_pattern_still_works():
    class _Probe(TauResetStrategy):
        token0_decimals = 6
        token1_decimals = 18
        tick_spacing = 60

    s = _Probe(params=TauResetParams(TAU=15, INITIAL_BALANCE=100_000.0))
    assert s._token0_decimals == 6
    assert s._token1_decimals == 18
    assert s._tick_spacing == 60


@pytest.mark.core
def test_tau_reset_range_matches_uniswap_v3_tick_formula_for_tau_one():
    # TAU=1, tick_spacing=60 → range = ±60 ticks ≈ ±0.6%.
    s = TauResetStrategy(
        params=TauResetParams(TAU=1, INITIAL_BALANCE=100_000.0),
        token0_decimals=6, token1_decimals=18, tick_spacing=60,
    )
    s.run([
        Observation(timestamp=datetime(2024, 1, 1), states={
            'UNISWAP_V3': UniswapV3LPGlobalState(
                price=1.0, tvl=1_000_000.0, volume=0.0,
                fees=0.0, liquidity=1_000_000.0,
            ),
        }),
        Observation(timestamp=datetime(2024, 1, 2), states={
            'UNISWAP_V3': UniswapV3LPGlobalState(
                price=2.0, tvl=1_000_000.0, volume=0.0,
                fees=0.0, liquidity=1_000_000.0,
            ),
        }),
    ])
    e = s.get_entity('UNISWAP_V3')
    assert e.internal_state.price_upper == pytest.approx(2.0 * (1.0001 ** 60), rel=1e-9)
    assert e.internal_state.price_lower == pytest.approx(2.0 * (1.0001 ** -60), rel=1e-9)
