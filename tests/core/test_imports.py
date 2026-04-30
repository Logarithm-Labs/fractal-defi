"""Import-smoke tests for the public core/base + entities API.

These guard against accidental breakage when files are moved, classes
renamed, or :data:`__all__` drifts away from the actual exports.
"""
import importlib

import pytest


# --------------------------------------------------------- core.base
@pytest.mark.core
def test_fractal_core_base_exports_load():
    from fractal.core.base import (Action, ActionToTake, BaseEntity,
                                   BaseStrategy, BaseStrategyParams,
                                   EntityException, GlobalState,
                                   InternalState, NamedEntity, Observation,
                                   ObservationsStorage)
    assert Action and ActionToTake and BaseEntity and BaseStrategy
    assert BaseStrategyParams and EntityException and GlobalState
    assert InternalState and NamedEntity and Observation and ObservationsStorage


@pytest.mark.core
def test_fractal_core_base_observations_exports():
    from fractal.core.base.observations import (Observation,
                                                ObservationsStorage,
                                                SQLiteObservationsStorage)
    assert Observation and ObservationsStorage and SQLiteObservationsStorage


@pytest.mark.core
def test_fractal_core_base_strategy_exports():
    from fractal.core.base.strategy import (ActionToTake, BaseStrategy,
                                            BaseStrategyParams, NamedEntity,
                                            StrategyMetrics, StrategyResult)
    assert ActionToTake and BaseStrategy and BaseStrategyParams
    assert NamedEntity and StrategyMetrics and StrategyResult


# --------------------------------------------------------- entities
@pytest.mark.core
def test_entities_top_level_all_consistent():
    """Every name in ``__all__`` must resolve to a real attribute."""
    mod = importlib.import_module("fractal.core.entities")
    for name in mod.__all__:
        assert hasattr(mod, name), f"{name} declared in __all__ but missing"


@pytest.mark.core
def test_entities_base_subpackage():
    from fractal.core.entities.base import (BaseHedgeEntity, BaseLendingEntity,
                                            BasePerpEntity,
                                            BasePerpInternalState,
                                            BasePoolEntity, BaseSpotEntity,
                                            BaseSpotInternalState)
    # BaseHedgeEntity is a transparent alias of BasePerpEntity
    assert BaseHedgeEntity is BasePerpEntity
    assert BaseLendingEntity and BasePoolEntity
    assert BaseSpotEntity and BaseSpotInternalState
    assert BasePerpInternalState


@pytest.mark.core
def test_entities_simple_subpackage():
    from fractal.core.entities.simple import (
        SimpleLendingEntity, SimpleLendingGlobalState, SimpleLendingInternalState,
        SimpleLiquidStakingToken, SimpleLiquidStakingTokenGlobalState,
        SimpleLiquidStakingTokenInternalState,
        SimplePerpEntity, SimplePerpGlobalState, SimplePerpInternalState,
        SimplePoolEntity, SimplePoolGlobalState, SimplePoolInternalState,
        SimpleSpotExchange, SimpleSpotExchangeGlobalState,
        SimpleSpotExchangeInternalState,
    )
    # All five simple categories present and instantiable.
    for cls in (SimpleLendingEntity, SimpleLiquidStakingToken,
                SimplePerpEntity, SimplePoolEntity, SimpleSpotExchange):
        assert cls() is not None
    # State pairs exist
    assert (SimpleLendingGlobalState and SimpleLendingInternalState
            and SimpleLiquidStakingTokenGlobalState and SimpleLiquidStakingTokenInternalState
            and SimplePerpGlobalState and SimplePerpInternalState
            and SimplePoolGlobalState and SimplePoolInternalState
            and SimpleSpotExchangeGlobalState and SimpleSpotExchangeInternalState)


@pytest.mark.core
def test_entities_protocols_subpackage():
    from fractal.core.entities.protocols import (AaveEntity, GMXV2Entity,
                                                 HyperliquidEntity,
                                                 StakedETHEntity,
                                                 UniswapV2LPEntity,
                                                 UniswapV3LPEntity,
                                                 UniswapV3SpotEntity)
    assert all((AaveEntity, GMXV2Entity, HyperliquidEntity, StakedETHEntity,
                UniswapV2LPEntity, UniswapV3LPEntity, UniswapV3SpotEntity))


# ------------------------------------------------- deprecated aliases
@pytest.mark.core
def test_single_spot_exchange_alias_resolves():
    from fractal.core.entities.simple.spot import SimpleSpotExchange
    from fractal.core.entities.single_spot_exchange import SingleSpotExchange
    # alias subclasses the new class
    assert issubclass(SingleSpotExchange, SimpleSpotExchange)


@pytest.mark.core
def test_base_hedge_entity_alias_resolves():
    from fractal.core.entities import BaseHedgeEntity, BasePerpEntity
    assert BaseHedgeEntity is BasePerpEntity


# ----------------------------- nested base imports through old paths
@pytest.mark.core
def test_old_module_paths_no_longer_exist():
    """Files moved into base/, simple/, protocols/. Old top-level paths
    should fail loudly (not silently shadow the moved modules)."""
    for old in (
        "fractal.core.entities.perp",
        "fractal.core.entities.spot",
        "fractal.core.entities.lending",
        "fractal.core.entities.pool",
        "fractal.core.entities.hedge",
        "fractal.core.entities.simple_perp",
        "fractal.core.entities.simple_spot",
        "fractal.core.entities.hyperliquid",
        "fractal.core.entities.gmx_v2",
        "fractal.core.entities.aave",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old)
