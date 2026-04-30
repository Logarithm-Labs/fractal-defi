"""Lock-in: class-level state annotations must not leak state between instances.

The protocol entities declare ``_internal_state: <Type>`` / ``_global_state:
<Type>`` at class scope. These are *annotations*, not assignments — so each
instance still gets its own object via ``_initialize_states``. This file
guards against a regression where someone changes the annotation to an
assignment (``_internal_state: <Type> = <Type>()``) and accidentally makes
state class-shared. That bug would only show up the second time someone
constructed two entities of the same type in one backtest.
"""
import pytest

from fractal.core.entities.protocols.aave import AaveEntity
from fractal.core.entities.protocols.gmx_v2 import GMXV2Entity
from fractal.core.entities.protocols.hyperliquid import HyperliquidEntity
from fractal.core.entities.protocols.steth import StakedETHEntity
from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                            UniswapV2LPEntity)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                            UniswapV3LPEntity)
from fractal.core.entities.protocols.uniswap_v3_spot import UniswapV3SpotEntity


def _make_aave():
    return AaveEntity()


def _make_gmx():
    return GMXV2Entity()


def _make_hl():
    return HyperliquidEntity()


def _make_steth():
    return StakedETHEntity()


def _make_uni_v2():
    return UniswapV2LPEntity(UniswapV2LPConfig())


def _make_uni_v3_lp():
    return UniswapV3LPEntity(UniswapV3LPConfig())


def _make_uni_v3_spot():
    return UniswapV3SpotEntity()


@pytest.mark.core
@pytest.mark.parametrize("factory", [
    _make_aave,
    _make_gmx,
    _make_hl,
    _make_steth,
    _make_uni_v2,
    _make_uni_v3_lp,
    _make_uni_v3_spot,
])
def test_internal_state_is_per_instance(factory):
    e1 = factory()
    e2 = factory()
    assert e1._internal_state is not e2._internal_state, (
        f"{type(e1).__name__}: internal_state shared across instances "
        "— class-level annotation likely became class-level assignment."
    )


@pytest.mark.core
@pytest.mark.parametrize("factory", [
    _make_aave,
    _make_gmx,
    _make_hl,
    _make_steth,
    _make_uni_v2,
    _make_uni_v3_lp,
    _make_uni_v3_spot,
])
def test_global_state_is_per_instance(factory):
    e1 = factory()
    e2 = factory()
    assert e1._global_state is not e2._global_state, (
        f"{type(e1).__name__}: global_state shared across instances "
        "— class-level annotation likely became class-level assignment."
    )


@pytest.mark.core
def test_aave_internal_state_mutation_does_not_leak():
    e1 = AaveEntity()
    e2 = AaveEntity()
    e1._internal_state.collateral = 100.0
    e2._internal_state.collateral = 200.0
    assert e1._internal_state.collateral == 100.0
    assert e2._internal_state.collateral == 200.0


@pytest.mark.core
def test_uniswap_v2_internal_state_mutation_does_not_leak():
    e1 = UniswapV2LPEntity(UniswapV2LPConfig())
    e2 = UniswapV2LPEntity(UniswapV2LPConfig())
    e1._internal_state.cash = 1000.0
    e2._internal_state.cash = 2000.0
    assert e1._internal_state.cash == 1000.0
    assert e2._internal_state.cash == 2000.0


@pytest.mark.core
def test_hyperliquid_positions_list_is_per_instance():
    """Lists are mutable — extra-important they are not shared."""
    from fractal.core.entities.protocols.hyperliquid import HyperLiquidPosition

    e1 = HyperliquidEntity()
    e2 = HyperliquidEntity()
    e1._internal_state.positions.append(
        HyperLiquidPosition(amount=1.0, entry_price=100.0, max_leverage=10)
    )
    assert len(e1._internal_state.positions) == 1
    assert len(e2._internal_state.positions) == 0
    assert e1._internal_state.positions is not e2._internal_state.positions
