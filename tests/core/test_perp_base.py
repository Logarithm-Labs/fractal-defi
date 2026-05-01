"""Smoke tests for the :class:`BasePerpEntity` interface and the
back-compat alias :class:`BaseHedgeEntity`.
"""
import pytest

from fractal.core.entities import BaseHedgeEntity, BasePerpEntity, HyperliquidEntity, SimplePerpEntity


@pytest.mark.core
def test_base_hedge_entity_is_alias_of_base_perp_entity():
    assert BaseHedgeEntity is BasePerpEntity


@pytest.mark.core
@pytest.mark.parametrize("cls", [HyperliquidEntity, SimplePerpEntity])
def test_concrete_perps_are_base_perp_entities(cls):
    assert issubclass(cls, BasePerpEntity)
    # Old name still works for isinstance / issubclass:
    assert issubclass(cls, BaseHedgeEntity)


@pytest.mark.core
@pytest.mark.parametrize("cls", [HyperliquidEntity, SimplePerpEntity])
def test_concrete_perps_expose_uniform_interface(cls):
    """Every concrete perp must declare the four properties + close action."""
    instance = cls()
    # Properties — readable on a fresh instance:
    assert isinstance(instance.size, (int, float))
    assert isinstance(instance.leverage, (int, float))
    assert isinstance(instance.pnl, (int, float))
    # `action_close_position` on a flat entity is a no-op.
    instance.action_close_position()
    assert instance.size == 0
