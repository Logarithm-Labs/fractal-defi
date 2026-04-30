"""Tests for the ``BaseStrategy[Params]`` generic + auto-derived ``PARAMS_CLS``
+ ``params=None`` fallback to defaults.
"""
from dataclasses import dataclass

import pytest

from fractal.core.base import (BaseStrategy, BaseStrategyParams, NamedEntity)
from fractal.core.entities import SimpleSpotExchange


# --------------------------------------------- A) auto-derived PARAMS_CLS
@dataclass
class _ParamsWithDefaults(BaseStrategyParams):
    BUY_THRESHOLD: float = 1500.0
    SELL_THRESHOLD: float = 2500.0


@dataclass
class _ParamsRequired(BaseStrategyParams):
    LEVERAGE: float  # no default → must be provided


class _StratWithDefaults(BaseStrategy[_ParamsWithDefaults]):
    def set_up(self):
        self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


class _StratRequired(BaseStrategy[_ParamsRequired]):
    def set_up(self):
        self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


class _UntypedStrat(BaseStrategy):  # no generic param
    def set_up(self):
        self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

    def predict(self):
        return []


@pytest.mark.core
def test_params_cls_auto_derived_from_generic_argument():
    assert _StratWithDefaults.PARAMS_CLS is _ParamsWithDefaults
    assert _StratRequired.PARAMS_CLS is _ParamsRequired


@pytest.mark.core
def test_untyped_strategy_has_no_params_cls():
    """A strategy without generic param leaves PARAMS_CLS = None."""
    assert _UntypedStrat.PARAMS_CLS is None


@pytest.mark.core
def test_explicit_params_cls_override_wins():
    """Explicit class-attribute override beats auto-derivation."""
    class _Explicit(BaseStrategy[_ParamsWithDefaults]):
        PARAMS_CLS = _ParamsRequired  # weird, but author wins

        def set_up(self):
            self.register_entity(NamedEntity("X", SimpleSpotExchange(trading_fee=0.0)))

        def predict(self):
            return []

    assert _Explicit.PARAMS_CLS is _ParamsRequired


# ----------------------------------------------- B) params=None semantics
@pytest.mark.core
def test_params_none_uses_default_class_when_all_fields_have_defaults():
    s = _StratWithDefaults(params=None)
    assert isinstance(s._params, _ParamsWithDefaults)
    assert s._params.BUY_THRESHOLD == 1500.0
    assert s._params.SELL_THRESHOLD == 2500.0


@pytest.mark.core
def test_params_none_raises_when_required_fields_missing():
    """A params class with required fields can't be defaulted; init must raise."""
    with pytest.raises(TypeError):
        _StratRequired(params=None)


@pytest.mark.core
def test_params_none_on_untyped_strategy_falls_back_to_empty_namespace():
    s = _UntypedStrat(params=None)
    assert isinstance(s._params, BaseStrategyParams)


# ---------------------------------------------- C) other params shapes
@pytest.mark.core
def test_params_dict_still_works():
    s = _StratWithDefaults(params={"BUY_THRESHOLD": 9000.0})
    # The base wraps dicts as BaseStrategyParams (not the typed subclass) —
    # historical behaviour preserved.
    assert s._params.BUY_THRESHOLD == 9000.0


@pytest.mark.core
def test_params_instance_used_as_is():
    custom = _ParamsWithDefaults(BUY_THRESHOLD=42.0)
    s = _StratWithDefaults(params=custom)
    assert s._params is custom
    assert s._params.BUY_THRESHOLD == 42.0


@pytest.mark.core
def test_invalid_params_type_raises_value_error():
    with pytest.raises(ValueError):
        _StratWithDefaults(params=42)  # type: ignore[arg-type]
