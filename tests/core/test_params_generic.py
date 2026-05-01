"""Tests for the ``BaseStrategy[Params]`` generic + auto-derived ``PARAMS_CLS``
+ ``params=None`` fallback to defaults.
"""
from dataclasses import dataclass

import pytest

from fractal.core.base import BaseStrategy, BaseStrategyParams, NamedEntity
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
def test_params_dict_on_typed_strategy_coerced_to_dataclass():
    """When the strategy declares ``BaseStrategy[Params]``, a dict
    must be splatted into ``PARAMS_CLS(**dict)`` so dataclass defaults,
    type hints and unknown-key rejection apply — not silently wrapped
    in a generic ``BaseStrategyParams``."""
    s = _StratWithDefaults(params={"BUY_THRESHOLD": 9000.0})
    assert isinstance(s._params, _ParamsWithDefaults)
    assert s._params.BUY_THRESHOLD == 9000.0
    # Default for unspecified field flows through.
    assert s._params.SELL_THRESHOLD == 2500.0


@pytest.mark.core
def test_params_dict_with_unknown_key_rejected_on_typed_strategy():
    """A typo in a dict key must fail loudly via dataclass ``__init__``
    rather than silently land in an untyped namespace."""
    with pytest.raises(TypeError):
        _StratWithDefaults(params={"BUY_THRSHOLD": 9000.0})  # typo


@pytest.mark.core
def test_params_dict_on_untyped_strategy_wraps_in_basestrategyparams():
    """Without ``PARAMS_CLS`` we keep the historical fallback —
    a dict is wrapped in a generic ``BaseStrategyParams``."""
    s = _UntypedStrat(params={"foo": 1})
    assert isinstance(s._params, BaseStrategyParams)
    assert s._params.foo == 1


@dataclass
class _ExtParams(_ParamsWithDefaults):
    EXECUTION_COST: float = 0.0005


class _ExtStrat(_StratWithDefaults):
    """Subclass that adds extra params via an explicit ``PARAMS_CLS`` override."""
    PARAMS_CLS = _ExtParams


@pytest.mark.core
def test_subclass_explicit_params_cls_override_accepts_extended_dict():
    """Regression: an ``sklearn.model_selection.ParameterGrid`` cell
    (a plain dict) for a strategy whose ``PARAMS_CLS`` extends the
    parent's params with extra fields must be coerced into the SUBCLASS
    params dataclass — not the parent's. The original symptom was
    ``HyperliquidBasis`` inheriting PARAMS_CLS from its parent and
    rejecting ``EXECUTION_COST`` from a grid cell.
    """
    grid_cell = {"BUY_THRESHOLD": 9000.0, "EXECUTION_COST": 0.001}
    s = _ExtStrat(params=grid_cell)
    assert isinstance(s._params, _ExtParams)
    assert s._params.BUY_THRESHOLD == 9000.0
    assert s._params.EXECUTION_COST == 0.001
    # default for unspecified field still flows through
    assert s._params.SELL_THRESHOLD == 2500.0


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
