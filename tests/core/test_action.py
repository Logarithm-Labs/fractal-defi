"""Tests for :class:`fractal.core.base.action.Action`."""
import pytest

from fractal.core.base import Action


@pytest.mark.core
def test_action_construction_with_args():
    a = Action("buy", {"amount_in_notional": 100.0})
    assert a.action == "buy"
    assert a.args == {"amount_in_notional": 100.0}


@pytest.mark.core
def test_action_default_empty_args():
    a = Action("close_position")
    assert a.action == "close_position"
    assert a.args == {}


@pytest.mark.core
def test_action_repr_contains_name_and_args():
    a = Action("buy", {"x": 1})
    r = repr(a)
    assert "buy" in r and "x" in r and "1" in r


@pytest.mark.core
def test_action_args_each_dataclass_gets_own_default():
    """``args=field(default_factory=dict)`` — each fresh Action must have
    its own dict, not a shared mutable default."""
    a = Action("buy")
    b = Action("sell")
    a.args["k"] = "v"
    assert "k" not in b.args


@pytest.mark.core
def test_action_supports_callable_args():
    """``args`` may carry callables; framework resolves them at execute time."""
    def delegate(obj):  # pylint: disable=unused-argument
        return 42
    a = Action("withdraw", {"amount_in_notional": delegate})
    assert callable(a.args["amount_in_notional"])
