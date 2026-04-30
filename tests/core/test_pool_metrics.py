"""Lock-in: pool-level IL/hodl/is_in_range and close-mutation contracts.

Covers the user-visible API added on top of UniswapV2LPEntity /
UniswapV3LPEntity:

* ``hodl_value`` — entry-composition × current price + cash.
* ``impermanent_loss`` — ``hodl_value - balance``; 0 without a position.
* ``is_in_range`` (V3 only) — pool price inside ``[price_lower, price_upper]``.
* ``action_close_position`` mutates ``_internal_state`` field-by-field
  rather than replacing the dataclass — subclass-added fields survive.
* Both V2/V3 ``GlobalState`` classes inherit from ``BasePoolGlobalState``.
"""
from dataclasses import dataclass

import pytest

from fractal.core.entities.base.pool import BasePoolGlobalState
from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                            UniswapV2LPEntity,
                                                            UniswapV2LPGlobalState,
                                                            UniswapV2LPInternalState)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                            UniswapV3LPEntity,
                                                            UniswapV3LPGlobalState,
                                                            UniswapV3LPInternalState)
from fractal.core.entities.simple.pool import SimplePoolGlobalState


# ============================================================ shared GlobalState
@pytest.mark.core
@pytest.mark.parametrize("cls", [UniswapV2LPGlobalState,
                                 UniswapV3LPGlobalState,
                                 SimplePoolGlobalState])
def test_pool_global_states_inherit_from_base(cls):
    assert issubclass(cls, BasePoolGlobalState)
    s = cls()
    # Required common fields:
    for f in ("tvl", "volume", "fees", "liquidity", "price"):
        assert hasattr(s, f), f"{cls.__name__} missing common field {f}"


# ============================================================ V2: hodl & IL
@pytest.fixture
def v2_open_at_1000():
    cfg = UniswapV2LPConfig(pool_fee_rate=0.003, slippage_pct=0.0,
                            token0_decimals=6, token1_decimals=18)
    e = UniswapV2LPEntity(config=cfg)
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          fees=0, price=1000, volume=0))
    e.action_deposit(1000)
    e.action_open_position(500)
    return e


@pytest.mark.core
def test_v2_hodl_equals_balance_when_price_unchanged(v2_open_at_1000):
    """Without price move and without fees, hodl == balance, IL == 0."""
    assert v2_open_at_1000.hodl_value == pytest.approx(v2_open_at_1000.balance)
    assert v2_open_at_1000.impermanent_loss == pytest.approx(0.0, abs=1e-9)


@pytest.mark.core
def test_v2_il_positive_after_price_move(v2_open_at_1000):
    """Constant-product LP underperforms 50/50 hodl on any price move."""
    e = v2_open_at_1000
    # Move price up 50%: pool tvl scales sub-proportionally vs hodl.
    e.update_state(UniswapV2LPGlobalState(tvl=10_000 * (2 * 1.5**0.5 / 2),
                                          liquidity=10_000, fees=0,
                                          price=1500, volume=0))
    assert e.impermanent_loss > 0
    assert e.hodl_value > e.balance


@pytest.mark.core
def test_v2_hodl_returns_cash_with_no_position():
    cfg = UniswapV2LPConfig()
    e = UniswapV2LPEntity(config=cfg)
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          price=1, volume=0, fees=0))
    e.action_deposit(500)
    assert e.hodl_value == 500
    assert e.impermanent_loss == 0


# ============================================================ V3: hodl, IL, is_in_range
@pytest.fixture
def v3_open_at_1():
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    e.update_state(UniswapV3LPGlobalState(price=1.0))
    e.action_deposit(1000)
    e.action_open_position(500, price_lower=0.9, price_upper=1.1)
    return e


@pytest.mark.core
def test_v3_hodl_equals_balance_when_price_unchanged(v3_open_at_1):
    assert v3_open_at_1.hodl_value == pytest.approx(v3_open_at_1.balance)
    assert v3_open_at_1.impermanent_loss == pytest.approx(0.0, abs=1e-9)


@pytest.mark.core
def test_v3_il_positive_when_price_moves_within_range(v3_open_at_1):
    e = v3_open_at_1
    e.update_state(UniswapV3LPGlobalState(price=1.05))
    assert e.impermanent_loss > 0


@pytest.mark.core
def test_v3_is_in_range_true_initially(v3_open_at_1):
    assert v3_open_at_1.is_in_range is True


@pytest.mark.core
def test_v3_is_in_range_false_when_price_below_lower(v3_open_at_1):
    e = v3_open_at_1
    e.update_state(UniswapV3LPGlobalState(price=0.85))
    assert e.is_in_range is False


@pytest.mark.core
def test_v3_is_in_range_false_when_price_above_upper(v3_open_at_1):
    e = v3_open_at_1
    e.update_state(UniswapV3LPGlobalState(price=1.15))
    assert e.is_in_range is False


@pytest.mark.core
def test_v3_is_in_range_false_with_no_position():
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    e.update_state(UniswapV3LPGlobalState(price=1.0))
    assert e.is_in_range is False


# ============================================================ is_position is property
@pytest.mark.core
def test_v2_is_position_is_property():
    """is_position should derive from liquidity, not be a settable attr."""
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    assert e.is_position is False
    with pytest.raises(AttributeError):
        e.is_position = True  # property has no setter


@pytest.mark.core
def test_v3_is_position_is_property():
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    assert e.is_position is False
    with pytest.raises(AttributeError):
        e.is_position = True


# ============================================================ close mutates, doesn't replace
@pytest.mark.core
def test_v2_close_position_preserves_subclass_fields():
    """Lock-in for V2-4: ``action_close_position`` must mutate, not replace,
    so subclass-added fields survive the close."""

    @dataclass
    class _ExtendedState(UniswapV2LPInternalState):
        my_extra_field: float = 0.0

    class _MyV2(UniswapV2LPEntity):
        def _initialize_states(self):
            self._internal_state = _ExtendedState()
            self._global_state = UniswapV2LPGlobalState()

    cfg = UniswapV2LPConfig()
    e = _MyV2(config=cfg)
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          price=1000, fees=0, volume=0))
    e.action_deposit(1000)
    e.action_open_position(500)
    e._internal_state.my_extra_field = 42.0
    e.action_close_position()
    assert isinstance(e._internal_state, _ExtendedState)
    assert e._internal_state.my_extra_field == 42.0


@pytest.mark.core
def test_v3_close_position_preserves_subclass_fields():
    """Lock-in for V3-4: same as V2."""

    @dataclass
    class _ExtendedState(UniswapV3LPInternalState):
        my_extra_field: float = 0.0

    class _MyV3(UniswapV3LPEntity):
        def _initialize_states(self):
            self._internal_state = _ExtendedState()
            self._global_state = UniswapV3LPGlobalState()

    cfg = UniswapV3LPConfig()
    e = _MyV3(config=cfg)
    e.update_state(UniswapV3LPGlobalState(price=1.0))
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    e._internal_state.my_extra_field = 42.0
    e.action_close_position()
    assert isinstance(e._internal_state, _ExtendedState)
    assert e._internal_state.my_extra_field == 42.0


@pytest.mark.core
def test_v2_close_resets_position_fields():
    cfg = UniswapV2LPConfig()
    e = UniswapV2LPEntity(config=cfg)
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          price=1000, fees=0, volume=0))
    e.action_deposit(1000)
    e.action_open_position(500)
    e.action_close_position()
    assert e._internal_state.token0_amount == 0
    assert e._internal_state.token1_amount == 0
    assert e._internal_state.entry_token0_amount == 0
    assert e._internal_state.entry_token1_amount == 0
    assert e._internal_state.price_init == 0
    assert e._internal_state.liquidity == 0
    assert e.is_position is False


@pytest.mark.core
def test_v3_close_resets_position_fields():
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    e.update_state(UniswapV3LPGlobalState(price=1.0))
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    e.action_close_position()
    assert e._internal_state.token0_amount == 0
    assert e._internal_state.token1_amount == 0
    assert e._internal_state.entry_token0_amount == 0
    assert e._internal_state.entry_token1_amount == 0
    assert e._internal_state.price_init == 0
    assert e._internal_state.price_lower == 0
    assert e._internal_state.price_upper == 0
    assert e._internal_state.liquidity == 0
    assert e.is_position is False
