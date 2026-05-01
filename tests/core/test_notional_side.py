"""Lock-in: ``notional_side`` config flag for UniV2/V3 LP entities.

Convention:
* ``price`` is always **notional per non-notional unit** — the same
  number regardless of which on-chain slot holds the notional.
* ``notional_side="token0"`` (default) — current/legacy behaviour: stable
  leg lives in the ``token0_amount`` field.
* ``notional_side="token1"`` — stable leg lives in the ``token1_amount``
  field; the volatile leg lives in ``token0_amount``.
* User-facing semantics (``balance``, ``hodl_value``, ``impermanent_loss``,
  ``stable_amount``, ``volatile_amount``) are identical between the two
  modes for the same pool data — only the internal slot mapping flips.
"""
import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.uniswap_v2_lp import UniswapV2LPConfig, UniswapV2LPEntity, UniswapV2LPGlobalState
from fractal.core.entities.protocols.uniswap_v3_lp import UniswapV3LPConfig, UniswapV3LPEntity, UniswapV3LPGlobalState


@pytest.mark.core
def test_v2_invalid_notional_side_rejected():
    with pytest.raises(EntityException, match="notional_side must be"):
        UniswapV2LPEntity(UniswapV2LPConfig(notional_side="weth"))


@pytest.mark.core
def test_v3_invalid_notional_side_rejected():
    with pytest.raises(EntityException, match="notional_side must be"):
        UniswapV3LPEntity(UniswapV3LPConfig(notional_side="weth"))


def _open_v2(notional_side):
    cfg = UniswapV2LPConfig(notional_side=notional_side)
    e = UniswapV2LPEntity(cfg)
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          price=1000, fees=0, volume=0))
    e.action_deposit(1000)
    e.action_open_position(500)
    return e


@pytest.mark.core
def test_v2_balance_invariant_under_notional_side_flip():
    """Same pool data, same deposit — same balance regardless of slot."""
    e0 = _open_v2("token0")
    e1 = _open_v2("token1")
    assert e0.balance == pytest.approx(e1.balance)


@pytest.mark.core
def test_v2_slots_swap_under_flip():
    """token0 mode stores stable in token0_amount; token1 mode stores it
    in token1_amount. Numerical values cross-equal."""
    e0 = _open_v2("token0")
    e1 = _open_v2("token1")
    assert e0._internal_state.token0_amount == pytest.approx(e1._internal_state.token1_amount)
    assert e0._internal_state.token1_amount == pytest.approx(e1._internal_state.token0_amount)


@pytest.mark.core
def test_v2_helper_properties_invariant():
    """Slot-aware accessors return the same numbers regardless of mode."""
    e0 = _open_v2("token0")
    e1 = _open_v2("token1")
    assert e0.stable_amount == pytest.approx(e1.stable_amount)
    assert e0.volatile_amount == pytest.approx(e1.volatile_amount)
    assert e0.entry_stable_amount == pytest.approx(e1.entry_stable_amount)
    assert e0.entry_volatile_amount == pytest.approx(e1.entry_volatile_amount)


@pytest.mark.core
def test_v2_il_invariant_under_flip():
    """IL is a property of the pool & deposit, not the slot mapping."""
    e0 = _open_v2("token0")
    e1 = _open_v2("token1")
    new_state0 = UniswapV2LPGlobalState(tvl=12_000, liquidity=10_000,
                                        price=1500, fees=0, volume=0)
    new_state1 = UniswapV2LPGlobalState(tvl=12_000, liquidity=10_000,
                                        price=1500, fees=0, volume=0)
    e0.update_state(new_state0)
    e1.update_state(new_state1)
    assert e0.impermanent_loss == pytest.approx(e1.impermanent_loss)
    assert e0.balance == pytest.approx(e1.balance)
    assert e0.hodl_value == pytest.approx(e1.hodl_value)


@pytest.mark.core
def test_v2_close_position_invariant_under_flip():
    e0 = _open_v2("token0")
    e1 = _open_v2("token1")
    e0.action_close_position()
    e1.action_close_position()
    assert e0._internal_state.cash == pytest.approx(e1._internal_state.cash)


def _open_v3(notional_side):
    cfg = UniswapV3LPConfig(notional_side=notional_side)
    e = UniswapV3LPEntity(cfg)
    e.update_state(UniswapV3LPGlobalState(price=1.0))
    e.action_deposit(1000)
    e.action_open_position(500, price_lower=0.9, price_upper=1.1)
    return e


@pytest.mark.core
def test_v3_balance_invariant_under_notional_side_flip():
    e0 = _open_v3("token0")
    e1 = _open_v3("token1")
    assert e0.balance == pytest.approx(e1.balance)


@pytest.mark.core
def test_v3_slots_swap_under_flip():
    e0 = _open_v3("token0")
    e1 = _open_v3("token1")
    assert e0._internal_state.token0_amount == pytest.approx(e1._internal_state.token1_amount)
    assert e0._internal_state.token1_amount == pytest.approx(e1._internal_state.token0_amount)


@pytest.mark.core
def test_v3_helper_properties_invariant():
    e0 = _open_v3("token0")
    e1 = _open_v3("token1")
    assert e0.stable_amount == pytest.approx(e1.stable_amount)
    assert e0.volatile_amount == pytest.approx(e1.volatile_amount)
    assert e0.entry_stable_amount == pytest.approx(e1.entry_stable_amount)
    assert e0.entry_volatile_amount == pytest.approx(e1.entry_volatile_amount)


@pytest.mark.core
def test_v3_il_invariant_under_flip():
    e0 = _open_v3("token0")
    e1 = _open_v3("token1")
    e0.update_state(UniswapV3LPGlobalState(price=1.05))
    e1.update_state(UniswapV3LPGlobalState(price=1.05))
    assert e0.impermanent_loss == pytest.approx(e1.impermanent_loss)
    assert e0.balance == pytest.approx(e1.balance)


@pytest.mark.core
def test_v3_is_in_range_invariant_under_flip():
    e0 = _open_v3("token0")
    e1 = _open_v3("token1")
    assert e0.is_in_range == e1.is_in_range
    e0.update_state(UniswapV3LPGlobalState(price=0.85))
    e1.update_state(UniswapV3LPGlobalState(price=0.85))
    assert e0.is_in_range is False
    assert e1.is_in_range is False


@pytest.mark.core
def test_v3_close_position_invariant_under_flip():
    e0 = _open_v3("token0")
    e1 = _open_v3("token1")
    e0.action_close_position()
    e1.action_close_position()
    assert e0._internal_state.cash == pytest.approx(e1._internal_state.cash)


@pytest.mark.core
def test_v3_below_range_volatile_in_correct_slot_token0_mode():
    """Below range: position holds 100% volatile, 0% stable."""
    e = _open_v3("token0")
    e.update_state(UniswapV3LPGlobalState(price=0.85))
    assert e.stable_amount == 0
    assert e.volatile_amount > 0
    # token0_mode: stable in token0_amount (zero), volatile in token1_amount.
    assert e._internal_state.token0_amount == 0
    assert e._internal_state.token1_amount > 0


@pytest.mark.core
def test_v3_below_range_volatile_in_correct_slot_token1_mode():
    e = _open_v3("token1")
    e.update_state(UniswapV3LPGlobalState(price=0.85))
    assert e.stable_amount == 0
    assert e.volatile_amount > 0
    # token1_mode: stable in token1_amount (zero), volatile in token0_amount.
    assert e._internal_state.token1_amount == 0
    assert e._internal_state.token0_amount > 0
