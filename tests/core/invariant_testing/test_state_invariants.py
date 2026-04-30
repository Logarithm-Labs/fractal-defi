"""State-machine and lifecycle invariants for V2/V3 LP entities.

These verify that after every action the entity is in a *valid* state —
no orphan position fields, no negative balances, idempotent transitions,
correct rejection of illegal operations. They are the structural
counterpart to ``test_pool_invariants.py`` (which covers value-level
invariants like IL and fee proportionality).
"""
import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                           UniswapV2LPEntity,
                                                           UniswapV2LPGlobalState)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                           UniswapV3LPEntity,
                                                           UniswapV3LPGlobalState)


def _v2():
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          price=1000, fees=0, volume=0))
    return e


def _v3():
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                          price=1.0, fees=0, volume=0))
    return e


@pytest.mark.core
def test_v2_initial_state_is_clean():
    """Fresh V2 entity has all internal state at zero, no position open."""
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    assert not e.is_position
    assert e._internal_state.cash == 0
    assert e._internal_state.token0_amount == 0
    assert e._internal_state.token1_amount == 0
    assert e._internal_state.entry_token0_amount == 0
    assert e._internal_state.entry_token1_amount == 0
    assert e._internal_state.price_init == 0
    assert e._internal_state.liquidity == 0
    assert e.balance == 0
    assert e.hodl_value == 0
    assert e.impermanent_loss == 0


@pytest.mark.core
def test_v3_initial_state_is_clean():
    """Fresh V3 entity has all internal state at zero, no position open."""
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    assert not e.is_position
    assert not e.is_in_range
    assert e._internal_state.cash == 0
    assert e._internal_state.token0_amount == 0
    assert e._internal_state.token1_amount == 0
    assert e._internal_state.entry_token0_amount == 0
    assert e._internal_state.entry_token1_amount == 0
    assert e._internal_state.price_init == 0
    assert e._internal_state.price_lower == 0
    assert e._internal_state.price_upper == 0
    assert e._internal_state.liquidity == 0
    assert e.balance == 0
    assert e.hodl_value == 0
    assert e.impermanent_loss == 0


@pytest.mark.core
def test_v2_state_after_open_has_position_fields_set():
    e = _v2()
    e.action_deposit(1000)
    e.action_open_position(500)
    assert e.is_position
    assert e._internal_state.token0_amount > 0
    assert e._internal_state.token1_amount > 0
    assert e._internal_state.entry_token0_amount > 0
    assert e._internal_state.entry_token1_amount > 0
    assert e._internal_state.price_init == e._global_state.price
    assert e._internal_state.liquidity > 0
    # cash drained by `amount`, with stable_leftover added back
    assert 0 < e._internal_state.cash <= 1000


@pytest.mark.core
def test_v3_state_after_open_has_position_fields_set():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    assert e.is_position
    assert e.is_in_range  # price=1.0 within [0.9, 1.1]
    assert e._internal_state.token0_amount > 0
    assert e._internal_state.token1_amount > 0
    assert e._internal_state.entry_token0_amount > 0
    assert e._internal_state.entry_token1_amount > 0
    assert e._internal_state.price_init == 1.0
    assert e._internal_state.price_lower == 0.9
    assert e._internal_state.price_upper == 1.1
    assert e._internal_state.liquidity > 0
    assert 0 < e._internal_state.cash <= 1000


@pytest.mark.core
def test_v2_state_after_close_resets_all_position_fields():
    e = _v2()
    e.action_deposit(1000)
    e.action_open_position(500)
    e.action_close_position()
    assert not e.is_position
    assert e._internal_state.token0_amount == 0
    assert e._internal_state.token1_amount == 0
    assert e._internal_state.entry_token0_amount == 0
    assert e._internal_state.entry_token1_amount == 0
    assert e._internal_state.price_init == 0
    assert e._internal_state.liquidity == 0
    # cash holds full proceeds
    assert e._internal_state.cash > 0


@pytest.mark.core
def test_v3_state_after_close_resets_all_position_fields():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    e.action_close_position()
    assert not e.is_position
    assert not e.is_in_range
    assert e._internal_state.token0_amount == 0
    assert e._internal_state.token1_amount == 0
    assert e._internal_state.entry_token0_amount == 0
    assert e._internal_state.entry_token1_amount == 0
    assert e._internal_state.price_init == 0
    assert e._internal_state.price_lower == 0
    assert e._internal_state.price_upper == 0
    assert e._internal_state.liquidity == 0
    assert e._internal_state.cash > 0


@pytest.mark.core
def test_v2_multiple_open_close_cycles_work():
    """Entity can be reopened/closed many times."""
    e = _v2()
    e.action_deposit(2000)
    for _ in range(3):
        e.action_open_position(500)
        assert e.is_position
        e.action_close_position()
        assert not e.is_position
    # Final cash == initial - sum of round-trip costs (small).
    assert e._internal_state.cash == pytest.approx(2000, rel=0.01)


@pytest.mark.core
def test_v3_multiple_open_close_cycles_work():
    e = _v3()
    e.action_deposit(2000)
    for _ in range(3):
        e.action_open_position(500, 0.9, 1.1)
        assert e.is_position
        e.action_close_position()
        assert not e.is_position
    assert e._internal_state.cash == pytest.approx(2000, rel=0.01)


@pytest.mark.core
def test_v2_open_while_already_in_position_raises():
    e = _v2()
    e.action_deposit(1000)
    e.action_open_position(500)
    with pytest.raises(EntityException, match="already open"):
        e.action_open_position(100)


@pytest.mark.core
def test_v3_open_while_already_in_position_raises():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    with pytest.raises(EntityException, match="already open"):
        e.action_open_position(100, 0.95, 1.05)


@pytest.mark.core
def test_v2_close_without_position_raises():
    e = _v2()
    with pytest.raises(EntityException, match="No position to close"):
        e.action_close_position()


@pytest.mark.core
def test_v3_close_without_position_raises():
    e = _v3()
    with pytest.raises(EntityException, match="No position to close"):
        e.action_close_position()


@pytest.mark.core
def test_v2_open_with_amount_exceeding_cash_raises():
    e = _v2()
    e.action_deposit(100)
    with pytest.raises(EntityException, match="Insufficient funds"):
        e.action_open_position(200)


@pytest.mark.core
def test_v3_open_with_amount_exceeding_cash_raises():
    e = _v3()
    e.action_deposit(100)
    with pytest.raises(EntityException, match="Insufficient funds"):
        e.action_open_position(200, 0.9, 1.1)


@pytest.mark.core
def test_v2_withdraw_exceeding_cash_raises():
    e = _v2()
    e.action_deposit(100)
    with pytest.raises(EntityException, match="Insufficient funds"):
        e.action_withdraw(200)


@pytest.mark.core
def test_v3_withdraw_exceeding_cash_raises():
    e = _v3()
    e.action_deposit(100)
    with pytest.raises(EntityException, match="Insufficient funds"):
        e.action_withdraw(200)


@pytest.mark.core
def test_v3_open_with_inverted_range_raises():
    e = _v3()
    e.action_deposit(1000)
    with pytest.raises(EntityException, match="price_lower must be less"):
        e.action_open_position(500, 1.1, 0.9)


@pytest.mark.core
def test_v3_open_with_zero_lower_bound_raises():
    e = _v3()
    e.action_deposit(1000)
    with pytest.raises(EntityException, match="price bounds must be positive"):
        e.action_open_position(500, 0.0, 1.0)


@pytest.mark.core
def test_v3_open_with_negative_upper_bound_raises():
    e = _v3()
    e.action_deposit(1000)
    with pytest.raises(EntityException, match="price_lower must be less"):
        e.action_open_position(500, 0.9, -1.0)


@pytest.mark.core
def test_v2_multiple_update_state_accumulates_fees():
    """Each ``update_state`` with non-zero pool fees credits cash by share × fees."""
    e = _v2()
    e.action_deposit(1000)
    e.action_open_position(500)
    cash_before = e._internal_state.cash
    pool_liq = e._global_state.liquidity
    pos_liq = e._internal_state.liquidity

    # 3 bars with 100 fees each
    for _ in range(3):
        e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                              price=1000, fees=100, volume=0))
    expected_share_per_bar = pos_liq / pool_liq
    expected_total = 3 * expected_share_per_bar * 100
    assert e._internal_state.cash == pytest.approx(cash_before + expected_total)


@pytest.mark.core
def test_v3_multiple_update_state_accumulates_fees_in_range():
    """When in range, V3 entity accrues fees on every bar."""
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    cash_before = e._internal_state.cash

    fee_per_bar = 0
    for _ in range(3):
        e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                              price=1.0, fees=100, volume=0))
        fee_per_bar = e.calculate_fees()  # last bar's fee
    # cash should have grown by 3 × per-bar fee (each update_state added fee_per_bar)
    assert e._internal_state.cash > cash_before
    assert fee_per_bar > 0


@pytest.mark.core
def test_v3_no_fee_accrual_when_out_of_range():
    """Out-of-range V3 position earns 0 fees."""
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    cash_before = e._internal_state.cash
    # Move price below range, then update with non-zero fees
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                          price=0.85, fees=1000, volume=0))
    # Cash must be unchanged (out-of-range → fees=0)
    assert e._internal_state.cash == pytest.approx(cash_before)


@pytest.mark.core
def test_v2_update_state_without_position_credits_no_fees():
    """No position → calculate_fees=0 → cash unchanged."""
    e = _v2()
    e.action_deposit(500)
    cash_before = e._internal_state.cash
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          price=1000, fees=100, volume=0))
    assert e._internal_state.cash == cash_before


@pytest.mark.core
def test_v3_update_state_without_position_credits_no_fees():
    e = _v3()
    e.action_deposit(500)
    cash_before = e._internal_state.cash
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                          price=1.0, fees=100, volume=0))
    assert e._internal_state.cash == cash_before


@pytest.mark.core
def test_v2_update_state_with_same_data_is_idempotent_modulo_fees():
    """Two consecutive ``update_state`` calls with identical state yield the
    same internal state (modulo accumulated fees if non-zero)."""
    e = _v2()
    e.action_deposit(1000)
    e.action_open_position(500)
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          price=1000, fees=0, volume=0))
    snapshot = (
        e._internal_state.token0_amount,
        e._internal_state.token1_amount,
        e._internal_state.liquidity,
        e._internal_state.cash,
    )
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          price=1000, fees=0, volume=0))
    assert (
        e._internal_state.token0_amount,
        e._internal_state.token1_amount,
        e._internal_state.liquidity,
        e._internal_state.cash,
    ) == pytest.approx(snapshot)


@pytest.mark.core
def test_v3_update_state_with_same_data_is_idempotent_modulo_fees():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                          price=1.0, fees=0, volume=0))
    snapshot = (
        e._internal_state.token0_amount,
        e._internal_state.token1_amount,
        e._internal_state.liquidity,
        e._internal_state.cash,
    )
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                          price=1.0, fees=0, volume=0))
    assert (
        e._internal_state.token0_amount,
        e._internal_state.token1_amount,
        e._internal_state.liquidity,
        e._internal_state.cash,
    ) == pytest.approx(snapshot)


@pytest.mark.core
@pytest.mark.parametrize("notional_side", ["token0", "token1"])
def test_v2_cash_non_negative_through_lifecycle(notional_side):
    """Through any sequence of legal actions, cash remains >= 0."""
    cfg = UniswapV2LPConfig(notional_side=notional_side)
    e = UniswapV2LPEntity(cfg)
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          price=1000, fees=10, volume=0))
    e.action_deposit(1000)
    assert e._internal_state.cash >= 0
    e.action_open_position(500)
    assert e._internal_state.cash >= 0
    e.update_state(UniswapV2LPGlobalState(tvl=15_000, liquidity=10_000,
                                          price=1500, fees=20, volume=0))
    assert e._internal_state.cash >= 0
    e.action_close_position()
    assert e._internal_state.cash >= 0


@pytest.mark.core
@pytest.mark.parametrize("notional_side", ["token0", "token1"])
def test_v3_cash_non_negative_through_lifecycle(notional_side):
    cfg = UniswapV3LPConfig(notional_side=notional_side)
    e = UniswapV3LPEntity(cfg)
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                          price=1.0, fees=10, volume=0))
    e.action_deposit(1000)
    assert e._internal_state.cash >= 0
    e.action_open_position(500, 0.9, 1.1)
    assert e._internal_state.cash >= 0
    e.update_state(UniswapV3LPGlobalState(tvl=1_500_000, liquidity=1_000_000,
                                          price=1.05, fees=20, volume=0))
    assert e._internal_state.cash >= 0
    e.action_close_position()
    assert e._internal_state.cash >= 0


@pytest.mark.core
def test_v2_liquidity_zero_iff_no_position():
    """``liquidity > 0`` ⟺ ``is_position``."""
    e = _v2()
    assert e._internal_state.liquidity == 0 and not e.is_position
    e.action_deposit(1000)
    assert e._internal_state.liquidity == 0 and not e.is_position
    e.action_open_position(500)
    assert e._internal_state.liquidity > 0 and e.is_position
    e.action_close_position()
    assert e._internal_state.liquidity == 0 and not e.is_position


@pytest.mark.core
def test_v3_liquidity_zero_iff_no_position():
    e = _v3()
    assert e._internal_state.liquidity == 0 and not e.is_position
    e.action_deposit(1000)
    assert e._internal_state.liquidity == 0 and not e.is_position
    e.action_open_position(500, 0.9, 1.1)
    assert e._internal_state.liquidity > 0 and e.is_position
    e.action_close_position()
    assert e._internal_state.liquidity == 0 and not e.is_position


@pytest.mark.core
def test_v2_entry_amounts_unchanged_by_update_state():
    """Entry token amounts (snapshot at open) must NOT change on update_state."""
    e = _v2()
    e.action_deposit(1000)
    e.action_open_position(500)
    entry0 = e._internal_state.entry_token0_amount
    entry1 = e._internal_state.entry_token1_amount
    e.update_state(UniswapV2LPGlobalState(tvl=20_000, liquidity=10_000,
                                          price=2000, fees=50, volume=0))
    assert e._internal_state.entry_token0_amount == entry0
    assert e._internal_state.entry_token1_amount == entry1


@pytest.mark.core
def test_v3_entry_amounts_unchanged_by_update_state():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    entry0 = e._internal_state.entry_token0_amount
    entry1 = e._internal_state.entry_token1_amount
    e.update_state(UniswapV3LPGlobalState(tvl=2_000_000, liquidity=1_000_000,
                                          price=1.05, fees=50, volume=0))
    assert e._internal_state.entry_token0_amount == entry0
    assert e._internal_state.entry_token1_amount == entry1


@pytest.mark.core
def test_v2_il_strictly_positive_after_price_change():
    """IL > 0 strictly when price moves (V2 50/50 always loses to hodl)."""
    e = _v2()
    e.action_deposit(1000)
    e.action_open_position(500)
    e.update_state(UniswapV2LPGlobalState(tvl=12_000, liquidity=10_000,
                                          price=1500, fees=0, volume=0))
    assert e.impermanent_loss > 0


@pytest.mark.core
def test_v3_il_strictly_positive_after_in_range_price_change():
    e = _v3()
    e.action_deposit(1000)
    e.action_open_position(500, 0.9, 1.1)
    e.update_state(UniswapV3LPGlobalState(tvl=1_000_000, liquidity=1_000_000,
                                          price=1.05, fees=0, volume=0))
    assert e.impermanent_loss > 0
