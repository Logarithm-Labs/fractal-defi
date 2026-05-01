"""Functional tests for :class:`StakedETHEntity`.

Mirrors the structure of ``test_uniswap_v3_spot.py`` since stETH is a
spot-traded asset. Adds tests for the LST-specific rebasing behaviour
(``staking_rate``).
"""
import pytest

from fractal.core.entities.protocols.steth import (
    StakedETHEntity,
    StakedETHEntityException,
    StakedETHGlobalState,
    StakedETHInternalState,
)


class TestStakedETHEntityCRUD:

    @pytest.mark.core
    def test_start_state(self):
        e = StakedETHEntity()
        assert e.global_state == StakedETHGlobalState()
        assert e.internal_state == StakedETHInternalState()
        assert e.balance == 0

    @pytest.mark.core
    def test_action_deposit(self):
        e = StakedETHEntity()
        e.action_deposit(amount_in_notional=1000)
        assert e.internal_state.cash == 1000
        assert e.balance == 1000

    @pytest.mark.core
    def test_action_buy(self):
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000))
        e.action_deposit(amount_in_notional=1000)
        e.action_buy(amount_in_notional=1000)
        assert e.internal_state.amount == (1000 / 2000) * (1 - e.trading_fee)
        assert e.internal_state.cash == 0
        assert e.balance == 1000 * (1 - e.trading_fee)

    @pytest.mark.core
    def test_action_sell(self):
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=4000))
        e.action_deposit(amount_in_notional=2000)
        e.action_buy(amount_in_notional=2000)
        product_to_sell = (2000 / 4000) * (1 - e.trading_fee)
        e.action_sell(amount_in_product=product_to_sell)
        assert e.internal_state.amount == 0
        assert e.internal_state.cash == 2000 * (1 - e.trading_fee) ** 2
        assert e.balance == 2000 * (1 - e.trading_fee) ** 2

    @pytest.mark.core
    def test_action_withdraw(self):
        e = StakedETHEntity()
        e.action_deposit(amount_in_notional=1000)
        e.action_withdraw(amount_in_notional=500)
        assert e.internal_state.cash == 500
        assert e.balance == 500

    @pytest.mark.core
    def test_update_state(self):
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))
        assert e.global_state.price == 2000


class TestStakedETHRebasing:
    """LST-specific: ``amount`` rebases by ``rate`` every ``update_state``."""

    @pytest.mark.core
    def test_amount_rebases_on_positive_rate(self):
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))
        e.action_deposit(2000)
        e.action_buy(2000)
        amount_before = e.internal_state.amount
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.001))
        # 0.1% rate → amount × 1.001
        assert e.internal_state.amount == pytest.approx(amount_before * 1.001)

    @pytest.mark.core
    def test_amount_shrinks_on_negative_rate_slashing(self):
        """Slashing modeled as negative rate (rate > -1)."""
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))
        e.action_deposit(2000)
        e.action_buy(2000)
        amount_before = e.internal_state.amount
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=-0.05))
        assert e.internal_state.amount == pytest.approx(amount_before * 0.95)

    @pytest.mark.core
    def test_rebase_with_no_position_is_noop(self):
        e = StakedETHEntity()
        e.action_deposit(1000)  # cash only, no buy
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.05))
        # No LST holdings → nothing to rebase, cash unchanged.
        assert e.internal_state.amount == 0
        assert e.internal_state.cash == 1000

    @pytest.mark.core
    def test_staking_rate_property_exposes_global_rate(self):
        """Polymorphic LST API: ``staking_rate`` mirrors ``global_state.staking_rate``."""
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0042))
        assert e.staking_rate == 0.0042

    @pytest.mark.core
    def test_update_state_rejects_rate_below_minus_one(self):
        """``rate < -1`` would flip amount negative — invariant guard."""
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))
        e.action_deposit(2000)
        e.action_buy(2000)
        with pytest.raises(StakedETHEntityException, match="staking_rate must be >= -1"):
            e.update_state(StakedETHGlobalState(price=2000, staking_rate=-1.5))


class TestStakedETHValidation:

    @pytest.mark.core
    def test_buy_rejects_negative(self):
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000))
        e.action_deposit(1000)
        with pytest.raises(StakedETHEntityException, match="buy amount must be >= 0"):
            e.action_buy(-1)

    @pytest.mark.core
    def test_buy_rejects_zero_price(self):
        e = StakedETHEntity()
        e.action_deposit(1000)
        with pytest.raises(StakedETHEntityException, match="price must be > 0"):
            e.action_buy(100)

    @pytest.mark.core
    def test_buy_rejects_overdraft(self):
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000))
        e.action_deposit(100)
        with pytest.raises(StakedETHEntityException, match="Not enough cash"):
            e.action_buy(200)

    @pytest.mark.core
    def test_sell_rejects_negative(self):
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000))
        with pytest.raises(StakedETHEntityException, match="sell amount must be >= 0"):
            e.action_sell(-1)

    @pytest.mark.core
    def test_sell_rejects_zero_price(self):
        """No selling at zero price (would yield zero proceeds)."""
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000))
        e.action_deposit(2000)
        e.action_buy(2000)
        # Now drop price to 0 (degenerate)
        e._global_state = StakedETHGlobalState(price=0)
        with pytest.raises(StakedETHEntityException, match="price must be > 0"):
            e.action_sell(0.5)

    @pytest.mark.core
    def test_sell_rejects_more_than_held(self):
        e = StakedETHEntity()
        e.update_state(StakedETHGlobalState(price=2000))
        with pytest.raises(StakedETHEntityException, match="Not enough product"):
            e.action_sell(1)

    @pytest.mark.core
    def test_deposit_rejects_negative(self):
        e = StakedETHEntity()
        with pytest.raises(StakedETHEntityException, match="deposit amount must be >= 0"):
            e.action_deposit(-1)

    @pytest.mark.core
    def test_withdraw_rejects_negative(self):
        e = StakedETHEntity()
        with pytest.raises(StakedETHEntityException, match="withdraw amount must be >= 0"):
            e.action_withdraw(-1)

    @pytest.mark.core
    def test_withdraw_rejects_overdraft(self):
        e = StakedETHEntity()
        e.action_deposit(100)
        with pytest.raises(StakedETHEntityException, match="Not enough cash"):
            e.action_withdraw(200)
