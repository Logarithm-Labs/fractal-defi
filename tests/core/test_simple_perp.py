"""Tests for :class:`SimplePerpEntity`."""
import pytest

from fractal.core.entities.simple.perp import SimplePerpEntity, SimplePerpEntityException, SimplePerpGlobalState


@pytest.fixture
def perp() -> SimplePerpEntity:
    e = SimplePerpEntity(trading_fee=0.0, max_leverage=50)
    e.update_state(SimplePerpGlobalState(mark_price=1000.0, funding_rate=0.0))
    return e


# -------------------------------------------------------------- account
@pytest.mark.core
def test_can_be_instantiated():
    e = SimplePerpEntity()
    assert e.balance == 0.0
    assert e.size == 0.0
    assert e.leverage == 0.0


@pytest.mark.core
def test_deposit_increases_balance(perp):
    perp.action_deposit(1000)
    assert perp.balance == 1000
    assert perp.size == 0
    assert perp.leverage == 0


@pytest.mark.core
def test_deposit_rejects_negative(perp):
    with pytest.raises(SimplePerpEntityException):
        perp.action_deposit(-1)


@pytest.mark.core
def test_withdraw_basic(perp):
    perp.action_deposit(1000)
    perp.action_withdraw(400)
    assert perp.balance == 600


@pytest.mark.core
def test_withdraw_rejects_negative(perp):
    perp.action_deposit(1000)
    with pytest.raises(SimplePerpEntityException):
        perp.action_withdraw(-1)


@pytest.mark.core
def test_withdraw_rejects_overdraft(perp):
    perp.action_deposit(1000)
    with pytest.raises(SimplePerpEntityException):
        perp.action_withdraw(1500)


@pytest.mark.core
def test_withdraw_blocks_below_maintenance_margin():
    """At max_leverage=10 and size=1 @ 1000, maintenance = 100. Cannot
    withdraw such that balance drops below 100."""
    e = SimplePerpEntity(trading_fee=0.0, max_leverage=10)
    e.update_state(SimplePerpGlobalState(mark_price=1000))
    e.action_deposit(200)
    e.action_open_position(1)  # leverage = 5x
    # Withdraw exactly to the maintenance line is allowed:
    e.action_withdraw(100)
    assert e.balance == pytest.approx(100)
    # Beyond → blocked
    with pytest.raises(SimplePerpEntityException):
        e.action_withdraw(0.01)


# -------------------------------------------------------------- opens
@pytest.mark.core
def test_open_long_sets_size_and_entry(perp):
    perp.action_deposit(1000)
    perp.action_open_position(0.5)
    assert perp.size == 0.5
    assert perp.internal_state.entry_price == 1000
    assert perp.balance == 1000  # zero-fee, mark unchanged


@pytest.mark.core
def test_open_short_sets_negative_size(perp):
    perp.action_deposit(1000)
    perp.action_open_position(-0.5)
    assert perp.size == -0.5
    assert perp.internal_state.entry_price == 1000


@pytest.mark.core
def test_trading_fee_charged_on_open():
    e = SimplePerpEntity(trading_fee=0.001, max_leverage=50)
    e.update_state(SimplePerpGlobalState(mark_price=1000))
    e.action_deposit(1000)
    e.action_open_position(1)
    # fee = |1| * 1000 * 0.001 = 1
    assert e.internal_state.collateral == pytest.approx(999)


@pytest.mark.core
def test_open_position_rejects_non_positive_mark(perp):
    perp._global_state = SimplePerpGlobalState(mark_price=0.0)
    perp.action_deposit(1000)
    with pytest.raises(SimplePerpEntityException):
        perp.action_open_position(1)


@pytest.mark.core
def test_open_zero_is_noop(perp):
    perp.action_deposit(1000)
    perp.action_open_position(0)
    assert perp.size == 0
    assert perp.balance == 1000


# -------------------------------------------------------------- aggregation
@pytest.mark.core
def test_same_side_aggregates_with_weighted_average_entry(perp):
    perp.action_deposit(10_000)
    perp.action_open_position(1)  # 1 long @ 1000
    perp.update_state(SimplePerpGlobalState(mark_price=1100))
    perp.action_open_position(1)  # 1 long @ 1100
    assert perp.size == 2
    assert perp.internal_state.entry_price == pytest.approx(1050)


@pytest.mark.core
def test_opposite_side_partial_close_realizes_pnl(perp):
    perp.action_deposit(10_000)
    perp.action_open_position(2)  # long 2 @ 1000
    perp.update_state(SimplePerpGlobalState(mark_price=1100))
    perp.action_open_position(-1)  # close 1 @ 1100, realized = 1*(1100-1000) = +100
    assert perp.size == 1
    assert perp.internal_state.entry_price == 1000  # remaining keeps original entry
    assert perp.internal_state.collateral == pytest.approx(10_100)


@pytest.mark.core
def test_opposite_side_full_close_realizes_full_pnl(perp):
    perp.action_deposit(10_000)
    perp.action_open_position(1)  # long 1 @ 1000
    perp.update_state(SimplePerpGlobalState(mark_price=1200))
    perp.action_open_position(-1)  # close @ 1200
    assert perp.size == 0
    assert perp.internal_state.entry_price == 0
    assert perp.internal_state.collateral == pytest.approx(10_200)
    assert perp.balance == pytest.approx(10_200)


@pytest.mark.core
def test_opposite_side_flips_direction_with_new_entry(perp):
    perp.action_deposit(10_000)
    perp.action_open_position(1)  # long 1 @ 1000
    perp.update_state(SimplePerpGlobalState(mark_price=1200))
    perp.action_open_position(-3)  # close 1 @ 1200 (+200), then open -2 @ 1200
    assert perp.size == -2
    assert perp.internal_state.entry_price == 1200
    assert perp.internal_state.collateral == pytest.approx(10_200)


# -------------------------------------------------------------- pnl + leverage
@pytest.mark.core
def test_pnl_long_positive_when_price_rises(perp):
    perp.action_deposit(10_000)
    perp.action_open_position(2)
    perp.update_state(SimplePerpGlobalState(mark_price=1100))
    assert perp.pnl == pytest.approx(2 * (1100 - 1000))
    assert perp.balance == pytest.approx(10_000 + 200)


@pytest.mark.core
def test_pnl_short_positive_when_price_falls(perp):
    perp.action_deposit(10_000)
    perp.action_open_position(-2)
    perp.update_state(SimplePerpGlobalState(mark_price=900))
    assert perp.pnl == pytest.approx(-2 * (900 - 1000))  # +200
    assert perp.balance == pytest.approx(10_200)


@pytest.mark.core
def test_leverage_scales_with_size_and_price(perp):
    perp.action_deposit(1000)
    perp.action_open_position(1)
    assert perp.leverage == pytest.approx(1 * 1000 / 1000)
    perp.update_state(SimplePerpGlobalState(mark_price=1500))
    # balance = 1000 + 1*(1500-1000) = 1500; leverage = 1*1500/1500 = 1
    assert perp.leverage == pytest.approx(1.0)


# -------------------------------------------------------------- funding
@pytest.mark.core
def test_funding_charged_on_long_when_rate_positive(perp):
    perp.action_deposit(1000)
    perp.action_open_position(1)
    # New tick: funding rate 0.001, longs pay → collateral decreases by 1*1000*0.001 = 1
    perp.update_state(SimplePerpGlobalState(mark_price=1000, funding_rate=0.001))
    assert perp.internal_state.collateral == pytest.approx(999)


@pytest.mark.core
def test_funding_paid_to_short_when_rate_positive(perp):
    perp.action_deposit(1000)
    perp.action_open_position(-1)
    perp.update_state(SimplePerpGlobalState(mark_price=1000, funding_rate=0.001))
    # short collateral increases by 1*1000*0.001 = 1
    assert perp.internal_state.collateral == pytest.approx(1001)


@pytest.mark.core
def test_funding_no_position_no_change(perp):
    perp.action_deposit(1000)
    perp.update_state(SimplePerpGlobalState(mark_price=1000, funding_rate=0.5))
    assert perp.internal_state.collateral == 1000


# -------------------------------------------------------------- liquidation
@pytest.mark.core
def test_liquidation_wipes_position_when_balance_below_maintenance():
    e = SimplePerpEntity(trading_fee=0.0, max_leverage=10)
    e.update_state(SimplePerpGlobalState(mark_price=1000))
    e.action_deposit(200)
    e.action_open_position(1)  # 5x leverage; maintenance @ 1000 = 100
    # Drop to 850: pnl = -150, balance = 50, maintenance = 85 → wipe
    e.update_state(SimplePerpGlobalState(mark_price=850))
    assert e.size == 0
    assert e.internal_state.collateral == 0


@pytest.mark.core
def test_liquidation_not_triggered_above_maintenance():
    e = SimplePerpEntity(trading_fee=0.0, max_leverage=10)
    e.update_state(SimplePerpGlobalState(mark_price=1000))
    e.action_deposit(200)
    e.action_open_position(1)
    # Drop to 950: pnl = -50, balance = 150, maintenance = 95 → safe
    e.update_state(SimplePerpGlobalState(mark_price=950))
    assert e.size == 1


@pytest.mark.core
def test_funding_settles_before_liquidation_check():
    """If a positive funding tick on a short position lifts balance back
    above maintenance, liquidation must NOT fire."""
    e = SimplePerpEntity(trading_fee=0.0, max_leverage=10)
    e.update_state(SimplePerpGlobalState(mark_price=1000))
    e.action_deposit(200)
    e.action_open_position(-1)  # short, maintenance = 100
    # Bring price up to a level where pnl alone would breach maintenance,
    # but a positive funding rate (which credits the short) keeps us above.
    # mark=1100 → pnl = -1*(1100-1000) = -100, balance pre-funding = 100.
    # funding 0.05 on short: collateral += 1*1100*0.05 = 55 → balance = 155.
    # maintenance @ 1100 = 110. 155 > 110 → safe.
    e.update_state(SimplePerpGlobalState(mark_price=1100, funding_rate=0.05))
    assert e.size == -1  # not liquidated


# -------------------------------------------------------------- close helper
@pytest.mark.core
def test_close_position_helper_flattens(perp):
    perp.action_deposit(10_000)
    perp.action_open_position(2)
    perp.update_state(SimplePerpGlobalState(mark_price=1100))
    perp.action_close_position()
    assert perp.size == 0
    assert perp.internal_state.entry_price == 0


@pytest.mark.core
def test_close_position_helper_is_noop_when_flat(perp):
    perp.action_deposit(1000)
    perp.action_close_position()
    assert perp.size == 0
    assert perp.internal_state.collateral == 1000


# -------------------------------------------------------------- H4 lock-ins
@pytest.mark.core
def test_open_position_rejects_above_max_leverage():
    """H4: opening a position whose post-trade leverage/margin would be
    invalid is rejected pre-trade and rolled back atomically.

    For SimplePerp ``MMR = 1/max_leverage`` (tighter than HL's
    ``1/(2·max_lev)``), so the maintenance-margin check fires before the
    leverage check — both flag the same illegal trade.
    """
    e = SimplePerpEntity(trading_fee=0.0, max_leverage=10)
    e.update_state(SimplePerpGlobalState(mark_price=1000.0))
    e.action_deposit(100)
    with pytest.raises(SimplePerpEntityException,
                       match="leverage|maintenance_margin"):
        e.action_open_position(2.0)  # notional 2000 → MM 200 > balance 100
    assert e.size == 0
    assert e.internal_state.collateral == 100


@pytest.mark.core
def test_open_position_rejects_without_deposit():
    """H4: opening any non-zero position with zero collateral is rejected."""
    e = SimplePerpEntity(trading_fee=0.0, max_leverage=10)
    e.update_state(SimplePerpGlobalState(mark_price=1000.0))
    with pytest.raises(SimplePerpEntityException):
        e.action_open_position(0.1)
    assert e.size == 0
    assert e.internal_state.collateral == 0


@pytest.mark.core
def test_open_position_at_max_leverage_passes():
    """H4 boundary: leverage == max_leverage is allowed (strict ``>`` check)."""
    e = SimplePerpEntity(trading_fee=0.0, max_leverage=10)
    e.update_state(SimplePerpGlobalState(mark_price=1000.0))
    e.action_deposit(100)
    e.action_open_position(1.0)  # notional 1000 → leverage exactly 10x
    assert e.size == 1.0


@pytest.mark.core
def test_close_position_passes_even_from_margin_bound_state():
    """H4: risk-reducing trades (close / partial close) must NOT be rejected
    by the leverage check, even when current leverage is at the cap."""
    e = SimplePerpEntity(trading_fee=0.0, max_leverage=10)
    e.update_state(SimplePerpGlobalState(mark_price=1000.0))
    e.action_deposit(100)
    e.action_open_position(1.0)  # opened at exactly 10x
    # Reduce by half — this is risk-decreasing, must pass.
    e.action_open_position(-0.5)
    assert e.size == 0.5
