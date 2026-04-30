"""Tests for :class:`SimpleLendingEntity`."""
import pytest

from fractal.core.entities.simple.lending import (SimpleLendingEntity,
                                                  SimpleLendingException,
                                                  SimpleLendingGlobalState)


@pytest.fixture
def lending() -> SimpleLendingEntity:
    e = SimpleLendingEntity(max_ltv=0.8, liq_thr=0.85)
    e.update_state(SimpleLendingGlobalState(
        collateral_price=1.0,    # collateral asset (USDC) priced at $1
        debt_price=1.0,     # borrowed asset (DAI) priced at $1
        lending_rate=0.0,
        borrowing_rate=0.0,
    ))
    return e


# ------------------------------------------------------- construction
@pytest.mark.core
def test_can_be_instantiated():
    e = SimpleLendingEntity()
    assert e.balance == 0


@pytest.mark.core
def test_invalid_max_ltv_rejected():
    with pytest.raises(SimpleLendingException):
        SimpleLendingEntity(max_ltv=0)
    with pytest.raises(SimpleLendingException):
        SimpleLendingEntity(max_ltv=1.5)


@pytest.mark.core
def test_invalid_liq_thr_rejected():
    with pytest.raises(SimpleLendingException):
        SimpleLendingEntity(max_ltv=0.8, liq_thr=0.7)


# ------------------------------------------------------- collateral
@pytest.mark.core
def test_deposit_increases_collateral(lending):
    lending.action_deposit(1000)
    assert lending.internal_state.collateral == 1000
    assert lending.balance == 1000


@pytest.mark.core
def test_deposit_rejects_negative(lending):
    with pytest.raises(SimpleLendingException):
        lending.action_deposit(-1)


@pytest.mark.core
def test_withdraw_basic(lending):
    lending.action_deposit(1000)
    lending.action_withdraw(400)
    assert lending.internal_state.collateral == 600


@pytest.mark.core
def test_withdraw_rejects_overdraft(lending):
    lending.action_deposit(100)
    with pytest.raises(SimpleLendingException):
        lending.action_withdraw(200)


@pytest.mark.core
def test_withdraw_blocks_when_post_ltv_exceeds_max(lending):
    lending.action_deposit(1000)
    lending.action_borrow(700)  # ltv = 0.7
    # Withdraw 100 → collateral 900, ltv = 700/900 ≈ 0.778 < 0.8 → ok
    lending.action_withdraw(100)
    # Withdraw another 50 → collateral 850, ltv = 700/850 ≈ 0.823 > 0.8 → blocked
    with pytest.raises(SimpleLendingException):
        lending.action_withdraw(50)


@pytest.mark.core
def test_cannot_withdraw_all_with_outstanding_debt(lending):
    lending.action_deposit(1000)
    lending.action_borrow(100)
    with pytest.raises(SimpleLendingException):
        lending.action_withdraw(1000)


# ------------------------------------------------------- debt
@pytest.mark.core
def test_borrow_basic(lending):
    lending.action_deposit(1000)
    lending.action_borrow(500)
    assert lending.internal_state.borrowed == 500
    assert lending.ltv == pytest.approx(0.5)


@pytest.mark.core
def test_borrow_rejects_no_collateral(lending):
    with pytest.raises(SimpleLendingException):
        lending.action_borrow(500)


@pytest.mark.core
def test_borrow_rejects_when_ltv_exceeds_max(lending):
    lending.action_deposit(1000)
    with pytest.raises(SimpleLendingException):
        lending.action_borrow(900)  # ltv = 0.9 > 0.8 max


# -------------------------------------------------- M4: price-zero guards
@pytest.mark.core
def test_borrow_rejects_zero_collateral_price():
    """M4: zero ``collateral_price`` makes the LTV check ill-defined.
    Must raise a domain exception, not a ZeroDivisionError."""
    e = SimpleLendingEntity()
    e.update_state(SimpleLendingGlobalState(collateral_price=0, debt_price=1))
    e._internal_state.collateral = 1000
    with pytest.raises(SimpleLendingException, match="collateral_price"):
        e.action_borrow(100)


@pytest.mark.core
def test_borrow_rejects_zero_debt_price():
    """M4: zero ``debt_price`` is rejected with a clear message."""
    e = SimpleLendingEntity()
    e.update_state(SimpleLendingGlobalState(collateral_price=1, debt_price=0))
    e._internal_state.collateral = 1000
    with pytest.raises(SimpleLendingException, match="debt_price"):
        e.action_borrow(100)


@pytest.mark.core
def test_withdraw_with_debt_rejects_zero_prices():
    """M4: withdraw against an existing debt must validate prices before
    computing the post-withdraw LTV."""
    e = SimpleLendingEntity()
    e.update_state(SimpleLendingGlobalState(collateral_price=1, debt_price=1))
    e._internal_state.collateral = 1000
    e._internal_state.borrowed = 100
    # Now flip debt_price to zero — same call shape would otherwise divide.
    e.update_state(SimpleLendingGlobalState(collateral_price=1, debt_price=0))
    with pytest.raises(SimpleLendingException, match="debt_price"):
        e.action_withdraw(100)


@pytest.mark.core
def test_repay_reduces_borrowed(lending):
    lending.action_deposit(1000)
    lending.action_borrow(500)
    lending.action_repay(200)
    assert lending.internal_state.borrowed == 300


@pytest.mark.core
def test_repay_rejects_overpayment(lending):
    lending.action_deposit(1000)
    lending.action_borrow(500)
    with pytest.raises(SimpleLendingException):
        lending.action_repay(800)


# ------------------------------------------------------- balance / ltv
@pytest.mark.core
def test_balance_reflects_collateral_minus_debt(lending):
    lending.action_deposit(1000)
    lending.action_borrow(300)
    assert lending.balance == pytest.approx(700)


@pytest.mark.core
def test_ltv_zero_when_no_debt(lending):
    lending.action_deposit(1000)
    assert lending.ltv == 0


@pytest.mark.core
def test_calculate_repay_to_target_ltv(lending):
    lending.action_deposit(1000)
    lending.action_borrow(700)  # ltv = 0.7
    # to drop ltv to 0.5, repay = 1000 * 1 * (0.7 - 0.5) / 1 = 200
    assert lending.calculate_repay(0.5) == pytest.approx(200)


# ------------------------------------------------------- interest
@pytest.mark.core
def test_interest_accrued_on_collateral_and_debt(lending):
    lending.action_deposit(1000)
    lending.action_borrow(500)
    lending.update_state(SimpleLendingGlobalState(
        collateral_price=1.0, debt_price=1.0,
        lending_rate=0.01, borrowing_rate=0.02,
    ))
    assert lending.internal_state.collateral == pytest.approx(1010)
    assert lending.internal_state.borrowed == pytest.approx(510)


# ------------------------------------------------------- liquidation
@pytest.mark.core
def test_liquidation_wipes_when_ltv_crosses_threshold():
    e = SimpleLendingEntity(max_ltv=0.8, liq_thr=0.85)
    e.update_state(SimpleLendingGlobalState(
        collateral_price=1, debt_price=1, lending_rate=0, borrowing_rate=0,
    ))
    e.action_deposit(1000)
    e.action_borrow(800)  # ltv = 0.8 (exactly at max)
    # Product price moves up — pushes ltv over liq_thr.
    e.update_state(SimpleLendingGlobalState(
        collateral_price=1, debt_price=1.1,  # debt now worth 880, ltv=0.88
        lending_rate=0, borrowing_rate=0,
    ))
    assert e.internal_state.collateral == 0
    assert e.internal_state.borrowed == 0


@pytest.mark.core
def test_liquidation_does_not_fire_below_threshold(lending):
    lending.action_deposit(1000)
    lending.action_borrow(700)  # ltv = 0.7 < 0.85
    lending.update_state(SimpleLendingGlobalState(
        collateral_price=1, debt_price=1.05,  # ltv ≈ 0.735
        lending_rate=0, borrowing_rate=0,
    ))
    assert lending.internal_state.collateral > 0
