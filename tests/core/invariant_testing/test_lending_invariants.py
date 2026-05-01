"""Invariant + parity tests for lending entities (Aave, SimpleLending).

* **State-machine invariants**: collateral/borrowed never negative, debt
  cannot outlive collateral after liquidation, deposit→withdraw round-trip.
* **Validation parity**: both entities reject the same illegal operations
  with semantically matching errors.
* **Math parity for the canonical case**: same inputs (collateral_price=1,
  debt_price=1) produce the same balance/LTV/calculate_repay across
  Aave and SimpleLending.
* **API surface check**: both expose the lending paradigm's actions and
  read-only properties.
"""
import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.aave import AaveEntity, AaveGlobalState
from fractal.core.entities.simple.lending import SimpleLendingEntity, SimpleLendingException, SimpleLendingGlobalState


def _aave():
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    return e


def _simple():
    e = SimpleLendingEntity()
    e.update_state(SimpleLendingGlobalState(collateral_price=1.0, debt_price=1.0))
    return e


@pytest.mark.core
def test_aave_initial_state_clean():
    e = AaveEntity()
    assert e._internal_state.collateral == 0
    assert e._internal_state.borrowed == 0
    assert e.balance == 0
    assert e.ltv == 0


@pytest.mark.core
def test_simple_lending_initial_state_clean():
    e = SimpleLendingEntity()
    assert e._internal_state.collateral == 0
    assert e._internal_state.borrowed == 0
    assert e.balance == 0
    assert e.ltv == 0


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", [(_aave, EntityException), (_simple, SimpleLendingException)])
def test_lending_negative_deposit_rejected(factory, exc):
    e = factory()
    with pytest.raises(exc, match="deposit amount must be >= 0"):
        e.action_deposit(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", [(_aave, EntityException), (_simple, SimpleLendingException)])
def test_lending_negative_withdraw_rejected(factory, exc):
    e = factory()
    e.action_deposit(100)
    with pytest.raises(exc, match="withdraw amount must be >= 0"):
        e.action_withdraw(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", [(_aave, EntityException), (_simple, SimpleLendingException)])
def test_lending_negative_borrow_rejected(factory, exc):
    e = factory()
    e.action_deposit(1000)
    with pytest.raises(exc, match="borrow amount must be >= 0"):
        e.action_borrow(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", [(_aave, EntityException), (_simple, SimpleLendingException)])
def test_lending_negative_repay_rejected(factory, exc):
    e = factory()
    e.action_deposit(1000)
    e.action_borrow(100)
    with pytest.raises(exc, match="repay amount must be >= 0"):
        e.action_repay(-1)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", [(_aave, EntityException), (_simple, SimpleLendingException)])
def test_lending_withdraw_overdraft_rejected(factory, exc):
    e = factory()
    e.action_deposit(100)
    with pytest.raises(exc):
        e.action_withdraw(200)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", [(_aave, EntityException), (_simple, SimpleLendingException)])
def test_lending_repay_more_than_borrowed_rejected(factory, exc):
    e = factory()
    e.action_deposit(1000)
    e.action_borrow(100)
    with pytest.raises(exc):
        e.action_repay(500)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", [(_aave, EntityException), (_simple, SimpleLendingException)])
def test_lending_borrow_above_max_ltv_rejected(factory, exc):
    e = factory()
    e.action_deposit(1000)
    # default max_ltv=0.8, so borrowing 900 (LTV=0.9) should fail
    with pytest.raises(exc):
        e.action_borrow(900)


@pytest.mark.core
@pytest.mark.parametrize("factory,exc", [(_aave, EntityException), (_simple, SimpleLendingException)])
def test_lending_withdraw_pushes_above_max_ltv_rejected(factory, exc):
    """Cannot withdraw collateral if it would breach max_ltv."""
    e = factory()
    e.action_deposit(1000)
    e.action_borrow(700)  # at LTV=0.7
    # Withdraw that pushes LTV past 0.8 (e.g. removing 200 → 800 collat, LTV=0.875)
    with pytest.raises(exc):
        e.action_withdraw(200)


@pytest.mark.core
def test_aave_rejects_liq_thr_below_max_ltv():
    """``liq_thr >= max_ltv`` invariant validated at construction."""
    with pytest.raises(EntityException, match="liq_thr.*must be >="):
        AaveEntity(max_ltv=0.8, liq_thr=0.7)


@pytest.mark.core
def test_simple_lending_rejects_liq_thr_below_max_ltv():
    with pytest.raises(SimpleLendingException, match="liq_thr.*must be >="):
        SimpleLendingEntity(max_ltv=0.8, liq_thr=0.7)


@pytest.mark.core
def test_aave_rejects_max_ltv_out_of_range():
    with pytest.raises(EntityException, match="max_ltv must be in"):
        AaveEntity(max_ltv=1.5)
    with pytest.raises(EntityException, match="max_ltv must be in"):
        AaveEntity(max_ltv=0.0)


@pytest.mark.core
def test_aave_rejects_negative_lending_rate_below_minus_one():
    """``lending_rate < -1`` would flip collateral negative."""
    e = AaveEntity()
    e.action_deposit(1000)
    with pytest.raises(EntityException, match="lending_rate must be >= -1"):
        e.update_state(AaveGlobalState(collateral_price=1, debt_price=1,
                                       lending_rate=-1.5, borrowing_rate=0))


@pytest.mark.core
def test_simple_lending_rejects_borrowing_rate_below_minus_one():
    e = SimpleLendingEntity()
    with pytest.raises(SimpleLendingException, match="borrowing_rate must be >= -1"):
        e.update_state(SimpleLendingGlobalState(collateral_price=1, debt_price=1,
                                                lending_rate=0, borrowing_rate=-2))


@pytest.mark.core
@pytest.mark.parametrize("factory", [_aave, _simple])
def test_lending_collateral_never_negative_through_lifecycle(factory):
    e = factory()
    assert e._internal_state.collateral >= 0
    e.action_deposit(1000)
    assert e._internal_state.collateral >= 0
    e.action_borrow(500)
    assert e._internal_state.collateral >= 0
    e.action_repay(500)
    assert e._internal_state.collateral >= 0
    e.action_withdraw(1000)
    assert e._internal_state.collateral >= 0


@pytest.mark.core
@pytest.mark.parametrize("factory", [_aave, _simple])
def test_lending_borrowed_never_negative_through_lifecycle(factory):
    e = factory()
    e.action_deposit(1000)
    assert e._internal_state.borrowed >= 0
    e.action_borrow(500)
    assert e._internal_state.borrowed >= 0
    e.action_repay(200)
    assert e._internal_state.borrowed >= 0
    e.action_repay(300)
    assert e._internal_state.borrowed == 0


@pytest.mark.core
@pytest.mark.parametrize("factory", [_aave, _simple])
def test_lending_full_repay_reduces_borrowed_to_zero(factory):
    e = factory()
    e.action_deposit(1000)
    e.action_borrow(700)
    e.action_repay(700)
    assert e._internal_state.borrowed == 0
    assert e.ltv == 0


@pytest.mark.core
def test_ltv_zero_when_no_debt_aave():
    e = _aave()
    e.action_deposit(1000)
    assert e.ltv == 0


@pytest.mark.core
def test_ltv_zero_when_no_debt_simple():
    e = _simple()
    e.action_deposit(1000)
    assert e.ltv == 0


@pytest.mark.core
def test_ltv_correct_with_debt_aave():
    e = _aave()
    e.action_deposit(1000)
    e.action_borrow(500)
    # ltv = 500 * 1 / (1000 * 1) = 0.5
    assert e.ltv == pytest.approx(0.5)


@pytest.mark.core
def test_ltv_correct_with_debt_simple():
    e = _simple()
    e.action_deposit(1000)
    e.action_borrow(500)
    assert e.ltv == pytest.approx(0.5)


@pytest.mark.core
def test_aave_ltv_inf_when_collateral_zero_with_debt():
    """LTV = inf, calculate_repay raises."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1))
    e._internal_state.collateral = 0
    e._internal_state.borrowed = 100
    assert e.ltv == float("inf")
    with pytest.raises(EntityException, match="non-finite"):
        e.calculate_repay(0.5)


@pytest.mark.core
def test_aave_liquidation_wipes_position_when_ltv_exceeds_threshold():
    e = _aave()
    e.action_deposit(1000)
    e.action_borrow(700)  # LTV 0.7 (below liq_thr 0.85)
    # Trigger liquidation by raising product_price
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=2))
    # Now ltv = 700 * 2 / (1000 * 1) = 1.4 — way above 0.85
    assert e._internal_state.collateral == 0
    assert e._internal_state.borrowed == 0


@pytest.mark.core
def test_simple_lending_liquidation_wipes_position():
    e = _simple()
    e.action_deposit(1000)
    e.action_borrow(700)
    e.update_state(SimpleLendingGlobalState(collateral_price=1, debt_price=2,
                                            lending_rate=0, borrowing_rate=0))
    assert e._internal_state.collateral == 0
    assert e._internal_state.borrowed == 0


@pytest.mark.core
def test_aave_simple_lending_match_balance_under_canonical_inputs():
    """At collateral_price=1, debt_price=1, lending_rate=0, borrowing_rate=0,
    Aave and SimpleLending must produce identical balance/LTV behaviour."""
    a = _aave()
    s = _simple()
    a.action_deposit(1000)
    s.action_deposit(1000)
    a.action_borrow(500)
    s.action_borrow(500)
    assert a.balance == s.balance
    assert a.ltv == s.ltv


@pytest.mark.core
def test_aave_simple_lending_match_calculate_repay():
    a = _aave()
    s = _simple()
    a.action_deposit(1000)
    s.action_deposit(1000)
    a.action_borrow(500)
    s.action_borrow(500)
    assert a.calculate_repay(0.3) == pytest.approx(s.calculate_repay(0.3))


@pytest.mark.core
def test_aave_check_liquidation_deprecated_alias_still_works():
    """``check_liquidation`` is deprecated public alias of ``_check_liquidation``."""
    import warnings
    e = _aave()
    e.action_deposit(1000)
    e.action_borrow(700)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e.check_liquidation()
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
