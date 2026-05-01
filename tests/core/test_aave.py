import pytest

from fractal.core.entities.protocols.aave import AaveEntity, AaveGlobalState, EntityException


@pytest.fixture
def aave_entity():
    return AaveEntity()


@pytest.mark.core
def test_action_repay(aave_entity: AaveEntity):
    aave_entity.internal_state.borrowed = 1000
    aave_entity.action_repay(500)
    assert aave_entity.internal_state.borrowed == 500


@pytest.mark.core
def test_action_repay_exceeds_borrowed_amount(aave_entity: AaveEntity):
    aave_entity.internal_state.borrowed = 1000
    with pytest.raises(EntityException):
        aave_entity.action_repay(1500)
    assert aave_entity.internal_state.borrowed == 1000


@pytest.mark.core
def test_action_redeem_deprecated_alias(aave_entity: AaveEntity):
    """Old ``action_redeem`` still works but emits a DeprecationWarning."""
    import warnings
    aave_entity.internal_state.borrowed = 1000
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        aave_entity.action_redeem(400)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert aave_entity.internal_state.borrowed == 600


@pytest.mark.core
def test_action_redeem_via_execute_routes_to_repay(aave_entity: AaveEntity):
    """``Action('redeem', ...)`` keeps working via the alias."""
    from fractal.core.base import Action
    aave_entity.internal_state.borrowed = 1000
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        aave_entity.execute(Action("redeem", {"amount_in_product": 250}))
    assert aave_entity.internal_state.borrowed == 750


@pytest.mark.core
def test_action_borrow(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.global_state.debt_price = 10
    aave_entity.global_state.collateral_price = 100
    aave_entity.max_ltv = 0.8
    aave_entity.action_borrow(500)
    assert aave_entity.internal_state.borrowed == 500


@pytest.mark.core
def test_action_borrow_exceeds_max_ltv(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.global_state.debt_price = 10
    aave_entity.global_state.collateral_price = 1
    aave_entity.max_ltv = 0.8
    with pytest.raises(EntityException):
        aave_entity.action_borrow(80.1)


@pytest.mark.core
def test_action_borrow_rejects_cumulative_ltv_over_max():
    """H2 lock-in: each borrow individually under ``max_ltv``, but
    together they exceed it. Pre-fix this passed silently because the
    LTV check used only ``amount_in_product`` instead of cumulative debt.
    """
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1))
    e.action_deposit(1000)
    e.max_ltv = 0.8

    e.action_borrow(600)  # LTV 0.6, below limit
    assert e.internal_state.borrowed == 600

    # Second borrow would push cumulative LTV to 1.2 — must reject.
    with pytest.raises(EntityException, match="loan-to-value"):
        e.action_borrow(600)
    assert e.internal_state.borrowed == 600  # state unchanged on rejection


@pytest.mark.core
def test_action_borrow_cumulative_parity_with_simple_lending():
    """Both Aave and SimpleLending must reject the same cumulative-LTV
    scenario identically (same convention)."""
    from fractal.core.entities.simple.lending import (
        SimpleLendingEntity,
        SimpleLendingException,
        SimpleLendingGlobalState,
    )

    aave = AaveEntity()
    simple = SimpleLendingEntity()
    aave.update_state(AaveGlobalState(collateral_price=1, debt_price=1))
    simple.update_state(SimpleLendingGlobalState(collateral_price=1, debt_price=1))
    aave.action_deposit(1000)
    simple.action_deposit(1000)
    aave.max_ltv = 0.8
    simple.max_ltv = 0.8

    aave.action_borrow(700)
    simple.action_borrow(700)

    with pytest.raises(EntityException):
        aave.action_borrow(200)  # cumulative 0.9 > 0.8
    with pytest.raises(SimpleLendingException):
        simple.action_borrow(200)


@pytest.mark.core
def test_action_deposit(aave_entity: AaveEntity):
    aave_entity.action_deposit(1000)
    assert aave_entity.internal_state.collateral == 1000


@pytest.mark.core
def test_action_withdraw(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 500
    aave_entity.global_state.debt_price = 10
    aave_entity.global_state.collateral_price = 100
    aave_entity.max_ltv = 0.8
    aave_entity.action_withdraw(500)
    assert aave_entity.internal_state.collateral == 500


@pytest.mark.core
def test_action_withdraw_exceeds_max_ltv(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 79.9
    aave_entity.global_state.debt_price = 10
    aave_entity.global_state.collateral_price = 1
    aave_entity.max_ltv = 0.8
    with pytest.raises(EntityException):
        aave_entity.action_withdraw(2)


@pytest.mark.core
def test_balance(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 50
    aave_entity.global_state.debt_price = 10
    aave_entity.global_state.collateral_price = 1
    assert aave_entity.balance == 500


@pytest.mark.core
def test_ltv(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 50
    aave_entity.global_state.debt_price = 10
    aave_entity.global_state.collateral_price = 1
    assert aave_entity.ltv == 0.5


@pytest.mark.core
def test_check_liquidation(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 84
    aave_entity.global_state.debt_price = 11
    aave_entity.global_state.collateral_price = 1
    aave_entity.liq_threshold = 0.85
    aave_entity.check_liquidation()
    assert aave_entity.internal_state.collateral == 0
    assert aave_entity.internal_state.borrowed == 0


@pytest.mark.core
def test_update_state(aave_entity: AaveEntity):
    state = AaveGlobalState(
        collateral_price=1,
        debt_price=10,
        lending_rate=0.01,
        borrowing_rate=0.02
    )
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 50
    aave_entity.update_state(state)
    assert aave_entity.internal_state.collateral == 1000 * (1 + 0.01)
    assert aave_entity.internal_state.borrowed == 50 * (1 + 0.02)


@pytest.mark.core
def test_calculate_repay(aave_entity: AaveEntity):
    aave_entity.internal_state.collateral = 1000
    aave_entity.internal_state.borrowed = 50
    aave_entity.global_state.debt_price = 10
    aave_entity.global_state.collateral_price = 1
    target_ltv = 0.4
    expected_repay = 10
    assert aave_entity.calculate_repay(target_ltv) == pytest.approx(expected_repay, 1e-6)


@pytest.mark.core
def test_positive_borrowing_rate_grows_debt():
    """H1 lock-in: positive ``borrowing_rate`` ⇒ debt grows.

    Prior to the fix, ``AaveV3RatesLoader`` flipped the sign and downstream
    backtests received the inverse — debt would *shrink*. Both legs use
    the convention ``balance *= 1 + rate``, matching ``SimpleLendingEntity``.
    """
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1))
    e.action_deposit(1000)
    e.action_borrow(500)
    e.update_state(AaveGlobalState(
        collateral_price=1, debt_price=1,
        lending_rate=0.0, borrowing_rate=0.01,  # +1% per step
    ))
    assert e.internal_state.borrowed == pytest.approx(500 * 1.01)


@pytest.mark.core
def test_aave_simple_lending_parity_under_same_rates():
    """H1 parity: ``AaveEntity`` and ``SimpleLendingEntity`` must apply
    the same positive ``borrowing_rate`` identically — both grow debt
    by the same factor."""
    from fractal.core.entities.simple.lending import SimpleLendingEntity, SimpleLendingGlobalState

    aave = AaveEntity()
    simple = SimpleLendingEntity()
    aave.update_state(AaveGlobalState(collateral_price=1, debt_price=1))
    simple.update_state(SimpleLendingGlobalState(collateral_price=1, debt_price=1))
    aave.action_deposit(1000)
    simple.action_deposit(1000)
    aave.action_borrow(400)
    simple.action_borrow(400)

    aave.update_state(AaveGlobalState(
        collateral_price=1, debt_price=1,
        lending_rate=0.005, borrowing_rate=0.02,
    ))
    simple.update_state(SimpleLendingGlobalState(
        collateral_price=1, debt_price=1,
        lending_rate=0.005, borrowing_rate=0.02,
    ))
    assert aave.internal_state.collateral == pytest.approx(simple._internal_state.collateral)
    assert aave.internal_state.borrowed == pytest.approx(simple._internal_state.borrowed)
