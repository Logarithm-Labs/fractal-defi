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


# --------------------------------------------------------------- supply cap
def _capped(supply_cap=1000.0, reserve_total_supply=600.0, **kwargs):
    e = AaveEntity()
    e.update_state(AaveGlobalState(
        collateral_price=1, debt_price=1,
        supply_cap=supply_cap, reserve_total_supply=reserve_total_supply, **kwargs,
    ))
    return e


@pytest.mark.core
def test_deposit_up_to_supply_cap_accepted():
    """Room = cap − others' supply = 1000 − 600 = 400; filling it exactly is allowed."""
    e = _capped()
    e.action_deposit(400)
    assert e.internal_state.collateral == 400


@pytest.mark.core
def test_deposit_over_supply_cap_rejected():
    e = _capped()
    with pytest.raises(EntityException, match="supply cap"):
        e.action_deposit(401)
    assert e.internal_state.collateral == 0


@pytest.mark.core
def test_supply_cap_counts_own_collateral():
    """Two deposits that fit one by one but not together: 600 + 300 + 101 > 1000."""
    e = _capped()
    e.action_deposit(300)
    with pytest.raises(EntityException, match="supply cap"):
        e.action_deposit(101)
    assert e.internal_state.collateral == 300


@pytest.mark.core
def test_withdraw_frees_supply_cap_room():
    e = _capped()
    e.action_deposit(400)
    e.action_withdraw(100)
    e.action_deposit(100)
    assert e.internal_state.collateral == 400


@pytest.mark.core
def test_accrual_may_exceed_supply_cap_but_blocks_new_deposits():
    """Aave checks the cap only on supply: interest can push past it, new supply cannot."""
    e = _capped()
    e.action_deposit(400)
    e.update_state(AaveGlobalState(
        collateral_price=1, debt_price=1, lending_rate=0.01,
        supply_cap=1000.0, reserve_total_supply=600.0,
    ))
    assert e.internal_state.collateral == pytest.approx(404)
    with pytest.raises(EntityException, match="supply cap"):
        e.action_deposit(1)


@pytest.mark.core
def test_update_state_rejects_negative_supply_cap():
    e = AaveEntity()
    with pytest.raises(EntityException, match="supply_cap must be >= 0"):
        e.update_state(AaveGlobalState(collateral_price=1, debt_price=1, supply_cap=-1.0))


@pytest.mark.core
def test_update_state_rejects_negative_reserve_total_supply():
    e = AaveEntity()
    with pytest.raises(EntityException, match="reserve_total_supply must be >= 0"):
        e.update_state(AaveGlobalState(collateral_price=1, debt_price=1, reserve_total_supply=-1.0))


# ------------------------------------------------------ partial liquidation
def _partial(**kwargs):
    kwargs.setdefault("max_ltv", 0.8)
    kwargs.setdefault("liq_thr", 0.85)
    kwargs.setdefault("liquidation_bonus", 1.05)
    e = AaveEntity(liquidation_mode="partial", **kwargs)
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1))
    return e


@pytest.mark.core
def test_partial_mode_keeps_position_at_health_factor_one():
    """Aave liquidates only below HF 1; the default ``full`` mode wipes at ltv == liq_thr."""
    partial = _partial(max_ltv=0.8, liq_thr=0.8)
    full = AaveEntity(max_ltv=0.8, liq_thr=0.8)
    full.update_state(AaveGlobalState(collateral_price=1, debt_price=1))
    for e in (partial, full):
        e.action_deposit(10_000)
        e.action_borrow(8_000)
        e.update_state(AaveGlobalState(collateral_price=1, debt_price=1))
    assert (partial.internal_state.collateral, partial.internal_state.borrowed) == (10_000, 8_000)
    assert (full.internal_state.collateral, full.internal_state.borrowed) == (0, 0)


@pytest.mark.core
def test_partial_mode_half_liquidation_restores_health():
    """HF 0.984 ⇒ half the debt is repaid at a 5% bonus and the position stays open."""
    e = _partial()
    e.action_deposit(10_000)
    e.action_borrow(8_000)
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1.08))
    # repay 4000, seize 4000 · 1.08 · 1.05 = 4536; new HF = 0.85 · 5464 / 4320 = 1.075
    assert e.internal_state.borrowed == pytest.approx(4_000)
    assert e.internal_state.collateral == pytest.approx(5_464)
    assert e.health_factor >= 1


@pytest.mark.core
def test_partial_mode_full_close_factor_returns_remaining_collateral():
    """HF 0.924 ≤ 0.95 ⇒ all debt repaid; the borrower keeps 10000 − 9660 = 340."""
    e = _partial()
    e.action_deposit(10_000)
    e.action_borrow(8_000)
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1.15))
    assert e.internal_state.borrowed == 0
    assert e.internal_state.collateral == pytest.approx(340)


@pytest.mark.core
def test_partial_mode_repeats_liquidation_until_healthy():
    """A first 50% call that leaves HF < 1 is followed by another one in the same step.

    Collateral 4000, debt 3400 at price 1.065, liq_thr 0.9, bonus 1.1:
    HF = 0.9 · 4000 / 3621 = 0.9942. Call 1 repays 1700 and seizes 1991.55, leaving
    HF = 0.9984 with debt $1810.5 < $2000, so call 2 repays the remaining 1700 and
    seizes another 1991.55: 4000 − 2 · 1991.55 = 16.9 of collateral is left.
    """
    e = _partial(max_ltv=0.85, liq_thr=0.9, liquidation_bonus=1.1)
    e.action_deposit(4_000)
    e.action_borrow(3_400)
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1.065))
    assert e.internal_state.borrowed == 0
    assert e.internal_state.collateral == pytest.approx(16.9)


@pytest.mark.core
def test_partial_mode_collateral_shortfall_clears_both_legs():
    """Debt · bonus above collateral: all collateral is seized and the rest of the debt written off."""
    e = _partial()
    e.action_deposit(10_000)
    e.action_borrow(8_000)
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1.25))  # 8000 · 1.25 · 1.05 = 10500 > 10000
    assert (e.internal_state.collateral, e.internal_state.borrowed) == (0, 0)


@pytest.mark.core
def test_default_liquidation_mode_is_full():
    """Without ``liquidation_mode`` the entity keeps the historical wipe-on-liquidation behaviour."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1))
    e.action_deposit(10_000)
    e.action_borrow(8_000)
    e.update_state(AaveGlobalState(collateral_price=1, debt_price=1.08))
    assert (e.internal_state.collateral, e.internal_state.borrowed) == (0, 0)


@pytest.mark.core
def test_rejects_unknown_liquidation_mode():
    with pytest.raises(EntityException, match="liquidation_mode"):
        AaveEntity(liquidation_mode="half")


@pytest.mark.core
def test_partial_mode_requires_liquidation_bonus():
    with pytest.raises(EntityException, match="liquidation_bonus is required"):
        AaveEntity(liquidation_mode="partial")


@pytest.mark.core
def test_liquidation_bonus_only_with_partial_mode():
    with pytest.raises(EntityException, match="liquidation_bonus only applies"):
        AaveEntity(liquidation_bonus=1.05)


@pytest.mark.core
def test_rejects_liquidation_bonus_not_above_one():
    """PoolConfigurator: the bonus must be bigger than 100%."""
    with pytest.raises(EntityException, match="liquidation_bonus must be > 1"):
        AaveEntity(liquidation_mode="partial", liquidation_bonus=1.0)


@pytest.mark.core
def test_rejects_threshold_times_bonus_above_one():
    """PoolConfigurator: liq_thr · bonus ≤ 100% (0.97 · 1.05 = 1.0185)."""
    with pytest.raises(EntityException, match="liq_thr · liquidation_bonus"):
        AaveEntity(max_ltv=0.9, liq_thr=0.97, liquidation_mode="partial", liquidation_bonus=1.05)
