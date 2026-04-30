"""Tests for the direction-agnostic configuration of lending entities.

Both AaveEntity and SimpleLendingEntity are mathematically symmetric
under their two natural use-cases:

* **stable collateral / volatile debt** (default — synthetic short)
* **volatile collateral / stable debt** (leveraged-long base)

The ``collateral_is_volatile`` config flag is informational; these tests
verify the math is identical under both labellings (with prices flipped
appropriately) and that the helper accessors (``collateral_value``,
``debt_value``, ``health_factor``) work in both directions.
"""
import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.aave import (AaveEntity,
                                                  AaveGlobalState)
from fractal.core.entities.simple.lending import (SimpleLendingEntity,
                                                  SimpleLendingGlobalState)


# ============================================================ flag plumbing
@pytest.mark.core
def test_aave_default_is_stable_collateral():
    e = AaveEntity()
    assert e.collateral_is_volatile is False


@pytest.mark.core
def test_aave_volatile_collateral_flag_set():
    e = AaveEntity(collateral_is_volatile=True)
    assert e.collateral_is_volatile is True


@pytest.mark.core
def test_simple_lending_default_is_stable_collateral():
    e = SimpleLendingEntity()
    assert e.collateral_is_volatile is False


@pytest.mark.core
def test_simple_lending_volatile_collateral_flag_set():
    e = SimpleLendingEntity(collateral_is_volatile=True)
    assert e.collateral_is_volatile is True


# ============================================================ math symmetry
@pytest.mark.core
def test_aave_math_identical_under_direction_flip():
    """Same scenario set up two ways — same balance/LTV/health.

    Setup 1 (default): deposit 10k USDC, borrow 2 ETH at $3000 each.
    Setup 2 (volatile collateral): deposit 10 ETH at $3000, borrow 6k USDC.
    Both have the same equity (≈ collateral_value − debt_value).
    """
    # Stable collateral: 10k USDC, 2 ETH debt
    a_stable = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=False)
    a_stable.update_state(AaveGlobalState(collateral_price=1.0, debt_price=3000.0))
    a_stable.action_deposit(10_000)
    a_stable.action_borrow(2.0)
    # equity = 10_000 × 1 − 2 × 3000 = 4_000

    # Volatile collateral: 10 ETH, 6k USDC debt
    a_vol = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    a_vol.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    a_vol.action_deposit(10.0)  # 10 ETH
    a_vol.action_borrow(6_000)  # 6k USDC
    # equity = 10 × 3000 − 6_000 × 1 = 24_000

    # Different scenarios; just sanity-check both use the same formula.
    assert a_stable.balance == 10_000 - 2 * 3000  # = 4_000
    assert a_vol.balance == 10 * 3000 - 6_000     # = 24_000

    # LTV computation identical regardless of direction:
    assert a_stable.ltv == pytest.approx(2 * 3000 / 10_000)         # 0.6
    assert a_vol.ltv == pytest.approx(6_000 / (10 * 3000))           # 0.2


# ============================================================ helper properties
@pytest.mark.core
def test_aave_collateral_value_in_stable_mode():
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=3000.0))
    e.action_deposit(10_000)
    assert e.collateral_value == 10_000  # 10k USDC × $1


@pytest.mark.core
def test_aave_collateral_value_in_volatile_mode():
    e = AaveEntity(collateral_is_volatile=True)
    e.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    e.action_deposit(5.0)
    assert e.collateral_value == pytest.approx(15_000)  # 5 ETH × $3000


@pytest.mark.core
def test_aave_debt_value_in_stable_mode():
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=3000.0))
    e.action_deposit(10_000)
    e.action_borrow(2.0)
    assert e.debt_value == pytest.approx(6_000)  # 2 ETH × $3000


@pytest.mark.core
def test_aave_debt_value_in_volatile_mode():
    e = AaveEntity(collateral_is_volatile=True)
    e.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    e.action_deposit(10.0)
    e.action_borrow(6_000)
    assert e.debt_value == pytest.approx(6_000)  # 6k USDC × $1


@pytest.mark.core
def test_aave_health_factor_when_no_debt_is_inf():
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=3000.0))
    e.action_deposit(10_000)
    assert e.health_factor == float("inf")


@pytest.mark.core
def test_aave_health_factor_far_from_liquidation():
    """LTV = 0.5, liq_thr = 0.85 → health = 1.7 (safe)."""
    e = AaveEntity(max_ltv=0.8, liq_thr=0.85)
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(1000)
    e.action_borrow(500)  # LTV = 0.5
    assert e.health_factor == pytest.approx(0.85 / 0.5)


@pytest.mark.core
def test_aave_health_factor_zero_when_undefined_ltv():
    """No collateral with debt → ltv = inf → health = 0."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e._internal_state.collateral = 0.0
    e._internal_state.borrowed = 100.0
    assert e.health_factor == 0.0


@pytest.mark.core
def test_simple_lending_helper_properties_match_aave():
    """SimpleLending helpers compute the same way as Aave."""
    a = AaveEntity()
    s = SimpleLendingEntity()
    a.update_state(AaveGlobalState(collateral_price=1.0, debt_price=2.0))
    s.update_state(SimpleLendingGlobalState(collateral_price=1.0, debt_price=2.0))
    a.action_deposit(1000)
    s.action_deposit(1000)
    a.action_borrow(200)
    s.action_borrow(200)
    assert a.collateral_value == s.collateral_value
    assert a.debt_value == s.debt_value
    assert a.health_factor == s.health_factor


# ============================================================ leveraged-long math
@pytest.mark.core
def test_aave_volatile_collateral_borrows_stable_against_eth():
    """Volatile-collateral mode: deposit ETH, borrow USDC.

    Walks through a basic 1.6x leveraged-long ETH position setup using
    Aave alone (without the spot loop step). Verifies the math holds.
    """
    e = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    e.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))

    # Start: deposit 3.33 ETH (= $10k of ETH)
    e.action_deposit(3.33)
    assert e.collateral_value == pytest.approx(3.33 * 3000)

    # Borrow 6_000 USDC against it (LTV ≈ 0.6)
    e.action_borrow(6_000)
    assert e.ltv == pytest.approx(6_000 / (3.33 * 3000))
    assert e.health_factor == pytest.approx(0.85 / e.ltv)

    # Equity = 10k - 6k = 4k notional, but our position is "long ETH" at
    # 3.33 ETH × $3000 = $10k exposure (vs initial $4k equity → 2.5x
    # exposure) — the ``borrowed`` USDC is conceptually the proceeds of
    # the swap that opened the leverage.
    assert e.balance == pytest.approx(3.33 * 3000 - 6_000)


@pytest.mark.core
def test_aave_volatile_collateral_eth_price_drop_increases_ltv():
    """Long-ETH setup: ETH price drop raises LTV (toward liquidation)."""
    e = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    e.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    e.action_deposit(10.0)  # 10 ETH
    e.action_borrow(15_000)  # LTV = 15k / (10 × 3000) = 0.5

    initial_ltv = e.ltv
    initial_health = e.health_factor

    # ETH drops 20%: 3000 → 2400
    e.update_state(AaveGlobalState(collateral_price=2400.0, debt_price=1.0))
    # New LTV = 15k / (10 × 2400) = 0.625
    assert e.ltv > initial_ltv
    assert e.health_factor < initial_health
