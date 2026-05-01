"""Cross-cutting entity-API parity lock-ins.

* ``effective_fee_rate`` property on spot/LST entities — parity with
  pool entities for polymorphic strategy code.
* ``max_borrow_amount`` and ``liquidation_price`` helpers on Aave /
  SimpleLending.
* ``TRADING_FEE`` / ``MAX_LEVERAGE`` UPPERCASE → lowercase rename on
  Hyperliquid + SimplePerp, with deprecated UPPERCASE property aliases.
"""
import math
import warnings

import pytest

from fractal.core.entities.protocols.aave import AaveEntity, AaveGlobalState
from fractal.core.entities.protocols.hyperliquid import HyperliquidEntity
from fractal.core.entities.protocols.steth import StakedETHEntity
from fractal.core.entities.protocols.uniswap_v3_spot import UniswapV3SpotEntity
from fractal.core.entities.simple.lending import SimpleLendingEntity, SimpleLendingGlobalState
from fractal.core.entities.simple.liquid_staking import SimpleLiquidStakingToken
from fractal.core.entities.simple.perp import SimplePerpEntity
from fractal.core.entities.simple.spot import SimpleSpotExchange


@pytest.mark.core
@pytest.mark.parametrize("factory,fee", [
    (lambda f: UniswapV3SpotEntity(trading_fee=f), 0.003),
    (lambda f: StakedETHEntity(trading_fee=f), 0.005),
    (lambda f: SimpleSpotExchange(trading_fee=f), 0.002),
    (lambda f: SimpleLiquidStakingToken(trading_fee=f), 0.001),
])
def test_spot_lst_effective_fee_rate_aliases_trading_fee(factory, fee):
    e = factory(fee)
    assert e.effective_fee_rate == pytest.approx(fee)


@pytest.mark.core
def test_effective_fee_rate_changes_when_trading_fee_changes():
    """Property tracks the underlying ``trading_fee``."""
    e = UniswapV3SpotEntity(trading_fee=0.003)
    assert e.effective_fee_rate == 0.003
    # Direct mutation through public attr (UniV3Spot exposes lowercase).
    e.trading_fee = 0.0005
    assert e.effective_fee_rate == 0.0005


@pytest.mark.core
def test_aave_max_borrow_amount_zero_when_no_collateral():
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    assert e.max_borrow_amount == 0


@pytest.mark.core
def test_aave_max_borrow_amount_zero_when_already_at_max_ltv():
    e = AaveEntity(max_ltv=0.8)
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(1000)
    e.action_borrow(800)  # exactly at max_ltv
    assert e.max_borrow_amount == pytest.approx(0)


@pytest.mark.core
def test_aave_max_borrow_amount_with_headroom():
    """1000 USDC collateral, max_ltv=0.8 → can borrow up to 800 USDC of debt."""
    e = AaveEntity(max_ltv=0.8)
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(1000)
    assert e.max_borrow_amount == pytest.approx(800)


@pytest.mark.core
def test_aave_max_borrow_amount_after_partial_borrow():
    e = AaveEntity(max_ltv=0.8)
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(1000)
    e.action_borrow(300)
    # Headroom: 800 - 300 = 500
    assert e.max_borrow_amount == pytest.approx(500)


@pytest.mark.core
def test_aave_max_borrow_amount_with_volatile_collateral():
    """ETH collateral × $3000, USDC debt: max_borrow in USDC count."""
    e = AaveEntity(max_ltv=0.75, collateral_is_volatile=True)
    e.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    e.action_deposit(1.0)  # 1 ETH = $3000 collat value
    # Cap = 0.75 × 3000 = 2250 USDC max debt → max_borrow = 2250
    assert e.max_borrow_amount == pytest.approx(2250)


@pytest.mark.core
def test_aave_liquidation_price_nan_when_no_debt():
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(1000)
    assert math.isnan(e.liquidation_price)


@pytest.mark.core
def test_aave_liquidation_price_short_volatile_default():
    """Default mode (USDC collat, volatile debt): liq is the volatile-asset
    price level that pushes LTV past liq_thr.

    Setup: 10000 USDC collat, 2 ETH debt, liq_thr=0.85.
    liq = 0.85 × 10000 × 1.0 / 2 = $4250 — at this ETH price, ltv = 0.85.
    """
    e = AaveEntity(max_ltv=0.8, liq_thr=0.85)
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=3000.0))
    e.action_deposit(10_000)
    e.action_borrow(2.0)
    assert e.liquidation_price == pytest.approx(0.85 * 10_000 / 2)


@pytest.mark.core
def test_aave_liquidation_price_long_volatile():
    """Volatile-collateral mode (ETH collat, USDC debt): liq is the
    collateral-asset price drop level that pushes LTV past liq_thr.

    Setup: 10 ETH collat, 15000 USDC debt, liq_thr=0.85.
    liq = 15000 × 1.0 / (10 × 0.85) ≈ $1764.7 — at this ETH price, ltv = 0.85.
    """
    e = AaveEntity(max_ltv=0.8, liq_thr=0.85, collateral_is_volatile=True)
    e.update_state(AaveGlobalState(collateral_price=3000.0, debt_price=1.0))
    e.action_deposit(10.0)
    e.action_borrow(15_000)
    expected = 15_000 / (10 * 0.85)
    assert e.liquidation_price == pytest.approx(expected)


@pytest.mark.core
def test_aave_liquidation_price_at_liq_triggers_wipe():
    """At the computed liquidation price, ``update_state`` wipes."""
    e = AaveEntity(max_ltv=0.8, liq_thr=0.85)
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=3000.0))
    e.action_deposit(10_000)
    e.action_borrow(2.0)
    liq = e.liquidation_price  # ≈ 4250
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=liq))
    # Position wiped on liquidation.
    assert e._internal_state.collateral == 0
    assert e._internal_state.borrowed == 0


@pytest.mark.core
def test_simple_lending_max_borrow_and_liquidation_price_match_aave():
    """SimpleLending exposes the same helpers with the same math."""
    a = AaveEntity(max_ltv=0.8, liq_thr=0.85)
    s = SimpleLendingEntity(max_ltv=0.8, liq_thr=0.85)
    a.update_state(AaveGlobalState(collateral_price=1.0, debt_price=2.0))
    s.update_state(SimpleLendingGlobalState(collateral_price=1.0, debt_price=2.0))
    a.action_deposit(1000)
    s.action_deposit(1000)
    a.action_borrow(200)
    s.action_borrow(200)
    assert a.max_borrow_amount == pytest.approx(s.max_borrow_amount)
    assert a.liquidation_price == pytest.approx(s.liquidation_price)


@pytest.mark.core
def test_hl_lowercase_attrs_present():
    """Hyperliquid uses lowercase ``trading_fee`` / ``max_leverage`` post-D5."""
    e = HyperliquidEntity(trading_fee=0.0007, max_leverage=25)
    assert e.trading_fee == pytest.approx(0.0007)
    assert e.max_leverage == 25


@pytest.mark.core
def test_sp_lowercase_attrs_present():
    e = SimplePerpEntity(trading_fee=0.0007, max_leverage=25)
    assert e.trading_fee == pytest.approx(0.0007)
    assert e.max_leverage == 25


@pytest.mark.core
def test_hl_uppercase_aliases_emit_deprecation_warning_but_still_work():
    e = HyperliquidEntity(trading_fee=0.0007, max_leverage=25)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf = e.TRADING_FEE
        ml = e.MAX_LEVERAGE
    assert tf == pytest.approx(0.0007)
    assert ml == 25
    deprec_msgs = [str(w.message) for w in caught
                   if issubclass(w.category, DeprecationWarning)]
    assert any("TRADING_FEE" in m for m in deprec_msgs)
    assert any("MAX_LEVERAGE" in m for m in deprec_msgs)


@pytest.mark.core
def test_sp_uppercase_aliases_emit_deprecation_warning_but_still_work():
    e = SimplePerpEntity(trading_fee=0.0007, max_leverage=25)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf = e.TRADING_FEE
        ml = e.MAX_LEVERAGE
    assert tf == pytest.approx(0.0007)
    assert ml == 25
    deprec_msgs = [str(w.message) for w in caught
                   if issubclass(w.category, DeprecationWarning)]
    assert any("TRADING_FEE" in m for m in deprec_msgs)
    assert any("MAX_LEVERAGE" in m for m in deprec_msgs)


@pytest.mark.core
def test_uppercase_aliases_have_no_setter():
    """Property has no setter → assignment raises (signals to migrate)."""
    e = HyperliquidEntity()
    with pytest.raises(AttributeError):
        e.TRADING_FEE = 0.001
    with pytest.raises(AttributeError):
        e.MAX_LEVERAGE = 10
