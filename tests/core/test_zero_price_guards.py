"""Lock-in: zero/negative-price rejection where division by price happens.

Convention: ``if price <= 0: raise <EntityException>(f"... price must be > 0, got {price}")``.

This catches the realistic bug of running ``action_*`` against a freshly
constructed entity whose ``GlobalState`` defaults to ``price=0`` /
``mark_price=0`` / ``notional_price=0`` / ``product_price=0``.
"""
import pytest

from fractal.core.base import EntityException
from fractal.core.entities.protocols.aave import AaveEntity, AaveGlobalState
from fractal.core.entities.protocols.gmx_v2 import (GMXV2Entity,
                                                    GMXV2EntityException,
                                                    GMXV2GlobalState)
from fractal.core.entities.protocols.hyperliquid import (
    HyperliquidEntity, HyperliquidEntityException, HyperLiquidGlobalState)
from fractal.core.entities.protocols.steth import (StakedETHEntity,
                                                   StakedETHEntityException,
                                                   StakedETHGlobalState)
from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                            UniswapV2LPEntity,
                                                            UniswapV2LPGlobalState)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                            UniswapV3LPEntity,
                                                            UniswapV3LPGlobalState)
from fractal.core.entities.protocols.uniswap_v3_spot import (
    UniswapV3SpotEntity, UniswapV3SpotEntityException, UniswapV3SpotGlobalState)


# ===================================================== Aave: borrow / withdraw / ltv / calculate_repay
@pytest.mark.core
def test_aave_borrow_rejects_zero_notional_price():
    e = AaveEntity()
    e.update_state(AaveGlobalState(notional_price=0.0, product_price=1.0))
    e.action_deposit(1000)
    with pytest.raises(EntityException, match="notional_price must be > 0"):
        e.action_borrow(100)


@pytest.mark.core
def test_aave_borrow_rejects_zero_product_price():
    e = AaveEntity()
    e.update_state(AaveGlobalState(notional_price=1.0, product_price=0.0))
    e.action_deposit(1000)
    with pytest.raises(EntityException, match="product_price must be > 0"):
        e.action_borrow(100)


@pytest.mark.core
def test_aave_withdraw_with_full_collateral_and_debt_raises_clearly():
    """Used to raise ZeroDivisionError; now explicit message."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(notional_price=1.0, product_price=1.0))
    e.action_deposit(1000)
    e.action_borrow(500)
    with pytest.raises(EntityException, match="cannot withdraw all collateral"):
        e.action_withdraw(1000)


@pytest.mark.core
def test_aave_withdraw_no_debt_zero_prices_ok():
    """No debt → no LTV check needed → zero prices don't matter."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(notional_price=0.0, product_price=0.0))
    e.action_deposit(1000)
    e.action_withdraw(500)  # must not raise
    assert e.internal_state.collateral == 500


@pytest.mark.core
def test_aave_ltv_returns_inf_when_collateral_zero_with_debt():
    """Used to ZeroDivisionError; now explicit ``+inf``."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(notional_price=1.0, product_price=1.0))
    e._internal_state.collateral = 0.0
    e._internal_state.borrowed = 100.0
    assert e.ltv == float("inf")


@pytest.mark.core
def test_aave_ltv_returns_inf_when_notional_price_zero_with_debt():
    e = AaveEntity()
    e.update_state(AaveGlobalState(notional_price=0.0, product_price=1.0))
    e._internal_state.collateral = 1000.0
    e._internal_state.borrowed = 100.0
    assert e.ltv == float("inf")


@pytest.mark.core
def test_aave_calculate_repay_raises_when_ltv_inf():
    e = AaveEntity()
    e.update_state(AaveGlobalState(notional_price=1.0, product_price=1.0))
    e._internal_state.collateral = 0.0
    e._internal_state.borrowed = 100.0
    with pytest.raises(EntityException, match="non-finite"):
        e.calculate_repay(0.5)


# ===================================================== Hyperliquid: open_position
@pytest.mark.core
def test_hyperliquid_open_position_rejects_zero_mark_price():
    e = HyperliquidEntity()
    # default global state mark_price = 0
    e.action_deposit(1000)
    with pytest.raises(HyperliquidEntityException, match="mark_price must be > 0"):
        e.action_open_position(1.0)


# ===================================================== GMXV2: open_position
@pytest.mark.core
def test_gmx_open_position_rejects_zero_price():
    e = GMXV2Entity()
    e.action_deposit(1000)
    with pytest.raises(GMXV2EntityException, match="price must be > 0"):
        e.action_open_position(1.0)


# ===================================================== UniV2 LP: open_position
@pytest.mark.core
def test_univ2_open_position_rejects_zero_price():
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    e.update_state(UniswapV2LPGlobalState(price=0.0, tvl=10_000, liquidity=10_000))
    e.action_deposit(1000)
    with pytest.raises(EntityException, match="price must be > 0"):
        e.action_open_position(500)


# ===================================================== UniV3 LP: calculate_fees short-circuits
@pytest.mark.core
def test_univ3lp_calculate_fees_returns_zero_when_price_zero():
    """Used to ZeroDivisionError on 1/p; now silently 0."""
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    e.update_state(UniswapV3LPGlobalState(price=1.0, tvl=10_000, liquidity=10_000, fees=100))
    e.action_deposit(1000)
    e.action_open_position(500, price_lower=0.9, price_upper=1.1)
    # Now flip the global price to 0 — calculate_fees must short-circuit.
    e._global_state.price = 0.0
    assert e.calculate_fees() == 0


# ===================================================== UniV3Spot: buy
@pytest.mark.core
def test_univ3spot_buy_rejects_zero_price():
    e = UniswapV3SpotEntity()
    e.action_deposit(1000)
    with pytest.raises(UniswapV3SpotEntityException, match="price must be > 0"):
        e.action_buy(100)


# ===================================================== stETH: buy
@pytest.mark.core
def test_steth_buy_rejects_zero_price():
    e = StakedETHEntity()
    e.action_deposit(1000)
    with pytest.raises(StakedETHEntityException, match="price must be > 0"):
        e.action_buy(100)
