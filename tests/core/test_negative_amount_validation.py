"""Lock-in: negative-amount rejection on every concrete entity action.

Convention: ``if amount < 0: raise <EntityException>``. Zero is a
legitimate no-op (we treat ``=0`` as identity, not invalid).

Perp ``action_open_position`` is intentionally **not** covered — there
``amount_in_product < 0`` means *short*, which is valid on
:class:`HyperliquidEntity` and :class:`GMXV2Entity`.
"""
import pytest

from fractal.core.base import EntityException
from fractal.core.entities.protocols.aave import AaveEntity, AaveGlobalState
from fractal.core.entities.protocols.gmx_v2 import (GMXV2Entity,
                                                    GMXV2EntityException,
                                                    GMXV2GlobalState)
from fractal.core.entities.protocols.steth import (StakedETHEntity,
                                                   StakedETHEntityException,
                                                   StakedETHGlobalState)
from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                            UniswapV2LPEntity)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                            UniswapV3LPEntity)
from fractal.core.entities.protocols.uniswap_v3_spot import (
    UniswapV3SpotEntity, UniswapV3SpotEntityException, UniswapV3SpotGlobalState)


# =================================================== Aave
@pytest.fixture
def aave() -> AaveEntity:
    e = AaveEntity()
    e.update_state(AaveGlobalState(
        notional_price=1.0, product_price=1.0,
        lending_rate=0.0, borrowing_rate=0.0,
    ))
    return e


@pytest.mark.core
def test_aave_deposit_rejects_negative(aave):
    with pytest.raises(EntityException, match=">= 0"):
        aave.action_deposit(-1)


@pytest.mark.core
def test_aave_withdraw_rejects_negative(aave):
    aave.action_deposit(1000)
    with pytest.raises(EntityException, match=">= 0"):
        aave.action_withdraw(-1)


@pytest.mark.core
def test_aave_borrow_rejects_negative(aave):
    aave.action_deposit(1000)
    with pytest.raises(EntityException, match=">= 0"):
        aave.action_borrow(-1)


@pytest.mark.core
def test_aave_repay_rejects_negative(aave):
    aave.action_deposit(1000)
    aave.action_borrow(100)
    with pytest.raises(EntityException, match=">= 0"):
        aave.action_repay(-1)


@pytest.mark.core
def test_aave_zero_amounts_are_noop(aave):
    """Zero amounts must succeed silently (identity)."""
    aave.action_deposit(0)
    aave.action_deposit(1000)
    aave.action_withdraw(0)
    aave.action_borrow(0)
    aave.action_repay(0)
    assert aave.internal_state.collateral == 1000


# =================================================== GMXV2
@pytest.fixture
def gmx() -> GMXV2Entity:
    e = GMXV2Entity()
    e.update_state(GMXV2GlobalState(price=1000.0))
    return e


@pytest.mark.core
def test_gmx_withdraw_rejects_negative(gmx):
    gmx.action_deposit(1000)
    with pytest.raises(GMXV2EntityException, match=">= 0"):
        gmx.action_withdraw(-1)


@pytest.mark.core
def test_gmx_open_position_negative_is_short_not_rejected(gmx):
    """Negative amount on perp open_position = short, must NOT raise."""
    gmx.action_deposit(1000)
    gmx.action_open_position(-0.1)  # short, must be allowed
    assert gmx.size == pytest.approx(-0.1)


# =================================================== UniV2LP
@pytest.fixture
def univ2() -> UniswapV2LPEntity:
    return UniswapV2LPEntity(UniswapV2LPConfig())


@pytest.mark.core
def test_univ2_deposit_rejects_negative(univ2):
    with pytest.raises(EntityException, match=">= 0"):
        univ2.action_deposit(-1)


@pytest.mark.core
def test_univ2_withdraw_rejects_negative(univ2):
    univ2.action_deposit(1000)
    with pytest.raises(EntityException, match=">= 0"):
        univ2.action_withdraw(-1)


@pytest.mark.core
def test_univ2_open_position_rejects_negative(univ2):
    univ2.action_deposit(1000)
    with pytest.raises(EntityException, match=">= 0"):
        univ2.action_open_position(-100)


# =================================================== UniV3LP
@pytest.fixture
def univ3lp() -> UniswapV3LPEntity:
    return UniswapV3LPEntity(UniswapV3LPConfig())


@pytest.mark.core
def test_univ3lp_deposit_rejects_negative(univ3lp):
    with pytest.raises(EntityException, match=">= 0"):
        univ3lp.action_deposit(-1)


@pytest.mark.core
def test_univ3lp_withdraw_rejects_negative(univ3lp):
    univ3lp.action_deposit(1000)
    with pytest.raises(EntityException, match=">= 0"):
        univ3lp.action_withdraw(-1)


@pytest.mark.core
def test_univ3lp_open_position_rejects_negative(univ3lp):
    univ3lp.action_deposit(1000)
    with pytest.raises(EntityException, match=">= 0"):
        univ3lp.action_open_position(-100, price_lower=0.9, price_upper=1.1)


# =================================================== UniV3Spot
@pytest.fixture
def v3spot() -> UniswapV3SpotEntity:
    e = UniswapV3SpotEntity()
    e.update_state(UniswapV3SpotGlobalState(price=2000.0))
    return e


@pytest.mark.core
def test_univ3spot_deposit_rejects_negative(v3spot):
    with pytest.raises(UniswapV3SpotEntityException, match=">= 0"):
        v3spot.action_deposit(-1)


@pytest.mark.core
def test_univ3spot_deposit_zero_is_noop(v3spot):
    """B19 fix: deposit(0) used to raise (`<= 0`); now zero is allowed."""
    v3spot.action_deposit(0)
    assert v3spot.internal_state.cash == 0


@pytest.mark.core
def test_univ3spot_withdraw_rejects_negative(v3spot):
    v3spot.action_deposit(1000)
    with pytest.raises(UniswapV3SpotEntityException, match=">= 0"):
        v3spot.action_withdraw(-1)


@pytest.mark.core
def test_univ3spot_buy_rejects_negative(v3spot):
    v3spot.action_deposit(1000)
    with pytest.raises(UniswapV3SpotEntityException, match=">= 0"):
        v3spot.action_buy(-1)


@pytest.mark.core
def test_univ3spot_sell_rejects_negative(v3spot):
    v3spot.action_deposit(2000)
    v3spot.action_buy(1000)
    with pytest.raises(UniswapV3SpotEntityException, match=">= 0"):
        v3spot.action_sell(-1)


# =================================================== stETH
@pytest.fixture
def steth() -> StakedETHEntity:
    e = StakedETHEntity()
    e.update_state(StakedETHGlobalState(price=2000.0, rate=0.0))
    return e


@pytest.mark.core
def test_steth_deposit_rejects_negative(steth):
    with pytest.raises(StakedETHEntityException, match=">= 0"):
        steth.action_deposit(-1)


@pytest.mark.core
def test_steth_deposit_zero_is_noop(steth):
    """B19 fix: deposit(0) used to raise (`<= 0`); now zero is allowed."""
    steth.action_deposit(0)
    assert steth.internal_state.cash == 0


@pytest.mark.core
def test_steth_withdraw_rejects_negative(steth):
    steth.action_deposit(1000)
    with pytest.raises(StakedETHEntityException, match=">= 0"):
        steth.action_withdraw(-1)


@pytest.mark.core
def test_steth_buy_rejects_negative(steth):
    steth.action_deposit(1000)
    with pytest.raises(StakedETHEntityException, match=">= 0"):
        steth.action_buy(-1)


@pytest.mark.core
def test_steth_sell_rejects_negative(steth):
    steth.action_deposit(2000)
    steth.action_buy(1000)
    with pytest.raises(StakedETHEntityException, match=">= 0"):
        steth.action_sell(-1)
