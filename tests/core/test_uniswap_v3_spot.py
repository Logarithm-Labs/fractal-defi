import pytest

from fractal.core.entities.uniswap_v3_spot import (UniswapV3SpotEntity,
                                                   UniswapV3SpotGlobalState,
                                                   UniswapV3SpotInternalState)


class TestUniswapV3SpotEntity:

    @pytest.mark.core
    def test_start_state(self):
        entity = UniswapV3SpotEntity()
        assert entity.global_state == UniswapV3SpotGlobalState()
        assert entity.internal_state == UniswapV3SpotInternalState()
        assert entity.balance == 0

    @pytest.mark.core
    def test_action_buy(self):
        entity = UniswapV3SpotEntity()
        entity.update_state(UniswapV3SpotGlobalState(price=2000))
        entity.action_deposit(amount_in_notional=1000)
        entity.action_buy(amount_in_notional=1000)
        assert entity.internal_state.amount == (1000 / 2000) * (1 - entity.TRADING_FEE)
        assert entity.internal_state.cash == 0
        assert entity.balance == 1000 * (1 - entity.TRADING_FEE)

    @pytest.mark.core
    def test_action_sell(self):
        entity = UniswapV3SpotEntity()
        entity.update_state(UniswapV3SpotGlobalState(price=4000))
        entity.action_deposit(amount_in_notional=2000)
        entity.action_buy(amount_in_notional=2000)
        entity.action_sell(amount_in_product=(2000 / 4000) * (1 - entity.TRADING_FEE))
        assert entity.internal_state.amount == 0
        assert entity.internal_state.cash == 2000 * (1 - entity.TRADING_FEE)**2
        assert entity.balance == 2000 * (1 - entity.TRADING_FEE)**2

    @pytest.mark.core
    def test_action_withdraw(self):
        entity = UniswapV3SpotEntity()
        entity.update_state(UniswapV3SpotGlobalState(price=2000))
        entity.action_deposit(amount_in_notional=1000)
        entity.action_withdraw(amount_in_notional=500)
        assert entity.internal_state.cash == 500
        assert entity.balance == 500

    @pytest.mark.core
    def test_action_deposit(self):
        entity = UniswapV3SpotEntity()
        entity.action_deposit(amount_in_notional=1000)
        assert entity.internal_state.cash == 1000
        assert entity.balance == 1000

    @pytest.mark.core
    def test_update_state(self):
        entity = UniswapV3SpotEntity()
        entity.update_state(UniswapV3SpotGlobalState(price=2000))
        assert entity.global_state == UniswapV3SpotGlobalState(price=2000)
