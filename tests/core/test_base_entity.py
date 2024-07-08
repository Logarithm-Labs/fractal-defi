from hodler import Hodler, HodlerGlobalState

from fractal.core.base import Action


class TestHodler:

    def test_start_state(self, hodler: Hodler):
        assert hodler.global_state.price == 0
        assert hodler.balance == 0

    def test_get_available_actions(self, hodler: Hodler):
        actions = hodler.get_available_actions()
        assert len(actions) == 4
        assert any(action == 'buy' for action in actions)
        assert any(action == 'sell' for action in actions)

    def test_update_initial_state(self, hodler: Hodler):
        hodler.update_state(HodlerGlobalState(price=2000))
        assert hodler.global_state == HodlerGlobalState(price=2000)

    def test_action_buy(self, hodler: Hodler):
        hodler.execute(Action(action='buy', args={'amount': 1000}))
        assert hodler.internal_state.amount == 1000
        assert hodler.balance == 1000 * 2000

    def test_update_state(self, hodler: Hodler):
        hodler.update_state(HodlerGlobalState(price=4000))
        assert hodler.balance == 1000 * 4000

    def test_action_sell(self, hodler: Hodler):
        hodler.execute(Action(action='sell', args={'amount': 500}))
        assert hodler.internal_state.amount == 500
        assert hodler.balance == 500 * 4000
        # GOOD HODLER!
