from datetime import datetime
from typing import List

from hodler import Hodler, HodlerGlobalState, HodlerParams, HodlerStrategy

from fractal.core.base import NamedEntity, Observation


class TestHodlerStrategy:

    def test_register_entity(self, hodler_strategy: HodlerStrategy):
        hodler_strategy.register_entity(NamedEntity(entity_name='stupid_hodler_2', entity=Hodler()))
        try:
            hodler_strategy.register_entity(NamedEntity(entity_name='stupid_hodler', entity=Hodler()))
            assert False
        except ValueError:
            assert True
        assert len(hodler_strategy._entities) == 2

    def test_estimate_predict(self, hodler_strategy: HodlerStrategy):
        actions = hodler_strategy.estimate_predict(
            Observation(timestamp=datetime(2020, 1, 2), states={
                'stupid_hodler': HodlerGlobalState(price=3001),
            }))
        assert len(actions) == 1
        assert actions[0].entity_name == 'stupid_hodler'
        assert actions[0].action.action == 'buy'
        assert actions[0].action.args == {'amount': 500}

    def test_step(self, hodler_strategy: HodlerStrategy):
        hodler_strategy.step(
            Observation(timestamp=datetime(2020, 1, 2), states={
                'stupid_hodler': HodlerGlobalState(price=3001),
            }))
        assert hodler_strategy.get_entity('stupid_hodler').internal_state.amount == 500
        assert hodler_strategy.get_entity('stupid_hodler').global_state.price == 3001
        assert hodler_strategy.get_entity('stupid_hodler').balance == 500 * 3001

    def test_full_pipeline(self):
        hodler_strategy = HodlerStrategy(debug=True, params=HodlerParams())
        observations: List[Observation] = [
            Observation(timestamp=datetime(2020, 1, 2), states={
                'stupid_hodler': HodlerGlobalState(price=3001),
            }),
            Observation(timestamp=datetime(2020, 1, 3), states={
                'stupid_hodler': HodlerGlobalState(price=3002),
            }),
            Observation(timestamp=datetime(2020, 1, 4), states={
                'stupid_hodler': HodlerGlobalState(price=3003),
            }),
        ]
        hodler = hodler_strategy.get_entity('stupid_hodler')
        hodler_strategy.run(observations)

        assert hodler.internal_state.amount == 1500
        assert hodler.global_state.price == 3003
        assert hodler.balance == 1500 * 3003
