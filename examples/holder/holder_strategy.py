from typing import List
from dataclasses import dataclass

from fractal.core.base import (
    BaseStrategy, Action, BaseStrategyParams,
    ActionToTake, NamedEntity, Observation)
from fractal.core.entities import BaseSpotEntity
from fractal.loaders import BinanceDayPriceLoader, LoaderType

from binance_entity import BinanceSpot, BinanceGlobalState


@dataclass
class HolderStrategyParams(BaseStrategyParams):
    BUY_PRICE: float
    SELL_PRICE: float
    TRADE_SHARE: float = 0.01
    INITIAL_BALANCE: float = 10_000


class HodlerStrategy(BaseStrategy[HolderStrategyParams]):

    def __init__(self, debug: bool = False, params: HolderStrategyParams | None = None, *args, **kwargs):
        super().__init__(params=params, debug=debug, *args, **kwargs)

    def set_up(self):
        # check that the entity 'exchange' is registered
        assert 'exchange' in self.get_all_available_entities()
        # deposit initial balance into the exchange
        if self._params is not None:
            self.__deposit_into_exchange()

    def predict(self) -> List[ActionToTake]:
        exchange: BaseSpotEntity = self.get_entity('exchange')
        if exchange.global_state.price < self._params.BUY_PRICE:
            # Spend a fraction of available cash on the buy.
            notional_to_spend = self._params.TRADE_SHARE * exchange.internal_state.cash
            if notional_to_spend < 1e-6:
                return []
            return [ActionToTake(
                entity_name='exchange',
                action=Action(action='buy', args={'amount_in_notional': notional_to_spend})
            )]
        elif exchange.global_state.price > self._params.SELL_PRICE:
            # Sell a fraction of held product.
            product_to_sell = self._params.TRADE_SHARE * exchange.internal_state.amount
            if product_to_sell < 1e-6:
                return []
            return [ActionToTake(
                entity_name='exchange',
                action=Action(action='sell', args={'amount_in_product': product_to_sell})
            )]
        else:
            # HODL
            return []

    def __deposit_into_exchange(self):
        exchange: BaseSpotEntity = self.get_entity('exchange')
        action: Action = Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        exchange.execute(action)


class BinanceHodlerStrategy(HodlerStrategy):
    def set_up(self):
        self.register_entity(NamedEntity(entity_name='exchange', entity=BinanceSpot()))
        super().set_up()


if __name__ == '__main__':
    # Load prices from Binance
    binance_prices = BinanceDayPriceLoader('BTCUSDT', loader_type=LoaderType.CSV).read(with_run=True)

    # Build observations list
    observations: List[Observation] = [
        Observation(timestamp=timestamp, states={'exchange': BinanceGlobalState(price=price)})
        for timestamp, price in zip(binance_prices.index, binance_prices['price'])
    ]

    # Run the strategy
    params: HolderStrategyParams = HolderStrategyParams(
        BUY_PRICE=50_000, SELL_PRICE=60_000,
        TRADE_SHARE=0.01, INITIAL_BALANCE=100_000
    )
    strategy = BinanceHodlerStrategy(debug=True, params=params)
    result = strategy.run(observations)
    print(result.get_default_metrics())  # show metrics
    result.to_dataframe().to_csv('result.csv')  # save results of strategy states
