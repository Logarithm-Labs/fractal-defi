from typing import List, Optional
from dataclasses import dataclass
from collections import deque

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.hyperliquid import HyperliquidPerpsKlinesLoader
from fractal.core.base import (
    BaseStrategy, Action, BaseStrategyParams,
    ActionToTake, NamedEntity, Observation)
from fractal.core.entities.single_spot_exchange import (
    SingleSpotExchangeGlobalState, 
    SingleSpotExchange
)

class SMA:
    def __init__(self, period: int):
        self.period = period
        self.window = deque()
        self.sum = 0.0

    def update(self, price: float) -> Optional[float]:
        self.window.append(price)
        self.sum += price

        if len(self.window) > self.period:
            self.sum -= self.window.popleft()

        if len(self.window) == self.period:
            return self.sum / self.period
        return None

@dataclass
class MACrossoverStrategyParams(BaseStrategyParams):
    SHORT_MA_PERIOD: int
    LONG_MA_PERIOD: int
    TRADE_SHARE: float
    INITIAL_BALANCE: float = 100_000

@dataclass
class MACrossoverStrategyState:
    short_ma: SMA
    long_ma: SMA
    prev_short_ma: float = None
    prev_long_ma: float = None

class MACrossoverStrategy(BaseStrategy):
    def __init__(self, debug: bool = False, params: MACrossoverStrategyParams | None = None):
        super().__init__(params=params, debug=debug)
        self.trades = []
        self.log_counter = 0

    def set_up(self):
        if 'exchange' not in self.get_all_available_entities():
            exchange = SingleSpotExchange()
            self.register_entity(NamedEntity(entity_name='exchange', entity=exchange))
        self.ma_crossover_strategy_state = MACrossoverStrategyState(
            short_ma=SMA(self._params.SHORT_MA_PERIOD),
            long_ma=SMA(self._params.LONG_MA_PERIOD)
        )
        if self._params is not None:
            self.__deposit_into_exchange()

    def predict(self) -> ActionToTake:
        exchange: SingleSpotExchange = self.get_entity('exchange')
        current_close = exchange.global_state.close
        timestamp = getattr(exchange.global_state, 'timestamp', self.log_counter)
        self.log_counter += 1

        current_short_ma = self.ma_crossover_strategy_state.short_ma.update(current_close)
        current_long_ma = self.ma_crossover_strategy_state.long_ma.update(current_close)

        if current_short_ma is None or current_long_ma is None:
            return []

        prev_short_ma = self.ma_crossover_strategy_state.prev_short_ma
        prev_long_ma = self.ma_crossover_strategy_state.prev_long_ma

        self.ma_crossover_strategy_state.prev_short_ma = current_short_ma
        self.ma_crossover_strategy_state.prev_long_ma = current_long_ma

        # Открытие позиции (покупка)
        if prev_short_ma is not None and prev_long_ma is not None:
            if prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma:
                amount_to_buy = self._params.TRADE_SHARE * exchange.internal_state.cash / current_close
                if amount_to_buy < 1e-6:
                    return []
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': current_close,
                    'amount': amount_to_buy,
                    'value': amount_to_buy * current_close,
                    'short_ma': current_short_ma,
                    'long_ma': current_long_ma
                })
                return [ActionToTake(
                    entity_name='exchange',
                    action=Action(action='buy', args={'amount': amount_to_buy})
                )]
            # Закрытие позиции (продажа)
            elif prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma:
                amount_to_sell = self._params.TRADE_SHARE * exchange.internal_state.amount
                if amount_to_sell < 1e-6:
                    return []
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': current_close,
                    'amount': amount_to_sell,
                    'value': amount_to_sell * current_close,
                    'short_ma': current_short_ma,
                    'long_ma': current_long_ma
                })
                return [ActionToTake(
                    entity_name='exchange',
                    action=Action(action='sell', args={'amount': amount_to_sell})
                )]
        return []

    def __deposit_into_exchange(self):
        exchange: SingleSpotExchange = self.get_entity('exchange')
        action: Action = Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        exchange.execute(action)

def build_observations() -> List[Observation]:
    hyperliquid_prices = HyperliquidPerpsKlinesLoader(
        ticker='BTC',
        interval='1h',
        loader_type=LoaderType.CSV
    ).read(with_run=True)

    observations: List[Observation] = [
        Observation(timestamp=timestamp, states={'exchange': SingleSpotExchangeGlobalState(
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price
        )})
        for timestamp, open_price, high_price, low_price, close_price in zip(
            hyperliquid_prices.index,
            hyperliquid_prices.open,
            hyperliquid_prices.high,
            hyperliquid_prices.low,
            hyperliquid_prices.close
        )
    ]
    return observations

if __name__ == '__main__':
    # Создание наблюдений
    observations = build_observations()

    # Создание параметров стратегии
    params = MACrossoverStrategyParams(
        SHORT_MA_PERIOD=9,
        LONG_MA_PERIOD=21,
        TRADE_SHARE=0.1,
        INITIAL_BALANCE=100_000
    )

    # Инициализация стратегии
    strategy = MACrossoverStrategy(debug=True, params=params)

    # Запуск стратегии
    result = strategy.run(observations)

    print(result.get_default_metrics())
    result.to_dataframe().to_csv('result_ma.csv')
