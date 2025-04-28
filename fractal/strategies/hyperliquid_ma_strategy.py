from typing import List
from dataclasses import dataclass
from collections import deque
from typing import Optional
import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.hyperliquid import HyperliquidPerpsKlinesLoader
from fractal.core.base import (
    BaseStrategy, Action, BaseStrategyParams,
    ActionToTake, NamedEntity, Observation)
from fractal.core.entities.single_spot_exchange import (
    SingleSpotExchangeGlobalState, 
    SingleSpotExchange
)
from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig
)

load_dotenv()

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
class MACrossoverStrategyState():
    short_ma: SMA
    long_ma: SMA
    prev_short_ma: float = None
    prev_long_ma: float = None
    
@dataclass
class MACrossoverStrategyParams(BaseStrategyParams):
    SHORT_MA_PERIOD: int
    LONG_MA_PERIOD: int
    INITIAL_BALANCE: float = 10_000
    TRADE_SHARE: float = 0.1


class MACrossoverStrategy(BaseStrategy):

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
        current_short_ma = self.ma_crossover_strategy_state.short_ma.update(exchange.global_state.close)
        current_long_ma = self.ma_crossover_strategy_state.long_ma.update(exchange.global_state.close)
        
        if current_short_ma is None or current_long_ma is None:
            return []
        
        prev_short_ma = self.ma_crossover_strategy_state.prev_short_ma
        prev_long_ma = self.ma_crossover_strategy_state.prev_long_ma
        
        self.ma_crossover_strategy_state.prev_short_ma = current_short_ma
        self.ma_crossover_strategy_state.prev_long_ma = current_long_ma
        
        if prev_short_ma is None or prev_long_ma is None:
            return []
        
        if prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma:
            amount_to_buy = self._params.TRADE_SHARE * exchange.internal_state.cash / exchange.global_state.close
            if amount_to_buy < 1e-6:
                return []
            return [ActionToTake(
                entity_name='exchange',
                action=Action(action='buy', args={'amount': amount_to_buy})
            )]
        elif prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma:
            amount_to_sell = self._params.TRADE_SHARE * exchange.internal_state.amount
            if amount_to_sell < 1e-6:
                return []
            return [ActionToTake(
                entity_name='exchange',
                action=Action(action='sell', args={'amount': amount_to_sell})
            )]
        else:
            return []

    def __deposit_into_exchange(self):
        exchange: SingleSpotExchange = self.get_entity('exchange')
        action: Action = Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        exchange.execute(action)


# Загрузка данных и создание списка наблюдений
def build_observations() -> List[Observation]:
    # Загрузка цен
    hyperliquid_prices = HyperliquidPerpsKlinesLoader(
        ticker='BTC',
        interval='1h',
        loader_type=LoaderType.CSV
    ).read(with_run=True)

    # Создание списка наблюдений
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


# Создание сетки параметров для перебора
def build_grid() -> ParameterGrid:
    grid = ParameterGrid({
        'SHORT_MA_PERIOD': [9, 12, 15],        # Короткие периоды MA
        'LONG_MA_PERIOD': [21, 30, 45],        # Длинные периоды MA
        'TRADE_SHARE': [0.1, 0.3, 0.5],        # Доля торговли
        'INITIAL_BALANCE': [100_000]           # Начальный баланс
    })
    return grid


if __name__ == '__main__':
    # Настройка MLflow
    mlflow_config = MLFlowConfig(
        mlflow_uri='https://mlflow.devcryptoservices.xyz/',
        experiment_name='hyperliquid_ma_strategy_btc_0',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    )
    
    # Настройка эксперимента
    experiment_config = ExperimentConfig(
        strategy_type=MACrossoverStrategy,
        backtest_observations=build_observations(),
        window_size=720,  # 30 дней при часовых данных
        params_grid=build_grid(),
        debug=True,
    )
    
    # Создание и запуск пайплайна
    pipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    
    # Запуск всех комбинаций параметров
    pipeline.run()
