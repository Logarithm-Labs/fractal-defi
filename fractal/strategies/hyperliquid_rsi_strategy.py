from typing import List
from dataclasses import dataclass
from collections import deque
from typing import Optional

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.hyperliquid import HyperliquidPerpsKlinesLoader
from fractal.loaders.structs import PriceHistory, RateHistory
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

class RSI:
    def __init__(self, period: int):
        self.period = period
        self.prev_price = None
        self.gains = deque()
        self.losses = deque()
        self.avg_gain = None
        self.avg_loss = None

    def update(self, price: float) -> Optional[float]:
        if self.prev_price is None:
            self.prev_price = price
            return None

        change = price - self.prev_price
        self.prev_price = price
        
        gain = max(0, change)
        loss = max(0, -change)
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        if len(self.gains) > self.period:
            self.gains.popleft()
            self.losses.popleft()
            
        if len(self.gains) < self.period:
            return None
            
        if self.avg_gain is None:
            self.avg_gain = sum(self.gains) / self.period
            self.avg_loss = sum(self.losses) / self.period
        else:
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
        
        if self.avg_loss == 0:
            return 100
            
        rs = self.avg_gain / self.avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
@dataclass
class RSIStrategyState:
    rsi: RSI
    sma: SMA
    prev_open: float = None
    prev_close: float = None
    in_position: bool = False
    entry_price: float = None
    
@dataclass
class RSIStrategyParams(BaseStrategyParams):
    RSI_PERIOD: int
    SMA_PERIOD: int
    RSI_OVERSOLD: float
    RSI_OVERBOUGHT: float
    INITIAL_BALANCE: float = 10_000
    TRADE_SHARE: float = 0.5
    TAKE_PROFIT_PCT: float = 0.07
    STOP_LOSS_PCT: float = 0.03

class RSIStrategy(BaseStrategy):

    def __init__(self, debug: bool = False, params: RSIStrategyParams | None = None):
        super().__init__(params=params, debug=debug)
        self.log_counter = 0
        self.trades = []

    def set_up(self):
        if 'exchange' not in self.get_all_available_entities():
            exchange = SingleSpotExchange()
            self.register_entity(NamedEntity(entity_name='exchange', entity=exchange))
            
        self.rsi_strategy_state = RSIStrategyState(
            rsi=RSI(self._params.RSI_PERIOD),
            sma=SMA(self._params.SMA_PERIOD)
        )
        if self._params is not None:
            self.__deposit_into_exchange()

    def predict(self) -> ActionToTake:
        exchange: SingleSpotExchange = self.get_entity('exchange')
        
        current_close = exchange.global_state.close
        current_open = exchange.global_state.open
        timestamp = exchange.global_state.timestamp if hasattr(exchange.global_state, 'timestamp') else self.log_counter
        
        self.log_counter += 1
        
        current_rsi = self.rsi_strategy_state.rsi.update(current_close)
        current_sma = self.rsi_strategy_state.sma.update(current_close)
        
        if current_rsi is None or current_sma is None:
            self.rsi_strategy_state.prev_close = current_close
            self.rsi_strategy_state.prev_open = current_open
            return []
        
        if not self.rsi_strategy_state.in_position:
            if (current_rsi < self._params.RSI_OVERSOLD and 
                (current_close > current_sma or current_close > current_open)):
                
                amount_to_buy = self._params.TRADE_SHARE * exchange.internal_state.cash / current_close
                if amount_to_buy < 1e-6:
                    return []
                    
                self.rsi_strategy_state.in_position = True
                self.rsi_strategy_state.entry_price = current_close
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': current_close,
                    'amount': amount_to_buy,
                    'value': amount_to_buy * current_close,
                    'rsi': current_rsi,
                    'sma': current_sma
                })
                
                return [ActionToTake(
                    entity_name='exchange',
                    action=Action(action='buy', args={'amount': amount_to_buy})
                )]
        else:
            amount_to_sell = exchange.internal_state.amount
            profit_pct = (current_close - self.rsi_strategy_state.entry_price) / self.rsi_strategy_state.entry_price * 100
            
            sell_signal = ""
            sell_reason = ""
            
            if current_close >= self.rsi_strategy_state.entry_price * (1 + self._params.TAKE_PROFIT_PCT):
                sell_signal = "TAKE PROFIT"
                sell_reason = f"Price {current_close:.2f} >= Entry {self.rsi_strategy_state.entry_price:.2f} + {self._params.TAKE_PROFIT_PCT*100}%"
            elif current_close <= self.rsi_strategy_state.entry_price * (1 - self._params.STOP_LOSS_PCT):
                sell_signal = "STOP LOSS"
                sell_reason = f"Price {current_close:.2f} <= Entry {self.rsi_strategy_state.entry_price:.2f} - {self._params.STOP_LOSS_PCT*100}%"
            elif current_rsi > self._params.RSI_OVERBOUGHT:
                sell_signal = "RSI OVERBOUGHT"
                sell_reason = f"RSI {current_rsi:.2f} > {self._params.RSI_OVERBOUGHT} (overbought)"
            elif current_close < current_sma:
                sell_signal = "PRICE BELOW SMA"
                sell_reason = f"Price {current_close:.2f} < SMA {current_sma:.2f}"
            
            if sell_signal and amount_to_sell >= 1e-6:
                self.rsi_strategy_state.in_position = False
                self.rsi_strategy_state.entry_price = None
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': f'SELL ({sell_signal})',
                    'price': current_close,
                    'amount': amount_to_sell,
                    'value': amount_to_sell * current_close,
                    'profit_pct': profit_pct,
                    'rsi': current_rsi,
                    'sma': current_sma
                })
                
                return [ActionToTake(
                    entity_name='exchange',
                    action=Action(action='sell', args={'amount': amount_to_sell})
                )]
        
        self.rsi_strategy_state.prev_close = current_close
        self.rsi_strategy_state.prev_open = current_open
        return []

    def __deposit_into_exchange(self):
        exchange: SingleSpotExchange = self.get_entity('exchange')
        action: Action = Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        exchange.execute(action)


if __name__ == '__main__':
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

    params: RSIStrategyParams = RSIStrategyParams(
        RSI_PERIOD=14,
        SMA_PERIOD=10,
        RSI_OVERSOLD=30,
        RSI_OVERBOUGHT=65,
        INITIAL_BALANCE=100_000,
        TRADE_SHARE=0.1,
        TAKE_PROFIT_PCT=0.07,
        STOP_LOSS_PCT=0.03
    )

    strategy = RSIStrategy(debug=True, params=params)
    
    result = strategy.run(observations)

    print(result.get_default_metrics())
    
    result.to_dataframe().to_csv('result.csv')
