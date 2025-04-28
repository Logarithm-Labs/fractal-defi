from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import numpy as np

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.hyperliquid import HyperliquidFundingRatesLoader, HyperLiquidPerpsPricesLoader
from fractal.loaders.structs import PriceHistory, RateHistory

from fractal.core.base import Observation, BaseStrategyParams, BaseStrategy, Action, ActionToTake
from fractal.core.entities import HyperliquidEntity, HyperLiquidGlobalState


@dataclass
class MomentumStrategyParams(BaseStrategyParams):
    """
    Parameters for the Momentum Strategy.
    
    INITIAL_BALANCE: Initial balance to deposit
    POSITION_SIZE: Percentage of balance to use for position (0-1)
    SHORT_WINDOW: Short-term moving average window
    LONG_WINDOW: Long-term moving average window
    TAKE_PROFIT: Take profit percentage (0-1)
    STOP_LOSS: Stop loss percentage (0-1)
    MAX_LEVERAGE: Maximum leverage to use
    """
    INITIAL_BALANCE: float
    POSITION_SIZE: float
    SHORT_WINDOW: int
    LONG_WINDOW: int
    TAKE_PROFIT: float
    STOP_LOSS: float
    MAX_LEVERAGE: float = 5.0


class MomentumStrategy(BaseStrategy):
    """
    A momentum strategy that uses moving average crossovers to make trading decisions.
    Trades on the Hyperliquid platform using the HyperliquidEntity.
    
    Strategy logic:
    1. When short MA crosses above long MA, go long
    2. When short MA crosses below long MA, go short
    3. Take profit or stop loss based on parameters
    """
    
    def __init__(self, *args, params: Optional[MomentumStrategyParams] = None, debug: bool = False, **kwargs):
        self._params: MomentumStrategyParams = None  # for type hinting
        self._prices_history = []
        self._position_open = False
        self._entry_price = 0.0
        self._position_side = 0  # 0 = no position, 1 = long, -1 = short
        super().__init__(params=params, debug=debug, *args, **kwargs)
    
    def set_up(self):
        """Set up the strategy by registering entities."""
        # Register HyperliquidEntity
        print(f"self.get_all_available_entities(): {self.get_all_available_entities()}")
        assert 'HYPERLIQ' in self.get_all_available_entities()
        hyperliq = self.get_entity('HYPERLIQ')
        hyperliq.MAX_LEVERAGE = self._params.MAX_LEVERAGE
        
    def predict(self) -> List[ActionToTake]:
        """
        Predict the next action based on moving average crossover strategy.
        
        Returns:
            List[ActionToTake]: Actions to take
        """
        hyperliq = self.get_entity('HYPERLIQ')
        current_price = hyperliq.global_state.mark_price
        
        # Store price history
        self._prices_history.append(current_price)
        
        # Wait until we have enough data for both moving averages
        if len(self._prices_history) < self._params.LONG_WINDOW:
            # If first observation, deposit initial balance
            if len(self._prices_history) == 1:
                self._debug(f"Depositing initial balance: {self._params.INITIAL_BALANCE}")
                return [
                    ActionToTake(
                        entity_name='HYPERLIQ',
                        action=Action('deposit', {'amount_in_notional': self._params.INITIAL_BALANCE})
                    )
                ]
            return []
        
        # Calculate moving averages
        short_ma = np.mean(self._prices_history[-self._params.SHORT_WINDOW:])
        long_ma = np.mean(self._prices_history[-self._params.LONG_WINDOW:])
        prev_short_ma = np.mean(self._prices_history[-self._params.SHORT_WINDOW-1:-1])
        prev_long_ma = np.mean(self._prices_history[-self._params.LONG_WINDOW-1:-1])
        
        # Check for crossovers
        actions = []
        
        # If we have an open position, check for take profit or stop loss
        if self._position_open:
            pnl_pct = (current_price - self._entry_price) / self._entry_price * self._position_side
            
            # Take profit condition
            if pnl_pct >= self._params.TAKE_PROFIT:
                self._debug(f"Taking profit at {current_price}, PnL: {pnl_pct:.2%}")
                self._position_open = False
                self._position_side = 0
                return [
                    ActionToTake(
                        entity_name='HYPERLIQ',
                        action=Action('open_position', {'amount_in_product': -hyperliq.size})
                    )
                ]
            
            # Stop loss condition
            if pnl_pct <= -self._params.STOP_LOSS:
                self._debug(f"Stopping loss at {current_price}, PnL: {pnl_pct:.2%}")
                self._position_open = False
                self._position_side = 0
                return [
                    ActionToTake(
                        entity_name='HYPERLIQ',
                        action=Action('open_position', {'amount_in_product': -hyperliq.size})
                    )
                ]
        
        # Check for crossover signals
        bullish_crossover = prev_short_ma <= prev_long_ma and short_ma > long_ma
        bearish_crossover = prev_short_ma >= prev_long_ma and short_ma < long_ma
        
        if bullish_crossover and self._position_side <= 0:
            # Close any existing short position
            if self._position_side < 0:
                actions.append(
                    ActionToTake(
                        entity_name='HYPERLIQ',
                        action=Action('open_position', {'amount_in_product': -hyperliq.size})
                    )
                )
            
            # Calculate position size based on account balance and leverage
            position_value = hyperliq.balance * self._params.POSITION_SIZE
            position_size = position_value / current_price
            
            self._debug(f"Bullish crossover at {current_price}, going long with size {position_size}")
            actions.append(
                ActionToTake(
                    entity_name='HYPERLIQ',
                    action=Action('open_position', {'amount_in_product': position_size})
                )
            )
            self._position_open = True
            self._entry_price = current_price
            self._position_side = 1
            
        elif bearish_crossover and self._position_side >= 0:
            # Close any existing long position
            if self._position_side > 0:
                actions.append(
                    ActionToTake(
                        entity_name='HYPERLIQ',
                        action=Action('open_position', {'amount_in_product': -hyperliq.size})
                    )
                )
            
            # Calculate position size based on account balance and leverage
            position_value = hyperliq.balance * self._params.POSITION_SIZE
            position_size = position_value / current_price
            
            self._debug(f"Bearish crossover at {current_price}, going short with size {-position_size}")
            actions.append(
                ActionToTake(
                    entity_name='HYPERLIQ',
                    action=Action('open_position', {'amount_in_product': -position_size})
                )
            )
            self._position_open = True
            self._entry_price = current_price
            self._position_side = -1
            
        return actions


def get_observations(
        rate_data: RateHistory, price_data: PriceHistory,
        start_time: datetime = None, end_time: datetime = None
    ) -> List[Observation]:
    """
    Get observations from price and funding rate data.
    
    Returns:
        List[Observation]: The observation list for MomentumStrategy.
    """
    observations_df: pd.DataFrame = price_data.join(rate_data)
    observations_df['rate'] = observations_df['rate'].fillna(0)
    observations_df = observations_df.loc[start_time:end_time]
    observations_df = observations_df.dropna()
    if start_time is None:
        start_time = observations_df.index.min()
    if end_time is None:
        end_time = observations_df.index.max()
    observations_df = observations_df.sort_index()
    
    return [
        Observation(
            timestamp=timestamp,
            states={
                'HYPERLIQ': HyperLiquidGlobalState(mark_price=price, funding_rate=rate)
            }
        ) for timestamp, (price, rate) in observations_df.iterrows()
    ]


def build_observations(
        ticker: str, start_time: datetime = None, end_time: datetime = None, fidelity: str = '1h',
    ) -> List[Observation]:
    """
    Build observations for the MomentumStrategy from the given time range.
    """
    rate_data: RateHistory = HyperliquidFundingRatesLoader(
        ticker, loader_type=LoaderType.CSV).read(with_run=True)
    
    # Get price data at specified fidelity
    prices: PriceHistory = HyperLiquidPerpsPricesLoader(
        ticker, interval=fidelity, loader_type=LoaderType.CSV,
        start_time=start_time, end_time=end_time).read(with_run=True)
    
    return get_observations(rate_data, prices, start_time, end_time)


if __name__ == '__main__':
    # Set up
    ticker: str = 'ETH'  # Ethereum
    start_time = datetime(2024, 1, 1)
    end_time = datetime(2024, 12, 31)
    fidelity = '1h'
    
    # Initialize strategy with parameters
    params: MomentumStrategyParams = MomentumStrategyParams(
        INITIAL_BALANCE=10000,  # $10,000 initial balance
        POSITION_SIZE=0.8,      # Use 80% of balance for position
        SHORT_WINDOW=12,        # 12-hour short moving average
        LONG_WINDOW=48,         # 48-hour long moving average
        TAKE_PROFIT=0.05,       # 5% take profit
        STOP_LOSS=0.03,         # 3% stop loss
        MAX_LEVERAGE=3.0        # 3x max leverage
    )
    
    # Create and configure strategy
    strategy = MomentumStrategy(debug=True, params=params)
    
    # Create and register HyperliquidEntity
    hl_entity = HyperliquidEntity(trading_fee=0.0005, max_leverage=params.MAX_LEVERAGE)
    strategy.register_entity(('HYPERLIQ', hl_entity))
    
    # Build observations
    observations: List[Observation] = build_observations(
        ticker=ticker,
        start_time=start_time,
        end_time=end_time,
        fidelity=fidelity
    )
    
    # Run the strategy
    result = strategy.run(observations)
    
    # Print results
    print(result.get_default_metrics())
    result.to_dataframe().to_csv(f'momentum_strategy_{ticker}.csv')
    print(result.to_dataframe().iloc[-1])  # Final state