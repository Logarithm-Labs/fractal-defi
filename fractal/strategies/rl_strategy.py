from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams
from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity
from fractal.core.base.observations import Observation
from fractal.core.base.strategy.result import StrategyResult
from copy import deepcopy
from fractal.rl_models.cppo import CVaRPPO
from fractal.rl_models.cpo import CPO


class UniswapV3Env(gym.Env):
    """
    Custom Gym environment for Uniswap V3 trading.
    """
    def __init__(
            self,
            uniswap_entity: UniswapV3LPEntity,
            initial_balance: float,
            observations: Optional[List[Observation]] = None,
            tick_spacing: int = 60,
            max_ticks: int = 10,  # Maximum number of ticks to deviate
            max_timesteps: int = 24 * 7,
    ):
        super().__init__()
        self.uniswap_entity = uniswap_entity
        self.initial_balance = initial_balance
        self.observations = observations
        self.current_observation_idx = 0
        self.tick_spacing = tick_spacing
        self.max_ticks = max_ticks
        
        # Define action space
        # First dimension: number of ticks below current price (0 to max_ticks)
        # Second dimension: number of ticks above current price (0 to max_ticks)
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        # self.action_space = gym.spaces.MultiDiscrete([max_ticks, max_ticks])
        self.action_space = gym.spaces.Discrete(max_ticks)
        
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(33,),
            # shape=(8,),
            dtype=np.float32
        )

        self.action_history = {}
        self.logger = configure_logger(
            verbose=0,
            tensorboard_log="./model_logs/",
            tb_log_name="run",
        )
        self.observation_history = []
        self.price_history = []  # Store price history for technical indicators
        self.high_price_history = []
        self.low_price_history = []
        self.close_price_history = []
        self.alpha = 0.05  # Smoothing factor for EWMA volatility
        self.max_timesteps = max_timesteps
        self.episode_number = 0


    def _calculate_ewma_volatility(self, prices: List[float]) -> float:
        """Calculate exponentially weighted moving average volatility."""
        if len(prices) < 2:
            return 0.0
        try:
            returns = np.diff(np.log(prices))
            ewma = np.zeros_like(returns)
            ewma[0] = returns[0] ** 2
            for i in range(1, len(returns)):
                ewma[i] = self.alpha * returns[i] ** 2 + (1 - self.alpha) * ewma[i-1]
            return np.sqrt(ewma[-1])
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_moving_averages(self, prices: List[float]) -> Tuple[float, float]:
        """Calculate 24 and 168 window moving averages."""
        if len(prices) < 24:
            return 0.0, 0.0
        try:
            ma24 = np.mean(prices[-24:])
            ma168 = np.mean(prices[-168:])
            return ma24, ma168
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0

    def _calculate_bollinger_bands(self, prices: List[float], window: int = 12) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < window:
            return 0.0, 0.0, 0.0
        try:
            ma = np.mean(prices[-window:])
            std = np.std(prices[-window:])
            upper_band = ma + 2 * std
            lower_band = ma - 2 * std
            return upper_band, ma, lower_band
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0, 0.0

    def _calculate_adxr(self, high_history: List[float], low_history: List[float], close_history: List[float], window: int = 12) -> float:
        """Calculate Average Directional Movement Index Rating."""
        if len(high_history) or len(low_history) or len(close_history) < window * 2:
            return 0.0
        try:
            high = np.array([p for p in high_history])
            low = np.array([p for p in low_history])
            close = np.array([p for p in close_history])
            
            # Calculate +DM and -DM
            plus_dm = np.zeros_like(high)
            minus_dm = np.zeros_like(high)
            for i in range(1, len(high)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            # Calculate True Range
            tr = np.zeros_like(high)
            for i in range(1, len(high)):
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            # Calculate smoothed values
            tr_mean = np.mean(tr[-window:])
            if tr_mean == 0:
                return 0.0
                
            plus_di = 100 * np.mean(plus_dm[-window:]) / tr_mean
            minus_di = 100 * np.mean(minus_dm[-window:]) / tr_mean
            
            # Calculate ADX
            if plus_di + minus_di == 0:
                return 0.0
                
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = np.mean(dx[-window:])
            
            return adx
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_bop(self, prices: List[float], global_state) -> float:
        """Calculate Balance of Power."""
        if len(prices) < 2:
            return 0.0
        try:
            high = global_state.high_price
            low = global_state.low_price
            close = global_state.close_price
            open_price = global_state.open_price
            
            if high - low == 0:
                return 0.0
                
            bop = (close - open_price) / (high - low)
            return bop
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_dx(self, high_history: List[float], low_history: List[float], close_history: List[float], window: int = 12) -> float:
        """Calculate Directional Movement Index."""
        if len(high_history) or len(low_history) or len(close_history) < window * 2:
            return 0.0
        try:
            high = np.array([p for p in high_history])
            low = np.array([p for p in low_history])
            close = np.array([p for p in close_history])
            
            # Calculate +DM and -DM
            plus_dm = np.zeros_like(high)
            minus_dm = np.zeros_like(high)
            for i in range(1, len(high)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            # Calculate True Range
            tr = np.zeros_like(high)
            for i in range(1, len(high)):
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            # Calculate smoothed values
            tr_mean = np.mean(tr[-window:])
            if tr_mean == 0:
                return 0.0
                
            plus_di = 100 * np.mean(plus_dm[-window:]) / tr_mean
            minus_di = 100 * np.mean(minus_dm[-window:]) / tr_mean
            
            # Calculate DX
            if plus_di + minus_di == 0:
                return 0.0
                
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            return dx
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_dema(self, prices: List[float], window: int = 12) -> float:
        """Calculate Double Exponential Moving Average."""
        if len(prices) < window:
            return 0.0
        try:
            prices_array = np.array(prices)
            ema1 = np.zeros_like(prices_array)
            ema2 = np.zeros_like(prices_array)
            alpha = 2 / (window + 1)
            
            # Calculate first EMA
            ema1[0] = prices_array[0]
            for i in range(1, len(prices_array)):
                ema1[i] = alpha * prices_array[i] + (1 - alpha) * ema1[i-1]
            
            # Calculate second EMA
            ema2[0] = ema1[0]
            for i in range(1, len(ema1)):
                ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
            
            # Calculate DEMA
            dema = 2 * ema1[-1] - ema2[-1]
            return dema
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_parabolic_sar(self, high_history: List[float], low_history: List[float], acceleration: float = 0.02, max_acceleration: float = 0.2) -> float:
        """Calculate Parabolic SAR."""
        if len(high_history) < 2:
            return 0.0
        try:
            high = np.array(high_history)
            low = np.array(low_history)
            
            # Initialize
            trend = 1  # 1 for uptrend, -1 for downtrend
            sar = low[0]
            extreme_point = high[0]
            current_acceleration = acceleration
            
            # Calculate SAR for the last point
            for i in range(1, len(high)):
                if trend == 1:
                    if low[i] < sar:
                        trend = -1
                        sar = extreme_point
                        extreme_point = low[i]
                        current_acceleration = acceleration
                    else:
                        if high[i] > extreme_point:
                            extreme_point = high[i]
                            current_acceleration = min(current_acceleration + acceleration, max_acceleration)
                        sar = sar + current_acceleration * (extreme_point - sar)
                else:
                    if high[i] > sar:
                        trend = 1
                        sar = extreme_point
                        extreme_point = high[i]
                        current_acceleration = acceleration
                    else:
                        if low[i] < extreme_point:
                            extreme_point = low[i]
                            current_acceleration = min(current_acceleration + acceleration, max_acceleration)
                        sar = sar + current_acceleration * (extreme_point - sar)
            
            return sar
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_apo(self, prices: List[float], fast_period: int = 12, slow_period: int = 26) -> float:
        """Calculate Absolute Price Oscillator."""
        if len(prices) < max(fast_period, slow_period):
            return 0.0
        try:
            prices_array = np.array(prices)
            fast_alpha = 2 / (fast_period + 1)
            slow_alpha = 2 / (slow_period + 1)
            
            # Calculate EMAs
            fast_ema = np.zeros_like(prices_array)
            slow_ema = np.zeros_like(prices_array)
            
            fast_ema[0] = prices_array[0]
            slow_ema[0] = prices_array[0]
            
            for i in range(1, len(prices_array)):
                fast_ema[i] = fast_alpha * prices_array[i] + (1 - fast_alpha) * fast_ema[i-1]
                slow_ema[i] = slow_alpha * prices_array[i] + (1 - slow_alpha) * slow_ema[i-1]
            
            return fast_ema[-1] - slow_ema[-1]
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_aroon_oscillator(self, high_history: List[float], low_history: List[float], period: int = 14) -> float:
        """Calculate Aroon Oscillator."""
        if len(high_history) < period or len(low_history) < period:
            return 0.0
        try:
            # Get last 'period' values
            high = np.array(high_history[-period:])
            low = np.array(low_history[-period:])
            
            # Find the number of periods since the highest high and lowest low
            high_days = period - np.argmax(high) - 1
            low_days = period - np.argmin(low) - 1
            
            # Calculate Aroon Up and Down
            aroon_up = (period - high_days) / period * 100
            aroon_down = (period - low_days) / period * 100
            
            # Calculate Aroon Oscillator
            return aroon_up - aroon_down
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_cci(self, high_history: List[float], low_history: List[float], close_history: List[float], period: int = 20) -> float:
        """Calculate Commodity Channel Index."""
        if len(high_history) < period or len(low_history) < period or len(close_history) < period:
            return 0.0
        try:
            # Get last 'period' values
            high = np.array(high_history[-period:])
            low = np.array(low_history[-period:])
            close = np.array(close_history[-period:])
            
            # Calculate typical price
            tp = (high + low + close) / 3
            
            # Calculate SMA of typical price
            sma_tp = np.mean(tp)
            
            # Calculate Mean Deviation
            mean_deviation = np.mean(np.abs(tp - sma_tp))
            
            if mean_deviation == 0:
                return 0.0
            
            # Calculate CCI
            cci = (tp[-1] - sma_tp) / (0.015 * mean_deviation)
            
            return cci
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_cmo(self, prices: List[float], period: int = 14) -> float:
        """Calculate Chande Momentum Oscillator."""
        if len(prices) < period + 1:
            return 0.0
        try:
            # Calculate price changes
            changes = np.diff(prices[-period-1:])
            
            # Separate positive and negative changes
            positive_sum = np.sum([change for change in changes if change > 0])
            negative_sum = abs(np.sum([change for change in changes if change < 0]))
            
            if positive_sum + negative_sum == 0:
                return 0.0
            
            # Calculate CMO
            cmo = 100 * (positive_sum - negative_sum) / (positive_sum + negative_sum)
            
            return cmo
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """Calculate Momentum."""
        if len(prices) <= period:
            return 0.0
        try:
            return prices[-1] - prices[-period-1]
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_trix(self, prices: List[float], period: int = 15) -> float:
        """Calculate TRIX indicator."""
        if len(prices) < period * 3:
            return 0.0
        try:
            prices_array = np.array(prices)
            alpha = 2 / (period + 1)
            
            # Calculate triple EMA
            ema1 = np.zeros_like(prices_array)
            ema2 = np.zeros_like(prices_array)
            ema3 = np.zeros_like(prices_array)
            
            # First EMA
            ema1[0] = prices_array[0]
            for i in range(1, len(prices_array)):
                ema1[i] = alpha * prices_array[i] + (1 - alpha) * ema1[i-1]
            
            # Second EMA
            ema2[0] = ema1[0]
            for i in range(1, len(ema1)):
                ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
            
            # Third EMA
            ema3[0] = ema2[0]
            for i in range(1, len(ema2)):
                ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
            
            # Calculate TRIX
            if ema3[-2] == 0:
                return 0.0
            trix = (ema3[-1] - ema3[-2]) / ema3[-2] * 100
            
            return trix
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_ultimate_oscillator(self, high_history: List[float], low_history: List[float], close_history: List[float],
                                    period1: int = 7, period2: int = 14, period3: int = 28) -> float:
        """Calculate Ultimate Oscillator."""
        if len(high_history) < max(period1, period2, period3):
            return 0.0
        try:
            high = np.array(high_history)
            low = np.array(low_history)
            close = np.array(close_history)
            
            # Calculate buying pressure (BP) and true range (TR)
            bp = close - np.minimum(low, np.roll(close, 1))
            tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
            
            # Calculate averages for different periods
            avg1 = np.sum(bp[-period1:]) / np.sum(tr[-period1:])
            avg2 = np.sum(bp[-period2:]) / np.sum(tr[-period2:])
            avg3 = np.sum(bp[-period3:]) / np.sum(tr[-period3:])
            
            # Calculate Ultimate Oscillator
            uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
            
            return uo
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_stochastic_momentum(self, high_history: List[float], low_history: List[float], close_history: List[float],
                                     k_period: int = 14, d_period: int = 3) -> Tuple[float, float, float]:
        """Calculate Stochastic Momentum Indicator."""
        if len(high_history) < k_period or len(low_history) < k_period or len(close_history) < k_period:
            return 0.0, 0.0, 0.0
        try:
            # Get relevant price ranges
            high = np.array(high_history[-k_period:])
            low = np.array(low_history[-k_period:])
            close = np.array(close_history[-k_period:])
            
            # Calculate %K
            lowest_low = np.min(low)
            highest_high = np.max(high)
            
            if highest_high - lowest_low == 0:
                return 0.0, 0.0, 0.0
            
            k = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
            
            # Calculate %D (3-period SMA of %K)
            if len(close_history) < k_period + d_period:
                return k, 0.0, 0.0
            
            d = np.mean([k for _ in range(d_period)])
            
            # Calculate Slow Stochastic
            slow_d = np.mean([d for _ in range(d_period)])
            
            return k, d, slow_d
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0, 0.0

    def _calculate_natr(self, high_history: List[float], low_history: List[float], close_history: List[float], period: int = 14) -> float:
        """Calculate Normalized Average True Range."""
        if len(high_history) < period or len(low_history) < period or len(close_history) < period:
            return 0.0
        try:
            high = np.array(high_history)
            low = np.array(low_history)
            close = np.array(close_history)
            
            # Calculate True Range
            tr = np.zeros(len(high))
            for i in range(1, len(high)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
            
            # Calculate ATR
            atr = np.mean(tr[-period:])
            
            # Normalize ATR
            if close[-1] == 0:
                return 0.0
            
            natr = 100 * atr / close[-1]
            
            return natr
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_true_range(self, high_history: List[float], low_history: List[float], close_history: List[float]) -> float:
        """Calculate True Range."""
        if len(high_history) < 2 or len(low_history) < 2 or len(close_history) < 2:
            return 0.0
        try:
            high = high_history[-1]
            low = low_history[-1]
            prev_close = close_history[-2]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            
            return tr
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_hilbert_transform(self, prices: List[float], period: int = 14) -> Tuple[float, float]:
        """Calculate Hilbert Transform - Dominant Cycle Period and Phase."""
        if len(prices) < period * 2:
            return 0.0, 0.0
        try:
            prices_array = np.array(prices[-period*2:])
            
            # Smooth prices
            alpha = 0.0962
            smooth = np.zeros_like(prices_array)
            smooth[0] = prices_array[0]
            for i in range(1, len(prices_array)):
                smooth[i] = (1 - alpha) * smooth[i-1] + alpha * prices_array[i]
            
            # Calculate quadrature
            quad = np.zeros_like(smooth)
            for i in range(period, len(smooth)):
                quad[i] = 1.25 * (smooth[i-7] - smooth[i-5]) - 0.25 * (smooth[i-3] - smooth[i-1])
            
            # Calculate dominant cycle period using autocorrelation
            corr = np.correlate(smooth, smooth, mode='full')
            corr = corr[len(corr)//2:]
            peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
            if len(peaks) > 0:
                dominant_cycle = peaks[0]
            else:
                dominant_cycle = period
            
            # Calculate dominant cycle phase
            if np.sum(quad[-period:] ** 2) == 0:
                phase = 0.0
            else:
                phase = np.arctan2(
                    np.sum(quad[-period:] * smooth[-period:]),
                    np.sum(quad[-period:] ** 2)
                ) * 180 / np.pi
            
            return float(dominant_cycle), float(phase)
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0

    def _get_observation(self):
        """Get current market state as observation."""
        global_state = self.uniswap_entity._global_state
        internal_state = self.uniswap_entity._internal_state

        # Update price history
        self.price_history.append(global_state.price)
        self.high_price_history.append(global_state.high_price)
        self.low_price_history.append(global_state.low_price)
        self.close_price_history.append(global_state.close_price)
        if len(self.price_history) > 168:  # Keep only last 168 prices
            self.price_history = self.price_history[-168:]
            self.high_price_history = self.high_price_history[-168:]
            self.low_price_history = self.low_price_history[-168:]
            self.close_price_history = self.close_price_history[-168:]

        # Calculate price ratios
        open_price = global_state.open_price
        if open_price == 0:
            high_to_open = 1.0
            low_to_open = 1.0
            close_to_open = 1.0
        else:
            high_to_open = global_state.high_price / open_price
            low_to_open = global_state.low_price / open_price
            close_to_open = global_state.close_price / open_price

        # Calculate technical indicators
        dema = self._calculate_dema(self.price_history)
        dema_ratio = dema / open_price if open_price != 0 else 1.0
        
        psar = self._calculate_parabolic_sar(self.high_price_history, self.low_price_history)
        psar_ratio = psar / open_price if open_price != 0 else 1.0
        
        apo = self._calculate_apo(self.price_history)
        aroon = self._calculate_aroon_oscillator(self.high_price_history, self.low_price_history)
        bop = self._calculate_bop(self.price_history, global_state)
        cci = self._calculate_cci(self.high_price_history, self.low_price_history, self.close_price_history)
        cmo = self._calculate_cmo(self.price_history)
        dx = self._calculate_dx(self.high_price_history, self.low_price_history, self.close_price_history)
        momentum = self._calculate_momentum(self.price_history)
        trix = self._calculate_trix(self.price_history)
        ultimate = self._calculate_ultimate_oscillator(self.high_price_history, self.low_price_history, self.close_price_history)

        ewma_volatility = self._calculate_ewma_volatility(self.price_history)
        ma24, ma168 = self._calculate_moving_averages(self.price_history)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(self.price_history)
        adxr = self._calculate_adxr(self.high_price_history, self.low_price_history, self.close_price_history)
        
        # Calculate Stochastic indicators
        k, d, slow_d = self._calculate_stochastic_momentum(
            self.high_price_history, 
            self.low_price_history, 
            self.close_price_history
        )
        
        # Calculate ranges
        natr = self._calculate_natr(self.high_price_history, self.low_price_history, self.close_price_history)
        true_range = self._calculate_true_range(self.high_price_history, self.low_price_history, self.close_price_history)
        
        # Calculate Hilbert Transform indicators
        dominant_cycle, dominant_phase = self._calculate_hilbert_transform(self.price_history)

        impermanent_loss = 0.0
        if self.uniswap_entity.is_position:
            impermanent_loss = abs(
                (
                    self.uniswap_entity._internal_state.token0_amount_position_init + 
                    self.uniswap_entity._internal_state.token1_amount_position_init * self.uniswap_entity._global_state.price
                ) - (
                    self.uniswap_entity._internal_state.token0_amount_position + 
                    self.uniswap_entity._internal_state.token1_amount_position * self.uniswap_entity._global_state.price
                )
            )

        current_observation = np.array([
            self.uniswap_entity.balance,  # balance
            # global_state.price,  # hourly price
            # global_state.centralized_price,  # centralized price
            # internal_state.price_init,  # initial price
            # internal_state.price_lower,  # lower price
            # internal_state.price_upper,  # upper price
            self.uniswap_entity.is_position,
            # global_state.open_price,  # hourly open price
            global_state.price,
            global_state.centralized_price,
            high_to_open,  # hourly highest price/hourly open price
            low_to_open,  # hourly lowest price/hourly open price
            close_to_open,  # hourly close price/hourly open price
            global_state.volume,  # hourly trading volume in USD
            dema_ratio,  # Double Exponential Moving Average/hourly open price
            psar_ratio,  # Parabolic SAR/hourly open price
            dx,  # Average Directional Movement Index
            apo,  # Absolute Price Oscillator
            aroon,  # Aroon Oscillator
            bop,  # Balance Of Power
            cci,  # Commodity Channel Index (2 features)
            cmo,  # Chande Momentum Oscillator
            momentum,  # Momentum
            trix,  # TRIX
            ultimate,  # Ultimate Oscillator
            k,  # Stochastic Momentum Indicator (3 features)
            d,
            slow_d,
            natr,  # Normalized Average True Range
            true_range,  # True Range
            dominant_cycle,  # Hilbert Transform - Dominant Cycle Period
            dominant_phase,  # Hilbert Transform - Dominant Cycle Phase
            ewma_volatility,  # Exponential Weighted Moving Average Volatility
            ma24, ma168,  # Moving Averages
            bb_upper, # Bollinger Bands
            bb_middle, 
            bb_lower,  
            adxr,  # Average Directional Index            
            # impermanent_loss,
        ], dtype=np.float32)
        # Handle NaN and infinite values
        current_observation = np.nan_to_num(current_observation, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

        return current_observation

    def _calculate_reward(self, rebalance, fees_earned_prev, done):
        """
        Calculate reward based on a combination of impermanent loss and earned fees.
        The reward is calculated as: fees_earned - absolute_impermanent_loss - rebalancing_penalty
        """
        # Calculate impermanent loss
        # current_price = self.uniswap_entity._global_state.price
        # initial_price = self.uniswap_entity._internal_state.price_init
        
        # # If we haven't opened a position yet, return 0
        # if initial_price == 0.0:
        #     return 0.0
            
        # Calculate price change ratio
        # price_ratio = current_price / initial_price
        
        # Calculate impermanent loss percentage
        # IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        # impermanent_loss_percentage = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        
        # # Convert to absolute value
        # impermanent_loss = abs(impermanent_loss_percentage) * self.uniswap_entity.balance

        impermanent_loss = 0.0
        rebalancing_penalty = 0.0
        earned_fees = 0.0
        instantaneous_lvr = 0.0
        holding_penalty = 0.0
        if self.uniswap_entity.is_position:
            # earned_fees = max(0, self.uniswap_entity._internal_state.earned_fees - fees_earned_prev)
            earned_fees = self.uniswap_entity._internal_state.earned_fees
            impermanent_loss = abs(
                (
                    self.uniswap_entity._internal_state.token0_amount_position_init + 
                    self.uniswap_entity._internal_state.token1_amount_position_init * self.uniswap_entity._global_state.price
                ) - (
                    self.uniswap_entity._internal_state.token0_amount_position + 
                    self.uniswap_entity._internal_state.token1_amount_position * self.uniswap_entity._global_state.price
                )
            )

            ewma_volatility = self._calculate_ewma_volatility(self.price_history)   
            liquidity = self.uniswap_entity._internal_state.liquidity
            sqrt_price = self.uniswap_entity._global_state.price ** 0.5
            instantaneous_lvr = liquidity * ewma_volatility**2 * sqrt_price / 4
            # instantaneous_lvr = liquidity / (2 * self.uniswap_entity._global_state.price ** 1.5)

            if rebalance or done:
                rebalancing_penalty = self.initial_balance - self.uniswap_entity.balance
                rebalancing_penalty = (
                    self.uniswap_entity._internal_state.token1_amount_position * 
                    self.uniswap_entity._global_state.price * 
                    self.uniswap_entity.trading_fee
                )
                rebalancing_penalty = 10
        else:
            holding_penalty = 0.1


        self.logger.record_mean(key="train/impermanent_loss", value=impermanent_loss)
        self.logger.record_mean(key="train/rebalancing_penalty", value=rebalancing_penalty)
        self.logger.record_mean(key="train/earned_fees", value=earned_fees)
        self.logger.record_mean(key="train/instantaneous_lvr", value=instantaneous_lvr)
        self.logger.record(key="train/earned_fees_abs", value=earned_fees)

        # instantaneous_lvr = 0.0
        # impermanent_loss = 0.0
        # rebalancing_penalty = 0.0
        # Calculate total reward
        # reward = earned_fees - instantaneous_lvr - rebalancing_penalty - impermanent_loss
        # reward = earned_fees / self.initial_balance - rebalancing_penalty / self.initial_balance + self.uniswap_entity.balance / self.initial_balance
        # reward = self.uniswap_entity.balance / self.initial_balance * 100
        reward = earned_fees - instantaneous_lvr

        self.logger.record_mean(key="train/balance_ratio_mean", value=self.uniswap_entity.balance / self.initial_balance / self.max_timesteps)
        self.logger.record(key="train/balance_ratio", value=self.uniswap_entity.balance / self.initial_balance)
        # reward = self.uniswap_entity.balance - self.initial_balance
        
        return reward

    def _get_price_from_tick(self, tick: int) -> float:
        """Convert tick to price using Uniswap V3 formula."""
        return 1.0001 ** tick

    def _get_tick_from_price(self, price: float) -> int:
        """Convert price to tick using Uniswap V3 formula."""
        return int(np.log(price) / np.log(1.0001))

    def step(self, action):
        """Execute one step in the environment."""
        fees_earned_prev = self.uniswap_entity._internal_state.earned_fees

        # Get current price and tick
        current_price = self.uniswap_entity._global_state.price
        current_tick = self._get_tick_from_price(current_price)

        # action = list(map(round, (action + 1) * self.max_ticks / 2))
        # assert isinstance(action[0], int) and isinstance(action[1], int)
        
        # Calculate new tick bounds
        lower_tick = current_tick - action * self.tick_spacing
        upper_tick = current_tick + action * self.tick_spacing

        
        # Convert ticks to prices
        price_lower = self._get_price_from_tick(lower_tick)
        price_upper = self._get_price_from_tick(upper_tick)

        rebalance = True

        if action == 0:
            rebalance = False

        if rebalance:
            if self.uniswap_entity.is_position:
                self.uniswap_entity.action_close_position()
            
            # Open new position with calculated range
            self.uniswap_entity.action_open_position(
                amount_in_notional=self.uniswap_entity._internal_state.cash,
                price_lower=price_lower,
                price_upper=price_upper
            )
        else:
            lower_bound = self.uniswap_entity._internal_state.price_lower
            upper_bound = self.uniswap_entity._internal_state.price_upper
            if current_price < lower_bound or current_price > upper_bound:
                if self.uniswap_entity.is_position:
                    self.uniswap_entity.action_close_position()

        # Get current observation and update entity state
        current_observation = self.observations[self.current_observation_idx]
        for entity_name, state in current_observation.states.items():
            self.uniswap_entity.update_state(state)

        # Get new state
        observation = self._get_observation()

        # Move to next observation
        self.current_observation_idx += 1
        truncated = self.current_observation_idx - self.start_observation_idx >= self.max_timesteps

        # Calculate reward before taking action
        reward = self._calculate_reward(rebalance, fees_earned_prev, truncated)
        
        return observation, reward, False, truncated, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # if self.episode_number % 100 == 0:
        #     self.max_timesteps *= 2 
        #     self.max_timesteps = min(self.max_timesteps, len(self.observations) - 1)
        #     print(f"Episode {self.episode_number} - Max timesteps: {self.max_timesteps}")

        self.episode_number += 1
        self.start_observation_idx = np.random.randint(0, len(self.observations) - self.max_timesteps - 1)
        self.current_observation_idx = self.start_observation_idx

        self.initial_balance = np.random.uniform(9 * 10**5, 12 * 10**5)
        # self.initial_balance = 10_000

        if self.uniswap_entity.is_position:
            self.uniswap_entity.action_close_position()
        
        self.uniswap_entity._initialize_states()
        
        current_observation = self.observations[self.current_observation_idx]
        for entity_name, state in current_observation.states.items():
            self.uniswap_entity.update_state(state)
        
        self.uniswap_entity.action_deposit(self.initial_balance)
        self.current_observation_idx += 1
        
        return self._get_observation(), {}



@dataclass
class RLStrategyParams(BaseStrategyParams):
    """
    Parameters for the Reinforcement Learning strategy:
    - INITIAL_BALANCE: The initial balance for liquidity allocation
    - LEARNING_RATE: The learning rate for the RL model
    - N_STEPS: Number of steps to run for each environment per update
    - BATCH_SIZE: Minibatch size
    - N_EPOCHS: Number of epochs when optimizing the surrogate loss
    - GAMMA: Discount factor
    - GAE_LAMBDA: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    - CLIP_RANGE: Clipping parameter for PPO
    """
    INITIAL_BALANCE: float
    LEARNING_RATE: float = 0.0003
    N_STEPS: int = 2048
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    SEED: int = 42
    FEES_RATE: float = 5.0
    ALPHA: float = 0.9
    BETA: float = 2800.0
    NU_LR: float = 1e-2
    LAM_LR: float = 1e-2
    NU_START: float = 0.0
    LAM_START: float = 0.5
    NU_DELAY: float = 0.8
    LAM_LOW_BOUND: float = 0.001
    DELAY: float = 1.0
    CVAR_CLIP_RATIO: float = 0.05
    MAX_CONSTRAINT_VALUE: float = 1000
    MAX_BACKTRACK_STEPS: int = 10
    BACKTRACK_COEFF: float = 0.8
    DAMPING_COEFF: float = 0.1
    CONSTRAINT_LR: float = 1e-2
    MAX_IMPERMANENT_LOSS: float = 500
    IMPERMANENT_LOSS_QUANTILE: float = 0.8
    


class RLStrategy(BaseStrategy):
    """
    A reinforcement learning based strategy for managing liquidity in Uniswap v3.
    The strategy uses PPO (Proximal Policy Optimization) from Stable Baselines3.
    """

    # Decimals for token0 and token1 for Uniswap V3 LP Config
    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1

    def __init__(self, params: RLStrategyParams, debug: bool = False, *args, **kwargs):
        self._params: RLStrategyParams = None  # set for type hinting
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial_funds = False
        self.env = None
        self.model = None
        self.price_history = []  # Store price history for technical indicators
        self.high_price_history = []
        self.low_price_history = []
        self.close_price_history = []
        self.action_history = []
        self.alpha = 0.05  # Smoothing factor for EWMA volatility
        self.max_ticks = 10


    def set_up(self):
        """
        Register the Uniswap V3 LP entity and initialize the RL environment and model.
        """
        self.register_entity(NamedEntity(
            entity_name='UNISWAP_V3',
            entity=UniswapV3LPEntity(
                UniswapV3LPConfig(
                    token0_decimals=self.token0_decimals,
                    token1_decimals=self.token1_decimals,
                    fees_rate=self._params.FEES_RATE,
                )
            )
        ))
        assert isinstance(self.get_entity('UNISWAP_V3'), UniswapV3LPEntity)

    def _calculate_ewma_volatility(self, prices: List[float]) -> float:
        """Calculate exponentially weighted moving average volatility."""
        if len(prices) < 2:
            return 0.0
        try:
            returns = np.diff(np.log(prices))
            ewma = np.zeros_like(returns)
            ewma[0] = returns[0] ** 2
            for i in range(1, len(returns)):
                ewma[i] = self.alpha * returns[i] ** 2 + (1 - self.alpha) * ewma[i-1]
            return np.sqrt(ewma[-1])
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_moving_averages(self, prices: List[float]) -> Tuple[float, float]:
        """Calculate 24 and 168 window moving averages."""
        if len(prices) < 24:
            return 0.0, 0.0
        try:
            ma24 = np.mean(prices[-24:])
            ma168 = np.mean(prices[-168:])
            return ma24, ma168
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0

    def _calculate_bollinger_bands(self, prices: List[float], window: int = 12) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < window:
            return 0.0, 0.0, 0.0
        try:
            ma = np.mean(prices[-window:])
            std = np.std(prices[-window:])
            upper_band = ma + 2 * std
            lower_band = ma - 2 * std
            return upper_band, ma, lower_band
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0, 0.0

    def _calculate_adxr(self, high_history: List[float], low_history: List[float], close_history: List[float], window: int = 12) -> float:
        """Calculate Average Directional Movement Index Rating."""
        if len(high_history) or len(low_history) or len(close_history) < window * 2:
            return 0.0
        try:
            high = np.array([p for p in high_history])
            low = np.array([p for p in low_history])
            close = np.array([p for p in close_history])
            
            # Calculate +DM and -DM
            plus_dm = np.zeros_like(high)
            minus_dm = np.zeros_like(high)
            for i in range(1, len(high)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            # Calculate True Range
            tr = np.zeros_like(high)
            for i in range(1, len(high)):
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            # Calculate smoothed values
            tr_mean = np.mean(tr[-window:])
            if tr_mean == 0:
                return 0.0
                
            plus_di = 100 * np.mean(plus_dm[-window:]) / tr_mean
            minus_di = 100 * np.mean(minus_dm[-window:]) / tr_mean
            
            # Calculate ADX
            if plus_di + minus_di == 0:
                return 0.0
                
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = np.mean(dx[-window:])
            
            return adx
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_bop(self, prices: List[float], global_state) -> float:
        """Calculate Balance of Power."""
        if len(prices) < 2:
            return 0.0
        try:
            high = global_state.high_price
            low = global_state.low_price
            close = global_state.close_price
            open_price = global_state.open_price
            
            if high - low == 0:
                return 0.0
                
            bop = (close - open_price) / (high - low)
            return bop
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_dx(self, high_history: List[float], low_history: List[float], close_history: List[float], window: int = 12) -> float:
        """Calculate Directional Movement Index."""
        if len(high_history) or len(low_history) or len(close_history) < window * 2:
            return 0.0
        try:
            high = np.array([p for p in high_history])
            low = np.array([p for p in low_history])
            close = np.array([p for p in close_history])
            
            # Calculate +DM and -DM
            plus_dm = np.zeros_like(high)
            minus_dm = np.zeros_like(high)
            for i in range(1, len(high)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
            
            # Calculate True Range
            tr = np.zeros_like(high)
            for i in range(1, len(high)):
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            # Calculate smoothed values
            tr_mean = np.mean(tr[-window:])
            if tr_mean == 0:
                return 0.0
                
            plus_di = 100 * np.mean(plus_dm[-window:]) / tr_mean
            minus_di = 100 * np.mean(minus_dm[-window:]) / tr_mean
            
            # Calculate DX
            if plus_di + minus_di == 0:
                return 0.0
                
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            return dx
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_dema(self, prices: List[float], window: int = 12) -> float:
        """Calculate Double Exponential Moving Average."""
        if len(prices) < window:
            return 0.0
        try:
            prices_array = np.array(prices)
            ema1 = np.zeros_like(prices_array)
            ema2 = np.zeros_like(prices_array)
            alpha = 2 / (window + 1)
            
            # Calculate first EMA
            ema1[0] = prices_array[0]
            for i in range(1, len(prices_array)):
                ema1[i] = alpha * prices_array[i] + (1 - alpha) * ema1[i-1]
            
            # Calculate second EMA
            ema2[0] = ema1[0]
            for i in range(1, len(ema1)):
                ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
            
            # Calculate DEMA
            dema = 2 * ema1[-1] - ema2[-1]
            return dema
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_parabolic_sar(self, high_history: List[float], low_history: List[float], acceleration: float = 0.02, max_acceleration: float = 0.2) -> float:
        """Calculate Parabolic SAR."""
        if len(high_history) < 2:
            return 0.0
        try:
            high = np.array(high_history)
            low = np.array(low_history)
            
            # Initialize
            trend = 1  # 1 for uptrend, -1 for downtrend
            sar = low[0]
            extreme_point = high[0]
            current_acceleration = acceleration
            
            # Calculate SAR for the last point
            for i in range(1, len(high)):
                if trend == 1:
                    if low[i] < sar:
                        trend = -1
                        sar = extreme_point
                        extreme_point = low[i]
                        current_acceleration = acceleration
                    else:
                        if high[i] > extreme_point:
                            extreme_point = high[i]
                            current_acceleration = min(current_acceleration + acceleration, max_acceleration)
                        sar = sar + current_acceleration * (extreme_point - sar)
                else:
                    if high[i] > sar:
                        trend = 1
                        sar = extreme_point
                        extreme_point = high[i]
                        current_acceleration = acceleration
                    else:
                        if low[i] < extreme_point:
                            extreme_point = low[i]
                            current_acceleration = min(current_acceleration + acceleration, max_acceleration)
                        sar = sar + current_acceleration * (extreme_point - sar)
            
            return sar
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_apo(self, prices: List[float], fast_period: int = 12, slow_period: int = 26) -> float:
        """Calculate Absolute Price Oscillator."""
        if len(prices) < max(fast_period, slow_period):
            return 0.0
        try:
            prices_array = np.array(prices)
            fast_alpha = 2 / (fast_period + 1)
            slow_alpha = 2 / (slow_period + 1)
            
            # Calculate EMAs
            fast_ema = np.zeros_like(prices_array)
            slow_ema = np.zeros_like(prices_array)
            
            fast_ema[0] = prices_array[0]
            slow_ema[0] = prices_array[0]
            
            for i in range(1, len(prices_array)):
                fast_ema[i] = fast_alpha * prices_array[i] + (1 - fast_alpha) * fast_ema[i-1]
                slow_ema[i] = slow_alpha * prices_array[i] + (1 - slow_alpha) * slow_ema[i-1]
            
            return fast_ema[-1] - slow_ema[-1]
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_aroon_oscillator(self, high_history: List[float], low_history: List[float], period: int = 14) -> float:
        """Calculate Aroon Oscillator."""
        if len(high_history) < period or len(low_history) < period:
            return 0.0
        try:
            # Get last 'period' values
            high = np.array(high_history[-period:])
            low = np.array(low_history[-period:])
            
            # Find the number of periods since the highest high and lowest low
            high_days = period - np.argmax(high) - 1
            low_days = period - np.argmin(low) - 1
            
            # Calculate Aroon Up and Down
            aroon_up = (period - high_days) / period * 100
            aroon_down = (period - low_days) / period * 100
            
            # Calculate Aroon Oscillator
            return aroon_up - aroon_down
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_cci(self, high_history: List[float], low_history: List[float], close_history: List[float], period: int = 20) -> float:
        """Calculate Commodity Channel Index."""
        if len(high_history) < period or len(low_history) < period or len(close_history) < period:
            return 0.0
        try:
            # Get last 'period' values
            high = np.array(high_history[-period:])
            low = np.array(low_history[-period:])
            close = np.array(close_history[-period:])
            
            # Calculate typical price
            tp = (high + low + close) / 3
            
            # Calculate SMA of typical price
            sma_tp = np.mean(tp)
            
            # Calculate Mean Deviation
            mean_deviation = np.mean(np.abs(tp - sma_tp))
            
            if mean_deviation == 0:
                return 0.0
            
            # Calculate CCI
            cci = (tp[-1] - sma_tp) / (0.015 * mean_deviation)
            
            return cci
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_cmo(self, prices: List[float], period: int = 14) -> float:
        """Calculate Chande Momentum Oscillator."""
        if len(prices) < period + 1:
            return 0.0
        try:
            # Calculate price changes
            changes = np.diff(prices[-period-1:])
            
            # Separate positive and negative changes
            positive_sum = np.sum([change for change in changes if change > 0])
            negative_sum = abs(np.sum([change for change in changes if change < 0]))
            
            if positive_sum + negative_sum == 0:
                return 0.0
            
            # Calculate CMO
            cmo = 100 * (positive_sum - negative_sum) / (positive_sum + negative_sum)
            
            return cmo
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """Calculate Momentum."""
        if len(prices) <= period:
            return 0.0
        try:
            return prices[-1] - prices[-period-1]
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_trix(self, prices: List[float], period: int = 15) -> float:
        """Calculate TRIX indicator."""
        if len(prices) < period * 3:
            return 0.0
        try:
            prices_array = np.array(prices)
            alpha = 2 / (period + 1)
            
            # Calculate triple EMA
            ema1 = np.zeros_like(prices_array)
            ema2 = np.zeros_like(prices_array)
            ema3 = np.zeros_like(prices_array)
            
            # First EMA
            ema1[0] = prices_array[0]
            for i in range(1, len(prices_array)):
                ema1[i] = alpha * prices_array[i] + (1 - alpha) * ema1[i-1]
            
            # Second EMA
            ema2[0] = ema1[0]
            for i in range(1, len(ema1)):
                ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
            
            # Third EMA
            ema3[0] = ema2[0]
            for i in range(1, len(ema2)):
                ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
            
            # Calculate TRIX
            if ema3[-2] == 0:
                return 0.0
            trix = (ema3[-1] - ema3[-2]) / ema3[-2] * 100
            
            return trix
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_ultimate_oscillator(self, high_history: List[float], low_history: List[float], close_history: List[float],
                                    period1: int = 7, period2: int = 14, period3: int = 28) -> float:
        """Calculate Ultimate Oscillator."""
        if len(high_history) < max(period1, period2, period3):
            return 0.0
        try:
            high = np.array(high_history)
            low = np.array(low_history)
            close = np.array(close_history)
            
            # Calculate buying pressure (BP) and true range (TR)
            bp = close - np.minimum(low, np.roll(close, 1))
            tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
            
            # Calculate averages for different periods
            avg1 = np.sum(bp[-period1:]) / np.sum(tr[-period1:])
            avg2 = np.sum(bp[-period2:]) / np.sum(tr[-period2:])
            avg3 = np.sum(bp[-period3:]) / np.sum(tr[-period3:])
            
            # Calculate Ultimate Oscillator
            uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
            
            return uo
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_stochastic_momentum(self, high_history: List[float], low_history: List[float], close_history: List[float],
                                     k_period: int = 14, d_period: int = 3) -> Tuple[float, float, float]:
        """Calculate Stochastic Momentum Indicator."""
        if len(high_history) < k_period or len(low_history) < k_period or len(close_history) < k_period:
            return 0.0, 0.0, 0.0
        try:
            # Get relevant price ranges
            high = np.array(high_history[-k_period:])
            low = np.array(low_history[-k_period:])
            close = np.array(close_history[-k_period:])
            
            # Calculate %K
            lowest_low = np.min(low)
            highest_high = np.max(high)
            
            if highest_high - lowest_low == 0:
                return 0.0, 0.0, 0.0
            
            k = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
            
            # Calculate %D (3-period SMA of %K)
            if len(close_history) < k_period + d_period:
                return k, 0.0, 0.0
            
            d = np.mean([k for _ in range(d_period)])
            
            # Calculate Slow Stochastic
            slow_d = np.mean([d for _ in range(d_period)])
            
            return k, d, slow_d
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0, 0.0

    def _calculate_natr(self, high_history: List[float], low_history: List[float], close_history: List[float], period: int = 14) -> float:
        """Calculate Normalized Average True Range."""
        if len(high_history) < period or len(low_history) < period or len(close_history) < period:
            return 0.0
        try:
            high = np.array(high_history)
            low = np.array(low_history)
            close = np.array(close_history)
            
            # Calculate True Range
            tr = np.zeros(len(high))
            for i in range(1, len(high)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
            
            # Calculate ATR
            atr = np.mean(tr[-period:])
            
            # Normalize ATR
            if close[-1] == 0:
                return 0.0
            
            natr = 100 * atr / close[-1]
            
            return natr
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_true_range(self, high_history: List[float], low_history: List[float], close_history: List[float]) -> float:
        """Calculate True Range."""
        if len(high_history) < 2 or len(low_history) < 2 or len(close_history) < 2:
            return 0.0
        try:
            high = high_history[-1]
            low = low_history[-1]
            prev_close = close_history[-2]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            
            return tr
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_hilbert_transform(self, prices: List[float], period: int = 14) -> Tuple[float, float]:
        """Calculate Hilbert Transform - Dominant Cycle Period and Phase."""
        if len(prices) < period * 2:
            return 0.0, 0.0
        try:
            prices_array = np.array(prices[-period*2:])
            
            # Smooth prices
            alpha = 0.0962
            smooth = np.zeros_like(prices_array)
            smooth[0] = prices_array[0]
            for i in range(1, len(prices_array)):
                smooth[i] = (1 - alpha) * smooth[i-1] + alpha * prices_array[i]
            
            # Calculate quadrature
            quad = np.zeros_like(smooth)
            for i in range(period, len(smooth)):
                quad[i] = 1.25 * (smooth[i-7] - smooth[i-5]) - 0.25 * (smooth[i-3] - smooth[i-1])
            
            # Calculate dominant cycle period using autocorrelation
            corr = np.correlate(smooth, smooth, mode='full')
            corr = corr[len(corr)//2:]
            peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
            if len(peaks) > 0:
                dominant_cycle = peaks[0]
            else:
                dominant_cycle = period
            
            # Calculate dominant cycle phase
            if np.sum(quad[-period:] ** 2) == 0:
                phase = 0.0
            else:
                phase = np.arctan2(
                    np.sum(quad[-period:] * smooth[-period:]),
                    np.sum(quad[-period:] ** 2)
                ) * 180 / np.pi
            
            return float(dominant_cycle), float(phase)
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0

    def _get_observation(self):
        """Get current market state as observation."""
        uniswap_entity = self.get_entity('UNISWAP_V3')
        global_state = uniswap_entity._global_state
        internal_state = uniswap_entity._internal_state

        # Update price history
        self.price_history.append(global_state.price)
        self.high_price_history.append(global_state.high_price)
        self.low_price_history.append(global_state.low_price)
        self.close_price_history.append(global_state.close_price)
        if len(self.price_history) > 168:  # Keep only last 168 prices
            self.price_history = self.price_history[-168:]
            self.high_price_history = self.high_price_history[-168:]
            self.low_price_history = self.low_price_history[-168:]
            self.close_price_history = self.close_price_history[-168:]

        # Calculate price ratios
        open_price = global_state.open_price
        if open_price == 0:
            high_to_open = 1.0
            low_to_open = 1.0
            close_to_open = 1.0
        else:
            high_to_open = global_state.high_price / open_price
            low_to_open = global_state.low_price / open_price
            close_to_open = global_state.close_price / open_price

        # Calculate technical indicators
        dema = self._calculate_dema(self.price_history)
        dema_ratio = dema / open_price if open_price != 0 else 1.0
        
        psar = self._calculate_parabolic_sar(self.high_price_history, self.low_price_history)
        psar_ratio = psar / open_price if open_price != 0 else 1.0
        
        apo = self._calculate_apo(self.price_history)
        aroon = self._calculate_aroon_oscillator(self.high_price_history, self.low_price_history)
        bop = self._calculate_bop(self.price_history, global_state)
        cci = self._calculate_cci(self.high_price_history, self.low_price_history, self.close_price_history)
        cmo = self._calculate_cmo(self.price_history)
        dx = self._calculate_dx(self.high_price_history, self.low_price_history, self.close_price_history)
        momentum = self._calculate_momentum(self.price_history)
        trix = self._calculate_trix(self.price_history)
        ultimate = self._calculate_ultimate_oscillator(self.high_price_history, self.low_price_history, self.close_price_history)

        ewma_volatility = self._calculate_ewma_volatility(self.price_history)
        ma24, ma168 = self._calculate_moving_averages(self.price_history)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(self.price_history)
        adxr = self._calculate_adxr(self.high_price_history, self.low_price_history, self.close_price_history)
        
        # Calculate Stochastic indicators
        k, d, slow_d = self._calculate_stochastic_momentum(
            self.high_price_history, 
            self.low_price_history, 
            self.close_price_history
        )
        
        # Calculate ranges
        natr = self._calculate_natr(self.high_price_history, self.low_price_history, self.close_price_history)
        true_range = self._calculate_true_range(self.high_price_history, self.low_price_history, self.close_price_history)


        impermanent_loss = 0.0
        if uniswap_entity.is_position:
            impermanent_loss = abs(
                (
                    uniswap_entity._internal_state.token0_amount_position_init + 
                    uniswap_entity._internal_state.token1_amount_position_init * uniswap_entity._global_state.price
                ) - (
                    uniswap_entity._internal_state.token0_amount_position + 
                    uniswap_entity._internal_state.token1_amount_position * uniswap_entity._global_state.price
                )
            )
        
        # Calculate Hilbert Transform indicators
        dominant_cycle, dominant_phase = self._calculate_hilbert_transform(self.price_history)
        current_observation = np.array([
            uniswap_entity.balance,  # balance
            # global_state.price,  # hourly price
            # global_state.centralized_price,  # centralized price
            # internal_state.price_init,  # initial price
            # internal_state.price_lower,  # lower price
            # internal_state.price_upper,  # upper price
            uniswap_entity.is_position,
            # global_state.open_price,  # hourly open price
            global_state.price,
            global_state.centralized_price,
            high_to_open,  # hourly highest price/hourly open price
            low_to_open,  # hourly lowest price/hourly open price
            close_to_open,  # hourly close price/hourly open price
            global_state.volume,  # hourly trading volume in USD
            dema_ratio,  # Double Exponential Moving Average/hourly open price
            psar_ratio,  # Parabolic SAR/hourly open price
            dx,  # Average Directional Movement Index
            apo,  # Absolute Price Oscillator
            aroon,  # Aroon Oscillator
            bop,  # Balance Of Power
            cci,  # Commodity Channel Index (2 features)
            cmo,  # Chande Momentum Oscillator
            momentum,  # Momentum
            trix,  # TRIX
            ultimate,  # Ultimate Oscillator
            k,  # Stochastic Momentum Indicator (3 features)
            d,
            slow_d,
            natr,  # Normalized Average True Range
            true_range,  # True Range
            dominant_cycle,  # Hilbert Transform - Dominant Cycle Period
            dominant_phase,  # Hilbert Transform - Dominant Cycle Phase
            ewma_volatility,  # Exponential Weighted Moving Average Volatility
            ma24, ma168,  # Moving Averages
            bb_upper, # Bollinger Bands
            bb_middle, 
            bb_lower,  
            adxr,  # Average Directional Index    
            # impermanent_loss,        
        ], dtype=np.float32)
        # Handle NaN and infinite values
        current_observation = np.nan_to_num(current_observation, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

        return current_observation


    def train(self, observations: List[Observation], total_timesteps: int = 100000):
        """
        Train the RL model using the provided observations.
        
        Args:
            observations: List of observations to use for training
            total_timesteps: Total number of timesteps to train for
        """
        # Create a new environment with the training observations
        uniswap_entity = self.get_entity('UNISWAP_V3')
        train_env = UniswapV3Env(deepcopy(uniswap_entity), self._params.INITIAL_BALANCE, observations, tick_spacing=self.tick_spacing)
        check_env(train_env)  # Verify the environment follows the Gym interface
        
        # Create a new model for training
        train_model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=self._params.LEARNING_RATE,
            n_steps=self._params.N_STEPS,
            batch_size=self._params.BATCH_SIZE,
            n_epochs=self._params.N_EPOCHS,
            gamma=self._params.GAMMA,
            gae_lambda=self._params.GAE_LAMBDA,
            clip_range=self._params.CLIP_RANGE,
            verbose=0,
            tensorboard_log="./model_logs/",
            seed=self._params.SEED,
            policy_kwargs={
                'net_arch': [256, 256]
            }
        )
        
        # train_model = DDPG(
        #     "MlpPolicy",
        #     train_env,
        #     verbose=0,
        #     tensorboard_log="./model_logs/",
        #     learning_rate=self._params.LEARNING_RATE,
        #     batch_size=self._params.BATCH_SIZE,
        #     action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=0.5 * np.ones(2)),
        # )

        # train_model =  CVaRPPO(
        #     "MlpPolicy",
        #     train_env,
        #     verbose=0,
        #     tensorboard_log="./model_logs/",
        #     learning_rate=self._params.LEARNING_RATE,
        #     n_steps=self._params.N_STEPS,
        #     batch_size=self._params.BATCH_SIZE,
        #     n_epochs=self._params.N_EPOCHS,
        #     gamma=self._params.GAMMA,
        #     gae_lambda=self._params.GAE_LAMBDA,
        #     clip_range=self._params.CLIP_RANGE,
        #     alpha=self._params.ALPHA,
        #     beta=self._params.BETA,
        #     nu_lr=self._params.NU_LR,
        #     lam_lr=self._params.LAM_LR,
        #     nu_start=self._params.NU_START,
        #     lam_start=self._params.LAM_START,
        #     nu_delay=self._params.NU_DELAY,
        #     lam_low_bound=self._params.LAM_LOW_BOUND,
        #     delay=self._params.DELAY,
        #     cvar_clip_ratio=self._params.CVAR_CLIP_RATIO,
        #     policy_kwargs={
        #         'net_arch': [256, 256]
        #     }
        # )
        train_model = CPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            tensorboard_log="./model_logs/",
            learning_rate=self._params.LEARNING_RATE,
            n_steps=self._params.N_STEPS,
            batch_size=self._params.BATCH_SIZE,
            n_epochs=self._params.N_EPOCHS,
            gamma=self._params.GAMMA,
            gae_lambda=self._params.GAE_LAMBDA,
            clip_range=self._params.CLIP_RANGE,
            max_constraint_value=self._params.MAX_CONSTRAINT_VALUE,
            max_backtrack_steps=self._params.MAX_BACKTRACK_STEPS,
            backtrack_coeff=self._params.BACKTRACK_COEFF,
            damping_coeff=self._params.DAMPING_COEFF,
            constraint_lr=self._params.CONSTRAINT_LR,
            max_impermanent_loss=self._params.MAX_IMPERMANENT_LOSS,
            impermanent_loss_quantile=self._params.IMPERMANENT_LOSS_QUANTILE,
        )
        train_model.set_logger(train_env.logger)
        
        # Train the model
        train_model.learn(total_timesteps=total_timesteps)
        
        # Update the main model with the trained weights
        self.model = train_model

    def predict(self) -> List[ActionToTake]:
        """
        Main logic of the strategy. Uses PPO to decide actions based on market state.
        """
        actions = []
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        
        # Check if we need to deposit funds into the LP before proceeding
        if not uniswap_entity.is_position and not self.deposited_initial_funds:
            self._debug("No active position. Depositing initial funds...")
            self.deposited_initial_funds = True
            return self._deposit_to_lp()
        
        # Get current market state
        global_state = uniswap_entity._global_state
        internal_state = uniswap_entity._internal_state
        
        observation = self._get_observation()

        # Get action from PPO model
        action, _ = self.model.predict(observation, deterministic=True)
        self.action_history.append(action)

        current_price = global_state.price
        current_tick = self._get_tick_from_price(current_price)
        
        # Calculate new tick bounds
        lower_tick = current_tick - action * self.tick_spacing
        upper_tick = current_tick + action * self.tick_spacing
        
        # Convert ticks to prices
        price_lower = self._get_price_from_tick(lower_tick)
        price_upper = self._get_price_from_tick(upper_tick)

        rebalance = True
        # if price_lower == uniswap_entity._internal_state.price_lower and price_upper == uniswap_entity._internal_state.price_upper:
        #     rebalance = False

        if action == 0:
            rebalance = False

        if rebalance:
            if uniswap_entity.is_position:
                actions.append(
                    ActionToTake(
                        entity_name='UNISWAP_V3',
                        action=Action(action='close_position', args={})
                    )
                )
                self._debug("Closing current position before opening new one.")
            
            delegate_get_cash = lambda obj: obj.get_entity('UNISWAP_V3').internal_state.cash
            actions.append(
                ActionToTake(
                    entity_name='UNISWAP_V3',
                    action=Action(
                        action='open_position',
                        args={
                            'amount_in_notional': delegate_get_cash,
                            'price_lower': price_lower,
                            'price_upper': price_upper
                        }
                    )
                )
            )
            self._debug(f"Opening new position with range [{price_lower:.2f}, {price_upper:.2f}]")  
        else:
            lower_bound = uniswap_entity._internal_state.price_lower
            upper_bound = uniswap_entity._internal_state.price_upper
            if current_price < lower_bound or current_price > upper_bound:
                if uniswap_entity.is_position:
                    actions.append(
                        ActionToTake(
                            entity_name='UNISWAP_V3',
                            action=Action(action='close_position', args={})
                        )
                    )

        return actions


    def _deposit_to_lp(self) -> List[ActionToTake]:
        """
        Deposit funds into the Uniswap LP if no position is currently open.
        """
        return [ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        )]
    
    def _get_price_from_tick(self, tick: int) -> float:
        """Convert tick to price using Uniswap V3 formula."""
        return 1.0001 ** tick

    def _get_tick_from_price(self, price: float) -> int:
        """Convert price to tick using Uniswap V3 formula."""
        return int(np.log(price) / np.log(1.0001))