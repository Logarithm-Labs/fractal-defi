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


class UniswapV3Env(gym.Env):
    """
    Custom Gym environment for Uniswap V3 trading.
    """
    def __init__(
            self,
            uniswap_entity: UniswapV3LPEntity,
            initial_balance: float,
            observations: Optional[List[Observation]] = None,
            tick_spacing: int = 1,
            max_ticks: int = 5,  # Maximum number of ticks to deviate
            max_timesteps: int = 1000,
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
        self.action_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([max_ticks, max_ticks]), shape=(2,))
        
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32
        )

        self.action_history = {}
        self.logger = configure_logger(
            verbose=1,
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
        if len(prices) < 168:
            return 0.0, 0.0
        try:
            ma24 = np.mean(prices[-24:])
            ma168 = np.mean(prices[-168:])
            return ma24, ma168
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0

    def _calculate_bollinger_bands(self, prices: List[float], window: int = 20) -> Tuple[float, float, float]:
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

    def _calculate_adxr(self, high_history: List[float], low_history: List[float], close_history: List[float], window: int = 14) -> float:
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

    def _calculate_dx(self, high_history: List[float], low_history: List[float], close_history: List[float], window: int = 14) -> float:
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

        # Calculate technical indicators
        ewma_volatility = self._calculate_ewma_volatility(self.price_history)
        ma24, ma168 = self._calculate_moving_averages(self.price_history)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(self.price_history)
        adxr = self._calculate_adxr(self.high_price_history, self.low_price_history, self.close_price_history)
        bop = self._calculate_bop(self.price_history, global_state)
        dx = self._calculate_dx(self.high_price_history, self.low_price_history, self.close_price_history)

        current_observation = np.array([
            global_state.price,
            global_state.centralized_price,
            global_state.open_price,
            global_state.close_price,
            global_state.high_price,
            global_state.low_price,
            internal_state.liquidity,
            internal_state.token0_amount,
            internal_state.token1_amount,
            internal_state.price_init,
            internal_state.price_lower,
            internal_state.price_upper,
            internal_state.earned_fees,
            ewma_volatility,
            ma24,
            ma168,
            bb_upper,
            bb_middle,
            bb_lower,
            adxr,
            bop,
            dx
        ], dtype=np.float32)

        # Handle NaN and infinite values
        current_observation = np.nan_to_num(current_observation, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

        # self.observation_history.append(current_observation)

        # # Get last 3 observations plus current
        # if len(self.observation_history) >= 4:
        #     last_observations = self.observation_history[-4:]
        # else:
        #     # Pad with zeros if we don't have enough history
        #     padding = [np.zeros_like(current_observation)] * (4 - len(self.observation_history))
        #     last_observations = padding + self.observation_history

        # # Concatenate the observations
        # observation = np.concatenate(last_observations)
    
        return current_observation

    def _calculate_reward(self, rebalance, fees_earned_prev):
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
        if self.uniswap_entity.is_position:
            earned_fees = max(0, self.uniswap_entity._internal_state.earned_fees - fees_earned_prev)
            impermanent_loss = abs(
                (
                    self.uniswap_entity._internal_state.token0_amount_init + 
                    self.uniswap_entity._internal_state.token1_amount_init * self.uniswap_entity._global_state.price
                ) - (
                    self.uniswap_entity._internal_state.token0_amount + 
                    self.uniswap_entity._internal_state.token1_amount * self.uniswap_entity._global_state.price
                )
            )

            ewma_volatility = self._calculate_ewma_volatility(self.price_history)   
            liquidity = self.uniswap_entity._internal_state.liquidity
            sqrt_price = self.uniswap_entity._global_state.price ** (1/2)
            instantaneous_lvr = liquidity * ewma_volatility**2 * sqrt_price / 4

            if rebalance:
                rebalancing_penalty = self.uniswap_entity.balance * self.uniswap_entity.trading_fee


        self.logger.record(key="train/impermanent_loss", value=impermanent_loss)
        self.logger.record(key="train/rebalancing_penalty", value=rebalancing_penalty)
        self.logger.record(key="train/earned_fees", value=earned_fees)
        self.logger.record(key="train/instantaneous_lvr", value=instantaneous_lvr)

        # Calculate total reward
        reward = 2 * earned_fees - instantaneous_lvr - 2 * rebalancing_penalty
        
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
        
        # Calculate new tick bounds
        lower_tick = current_tick - round(action[0]) * self.tick_spacing - self.tick_spacing
        upper_tick = current_tick + round(action[1]) * self.tick_spacing + self.tick_spacing
        
        # Convert ticks to prices
        price_lower = self._get_price_from_tick(lower_tick)
        price_upper = self._get_price_from_tick(upper_tick)

        rebalance = True

        if round(action[0]) == 0 and round(action[1]) == 0:
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

        # Get current observation and update entity state
        current_observation = self.observations[self.current_observation_idx]
        for entity_name, state in current_observation.states.items():
            self.uniswap_entity.update_state(state)

        # Get new state
        observation = self._get_observation()

        # Calculate reward before taking action
        reward = self._calculate_reward(rebalance, fees_earned_prev)
        
        # Move to next observation
        self.current_observation_idx += 1
        done = self.current_observation_idx - self.start_observation_idx >= self.max_timesteps
        
        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.start_observation_idx = np.random.randint(0, len(self.observations) - self.max_timesteps - 1)
        self.current_observation_idx = self.start_observation_idx

        # self.initial_balance = np.random.uniform(10 ** 5, 10 ** 6)

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
        self.alpha = 0.05  # Smoothing factor for EWMA volatility


    def set_up(self):
        """
        Register the Uniswap V3 LP entity and initialize the RL environment and model.
        """
        self.register_entity(NamedEntity(
            entity_name='UNISWAP_V3',
            entity=UniswapV3LPEntity(
                UniswapV3LPConfig(
                    token0_decimals=self.token0_decimals,
                    token1_decimals=self.token1_decimals
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
        if len(prices) < 168:
            return 0.0, 0.0
        try:
            ma24 = np.mean(prices[-24:])
            ma168 = np.mean(prices[-168:])
            return ma24, ma168
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0

    def _calculate_bollinger_bands(self, prices: List[float], window: int = 20) -> Tuple[float, float, float]:
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

    def _calculate_adxr(self, high_history: List[float], low_history: List[float], close_history: List[float], window: int = 14) -> float:
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

    def _calculate_dx(self, high_history: List[float], low_history: List[float], close_history: List[float], window: int = 14) -> float:
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

        # Calculate technical indicators
        ewma_volatility = self._calculate_ewma_volatility(self.price_history)
        ma24, ma168 = self._calculate_moving_averages(self.price_history)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(self.price_history)
        adxr = self._calculate_adxr(self.high_price_history, self.low_price_history, self.close_price_history)
        bop = self._calculate_bop(self.price_history, global_state)
        dx = self._calculate_dx(self.high_price_history, self.low_price_history, self.close_price_history)

        current_observation = np.array([
            global_state.price,
            global_state.centralized_price,
            global_state.open_price,
            global_state.close_price,
            global_state.high_price,
            global_state.low_price,
            internal_state.liquidity,
            internal_state.token0_amount,
            internal_state.token1_amount,
            internal_state.price_init,
            internal_state.price_lower,
            internal_state.price_upper,
            internal_state.earned_fees,
            ewma_volatility,
            ma24,
            ma168,
            bb_upper,
            bb_middle,
            bb_lower,
            adxr,
            bop,
            dx
        ], dtype=np.float32)

        # Handle NaN and infinite values
        current_observation = np.nan_to_num(current_observation, copy=True, nan=0.0, posinf=0.0, neginf=0.0)

        # self.observation_history.append(current_observation)

        # # Get last 3 observations plus current
        # if len(self.observation_history) >= 4:
        #     last_observations = self.observation_history[-4:]
        # else:
        #     # Pad with zeros if we don't have enough history
        #     padding = [np.zeros_like(current_observation)] * (4 - len(self.observation_history))
        #     last_observations = padding + self.observation_history

        # # Concatenate the observations
        # observation = np.concatenate(last_observations)
    
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
        # train_model = PPO(
        #     "MlpPolicy",
        #     train_env,
        #     learning_rate=self._params.LEARNING_RATE,
        #     n_steps=self._params.N_STEPS,
        #     batch_size=self._params.BATCH_SIZE,
        #     n_epochs=self._params.N_EPOCHS,
        #     gamma=self._params.GAMMA,
        #     gae_lambda=self._params.GAE_LAMBDA,
        #     clip_range=self._params.CLIP_RANGE,
        #     verbose=1,
        #     tensorboard_log="./model_logs/"
        # )
        train_model = DDPG(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log="./model_logs/",
            learning_rate=self._params.LEARNING_RATE,
            batch_size=self._params.BATCH_SIZE,
            action_noise=OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))
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


        current_price = global_state.price
        current_tick = self._get_tick_from_price(current_price)
        
        # Calculate new tick bounds
        lower_tick = current_tick - round(action[0]) * self.tick_spacing - self.tick_spacing
        upper_tick = current_tick + round(action[1]) * self.tick_spacing + self.tick_spacing
        
        # Convert ticks to prices
        price_lower = self._get_price_from_tick(lower_tick)
        price_upper = self._get_price_from_tick(upper_tick)

        rebalance = True
        # if price_lower == uniswap_entity._internal_state.price_lower and price_upper == uniswap_entity._internal_state.price_upper:
        #     rebalance = False

        if round(action[0]) == 0 and round(action[1]) == 0:
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