from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import configure_logger

from fractal.core.base.observations import Observation
from fractal.core.entities import UniswapV3LPEntity
from fractal.rl_core.features.uniswap_feature_extractor import \
    UniswapFeatureExtractor


class UniswapV3Env(gym.Env):
    """
    Custom Gym environment for Uniswap V3 trading.

    This environment simulates trading on Uniswap V3, allowing an agent to manage liquidity positions
    by adjusting price ranges around the current market price.

    Attributes:
        uniswap_entity (UniswapV3LPEntity): The Uniswap V3 LP entity being managed
        initial_balance (float): Initial balance for trading
        observations (Optional[List[Observation]]): List of market observations
        tick_spacing (int): Spacing between ticks in the price range
        max_ticks (int): Maximum number of ticks to deviate from current price
        max_timesteps (int): Maximum number of timesteps per episode
        reward_weights (Dict[str, float]): Weights for different reward components
    """

    def __init__(
        self,
        uniswap_entity: UniswapV3LPEntity,
        initial_balance: float,
        observations: Optional[List[Observation]] = None,
        tick_spacing: int = 60,
        max_ticks: int = 10,
        max_timesteps: int = 24 * 7,
        reward_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the Uniswap V3 environment.

        Args:
            uniswap_entity: The Uniswap V3 LP entity to manage
            initial_balance: Initial balance for trading
            observations: List of market observations
            tick_spacing: Spacing between ticks in the price range
            max_ticks: Maximum number of ticks to deviate from current price
            max_timesteps: Maximum number of timesteps per episode
            reward_weights: Weights for different reward components
        """
        super().__init__()
        self.uniswap_entity = uniswap_entity
        self.initial_balance = initial_balance
        self.observations = observations
        self.current_observation_idx = 0
        self.tick_spacing = tick_spacing
        self.max_ticks = max_ticks

        # Set default reward weights if not provided
        self.reward_weights = {
            "earned_fees": 1.0,  # positive
            "instantaneous_lvr": -1.0,  # negative
            "impermanent_loss": 0.0,  # negative
            "rebalancing_penalty": -0.5,  # negative
            "holding_penalty": 0.0,  # negative
        }
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

        # Define action space
        self.action_space = gym.spaces.Discrete(max_ticks)

        # Initialize feature extractor
        self.feature_extractor = UniswapFeatureExtractor()

        # Define observation space for processed features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )

        self.action_history = {}
        self.logger: Logger = configure_logger(
            verbose=0,
            tensorboard_log="./model_logs/",
            tb_log_name="run",
        )
        self.max_timesteps = max_timesteps
        self.episode_number = 0
        self.alpha = 0.05  # Smoothing factor for EWMA volatility
        self.price_history = []

    def _calculate_ewma_volatility(self, prices: List[float]) -> float:
        """
        Calculate exponentially weighted moving average volatility.

        Args:
            prices: List of historical prices

        Returns:
            float: The calculated EWMA volatility
        """
        if len(prices) < 2:
            return 0.0

        returns = np.diff(np.log(prices))
        ewma = np.zeros_like(returns)
        ewma[0] = returns[0] ** 2
        for i in range(1, len(returns)):
            ewma[i] = self.alpha * returns[i] ** 2 + (1 - self.alpha) * ewma[i - 1]
        return np.sqrt(ewma[-1])

    def _get_observation(self) -> np.ndarray:
        """
        Get current market state as observation.

        Returns:
            np.ndarray: Processed features from the feature extractor
        """
        global_state = self.uniswap_entity._global_state
        self.price_history.append(global_state.price)

        raw_observation = {
            "balance": float(self.uniswap_entity.balance),
            "is_position": float(self.uniswap_entity.is_position),
            "price": float(global_state.price),
            "centralized_price": float(global_state.centralized_price),
            "high_price": float(global_state.high_price),
            "low_price": float(global_state.low_price),
            "open_price": float(global_state.open_price),
            "close_price": float(global_state.close_price),
            "volume": float(global_state.volume),
        }

        # Process raw observation through feature extractor
        return self.feature_extractor.forward(raw_observation)

    def _calculate_impermanent_loss(self) -> float:
        """
        Calculate impermanent loss for the current position.

        Returns:
            float: The calculated impermanent loss
        """

        if not self.uniswap_entity.is_position:
            return 0.0

        impermanent_loss = abs(
            (
                self.uniswap_entity._internal_state.token0_amount_position_init
                + self.uniswap_entity._internal_state.token1_amount_position_init
                * self.uniswap_entity._global_state.price
            )
            - (
                self.uniswap_entity._internal_state.token0_amount_position
                + self.uniswap_entity._internal_state.token1_amount_position
                * self.uniswap_entity._global_state.price
            )
        )
        return impermanent_loss

    def _calculate_instantaneous_lvr(self) -> float:
        """
        Calculate instantaneous loss versus rebalancing.

        Returns:
            float: The calculated instantaneous LVR
        """

        if not self.uniswap_entity.is_position:
            return 0.0

        ewma_volatility = self._calculate_ewma_volatility(self.price_history)
        liquidity = self.uniswap_entity._internal_state.liquidity
        sqrt_price = self.uniswap_entity._global_state.price**0.5
        instantaneous_lvr = liquidity * ewma_volatility**2 * sqrt_price / 4
        return instantaneous_lvr

    def _calculate_rebalancing_penalty(self, rebalance: bool, truncated: bool) -> float:
        """
        Calculate rebalancing penalty.

        Args:
            rebalance: Whether a rebalancing action was taken
            truncated: Whether the episode was truncated

        Returns:
            float: The calculated rebalancing penalty
        """
        if not self.uniswap_entity.is_position or not (rebalance or truncated):
            return 0.0

        rebalancing_penalty = (
            self.uniswap_entity._internal_state.token1_amount_position
            * self.uniswap_entity._global_state.price
            * self.uniswap_entity.trading_fee
        )
        return rebalancing_penalty

    def _calculate_holding_penalty(self) -> float:
        """
        Calculate holding penalty for not being in a position.

        Returns:
            float: The calculated holding penalty
        """
        if self.uniswap_entity.is_position:
            return 0.0

        return 1.0

    def _calculate_reward(
        self, rebalance: bool, fees_earned_prev: float, truncated: bool
    ) -> float:
        """
        Calculate reward based on a combination of various factors.

        Args:
            rebalance: Whether a rebalancing action was taken
            fees_earned_prev: Previously earned fees
            truncated: Whether the episode was truncated

        Returns:
            float: The calculated reward
        """
        earned_fees = self.uniswap_entity._internal_state.position_fees
        impermanent_loss = self._calculate_impermanent_loss()
        instantaneous_lvr = self._calculate_instantaneous_lvr()
        rebalancing_penalty = self._calculate_rebalancing_penalty(rebalance, truncated)
        holding_penalty = self._calculate_holding_penalty()
        reward = (
            self.reward_weights["earned_fees"] * earned_fees
            + self.reward_weights["instantaneous_lvr"] * instantaneous_lvr
            + self.reward_weights["impermanent_loss"] * impermanent_loss
            + self.reward_weights["rebalancing_penalty"] * rebalancing_penalty
            + self.reward_weights["holding_penalty"] * holding_penalty
        )

        self.logger.record_mean(key="train/impermanent_loss", value=impermanent_loss)
        self.logger.record_mean(
            key="train/rebalancing_penalty", value=rebalancing_penalty
        )
        self.logger.record_mean(key="train/holding_penalty", value=holding_penalty)
        self.logger.record_mean(key="train/earned_fees", value=earned_fees)
        self.logger.record_mean(key="train/instantaneous_lvr", value=instantaneous_lvr)
        self.logger.record(
            key="train/balance_ratio",
            value=self.uniswap_entity.balance / self.initial_balance,
        )

        return reward

    def _get_price_from_tick(self, tick: int) -> float:
        """
        Convert tick to price using Uniswap V3 formula.

        Args:
            tick: The tick to convert

        Returns:
            float: The corresponding price
        """
        return 1.0001**tick

    def _get_tick_from_price(self, price: float) -> int:
        """
        Convert price to tick using Uniswap V3 formula.

        Args:
            price: The price to convert

        Returns:
            int: The corresponding tick
        """
        return int(np.log(price) / np.log(1.0001))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: The action to take (number of ticks to deviate)

        Returns:
            Tuple containing:
                - observation: Current market state
                - reward: Reward for the action
                - terminated: Whether the episode is done
                - truncated: Whether the episode was truncated
                - info: Additional information
        """
        fees_earned_prev = self.uniswap_entity._internal_state.earned_fees

        # Get current price and tick
        current_price = self.uniswap_entity._global_state.price
        current_tick = self._get_tick_from_price(current_price)

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
                price_upper=price_upper,
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
        truncated = (
            self.current_observation_idx - self.start_observation_idx
            >= self.max_timesteps
        )

        # Calculate reward before taking action
        reward = self._calculate_reward(rebalance, fees_earned_prev, truncated)

        return observation, reward, False, truncated, {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple containing:
                - observation: Initial market state
                - info: Additional information
        """
        super().reset(seed=seed)

        self.episode_number += 1
        self.start_observation_idx = np.random.randint(
            0, len(self.observations) - self.max_timesteps - 1
        )
        self.current_observation_idx = self.start_observation_idx

        initial_balance = np.random.uniform(0.8 * self.initial_balance, 1.2 * self.initial_balance)

        if self.uniswap_entity.is_position:
            self.uniswap_entity.action_close_position()

        self.uniswap_entity._initialize_states()

        current_observation = self.observations[self.current_observation_idx]
        for entity_name, state in current_observation.states.items():
            self.uniswap_entity.update_state(state)

        self.uniswap_entity.action_deposit(initial_balance)
        self.current_observation_idx += 1

        return self._get_observation(), {}
