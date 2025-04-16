from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
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
            max_ticks: int = 5  # Maximum number of ticks to deviate
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
        self.action_space = gym.spaces.MultiDiscrete([max_ticks + 1, max_ticks + 1])
        
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),  # [price, volume, liquidity, token0_balance, token1_balance, price_init, price_lower, price_upper]
            dtype=np.float32
        )

        self.action_history = {}

    def _get_observation(self):
        """Get current market state as observation."""
        global_state = self.uniswap_entity._global_state
        internal_state = self.uniswap_entity._internal_state
        return np.array([
            global_state.price,
            global_state.volume,
            global_state.liquidity,
            internal_state.token0_amount,
            internal_state.token1_amount,
            internal_state.price_init,
            internal_state.price_lower,
            internal_state.price_upper
        ], dtype=np.float32)

    def _calculate_reward(self, price_lower_prev, price_upper_prev):
        """
        Calculate reward based on a combination of impermanent loss and earned fees.
        The reward is calculated as: fees_earned - absolute_impermanent_loss - rebalancing_penalty
        """
        # Calculate impermanent loss
        current_price = self.uniswap_entity._global_state.price
        initial_price = self.uniswap_entity._internal_state.price_init
        
        # If we haven't opened a position yet, return 0
        if initial_price == 0.0:
            return 0.0
            
        # Calculate price change ratio
        price_ratio = current_price / initial_price
        
        # Calculate impermanent loss percentage
        # IL = 2 * sqrt(price_ratio) / (1 + price_ratio) - 1
        impermanent_loss_percentage = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        
        # Convert to absolute value
        impermanent_loss = abs(impermanent_loss_percentage) * self.uniswap_entity.balance
        
        # Get earned fees
        earned_fees = self.uniswap_entity._internal_state.earned_fees
        
        # Calculate rebalancing penalty
        # Penalize each rebalancing action with a fixed cost
        rebalancing_penalty = 0.0
        if self.uniswap_entity.is_position:
            current_lower = price_lower_prev
            current_upper = price_upper_prev
            new_lower = self.uniswap_entity._internal_state.price_lower
            new_upper = self.uniswap_entity._internal_state.price_upper
            
            # If the range has changed, apply rebalancing penalty
            if current_lower != new_lower or current_upper != new_upper:
                rebalancing_penalty = self.uniswap_entity.balance * self.uniswap_entity.trading_fee
        
        # Calculate total reward
        reward = earned_fees - impermanent_loss - rebalancing_penalty

        if self.current_observation_idx % 1000 == 0:
            print(f"earned_fees: {earned_fees:.2f}, "
                  f"impermanent_loss: {impermanent_loss:.2f}, "
                  f"rebalancing_penalty: {rebalancing_penalty:.2f}, "
                  f"reward: {reward:.2f}")
            print(f"action_history: {self.action_history}")
        
        return reward

    def _get_price_from_tick(self, tick: int) -> float:
        """Convert tick to price using Uniswap V3 formula."""
        return 1.0001 ** tick

    def _get_tick_from_price(self, price: float) -> int:
        """Convert price to tick using Uniswap V3 formula."""
        return int(np.log(price) / np.log(1.0001))

    def step(self, action):
        """Execute one step in the environment."""

        price_lower_prev = self.uniswap_entity._internal_state.price_lower
        price_upper_prev = self.uniswap_entity._internal_state.price_upper

        # Get current price and tick
        current_price = self.uniswap_entity._global_state.price
        current_tick = self._get_tick_from_price(current_price)
        
        # Calculate new tick bounds
        lower_tick = current_tick - action[0] * self.tick_spacing - self.tick_spacing
        upper_tick = current_tick + action[1] * self.tick_spacing + self.tick_spacing
        
        # Convert ticks to prices
        price_lower = self._get_price_from_tick(lower_tick)
        price_upper = self._get_price_from_tick(upper_tick)

        rebalance = True
        # if price_lower == self.uniswap_entity._internal_state.price_lower and price_upper == self.uniswap_entity._internal_state.price_upper:
        #     rebalance = False

        if action[0] == 0 and action[1] == 0:
            rebalance = False

        self.action_history[tuple(action)] = self.action_history.get(tuple(action), 0) + 1

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
        reward = self._calculate_reward(price_lower_prev, price_upper_prev)
        
        # Move to next observation
        self.current_observation_idx += 1
        done = self.current_observation_idx >= len(self.observations)
        
        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_observation_idx = 0
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
            verbose=1,
            tensorboard_log="./model_logs/"
        )
        
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
        
        # Prepare observation for the model
        observation = np.array([
            global_state.price,
            global_state.volume,
            global_state.liquidity,
            internal_state.token0_amount,
            internal_state.token1_amount,
            internal_state.price_init,
            internal_state.price_lower,
            internal_state.price_upper
        ], dtype=np.float32)

        # Get action from PPO model
        action, _ = self.model.predict(observation, deterministic=False)


        current_price = global_state.price
        current_tick = self._get_tick_from_price(current_price)
        
        # Calculate new tick bounds
        lower_tick = current_tick - action[0] * self.tick_spacing - self.tick_spacing
        upper_tick = current_tick + action[1] * self.tick_spacing + self.tick_spacing
        
        # Convert ticks to prices
        price_lower = self._get_price_from_tick(lower_tick)
        price_upper = self._get_price_from_tick(upper_tick)

        rebalance = True
        # if price_lower == uniswap_entity._internal_state.price_lower and price_upper == uniswap_entity._internal_state.price_upper:
        #     rebalance = False

        if action[0] == 0 and action[1] == 0:
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