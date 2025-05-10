from copy import deepcopy
from dataclasses import dataclass
from typing import List, Type

import numpy as np
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_checker import check_env

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams)
from fractal.core.base.observations import Observation
from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity
from fractal.rl_core.config import (BaseModelConfig, CPOConfig, CVaRPPOConfig,
                                    DDPGConfig, PPOConfig)
from fractal.rl_core.envs.uniswap_v3_env import UniswapV3Env
from fractal.rl_core.models.cpo import CPO
from fractal.rl_core.models.cppo import CVaRPPO


@dataclass
class RLStrategyParams(BaseStrategyParams):
    """
    Parameters for the Reinforcement Learning strategy.
    """
    INITIAL_BALANCE: float
    MODEL_CLASS: Type[BaseAlgorithm] = PPO
    MODEL_CONFIG: BaseModelConfig = None

    def __post_init__(self):
        if self.MODEL_CONFIG is None:
            if self.MODEL_CLASS == PPO:
                self.MODEL_CONFIG = PPOConfig()
            elif self.MODEL_CLASS == DDPG:
                self.MODEL_CONFIG = DDPGConfig()
            elif self.MODEL_CLASS == CVaRPPO:
                self.MODEL_CONFIG = CVaRPPOConfig()
            elif self.MODEL_CLASS == CPO:
                self.MODEL_CONFIG = CPOConfig()
            else:
                raise ValueError(f"Unsupported model class: {self.MODEL_CLASS}")


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
                    token1_decimals=self.token1_decimals,
                )
            )
        ))
        assert isinstance(self.get_entity('UNISWAP_V3'), UniswapV3LPEntity)

    def _get_observation(self):
        """Get current market state as observation."""
        uniswap_entity = self.get_entity('UNISWAP_V3')
        global_state = uniswap_entity._global_state

        return {
            'balance': np.array([uniswap_entity.balance], dtype=np.float32),
            'is_position': np.array([uniswap_entity.is_position], dtype=np.float32),
            'price': np.array([global_state.price], dtype=np.float32),
            'centralized_price': np.array([global_state.centralized_price], dtype=np.float32),
            'high_price': np.array([global_state.high_price], dtype=np.float32),
            'low_price': np.array([global_state.low_price], dtype=np.float32),
            'open_price': np.array([global_state.open_price], dtype=np.float32),
            'close_price': np.array([global_state.close_price], dtype=np.float32),
            'volume': np.array([global_state.volume], dtype=np.float32),
        }

    def train(self, observations: List[Observation], total_timesteps: int = 100000):
        """
        Train the RL model using the provided observations.

        Args:
            observations: List of observations to use for training
            total_timesteps: Total number of timesteps to train for
        """
        # Create a new environment with the training observations
        uniswap_entity = self.get_entity('UNISWAP_V3')
        train_env = UniswapV3Env(
            deepcopy(uniswap_entity),
            self._params.INITIAL_BALANCE,
            observations,
            tick_spacing=self.tick_spacing
        )
        check_env(train_env)  # Verify the environment follows the Gym interface

        # Create model using config
        model_kwargs = self._params.MODEL_CONFIG.to_model_kwargs()
        model_kwargs['env'] = train_env
        train_model = self._params.MODEL_CLASS("MultiInputPolicy", **model_kwargs)

        # Set logger
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

        observation = self._get_observation()

        # Get action from PPO model
        action, _ = self.model.predict(observation, deterministic=True)

        current_price = global_state.price
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
