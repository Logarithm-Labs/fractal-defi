from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from fractal.rl_core.config.base import BaseModelConfig


@dataclass
class DDPGConfig(BaseModelConfig):
    """Configuration for DDPG model."""

    action_noise_sigma: float = 0.5

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Convert config to model initialization kwargs with action noise."""
        kwargs = super().to_model_kwargs()
        kwargs["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(2), sigma=self.action_noise_sigma * np.ones(2)
        )
        return kwargs
