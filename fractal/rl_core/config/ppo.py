from dataclasses import dataclass

from fractal.rl_core.config.base import BaseModelConfig


@dataclass
class PPOConfig(BaseModelConfig):
    """Configuration for PPO model."""

    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2


@dataclass
class CVaRPPOConfig(PPOConfig):
    """Configuration for CVaR-PPO model."""

    alpha: float = 0.9
    beta: float = 2800.0
    nu_lr: float = 1e-2
    lam_lr: float = 1e-2
    nu_start: float = 0.0
    lam_start: float = 0.5
    nu_delay: float = 0.8
    lam_low_bound: float = 0.001
    delay: float = 1.0
    cvar_clip_ratio: float = 0.05


@dataclass
class CPOConfig(PPOConfig):
    """Configuration for CPO model."""

    max_constraint_value: float = 1000
    max_backtrack_steps: int = 10
    backtrack_coeff: float = 0.8
    damping_coeff: float = 0.1
    constraint_lr: float = 1e-2
    max_impermanent_loss: float = 500
    impermanent_loss_quantile: float = 0.8
