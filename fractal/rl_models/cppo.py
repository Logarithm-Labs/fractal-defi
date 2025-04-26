import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym


class CVaRPPO(PPO):
    """
    Proximal Policy Optimization algorithm with CVaR (Conditional Value at Risk) optimization.
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        # CVaR specific parameters
        alpha: float = 0.9,
        beta: float = 2800.0,
        nu_lr: float = 1e-2,
        lam_lr: float = 1e-2,
        nu_start: float = 0.0,
        lam_start: float = 0.5,
        nu_delay: float = 0.8,
        lam_low_bound: float = 0.001,
        delay: float = 1.0,
        cvar_clip_ratio: float = 0.05,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        
        # CVaR specific parameters
        self.alpha = alpha
        self.beta = beta
        self.nu_lr = nu_lr
        self.lam_lr = lam_lr
        self.nu_delay = nu_delay
        self.lam_low_bound = lam_low_bound
        self.delay = delay
        self.cvar_clip_ratio = cvar_clip_ratio
        
        # Initialize CVaR parameters as tensors with gradients
        self.nu = torch.tensor(nu_start, requires_grad=True, device=self.device)
        self.cvarlam = torch.tensor(lam_start, requires_grad=True, device=self.device)
        
        # Initialize optimizers for CVaR parameters
        self.nu_optimizer = torch.optim.Adam([self.nu], lr=nu_lr)
        self.lam_optimizer = torch.optim.Adam([self.cvarlam], lr=lam_lr)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate CVaR loss
                cvar_loss = self._compute_cvar_loss(rollout_data, values)
                loss += cvar_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                approx_kl_divs.append(torch.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                continue_training = False
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/cvar_nu", self.nu.item())
        self.logger.record("train/cvar_lambda", self.cvarlam.item())

    def _compute_cvar_loss(self, rollout_data, values):
        """
        Compute the CVaR loss component.
        """
        clip_range = self.clip_range(self._current_progress_remaining)
        # Calculate CVaR components
        returns = rollout_data.returns
        advantages = rollout_data.advantages
        
        # Update nu (VaR estimate)
        nu_loss = torch.mean(torch.relu(returns - self.nu)) - (1 - self.alpha) * self.nu
        self.nu_optimizer.zero_grad()
        nu_loss.backward()
        self.nu_optimizer.step()
        
        # Update lambda (CVaR multiplier)
        lam_loss = self.beta * self.cvarlam - torch.mean(torch.relu(returns - self.nu))
        self.lam_optimizer.zero_grad()
        lam_loss.backward()
        self.lam_optimizer.step()
        
        # Clip lambda to ensure it stays positive
        with torch.no_grad():
            self.cvarlam.data = torch.clamp(self.cvarlam, min=self.lam_low_bound)
        
        # Calculate CVaR advantage
        cvar_advantage = torch.where(
            returns < self.nu,
            self.delay * self.cvarlam / (1 - self.alpha) * (self.nu - returns),
            torch.zeros_like(returns)
        )
        
        # Clip CVaR advantage
        cvar_advantage = torch.clamp(cvar_advantage, -self.cvar_clip_ratio * torch.abs(values), 
                                   self.cvar_clip_ratio * torch.abs(values))
        
        # Combine with original advantage
        total_advantage = advantages + cvar_advantage
        
        # Calculate CVaR policy loss
        ratio = torch.exp(rollout_data.old_log_prob - rollout_data.old_log_prob)
        policy_loss_1 = total_advantage * ratio
        policy_loss_2 = total_advantage * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        cvar_policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        return cvar_policy_loss 