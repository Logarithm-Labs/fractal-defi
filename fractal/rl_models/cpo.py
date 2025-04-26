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


class CPO(PPO):
    """
    Constrained Policy Optimization algorithm for portfolio management.
    Implements the CPO algorithm from "Constrained Policy Optimization" paper.
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
        # CPO specific parameters
        max_constraint_value: float = 0.1,  # Maximum allowed constraint violation
        max_backtrack_steps: int = 10,  # Maximum number of backtracking steps
        backtrack_coeff: float = 0.8,  # Backtracking coefficient
        damping_coeff: float = 0.1,  # Damping coefficient for Fisher matrix
        constraint_lr: float = 1e-2,  # Learning rate for constraint parameters
        # Domain-specific constraints
        max_impermanent_loss: float = 0.05,  # Maximum allowed impermanent loss (5%)
        impermanent_loss_quantile: float = 0.95,  # Quantile for impermanent loss constraint
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
        
        # CPO specific parameters
        self.max_constraint_value = max_constraint_value
        self.max_backtrack_steps = max_backtrack_steps
        self.backtrack_coeff = backtrack_coeff
        self.damping_coeff = damping_coeff
        self.constraint_lr = constraint_lr
        
        # Domain-specific constraint parameters
        self.max_impermanent_loss = max_impermanent_loss
        self.impermanent_loss_quantile = impermanent_loss_quantile
        
        # Initialize constraint parameters
        self.lambda_ = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.nu = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.impermanent_loss_lambda = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Initialize optimizers for constraint parameters
        self.constraint_optimizer = torch.optim.Adam(
            [self.lambda_, self.nu, self.impermanent_loss_lambda], 
            lr=constraint_lr
        )

    def _compute_constraint_violation(self, rollout_data):
        """
        Compute the constraint violation for the current policy.
        Includes both value function and domain-specific constraints.
        """
        # Value at Risk (VaR) constraint
        returns = rollout_data.returns
        var_threshold = torch.quantile(returns, 0.05)  # 5% VaR
        value_constraint = torch.relu(-var_threshold - self.max_constraint_value)
        
        # Extract impermanent loss from observations (last dimension)
        observations = rollout_data.observations
        impermanent_losses = observations[..., -1]  # Assuming IL is the last feature in observations
        il_quantile = torch.quantile(impermanent_losses, self.impermanent_loss_quantile)
        impermanent_loss_constraint = torch.relu(il_quantile - self.max_impermanent_loss)
        
        return value_constraint, impermanent_loss_constraint

    def _compute_cpo_loss(self, ratio, advantages, values, rollout_data, clip_range, value_constraint, impermanent_loss_constraint):
        """
        Compute the CPO loss components including domain-specific constraints.
        """
        # Policy loss with constraints
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Add constraint terms
        policy_loss += (
            self.lambda_ * value_constraint + 
            self.impermanent_loss_lambda * impermanent_loss_constraint
        )
        
        # Value loss
        if self.clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + torch.clamp(
                values - rollout_data.old_values, -self.clip_range_vf, self.clip_range_vf
            )
        value_loss = F.mse_loss(rollout_data.returns, values_pred)
        
        # Entropy loss - use the entropy from policy evaluation
        entropy_loss = -torch.mean(-rollout_data.old_log_prob)
        
        return policy_loss, value_loss, entropy_loss

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer with CPO constraints.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        value_constraint_violations = []
        impermanent_loss_violations = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                
                # Calculate constraint violations
                value_constraint, impermanent_loss_constraint = self._compute_constraint_violation(rollout_data)
                value_constraint_violations.append(value_constraint.item())
                impermanent_loss_violations.append(impermanent_loss_constraint.item())
                
                # Calculate CPO loss
                policy_loss, value_loss, entropy_loss = self._compute_cpo_loss(
                    ratio, advantages, values, rollout_data, clip_range,
                    value_constraint, impermanent_loss_constraint
                )
                
                # Logging
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                # Combine losses
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                
                approx_kl_divs.append(torch.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                continue_training = False
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/value_constraint_violation", np.mean(value_constraint_violations))
        self.logger.record("train/impermanent_loss_violation", np.mean(impermanent_loss_violations))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item()) 