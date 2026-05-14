"""MASAC: SAC with a centralized critic (CTDE) for 2-agent parameter sharing.

Subclasses SB3's SAC and overrides only `train()`. The actor update and the
critic targets reshape between the per-agent actor outputs (B*2, 6) and the
joint critic inputs (obs (B, 2, 38), action (B, 12)). The actor gradient flows
through the *joint* critic, so each agent's action gradient is shaped by the
team Q-value — the credit-assignment fix that independent SAC lacks.

Two stabilizers on top of stock SAC, both targeting the Q-overestimation that
showed up when a fresh centralized critic was paired with a transferred,
already-competent actor (actor exploits the immature critic -> critic_loss and
actor_loss both diverge):

  * critic_warmup_grad_steps: for the first N gradient steps, update only the
    critic. The critic learns real Q-values for the transferred actor's
    behaviour before the actor starts moving against it.
  * max_grad_norm: clip the critic gradient norm — a cheap safety net against
    Q-divergence.

Everything else (rollout collection, target polyak update, ent_coef optimizer,
save/load) is inherited unchanged. `target_entropy="auto"` resolves to
-prod(action_space.shape) = -12, which is correct for the joint log-prob
(summed over both agents' 6 action dims).
"""

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from masac_policy import N_AGENTS, SINGLE_ACT_DIM, MASACPolicy


class MASAC(SAC):
    """SAC variant with a centralized critic over the 2-agent joint space."""

    policy_aliases = {"MlpPolicy": MASACPolicy}

    def __init__(
        self, *args,
        critic_warmup_grad_steps: int = 0,
        max_grad_norm: float = 0.0,
        **kwargs,
    ):
        # Stored before super().__init__ so they survive _setup_model; both
        # are plain scalars, so SB3 save/load round-trips them via __dict__.
        self.critic_warmup_grad_steps = int(critic_warmup_grad_steps)
        self.max_grad_norm = float(max_grad_norm)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (affects batch norm / dropout).
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        act_joint_dim = N_AGENTS * SINGLE_ACT_DIM  # 12

        for gradient_step in range(gradient_steps):
            # Global gradient-step index across all train() calls.
            global_step = self._n_updates + gradient_step
            in_warmup = global_step < self.critic_warmup_grad_steps

            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )
            batch = replay_data.observations.shape[0]

            if self.use_sde:
                self.actor.reset_noise()

            # ent_coef: during warmup we still need its current value for the
            # target, but we don't optimize it (the actor isn't moving yet).
            if (
                self.ent_coef_optimizer is not None
                and self.log_ent_coef is not None
            ):
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            with th.no_grad():
                # Next joint action from the current actor.
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_actions_joint = next_actions.reshape(batch, act_joint_dim)
                next_log_prob_joint = next_log_prob.reshape(
                    batch, N_AGENTS
                ).sum(dim=1, keepdim=True)
                # Centralized target Q over the joint next state-action.
                next_q_values = th.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions_joint
                    ),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob_joint
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            # Current Q from buffer actions (already joint (B, 12)).
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm > 0.0:
                th.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
            self.critic.optimizer.step()

            # Polyak-update the target critic — must run during warmup too,
            # otherwise the target stays at random init while the critic
            # learns and the warmup targets are garbage.
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats, self.batch_norm_stats_target, 1.0
                )

            # Critic-warmup: hold the actor (and ent_coef) frozen so the
            # centralized critic can learn real Q-values for the transferred
            # actor before the actor starts optimizing against it.
            if in_warmup:
                continue

            # Current actor: per-agent (B*2, 6) actions + (B*2,) log-prob.
            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            actions_pi_joint = actions_pi.reshape(batch, act_joint_dim)
            # Joint log-prob: sum the 2 agents' log-probs per sample -> (B, 1).
            log_prob_joint = log_prob.reshape(batch, N_AGENTS).sum(
                dim=1, keepdim=True
            )

            ent_coef_loss = None
            if (
                self.ent_coef_optimizer is not None
                and self.log_ent_coef is not None
            ):
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob_joint + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # Actor loss: gradient flows through the *joint* critic, so each
            # agent's action gradient is shaped by the team Q-value.
            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi_joint), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob_joint - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm > 0.0:
                th.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
            self.actor.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record(
                "train/ent_coef_loss", np.mean(ent_coef_losses)
            )
