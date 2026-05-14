"""MASAC: SAC with a centralized critic (CTDE) for 2-agent parameter sharing.

Subclasses SB3's SAC and overrides only `train()`. The actor update and the
critic targets reshape between the per-agent actor outputs (B*2, 6) and the
joint critic inputs (obs (B, 2, 38), action (B, 12)). The actor gradient flows
through the *joint* critic, so each agent's action gradient is shaped by the
team Q-value — the credit-assignment fix that independent SAC lacks.

Everything else (rollout collection, target polyak update, ent_coef optimizer,
save/load) is inherited unchanged. `target_entropy="auto"` resolves to
-prod(action_space.shape) = -12, which is already correct for the joint
log-prob (summed over both agents' 6 action dims).
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
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob_joint + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

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
            self.critic.optimizer.step()

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
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats, self.batch_norm_stats_target, 1.0
                )

        self._n_updates += gradient_steps

        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record(
                "train/ent_coef_loss", np.mean(ent_coef_losses)
            )
