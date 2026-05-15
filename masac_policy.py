"""MASAC policy: parameter-shared decentralized actor + centralized twin-Q critic.

The actor sees one agent's 38-dim observation and outputs a 6-dim action;
both yellow agents share its weights (parameter sharing). The critic is
*centralized* — it sees the joint observation (2x38) and joint action (2x6)
and outputs a single team Q-value. This CTDE structure gives the policy
gradient access to team-level credit assignment, which independent SAC
critics structurally cannot do.

Shapes through the system:
  - env / vec env:   obs (n_envs, 2, 38), action (n_envs, 2, 6)
  - replay buffer:   obs (B, 2, 38), action (B, 12)   [SB3 flattens actions]
  - actor:           joint obs (B, 2, 38) -> reshape (B*2, 38) -> (B*2, 6)
  - critic:          obs (B, 2, 38) -> Flatten (B, 76); action (B, 12) -> Q (B, 1)

Only `make_actor` is overridden. `make_critic` stays stock: a `ContinuousCritic`
built on the joint spaces automatically gets `features_dim=76` (FlattenExtractor
over the (2,38) obs) and `action_dim=12`, i.e. exactly Q(joint_obs, joint_act).
"""

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.sac.policies import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    Actor,
    SACPolicy,
)

N_AGENTS = 2
SINGLE_OBS_DIM = 38
SINGLE_ACT_DIM = 6


class MASACActor(Actor):
    """Parameter-shared decentralized actor.

    Constructed with single-agent spaces (38 -> 6). It accepts a joint
    (B, 2, 38) observation, reshapes to (B*2, 38), runs the shared network,
    and returns per-agent-flattened (B*2, 6) outputs. SB3's `predict()` then
    reshapes (B*2, 6) -> (B, 2, 6) via the joint action space.
    """

    def get_action_dist_params(self, obs: PyTorchObs):
        # Joint obs (B, 2, 38) -> per-agent (B*2, 38).
        obs = obs.reshape(-1, SINGLE_OBS_DIM)
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}


class MASACCritic(ContinuousCritic):
    """Centralized twin-Q critic with LayerNorm between layers.

    Same interface as `ContinuousCritic` (joint obs (2,38) -> FlattenExtractor
    -> 76; joint action 12; input 88), but each Q-network interleaves a
    `LayerNorm` after every hidden linear. LayerNorm bounds the activations
    and is the standard, well-established fix for the unbounded Q-value
    growth that plain SAC critics fall into on this task.
    """

    def __init__(
        self, observation_space, action_space, net_arch, features_extractor,
        features_dim, activation_fn=nn.ReLU, normalize_images=True,
        n_critics=2, share_features_extractor=True,
    ):
        super().__init__(
            observation_space, action_space, net_arch, features_extractor,
            features_dim, activation_fn, normalize_images, n_critics,
            share_features_extractor,
        )
        # Rebuild the q-networks with LayerNorm; add_module overwrites the
        # stock qf{idx} modules created by ContinuousCritic.__init__.
        action_dim = get_action_dim(self.action_space)
        self.q_networks = []
        for idx in range(n_critics):
            q_net = self._build_ln_qnet(
                features_dim + action_dim, net_arch, activation_fn
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    @staticmethod
    def _build_ln_qnet(in_dim, net_arch, activation_fn):
        layers = []
        last = in_dim
        for hidden in net_arch:
            layers += [
                nn.Linear(last, hidden),
                nn.LayerNorm(hidden),
                activation_fn(),
            ]
            last = hidden
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)


class MASACPolicy(SACPolicy):
    """SAC policy with a parameter-shared actor and a centralized critic.

    `make_actor` builds the actor on single-agent spaces. `make_critic` stays
    stock: `ContinuousCritic` on the joint spaces gives `features_dim=76`
    (FlattenExtractor over (2,38)) and `action_dim=12`, i.e. a centralized
    Q(joint_obs, joint_action). `MASACCritic` (with LayerNorm) is kept above
    for reference / re-enabling if Q-divergence resurfaces.
    """

    def make_actor(self, features_extractor=None) -> MASACActor:
        single_obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(SINGLE_OBS_DIM,), dtype=np.float32,
        )
        single_act_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(SINGLE_ACT_DIM,), dtype=np.float32,
        )
        actor_kwargs = self.actor_kwargs.copy()
        actor_kwargs["observation_space"] = single_obs_space
        actor_kwargs["action_space"] = single_act_space
        fe = FlattenExtractor(single_obs_space)
        actor_kwargs["features_extractor"] = fe
        actor_kwargs["features_dim"] = fe.features_dim  # 38
        return MASACActor(**actor_kwargs).to(self.device)


MlpPolicy = MASACPolicy
