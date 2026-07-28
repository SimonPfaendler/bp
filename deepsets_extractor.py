"""Deep-Sets features extractor for the 2v2 obs layout.

Slices the flat 52-dim obs (see SSL2v2SelfPlayEnv._egocentric_obs) into
entity tokens and encodes the opponents permutation-invariantly:

    ego block  = BALL + SELF + MATE + OPP1-extra + PRED + TEAM (+ any
                 trailing dims such as the role-index one-hot) — raw
    opp tokens = OPP1[:9], OPP2[:9] -> shared opponent encoder phi_opp,
                 pooled over the set with mean AND max, concatenated

Pooling over the opponents removes the distance-sort discontinuity: when
the two blues swap their closest/farther ranking, the flat layout's slot
contents jump, while the pooled representation is unchanged. It also makes
the extractor size-agnostic over opponents (2v2 -> NvN needs no
architecture change, only more tokens).

`pooling="meanmax"` (default) concatenates mean- and max-pooling. Mean
alone is a real bottleneck at N=2: it cannot separate "one opponent right
on me, one far away" from "both at medium range" — exactly the asymmetry
that drives open-play decisions. Max recovers the per-feature extreme
(effectively "the most threatening opponent"), and is just as
permutation-invariant. `pooling="mean"` reproduces the first-generation
behaviour for the ablation.

The mate is a single entity, so there is nothing to be invariant over —
it goes into the ego block raw instead of through an encoder that would
only act as a bottleneck.

The obs itself stays untouched, so demos, replay buffers, frozen opponents
and the env are all unaffected — only the network input structure changes.

Usage (SB3):
    policy_kwargs = dict(
        net_arch=[512, 512, 512],
        features_extractor_class=DeepSetsExtractor,
        features_extractor_kwargs=dict(embed_dim=64, pooling="meanmax"),
    )
"""
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Slot layout of the 52-dim base obs (kept in sync with
# SSL2v2SelfPlayEnv._egocentric_obs; verified against slots 0..51).
BALL = slice(0, 5)
SELF = slice(5, 18)
MATE = slice(18, 27)          # pos, sin/cos, vel, v_theta, infrared, dist_ball
OPP1 = slice(27, 36)          # same 9-dim semantic layout as MATE/OPP2
OPP1_EXTRA = slice(36, 37)    # (self_dist - opp1_dist) -> ego block
OPP2 = slice(37, 46)
PRED = slice(46, 50)
TEAM = slice(50, 52)
BASE_DIM = 52                 # trailing dims (role-index one-hot) -> ego block
ENTITY_DIM = 9
# 5 (ball) + 13 (self) + 9 (mate) + 1 (opp1-extra) + 4 (pred) + 2 (team)
EGO_BASE_DIM = 34


class DeepSetsExtractor(BaseFeaturesExtractor):
    """Permutation-invariant opponent encoder for the 52-dim 2v2 obs."""

    def __init__(self, observation_space, embed_dim: int = 64,
                 pooling: str = "meanmax"):
        assert pooling in ("meanmax", "mean"), pooling
        obs_dim = int(observation_space.shape[-1])
        assert obs_dim >= BASE_DIM, (
            f"DeepSetsExtractor expects the 52-dim 2v2 layout, got {obs_dim}"
        )
        ego_dim = EGO_BASE_DIM + (obs_dim - BASE_DIM)
        n_pool = 2 if pooling == "meanmax" else 1
        super().__init__(
            observation_space, features_dim=ego_dim + n_pool * embed_dim
        )
        self.obs_dim = obs_dim
        self.pooling = pooling

        self.opp_enc = nn.Sequential(
            nn.Linear(ENTITY_DIM, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        ego = th.cat(
            [
                obs[..., BALL], obs[..., SELF], obs[..., MATE],
                obs[..., OPP1_EXTRA], obs[..., PRED], obs[..., TEAM],
                obs[..., BASE_DIM:],
            ],
            dim=-1,
        )
        # (B, 2, 9) -> shared encoder -> pool over the opponent set.
        opps = th.stack([obs[..., OPP1], obs[..., OPP2]], dim=-2)
        emb = self.opp_enc(opps)
        pooled = emb.mean(dim=-2)
        if self.pooling == "meanmax":
            pooled = th.cat([pooled, emb.max(dim=-2).values], dim=-1)
        return th.cat([ego, pooled], dim=-1)
