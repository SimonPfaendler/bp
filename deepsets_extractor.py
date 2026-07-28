"""Deep-Sets features extractor for the 2v2 obs layout.

Slices the flat 52-dim obs (see SSL2v2SelfPlayEnv._egocentric_obs) into
entity tokens and encodes them permutation-invariantly:

    ego block  = BALL + SELF + OPP1-extra + PRED + TEAM (+ any trailing
                 dims such as the role-index one-hot) — passed through raw
    mate token = MATE (9) -> shared teammate encoder phi_mate
    opp tokens = OPP1[:9], OPP2[:9] -> shared opponent encoder phi_opp,
                 mean-pooled over the set

Mean pooling over the opponents removes the distance-sort discontinuity:
when the two blues swap their closest/farther ranking, the flat layout's
slot contents jump, while the pooled representation is unchanged. It also
makes the extractor size-agnostic over opponents (2v2 -> NvN needs no
architecture change, only more tokens).

The obs itself stays untouched, so demos, replay buffers, frozen opponents
and the env are all unaffected — only the network input structure changes.

Usage (SB3):
    policy_kwargs = dict(
        net_arch=[512, 512, 512],
        features_extractor_class=DeepSetsExtractor,
        features_extractor_kwargs=dict(embed_dim=64),
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


class DeepSetsExtractor(BaseFeaturesExtractor):
    """Permutation-invariant entity encoder for the 52-dim 2v2 obs."""

    def __init__(self, observation_space, embed_dim: int = 64):
        obs_dim = int(observation_space.shape[-1])
        assert obs_dim >= BASE_DIM, (
            f"DeepSetsExtractor expects the 52-dim 2v2 layout, got {obs_dim}"
        )
        # 5 + 13 + 1 + 4 + 2 = 25 ego dims, plus role-index etc. if present.
        ego_dim = 25 + (obs_dim - BASE_DIM)
        super().__init__(
            observation_space, features_dim=ego_dim + 2 * embed_dim
        )
        self.obs_dim = obs_dim

        def entity_encoder():
            return nn.Sequential(
                nn.Linear(ENTITY_DIM, embed_dim), nn.ReLU(),
                nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            )

        # Separate encoders per entity type: teammates and opponents carry
        # different tactical meaning even with identical feature layout.
        self.mate_enc = entity_encoder()
        self.opp_enc = entity_encoder()

    def forward(self, obs: th.Tensor) -> th.Tensor:
        ego = th.cat(
            [
                obs[..., BALL], obs[..., SELF], obs[..., OPP1_EXTRA],
                obs[..., PRED], obs[..., TEAM], obs[..., BASE_DIM:],
            ],
            dim=-1,
        )
        mate = self.mate_enc(obs[..., MATE])
        # (B, 2, 9) -> shared encoder -> mean over the opponent set.
        opps = th.stack([obs[..., OPP1], obs[..., OPP2]], dim=-2)
        opp_pool = self.opp_enc(opps).mean(dim=-2)
        return th.cat([ego, mate, opp_pool], dim=-1)
