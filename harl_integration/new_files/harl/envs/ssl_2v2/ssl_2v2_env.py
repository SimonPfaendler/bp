"""HARL-side wrapper around SSL2v2SelfPlayEnv (defined in ~/dev/bp).

HARL contract per `step`:
    return local_obs, global_state, rewards, dones, infos, available_actions
Each entry is a list of length n_agents (None for available_actions on
continuous control).
"""
import copy
import os
import sys

import numpy as np
# rSoccer uses gymnasium internally; HARL's runtime path for --env ssl_2v2
# never loads any HARL env that requires old gym, so gymnasium suffices.
from gymnasium.spaces import Box

# Make the bp project importable. The env code (rsoccer wiring, reward shaping,
# curriculum, kick logic) all stays in bp/ as the single source of truth — this
# wrapper just adapts its (2,52) obs / (2,6) action shapes to HARL's per-agent
# list interface.
_BP_DIR = os.environ.get("BP_DIR", "/home/simon/dev/bp")
if _BP_DIR not in sys.path:
    sys.path.insert(0, _BP_DIR)

from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv  # noqa: E402

N_AGENTS = 2
SINGLE_OBS_DIM = 52
SINGLE_ACT_DIM = 6


class SSL2v2Env:
    """Adapter: per-agent obs (52,) / action (6,) lists for HARL.

    State for the centralized critic is the concatenated joint obs (104,).
    Reward is the shared scalar from the env (yellows' shaped+sparse team
    reward), exposed identically to both agents per HARL's common-reward
    convention.
    """

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        # `curriculum_level` (legacy name) and `curriculum_start_level` both
        # accepted as the starting spawn level. The env then auto-promotes
        # itself to `curriculum_target_level` once rolling success rate over
        # the last `curriculum_window` episodes clears `curriculum_threshold`.
        start_level = (
            self.args.get("curriculum_start_level")
            or self.args.get("curriculum_level")
        )
        env_kwargs = {
            "reward_type": self.args.get("reward_type", "dense"),
            "frozen_path": self.args.get("frozen_path", None),
            "role_index": self.args.get("role_index", False),
            "oob_grace_steps": self.args.get("oob_grace_steps", 0),
            "curriculum_start_level": start_level,
            "curriculum_target_level": self.args.get(
                "curriculum_target_level", 5
            ),
            "curriculum_threshold": self.args.get(
                "curriculum_threshold", 0.9
            ),
            "curriculum_window": self.args.get(
                "curriculum_window", 200
            ),
        }
        self.env = SSL2v2SelfPlayEnv(**env_kwargs)

        self.n_agents = N_AGENTS

        # HARL expects gym (not gymnasium) Box. Bounds are the env's NORM_BOUNDS,
        # but stating ±inf is safer because we clip in the env already and HARL's
        # value-norm doesn't care about the box bounds.
        single_obs = Box(
            low=-np.inf, high=np.inf, shape=(SINGLE_OBS_DIM,), dtype=np.float32
        )
        single_act = Box(
            low=-1.0, high=1.0, shape=(SINGLE_ACT_DIM,), dtype=np.float32
        )
        joint_state = Box(
            low=-np.inf, high=np.inf,
            shape=(N_AGENTS * SINGLE_OBS_DIM,), dtype=np.float32,
        )
        self.observation_space = [single_obs for _ in range(N_AGENTS)]
        self.share_observation_space = [joint_state for _ in range(N_AGENTS)]
        self.action_space = [single_act for _ in range(N_AGENTS)]
        self.discrete = False

        self._seed = 0

    # ---------- HARL API ----------

    def _split_obs(self, joint_obs):
        """(2, 52) -> [agent0 (52,), agent1 (52,)]"""
        return [joint_obs[i].astype(np.float32) for i in range(N_AGENTS)]

    def _joint_state(self, joint_obs):
        return joint_obs.reshape(-1).astype(np.float32)

    def reset(self):
        joint_obs, _ = self.env.reset(seed=self._seed)
        local_obs = self._split_obs(joint_obs)
        s_obs = [self._joint_state(joint_obs) for _ in range(N_AGENTS)]
        return local_obs, s_obs, self.get_avail_actions()

    def step(self, actions):
        # actions: np.ndarray (n_agents, 6) or list-of-arrays.
        joint_action = np.asarray(actions, dtype=np.float32).reshape(
            N_AGENTS, SINGLE_ACT_DIM
        )
        joint_obs, reward, done, trunc, info = self.env.step(joint_action)

        local_obs = self._split_obs(joint_obs)
        s_obs = [self._joint_state(joint_obs) for _ in range(N_AGENTS)]
        # Reward from SSL env is a (2,) array — both entries identical (team
        # reward). HARL wants per-agent [r] lists.
        r_scalar = float(np.asarray(reward).mean())
        rewards = [[r_scalar] for _ in range(N_AGENTS)]
        # HARL's `dones` are per agent; same termination for both.
        dones = [bool(done) for _ in range(N_AGENTS)]
        # Forward truncation as info["bad_transition"] — HARL's standard flag
        # for distinguishing TimeLimit-truncation from real termination so the
        # critic bootstraps the terminal value.
        infos = [dict(info) for _ in range(N_AGENTS)]
        if trunc:
            for ag_info in infos:
                ag_info["bad_transition"] = True
        # On truncation HARL also expects done=True so the rollout buffer
        # closes the episode and resets.
        if trunc:
            dones = [True for _ in range(N_AGENTS)]

        return local_obs, s_obs, rewards, dones, infos, self.get_avail_actions()

    def get_avail_actions(self):
        # Continuous control: no action masking.
        return None

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        self._seed = int(seed)
