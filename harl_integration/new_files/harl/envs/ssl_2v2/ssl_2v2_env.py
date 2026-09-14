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
# wrapper just adapts its (2,obs) / (2,6) shapes to HARL's per-agent
# list interface.
_BP_DIR = os.environ.get("BP_DIR", "/home/simon/dev/bp")
if _BP_DIR not in sys.path:
    sys.path.insert(0, _BP_DIR)

from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv, N_YELLOW  # noqa: E402

N_AGENTS = N_YELLOW


class SSL2v2Env:
    """Adapter: per-agent obs / action lists for HARL.

    State for the centralized critic is the concatenated joint obs.
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
        #
        # Eval-mode override: when the HARL runner sets `eval_mode=True` in
        # env_args, the eval envs spawn directly at `curriculum_target_level`
        # so eval always measures real-task performance regardless of where
        # training-side curriculum currently is. Without this, eval envs
        # would need ~200 episodes of training-equivalent success rate
        # before they self-promote, which lags training by many evals.
        target_level = self.args.get("curriculum_target_level", 5)
        if self.args.get("eval_mode"):
            start_level = target_level
        else:
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
            "blue_heuristic": self.args.get("blue_heuristic", None),
            # Level-5 spawn mix and terminal payoff. These are the knobs the
            # SB3 runs were swept on (pass_scenario_prob=0.35 throughout;
            # goal_reward_solo=None keeps the symmetric payoff, which is the
            # config that won the ablation) — without forwarding them a HARL
            # run silently trains on 100% chaos spawns.
            "pass_scenario_prob": self.args.get("pass_scenario_prob", 0.0),
            "goal_reward": self.args.get("goal_reward", 10.0),
            "goal_reward_solo": self.args.get("goal_reward_solo", None),
        }
        self.env = SSL2v2SelfPlayEnv(**env_kwargs)

        self.n_agents = N_AGENTS
        # Read the per-agent dims off the env instead of hardcoding them: the
        # obs layout has grown twice (role_index, then EPISODE_STATE_DIM in
        # the Markov repair) and a stale constant here builds actor/critic
        # nets of the wrong width without any error until the first step.
        self.single_obs_dim = int(self.env.single_observation_space.shape[0])
        self.single_act_dim = int(self.env.single_action_space.shape[0])

        # HARL expects gym (not gymnasium) Box. Bounds are the env's NORM_BOUNDS,
        # but stating ±inf is safer because we clip in the env already and HARL's
        # value-norm doesn't care about the box bounds.
        single_obs = Box(
            low=-np.inf, high=np.inf,
            shape=(self.single_obs_dim,), dtype=np.float32,
        )
        single_act = Box(
            low=-1.0, high=1.0, shape=(self.single_act_dim,), dtype=np.float32
        )
        joint_state = Box(
            low=-np.inf, high=np.inf,
            shape=(N_AGENTS * self.single_obs_dim,), dtype=np.float32,
        )
        self.observation_space = [single_obs for _ in range(N_AGENTS)]
        self.share_observation_space = [joint_state for _ in range(N_AGENTS)]
        self.action_space = [single_act for _ in range(N_AGENTS)]
        self.discrete = False

        self._seed = 0
        self._seed_pending = True

    # ---------- HARL API ----------

    def _split_obs(self, joint_obs):
        """(2, obs_dim) -> [agent0 (obs_dim,), agent1 (obs_dim,)]"""
        return [joint_obs[i].astype(np.float32) for i in range(N_AGENTS)]

    def _joint_state(self, joint_obs):
        return joint_obs.reshape(-1).astype(np.float32)

    def reset(self):
        # gymnasium re-seeds np_random every time reset() is handed a seed, so
        # passing it on every episode would replay one fixed spawn — and one
        # fixed pass-vs-chaos roll — for the whole life of this worker. Seed
        # the stream once after seed(), then let it run.
        seed = self._seed if self._seed_pending else None
        self._seed_pending = False
        joint_obs, _ = self.env.reset(seed=seed)
        local_obs = self._split_obs(joint_obs)
        s_obs = [self._joint_state(joint_obs) for _ in range(N_AGENTS)]
        return local_obs, s_obs, self.get_avail_actions()

    def step(self, actions):
        # actions: np.ndarray (n_agents, 6) or list-of-arrays.
        joint_action = np.asarray(actions, dtype=np.float32).reshape(
            N_AGENTS, self.single_act_dim
        )
        # HARL's on-policy Gaussian actor is unsquashed (only HASAC's has the
        # final tanh). The env caps just the translational speed norm; v_theta
        # and the kick/dribble triggers take the raw values. Clip to the
        # declared Box bounds here — what SB3 does for Box spaces — so MAPPO
        # cannot command turn rates the SAC policies never could.
        joint_action = np.clip(joint_action, -1.0, 1.0)
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
        self._seed_pending = True
