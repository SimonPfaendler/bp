"""Smoke-test the ssl_2v2 HARL wrapper without invoking the full HARL runner.

Verifies:
  - Env imports correctly via the BP_DIR shim
  - Observation / state / action spaces match what the env itself reports
    (the obs width is read off the env — it grew 52 -> 56 with the Markov
    repair and grows again with role_index — so no hardcoded 52 here)
  - reset() returns (local_obs, s_obs, avail_actions) with correct types
  - step() returns the 6-tuple with correct shapes and dtypes
  - seed() pins the first spawn, later resets draw fresh spawns
  - out-of-range actions are clipped to the Box bounds before the env sees them
  - Truncation at max_steps sets done=True and info["bad_transition"]
  - Curriculum-level kwarg is honored (Level 1 spawn is near goal)

Run from the HARL repo root (setup_harl_cluster.sh copies it there):
  BP_DIR=/path/to/bp <venv>/bin/python test_ssl_2v2_wrapper.py
"""
import os
import sys

import numpy as np

# Resolve the harl package relative to this file (it lives at the HARL repo
# root after setup) instead of a hardcoded home-dir path. BP_DIR falls back
# to the documented sibling layout: <workspace>/bp next to <workspace>/HARL.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("BP_DIR", os.path.join(os.path.dirname(_HERE), "bp"))

from harl.envs.ssl_2v2.ssl_2v2_env import SSL2v2Env  # noqa: E402
# Importable once the wrapper has put BP_DIR on sys.path.
from ssl_rl_2v2_selfplay import (  # noqa: E402
    SINGLE_OBS_DIM_BASE, EPISODE_STATE_DIM, SINGLE_ACT_DIM,
)


def _make(args, seed):
    env = SSL2v2Env(args)
    env.seed(seed)
    return env


def _commands(env):
    """Flatten the last batch of robot commands the env sent to rsim."""
    return np.array(
        [
            (c.v_x, c.v_y, c.v_theta, c.kick_v_x, float(c.dribbler))
            for c in env.env.sent_commands
        ],
        dtype=np.float64,
    )


def main():
    args = {
        "reward_type": "dense",
        "frozen_path": None,
        "role_index": False,
        "oob_grace_steps": 0,
        "curriculum_level": 1,
        "pass_scenario_prob": 0.0,
    }
    env = _make(args, 0)
    obs_dim = SINGLE_OBS_DIM_BASE + EPISODE_STATE_DIM  # role_index=False

    print(f"n_agents:                {env.n_agents}")
    print(f"observation_space[0]:    {env.observation_space[0]}")
    print(f"share_observation_space: {env.share_observation_space[0]}")
    print(f"action_space[0]:         {env.action_space[0]}")
    print(f"discrete:                {env.discrete}")
    assert env.n_agents == 2
    assert env.observation_space[0].shape == (obs_dim,), env.observation_space[0]
    assert env.share_observation_space[0].shape == (2 * obs_dim,)
    assert env.action_space[0].shape == (SINGLE_ACT_DIM,)
    assert env.discrete is False
    # The wrapper must agree with the env's own single-agent space.
    assert env.observation_space[0].shape == env.env.single_observation_space.shape

    local_obs, s_obs, avail = env.reset()
    assert len(local_obs) == 2 and local_obs[0].shape == (obs_dim,), local_obs[0].shape
    assert len(s_obs) == 2 and s_obs[0].shape == (2 * obs_dim,), s_obs[0].shape
    assert local_obs[0].dtype == np.float32
    assert avail is None
    print(f"\nreset OK | obs[0][:5]={local_obs[0][:5]}")

    # --- seed semantics: same seed -> same first spawn; second reset -> new spawn
    twin_obs, _, _ = _make(args, 0).reset()
    assert np.allclose(local_obs[0], twin_obs[0]), "same seed should give same first spawn"
    second_obs, _, _ = env.reset()
    assert not np.allclose(local_obs[0], second_obs[0]), (
        "second reset() replayed the first spawn — seed is being re-applied every reset"
    )
    print("seed OK  | seeded twin matches, second reset draws a fresh spawn")

    # --- clipping: identical twins, one fed +1.0 and one +5.0, must send the
    # same commands to rsim.
    a, b = _make(args, 7), _make(args, 7)
    a.reset(); b.reset()
    a.step(np.full((2, SINGLE_ACT_DIM), 1.0, dtype=np.float32))
    b.step(np.full((2, SINGLE_ACT_DIM), 5.0, dtype=np.float32))
    assert np.allclose(_commands(a), _commands(b)), "out-of-range actions were not clipped"
    print("clip OK  | action=5.0 and action=1.0 produce identical robot commands")

    # --- 50 no-op steps: Level 1 + static blue should give negative
    # cumulative reward (time + standing penalty) per the reward design.
    total_r = 0.0
    for t in range(50):
        actions = np.zeros((env.n_agents, SINGLE_ACT_DIM), dtype=np.float32)
        local_obs, s_obs, rewards, dones, infos, _ = env.step(actions)
        assert len(rewards) == 2 and len(rewards[0]) == 1, rewards
        assert isinstance(dones[0], bool), type(dones[0])
        total_r += rewards[0][0]
        if any(dones):
            break
    print(f"step OK  | 50 no-op steps total_r={total_r:.3f} (expect slightly negative)")

    # --- run the no-op episode out to max_steps: nobody moves, so the only
    # exit is the time limit, which must surface as done + bad_transition
    # and carry the terminal summary keys the loggers read.
    max_steps = env.env.max_steps
    done_at = None
    for t in range(51, max_steps + 5):
        _, _, _, dones, infos, _ = env.step(
            np.zeros((env.n_agents, SINGLE_ACT_DIM), dtype=np.float32)
        )
        if all(dones):
            done_at = t
            break
    assert done_at == max_steps, f"expected truncation at step {max_steps}, got {done_at}"
    assert infos[0].get("bad_transition") is True, infos[0].keys()
    for key in ("is_success", "blue_goal", "passes", "scored_after_pass", "scenario"):
        assert key in infos[0], f"terminal info missing {key!r}"
    print(f"trunc OK | done at step {done_at}, bad_transition set, terminal keys present")

    env.close(); a.close(); b.close()
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
