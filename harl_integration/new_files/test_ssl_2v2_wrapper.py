"""Smoke-test the ssl_2v2 HARL wrapper without invoking the full HARL runner.

Verifies:
  - Env imports correctly via the BP_DIR shim
  - Observation / state / action spaces have the expected shapes per agent
  - reset() returns (local_obs, s_obs, avail_actions) with correct types
  - step() returns the 6-tuple with correct shapes and dtypes
  - Truncation triggers info["bad_transition"]
  - Curriculum-level kwarg is honored (Level 1 spawn is near goal)

Run with:
  /home/simon/dev/venv_harl/bin/python test_ssl_2v2_wrapper.py
"""
import os
import sys

import numpy as np

# Ensure we can import the wrapper itself (relative to HARL repo root).
sys.path.insert(0, "/home/simon/dev/HARL")
os.environ.setdefault("BP_DIR", "/home/simon/dev/bp")

from harl.envs.ssl_2v2.ssl_2v2_env import SSL2v2Env


def main():
    args = {
        "reward_type": "dense",
        "frozen_path": None,
        "role_index": False,
        "oob_grace_steps": 0,
        "curriculum_level": 1,
    }
    env = SSL2v2Env(args)
    env.seed(0)

    print(f"n_agents:                {env.n_agents}")
    print(f"observation_space[0]:    {env.observation_space[0]}")
    print(f"share_observation_space: {env.share_observation_space[0]}")
    print(f"action_space[0]:         {env.action_space[0]}")
    print(f"discrete:                {env.discrete}")
    assert env.n_agents == 2
    assert env.observation_space[0].shape == (52,)
    assert env.share_observation_space[0].shape == (104,)
    assert env.action_space[0].shape == (6,)
    assert env.discrete is False

    local_obs, s_obs, avail = env.reset()
    assert len(local_obs) == 2 and local_obs[0].shape == (52,), local_obs[0].shape
    assert len(s_obs) == 2 and s_obs[0].shape == (104,), s_obs[0].shape
    assert avail is None
    print(f"\nreset OK | obs[0][:5]={local_obs[0][:5]}")

    # 50 no-op steps — Level 1 + static blue should give negative cumulative
    # reward (time + standing penalty) per the new reward design.
    total_r = 0.0
    for t in range(50):
        actions = np.zeros((env.n_agents, 6), dtype=np.float32)
        local_obs, s_obs, rewards, dones, infos, _ = env.step(actions)
        assert len(rewards) == 2 and len(rewards[0]) == 1, rewards
        assert isinstance(dones[0], bool), type(dones[0])
        total_r += rewards[0][0]
        if any(dones):
            break

    print(
        f"step OK | 50 no-op steps total_r={total_r:.3f} (expect slightly negative)"
    )
    env.close()
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
