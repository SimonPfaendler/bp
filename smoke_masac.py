"""Smoke test for the MASAC / CTDE stack.

Builds a tiny JointDummyPairVecEnv, runs MASAC.learn past learning_starts so
both the random-action rollout path and the gradient train() path execute,
then exercises save/load and predict. Catches SB3-internals breakage before
spending a SLURM slot.
"""

import numpy as np

from masac import MASAC
from masac_policy import MASACPolicy
from pair_vec_env import JointDummyPairVecEnv
from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv


def make_env_fn(seed):
    def _init():
        env = SSL2v2SelfPlayEnv(reward_type="dense", frozen_path=None)
        env.reset(seed=seed)
        return env
    return _init


def main():
    n_pairs = 2
    env = JointDummyPairVecEnv([make_env_fn(s) for s in range(n_pairs)])
    print(f"vec env: num_envs={env.num_envs}, obs={env.observation_space}, "
          f"act={env.action_space}")

    model = MASAC(
        policy=MASACPolicy, env=env, verbose=1, device="cpu", seed=0,
        train_freq=1, gradient_steps=1, batch_size=32,
        buffer_size=5000, learning_rate=3e-4,
        learning_starts=120, ent_coef=0.2, target_entropy="auto",
        policy_kwargs=dict(net_arch=[64, 64]), gamma=0.99,
    )
    print("target_entropy:", model.target_entropy)
    print("actor:", type(model.actor).__name__,
          "| critic:", type(model.critic).__name__)

    # Run past learning_starts so train() executes at least a few grad steps.
    model.learn(total_timesteps=400, log_interval=1)
    print("learn() OK")

    # Save / load round-trip.
    model.save("/tmp/masac_smoke")
    reloaded = MASAC.load("/tmp/masac_smoke", device="cpu")
    print("save/load OK")

    # Predict on a single joint obs (2, 38) -> (2, 6).
    obs = env.reset()
    print("reset obs shape:", obs.shape)
    action, _ = reloaded.predict(obs[0], deterministic=True)
    print("predict single (2,38) -> action:", action.shape)
    assert action.shape == (2, 6), action.shape

    # Predict on the full vec batch (n_pairs, 2, 38) -> (n_pairs, 2, 6).
    action_batch, _ = reloaded.predict(obs, deterministic=True)
    print("predict batch (n,2,38) -> action:", action_batch.shape)
    assert action_batch.shape == (n_pairs, 2, 6), action_batch.shape

    # One env step with the predicted action.
    obs2, r, d, infos = env.step(action_batch)
    print("step OK | obs:", obs2.shape, "| reward:", r.shape, "| dones:", d.shape)

    env.close()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
