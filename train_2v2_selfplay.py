"""Train SAC self-play in 2v2 SSL.

Blue side controlled by a frozen SAC checkpoint (no gradients). Yellow side
is the training side; same checkpoint is loaded as yellow initialization for
the first iteration so both teams start at parity.

For subsequent iterations, set --frozen_path to the previous run's _final.zip
to keep climbing the response ladder.
"""

import argparse
import datetime
import os
import time
from collections import deque

import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from pair_vec_env import DummyPairVecEnv, SubprocPairVecEnv
from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
print(f"Detected CPUs: {slurm_cpus}")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env_fn(reward_type, seed, frozen_path):
    def _init():
        env = SSL2v2SelfPlayEnv(
            reward_type=reward_type, frozen_path=frozen_path
        )
        env.reset(seed=seed)
        return env
    return _init


class StatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_buffer = deque(maxlen=300)
        self.passes_buffer = deque(maxlen=300)
        self.scored_after_pass_buffer = deque(maxlen=300)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, done in enumerate(dones):
            if not done:
                continue
            if "is_success" in infos[i]:
                self.success_buffer.append(float(infos[i]["is_success"]))
            if "passes" in infos[i]:
                self.passes_buffer.append(float(infos[i]["passes"]))
            if "scored_after_pass" in infos[i]:
                self.scored_after_pass_buffer.append(
                    float(infos[i]["scored_after_pass"])
                )
        if self.success_buffer:
            self.logger.record(
                "selfplay/live_success_rate",
                float(np.mean(self.success_buffer)),
            )
        if self.passes_buffer:
            self.logger.record(
                "rollout/passes_per_episode",
                float(np.mean(self.passes_buffer)),
            )
        if self.scored_after_pass_buffer:
            self.logger.record(
                "rollout/scored_after_pass_rate",
                float(np.mean(self.scored_after_pass_buffer)),
            )
        return True


def build_vec_env(n_envs, reward_type, seed, frozen_path, use_subproc):
    fns = [
        make_env_fn(reward_type, seed + i, frozen_path)
        for i in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        return SubprocPairVecEnv(fns)
    return DummyPairVecEnv(fns)


def train(reward_type, seed, n_envs, frozen_path, init_path=None,
          total_steps=5_000_000):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"2v2_selfplay_SAC_{reward_type}_seed{seed}_{timestamp}"
    log_dir = os.path.join(LOG_DIR, run_name)

    wandb.init(
        project="ssl-rl-2v2-selfplay",
        name=run_name,
        sync_tensorboard=True,
        config={
            "algo": "SAC",
            "reward_type": reward_type,
            "seed": seed,
            "n_envs": n_envs,
            "frozen_path": frozen_path,
            "init_path": init_path,
            "total_steps": total_steps,
        },
    )

    env = build_vec_env(
        n_envs=n_envs, reward_type=reward_type, seed=seed,
        frozen_path=frozen_path, use_subproc=True,
    )
    print(
        f"2v2 self-play | frozen={frozen_path} | seed={seed} | "
        f"envs={n_envs} | slots={env.num_envs}"
    )

    init_load = init_path or frozen_path
    if init_load and os.path.exists(init_load):
        print(f"Loading yellow init from {init_load}")
        model = SAC.load(
            init_load, env=env, device="auto",
            tensorboard_log=log_dir,
            custom_objects={"learning_rate": 3e-4, "ent_coef": 0.10},
        )
    else:
        print(f"No init checkpoint at {init_load}; training from scratch")
        policy_kwargs = dict(net_arch=[512, 512, 512])
        model = SAC(
            policy="MlpPolicy", env=env, verbose=1, device="cuda",
            tensorboard_log=log_dir, seed=seed,
            train_freq=1, gradient_steps=1, batch_size=2048,
            buffer_size=1_000_000, learning_rate=3e-4,
            learning_starts=10000, ent_coef=0.10, target_entropy="auto",
            policy_kwargs=policy_kwargs, gamma=0.99,
        )

    callbacks = CallbackList([
        StatsCallback(),
        CheckpointCallback(
            save_freq=20000, save_path=MODEL_DIR,
            name_prefix=run_name, save_replay_buffer=True,
        ),
    ])

    model.learn(
        total_timesteps=total_steps,
        reset_num_timesteps=False,
        log_interval=10,
        callback=callbacks,
    )
    final = f"{MODEL_DIR}/{run_name}_final"
    model.save(final)
    model.save_replay_buffer(f"{final}_replay_buffer")
    print(f"Saved {final}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train 2v2 SAC self-play with frozen opponent."
    )
    parser.add_argument("--frozen_path", required=True,
                        help="Path to frozen SAC checkpoint for blue side.")
    parser.add_argument("--init_path", default=None,
                        help="Yellow init checkpoint. Defaults to frozen_path.")
    parser.add_argument("--reward_type", default="dense",
                        choices=["sparse", "dense"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_pairs", type=int,
                        default=max(1, slurm_cpus // 2))
    parser.add_argument("--total_steps", type=int, default=5_000_000)
    args = parser.parse_args()

    train(
        reward_type=args.reward_type, seed=args.seed,
        n_envs=args.n_pairs, frozen_path=args.frozen_path,
        init_path=args.init_path, total_steps=args.total_steps,
    )
