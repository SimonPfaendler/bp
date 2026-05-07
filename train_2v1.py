"""Train a parameter-sharing SAC for 2v1 (2 yellow attackers vs 1 blue heuristic).

Usage:
  python train_2v1.py SAC -t                 # start training
  python train_2v1.py SAC -s models/run_final.zip   # render a trained model

Both yellow agents share a single SAC policy. The PairVecEnv wraps each
physics simulator into 2 SB3 agent slots, so both agents' transitions
flow into the same replay buffer.
"""

import argparse
import datetime
import os
import time
from collections import deque

import numpy as np
import torch
import wandb
from sb3_contrib import CrossQ
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from pair_vec_env import DummyPairVecEnv, SubprocPairVecEnv
from ssl_rl_2v1_continuous import SSL2v1SharedEnv

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


def make_env_fn(reward_type: str, seed: int):
    def _init():
        env = SSL2v1SharedEnv(reward_type=reward_type)
        env.reset(seed=seed)
        return env

    return _init


class CurriculumCallback(BaseCallback):
    """Same logic as 1v1, but is_success now appears twice per episode
    (once per agent slot). The success-rate computation still works:
    duplicates have identical values, so the running mean is unchanged.
    """

    def __init__(self, start_level=1, verbose=0):
        super().__init__(verbose)
        self.current_level = start_level
        self.success_buffer = deque(maxlen=300)

    def _on_training_start(self) -> None:
        if self.current_level > 1:
            self.training_env.env_method(
                "set_curriculum_level", self.current_level
            )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, done in enumerate(dones):
            if done and "is_success" in infos[i]:
                self.success_buffer.append(float(infos[i]["is_success"]))

        if (
            len(self.success_buffer) == self.success_buffer.maxlen
            and self.current_level < 5
        ):
            success_rate = np.mean(self.success_buffer)
            new_level = self.current_level
            if self.current_level == 1 and success_rate >= 0.90:
                new_level = 2
            elif self.current_level == 2 and success_rate >= 0.75:
                new_level = 3
            elif self.current_level == 3 and success_rate >= 0.60:
                new_level = 4
            elif self.current_level == 4 and success_rate >= 0.50:
                new_level = 5

            if new_level != self.current_level:
                self.current_level = new_level
                self.success_buffer.clear()
                self.training_env.env_method(
                    "set_curriculum_level", self.current_level
                )

        self.logger.record("curriculum/level", self.current_level)
        if len(self.success_buffer) > 0:
            self.logger.record(
                "curriculum/live_success_rate", float(np.mean(self.success_buffer))
            )
        return True


def build_vec_env(n_pairs: int, reward_type: str, seed: int, use_subproc: bool):
    fns = [make_env_fn(reward_type, seed + i) for i in range(n_pairs)]
    if use_subproc and n_pairs > 1:
        return SubprocPairVecEnv(fns)
    return DummyPairVecEnv(fns)


def train(sb3_algo, reward_type, seed, n_pairs, load_path=None, start_level=1):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"2v1_{sb3_algo}_{reward_type}_seed{seed}_{timestamp}"
    current_log_dir = os.path.join(LOG_DIR, run_name)

    wandb.init(
        project="ssl-rl-2v1",
        name=run_name,
        sync_tensorboard=True,
        config={
            "algo": sb3_algo, "reward_type": reward_type,
            "seed": seed,
            "n_pairs": n_pairs,
            "n_agent_slots": 2 * n_pairs,
        },
    )

    env = build_vec_env(
        n_pairs=n_pairs, reward_type=reward_type, seed=seed, use_subproc=True
    )
    print(
        f"Training {sb3_algo} | reward={reward_type} | seed={seed} | "
        f"pairs={n_pairs} | slots={env.num_envs}"
    )

    if load_path and os.path.exists(load_path):
        print(f"Loading model {load_path}")
        algo_class = CrossQ if sb3_algo == "CrossQ" else SAC
        new_ent = 0.05
        model = algo_class.load(
            load_path,
            env=env,
            device="auto",
            tensorboard_log=current_log_dir,
            custom_objects={"learning_rate": 3e-4, "ent_coef": new_ent},
        )
        if sb3_algo == "SAC" and hasattr(model, "ent_coef_tensor"):
            model.ent_coef_tensor = torch.tensor(
                float(new_ent), device=model.device
            )
    else:
        policy_kwargs = dict(net_arch=[512, 512, 512])
        common = dict(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device="cuda",
            tensorboard_log=current_log_dir,
            seed=seed,
            train_freq=48,
            gradient_steps=96,
            batch_size=2048,
            buffer_size=1_000_000,
            learning_rate=3e-4,
            learning_starts=10000,
            ent_coef=0.05,
            target_entropy="auto",
            policy_kwargs=policy_kwargs,
        )
        if sb3_algo == "CrossQ":
            model = CrossQ(**common)
        elif sb3_algo == "SAC":
            model = SAC(gamma=0.99, **common)
        else:
            raise ValueError(f"Unsupported algo for 2v1: {sb3_algo}")

    callbacks = CallbackList(
        [
            CurriculumCallback(start_level=start_level),
            CheckpointCallback(
                save_freq=20000,
                save_path=MODEL_DIR,
                name_prefix=run_name,
                save_replay_buffer=False,
            ),
        ]
    )

    TOTAL_STEPS = 5_000_000
    model.learn( total_timesteps=TOTAL_STEPS,
        reset_num_timesteps=False,
        log_interval=10,
        callback=callbacks,
    )
    final = f"{MODEL_DIR}/{run_name}_final"
    model.save(final)
    print(f"Saved {final}")


def test(sb3_algo, reward_type, model_path, test_level=4):
    """Render one trained shared policy controlling both yellows."""
    env = SSL2v1SharedEnv(reward_type=reward_type, render_mode="human")
    env.set_curriculum_level(test_level)
    env.reset()

    algo_class = CrossQ if sb3_algo == "CrossQ" else SAC
    model = algo_class.load(model_path, device="cpu")
    obs, _ = env.reset()

    total = 0.0
    while True:
        # obs is (2, OBS); SB3 predict expects batched shape, which works directly.
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, truncated, info = env.step(actions)
        env.render()
        time.sleep(0.025)
        total += float(rewards[0])
        print(f"r={rewards[0]:.2f} sum={total:.2f}", end="\r")
        if done or truncated:
            print(f"\nepisode end: {info}")
            total = 0.0
            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or test SSL 2v1 parameter-sharing model."
    )
    parser.add_argument("sb3_algo", choices=["SAC", "CrossQ"])
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-s", "--test", metavar="path_to_model")
    parser.add_argument(
        "--reward_type", choices=["sparse", "dense"], default="dense"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--start_level", type=int, default=1, choices=[1, 2, 3, 4, 5]
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=max(1, slurm_cpus // 2),
        help="Number of pair envs (each = 1 physics sim, 2 agent slots).",
    )
    parser.add_argument(
        "--test_level", type=int, default=5, choices=[1, 2, 3, 4, 5]
    )
    args = parser.parse_args()

    if args.train:
        train(
            sb3_algo=args.sb3_algo,
            reward_type=args.reward_type,
            seed=args.seed,
            n_pairs=args.n_pairs,
            start_level=args.start_level,
            load_path="models/2v1_SAC_dense_seed820_20260507-170258_14361600_steps.zip",
        )
    if args.test:
        if os.path.isfile(args.test):
            test(
                sb3_algo=args.sb3_algo,
                reward_type=args.reward_type,
                model_path=args.test,
                test_level=args.test_level,
            )
        else:
            print(f"file {args.test} not found")
