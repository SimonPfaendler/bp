"""Train MASAC self-play in 2v2 SSL (CTDE — centralized critic).

Blue side controlled by a frozen SAC/MASAC checkpoint (no gradients). Yellow
side is the training side; same checkpoint is loaded as yellow initialization
for the first iteration so both teams start at parity.

MASAC keeps a parameter-shared decentralized actor (38 -> 6, transfers cleanly
from prior SAC checkpoints) but replaces the per-agent critic with a
centralized one that scores the joint (2x38) obs + (2x6) action — giving the
policy gradient team-level credit assignment.

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
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from masac import MASAC
from masac_policy import MASACPolicy
from pair_vec_env import JointDummyPairVecEnv, JointSubprocPairVecEnv
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


def _load_any(path):
    """Load a checkpoint that may be a stock-SAC run or a MASAC iteration.

    SB3 stores the policy class in the zip, so SAC.load reconstructs whichever
    one it finds. MASAC.load is the fallback for any algorithm-level mismatch.
    The caller only reads `.policy.state_dict()`.
    """
    from stable_baselines3 import SAC

    try:
        return SAC.load(path, device="cpu")
    except Exception as e:
        print(f"  SAC.load failed ({e}); retrying with MASAC.load")
        return MASAC.load(path, device="cpu")


class StatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_buffer = deque(maxlen=300)
        self.blue_goal_buffer = deque(maxlen=300)
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
            if "blue_goal" in infos[i]:
                self.blue_goal_buffer.append(float(infos[i]["blue_goal"]))
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
        if self.blue_goal_buffer:
            self.logger.record(
                "selfplay/blue_goal_rate",
                float(np.mean(self.blue_goal_buffer)),
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
    # Joint variants keep the 2-agent pairing intact so the replay buffer
    # stores joint transitions for the centralized critic.
    if use_subproc and n_envs > 1:
        return JointSubprocPairVecEnv(fns)
    return JointDummyPairVecEnv(fns)


def train(reward_type, seed, n_envs, frozen_path, init_path=None,
          total_steps=5_000_000):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"2v2_selfplay_MASAC_{reward_type}_seed{seed}_{timestamp}"
    log_dir = os.path.join(LOG_DIR, run_name)

    wandb.init(
        project="ssl-rl-2v2-selfplay",
        name=run_name,
        sync_tensorboard=True,
        config={
            "algo": "MASAC",
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
        f"2v2 MASAC self-play | frozen={frozen_path} | seed={seed} | "
        f"envs={n_envs} | pairs={env.num_envs}"
    )

    init_load = init_path or frozen_path

    policy_kwargs = dict(net_arch=[512, 512, 512])
    model = MASAC(
        policy=MASACPolicy, env=env, verbose=1, device="cuda",
        tensorboard_log=log_dir, seed=seed,
        train_freq=1, gradient_steps=1, batch_size=2048,
        buffer_size=200_000, learning_rate=3e-4,
        learning_starts=10000, ent_coef=0.1, target_entropy="auto",
        critic_warmup_grad_steps=0, max_grad_norm=0.0,
        policy_kwargs=policy_kwargs, gamma=0.995,
    )
    if init_load and os.path.exists(init_load):
        print(f"Transferring actor weights from {init_load}")
        old_model = _load_any(init_load)
        new_state = model.policy.state_dict()
        old_state = old_model.policy.state_dict()
        transferred, skipped = [], []
        for k, v in old_state.items():
            if (
                k.startswith("actor.")
                and k in new_state
                and new_state[k].shape == v.shape
            ):
                new_state[k] = v
                transferred.append(k)
            else:
                skipped.append(k)
        model.policy.load_state_dict(new_state)
        print(
            f"Actor transfer: {len(transferred)} params copied, "
            f"{len(skipped)} skipped (critic + mismatches)"
        )
        del old_model
    else:
        print(f"No init checkpoint at {init_load}; training from scratch")

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
    parser.add_argument("--frozen_path", default=None,
                        help="Path to frozen SAC checkpoint for blue side. "
                             "Omit for stationary Blue (warmup phase).")
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
