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
from pair_vec_env import (
    DummyPairVecEnv,
    JointDummyPairVecEnv,
    JointSubprocPairVecEnv,
    SubprocPairVecEnv,
)
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


class CurriculumCallback(BaseCallback):
    """Rolling-window curriculum: start at Level 1 (easy scoring chance),
    flip to Level 5 (chaos) once rolling success_rate clears `threshold`.

    Works with both VecEnv variants — env_method dispatches to all envs.
    """

    def __init__(
        self, start_level=1, target_level=5,
        threshold=0.9, window=300, verbose=1,
    ):
        super().__init__(verbose)
        self.start_level = int(start_level)
        self.target_level = int(target_level)
        self.threshold = float(threshold)
        self.window = int(window)
        self.success_buffer = deque(maxlen=self.window)
        self.current_level = self.start_level

    def _on_training_start(self) -> None:
        self.training_env.env_method(
            "set_curriculum_level", self.start_level
        )
        if self.verbose:
            print(f"[Curriculum] starting at level={self.start_level}")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for i, done in enumerate(dones):
            if done and "is_success" in infos[i]:
                self.success_buffer.append(float(infos[i]["is_success"]))
        if (
            self.current_level < self.target_level
            and len(self.success_buffer) >= self.window
        ):
            sr = float(np.mean(self.success_buffer))
            if sr >= self.threshold:
                self.current_level = self.target_level
                self.training_env.env_method(
                    "set_curriculum_level", self.current_level
                )
                self.success_buffer.clear()
                print(
                    f"[Curriculum] success_rate={sr:.2f} >= "
                    f"{self.threshold} → level={self.current_level}"
                )
        self.logger.record("curriculum/level", self.current_level)
        return True


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


def build_vec_env(n_envs, reward_type, seed, frozen_path, use_subproc, algo):
    fns = [
        make_env_fn(reward_type, seed + i, frozen_path)
        for i in range(n_envs)
    ]
    if algo == "masac":
        # Joint variants keep the 2-agent pairing intact so the replay buffer
        # stores joint transitions for the centralized critic.
        if use_subproc and n_envs > 1:
            return JointSubprocPairVecEnv(fns)
        return JointDummyPairVecEnv(fns)
    # Independent SAC: unstack each pair into 2N SB3 slots. Each slot is one
    # agent, sees its own (52,) obs, outputs its own (6,) action. Parameter
    # sharing happens automatically via the single shared policy.
    if use_subproc and n_envs > 1:
        return SubprocPairVecEnv(fns)
    return DummyPairVecEnv(fns)


def train(reward_type, seed, n_envs, frozen_path, init_path=None,
          total_steps=5_000_000, algo="masac"):
    assert algo in ("masac", "sac"), algo
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    algo_tag = algo.upper()
    run_name = f"2v2_selfplay_{algo_tag}_{reward_type}_seed{seed}_{timestamp}"
    log_dir = os.path.join(LOG_DIR, run_name)

    wandb.init(
        project="ssl-rl-2v2-selfplay",
        name=run_name,
        sync_tensorboard=True,
        config={
            "algo": algo_tag,
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
        frozen_path=frozen_path, use_subproc=True, algo=algo,
    )
    print(
        f"2v2 {algo_tag} self-play | frozen={frozen_path} | seed={seed} | "
        f"envs={n_envs} | vec_slots={env.num_envs}"
    )

    init_load = init_path or frozen_path

    policy_kwargs = dict(net_arch=[512, 512, 512])
    if algo == "masac":
        model = MASAC(
            policy=MASACPolicy, env=env, verbose=1, device="cuda",
            tensorboard_log=log_dir, seed=seed,
            train_freq=1, gradient_steps=1, batch_size=1024,
            buffer_size=200_000, learning_rate=1e-4,
            learning_starts=10000, ent_coef=0.05, target_entropy="auto",
            critic_warmup_grad_steps=0, max_grad_norm=10.0,
            policy_kwargs=policy_kwargs, gamma=0.995,
        )
    else:
        # Independent SAC diagnostic: stock SB3 SAC over per-agent (52,)
        # slots — the warmup setup, but with the current world-frame obs
        # and current reward / kick logic. If this scores against static
        # Blue and MASAC does not, the issue is in the custom MASAC stack,
        # not in env/reward/kick.
        from stable_baselines3 import SAC
        model = SAC(
            policy="MlpPolicy", env=env, verbose=1, device="cuda",
            tensorboard_log=log_dir, seed=seed,
            train_freq=1, gradient_steps=1, batch_size=2048,
            buffer_size=200_000, learning_rate=3e-4,
            learning_starts=10000, ent_coef=0.05, target_entropy="auto",
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
        CurriculumCallback(start_level=1, target_level=5, threshold=0.9),
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
    parser.add_argument("--algo", default="masac", choices=["masac", "sac"],
                        help="masac: custom CTDE (default). "
                             "sac: stock Independent SAC diagnostic.")
    args = parser.parse_args()

    train(
        reward_type=args.reward_type, seed=args.seed,
        n_envs=args.n_pairs, frozen_path=args.frozen_path,
        init_path=args.init_path, total_steps=args.total_steps,
        algo=args.algo,
    )
