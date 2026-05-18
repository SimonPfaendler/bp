import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from sb3_contrib import CrossQ
import os
import shutil
import argparse
import time
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
import datetime
import numpy as np
import math
import random
from collections import deque
import wandb
from ssl_rl_1v1_continuous import SSL1v1ContinuousEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
print(f"Detected CPUs: {slurm_cpus}")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print(f"Torch threads: {torch.get_num_threads()}")
print(f"Torch inter-op threads: {torch.get_num_interop_threads()}")
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")


model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

class CurriculumCallback(BaseCallback):
    def __init__(self, start_level=1, verbose=0):
        super().__init__(verbose)
        self.current_level = start_level
        self.success_buffer = deque(maxlen=150)

    def _on_training_start(self) -> None:
        if self.current_level > 1:
            self.training_env.env_method("set_curriculum_level", self.current_level)
        
    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        
        for i, done in enumerate(dones):
            if done:
                is_success = float(infos[i].get("is_success", 0.0))
                self.success_buffer.append(is_success)

        if len(self.success_buffer) == self.success_buffer.maxlen and self.current_level < 5:
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
                self.training_env.env_method("set_curriculum_level", self.current_level)

        self.logger.record("curriculum/level", self.current_level)
        if len(self.success_buffer) > 0:
            self.logger.record("curriculum/live_success_rate", np.mean(self.success_buffer))

        return True

class PoolSnapshotCallback(BaseCallback):
    """Snapshot the live policy into the opponent pool every `save_freq` steps.

    Atomic write: save to .tmp.zip then rename so env workers (which scan the
    dir on reset) never see a half-written file.
    """
    def __init__(self, save_freq, pool_dir, name_prefix, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.pool_dir = pool_dir
        self.name_prefix = name_prefix
        os.makedirs(pool_dir, exist_ok=True)
        self.next_save_step = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_save_step:
            tmp = os.path.join(self.pool_dir, f"{self.name_prefix}_{self.num_timesteps}.tmp.zip")
            final = os.path.join(self.pool_dir, f"{self.name_prefix}_{self.num_timesteps}.zip")
            self.model.save(tmp)
            os.replace(tmp, final)
            if self.verbose:
                print(f"[pool] snapshot -> {final}")
            self.next_save_step += self.save_freq
        return True


def train(sb3_algo, action_type, reward_type, seed, load_path=None, start_level=1,
          selfplay=False, opponent_path=None, pool_snapshot_freq=200_000):

    log_freq = 10
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{sb3_algo}_{action_type}_{reward_type}_seed{seed}_{timestamp}"
    current_log_dir = os.path.join(log_dir, run_name)

    run = wandb.init(
        project="ssl-rl-1v1",
        name=run_name,
        sync_tensorboard=True,
        config={
            "algo": sb3_algo,
            "action_type": action_type,
            "reward_type": reward_type,
            "seed": seed,
            "cpus": slurm_cpus
        }
    )
    env_kwargs = dict(action_type=action_type, reward_type=reward_type)

    pool_dir = None
    if selfplay:
        if not opponent_path or not os.path.isfile(opponent_path):
            raise ValueError(f"--selfplay requires --opponent <path>, got {opponent_path!r}")
        pool_dir = os.path.join(model_dir, f"pool_{run_name}")
        os.makedirs(pool_dir, exist_ok=True)
        # Seed the pool so v0 is sampled alongside future snapshots.
        seed_dst = os.path.join(pool_dir, f"v0_{os.path.basename(opponent_path)}")
        if not os.path.isfile(seed_dst):
            shutil.copyfile(opponent_path, seed_dst)
        env_kwargs.update(
            blue_mode="selfplay",
            opponent_pool_dir=pool_dir,
            seed_opponent_path=opponent_path,
        )
        print(f"Self-play enabled. Pool dir: {pool_dir} (seeded with v0)")


    env = make_vec_env(
        SSL1v1ContinuousEnv,
        n_envs=slurm_cpus,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        monitor_kwargs={"info_keywords": ("is_success", "match_result", "possession_ratio")}
    )
    print(f"Starte Training: {sb3_algo} | Modus: {action_type} | Reward: {reward_type} | Seed: {seed}")

    if load_path and os.path.exists(load_path):
        print(f"Lade existierendes Modell von {load_path} zum Weitertrainieren...")
        algo_class = CrossQ if sb3_algo == 'CrossQ' else globals()[sb3_algo]
        new_ent_coef = 0.05
        model = algo_class.load(load_path, env=env, device='auto', tensorboard_log=current_log_dir,
                                custom_objects={'learning_rate': 0.0003, 'ent_coef': new_ent_coef})
        if sb3_algo == 'SAC' and hasattr(model, 'ent_coef_tensor'):
            model.ent_coef_tensor = torch.tensor(float(new_ent_coef), device=model.device)
            print(f"Overwrote ent_coef_tensor -> {new_ent_coef}")
    else:
        print("Start new Training")

        custom_policy_kwargs = dict(net_arch=[512, 512, 512])
        if sb3_algo == 'CrossQ':
            model = CrossQ('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=current_log_dir, seed=seed,
                            train_freq=48,
                            gradient_steps=96,
                            batch_size=2048,
                            buffer_size=1_000_000,
                            learning_rate=3e-4,
                            learning_starts=10000,
                            ent_coef=0.05,
                            target_entropy='auto',
                            policy_kwargs=custom_policy_kwargs)
        elif sb3_algo == 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=current_log_dir, seed=seed,
                        train_freq=48,
                        gradient_steps=96,
                        batch_size=2048,
                        policy_kwargs=custom_policy_kwargs,
                        buffer_size=1_000_000,
                        learning_rate=3e-4,
                        learning_starts=10000,
                        ent_coef=0.05,
                        target_entropy='auto',
                        gamma=0.99
                    )


        elif sb3_algo == 'PPO':
            model = PPO('MlpPolicy', env, verbose=1, device='auto', tensorboard_log=current_log_dir, seed=seed)

        else:
            print(f"Algo {sb3_algo} nicht gefunden")
            return

    TOTAL_STEPS = 3800000

    curriculum_callback = CurriculumCallback(start_level=start_level)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=model_dir,
        name_prefix=run_name,
        save_replay_buffer=False
    )

    
    callbacks = [curriculum_callback, checkpoint_callback]
    if selfplay:
        callbacks.append(PoolSnapshotCallback(
            save_freq=pool_snapshot_freq,
            pool_dir=pool_dir,
            name_prefix=run_name,
            verbose=1,
        ))
    callback_list = CallbackList(callbacks)
    
    
    model.learn(
        total_timesteps=TOTAL_STEPS, 
        reset_num_timesteps=False, 
        log_interval=log_freq,
        callback=callback_list
    )
    final_save_path = f"{model_dir}/{run_name}_final"
    model.save(final_save_path)
    print(f"Training done: {final_save_path}")

def test(sb3_algo, action_type, reward_type, path_to_model, test_level=4,
         selfplay=False, opponent_path=None, opponent_pool_dir=None):
    env_kwargs = dict(action_type=action_type, reward_type=reward_type, render_mode="human")
    if selfplay:
        if not opponent_path and not opponent_pool_dir:
            raise ValueError("--selfplay needs --opponent <path> and/or --opponent_pool_dir <dir>")
        env_kwargs.update(
            blue_mode="selfplay",
            opponent_pool_dir=opponent_pool_dir,
            seed_opponent_path=opponent_path,
        )
    env = SSL1v1ContinuousEnv(**env_kwargs)
    if hasattr(env, 'set_curriculum_level'):
        env.set_curriculum_level(test_level)
    algo_class = CrossQ if sb3_algo == 'CrossQ' else globals()[sb3_algo]
    model = algo_class.load(path_to_model, env=env, device='cpu')
    obs, info = env.reset()

    print(f"Test Model: {path_to_model}")
    if selfplay:
        print(f"Blue opponent: {env._opponent_path}")
    summe = 0.0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env.render()
        time.sleep(0.025)
        summe += reward
        print(f"Reward: {reward:.2f}, Gesamt: {summe:.2f}, Action: {action}", end="\r")

        if done:
            print("\nEpisode done")
            if selfplay:
                print(f"Next opponent: {env._opponent_path}")
            summe = 0.0
            obs, info = env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test SSL 1v1 Model.')
    parser.add_argument('sb3_algo', help='RL Algorithmus (PPO, SAC, TD3, A2C, DDPG)')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')

    parser.add_argument('--action_type', choices=['skills', 'low_level'], default='skills')
    parser.add_argument('--reward_type', choices=['sparse', 'dense'], default='dense')
    parser.add_argument('--seed', type=int, default=0, help='Zufalls-Seed für das Training')
    parser.add_argument('--start_level', type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help='Curriculum-Level beim Start (1-5). Nützlich beim Weitertrainieren.')
    parser.add_argument('--selfplay', action='store_true',
                        help='Enable 1v1 self-play; blue is a frozen SAC sampled from the pool.')
    parser.add_argument('--opponent', type=str, default=None,
                        help='Seed opponent .zip (also used as the v0 entry in the pool).')
    parser.add_argument('--pool_snapshot_freq', type=int, default=200_000,
                        help='Snapshot the live policy into the opponent pool every N steps.')
    parser.add_argument('--opponent_pool_dir', type=str, default=None,
                        help='Directory of frozen .zip opponents (test mode: sampled each episode).')
    parser.add_argument('--test_level', type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help='Curriculum level used by --test (default 5).')
    args = parser.parse_args()

    if args.train:
        path = ""  # Path to model for continued training
        if args.selfplay and not path:
            # Common case: warm-start training from the seed opponent.
            path = args.opponent or ""
        train(args.sb3_algo, args.action_type, args.reward_type, args.seed,
              load_path=path if os.path.isfile(path) else None,
              start_level=args.start_level,
              selfplay=args.selfplay,
              opponent_path=args.opponent,
              pool_snapshot_freq=args.pool_snapshot_freq)


    if args.test:
        if os.path.isfile(args.test):
            test(args.sb3_algo, args.action_type, args.reward_type, path_to_model=args.test,
                 test_level=args.test_level,
                 selfplay=args.selfplay,
                 opponent_path=args.opponent,
                 opponent_pool_dir=args.opponent_pool_dir)
        else:
            print(f'Datei {args.test} wurde nicht gefunden.')
