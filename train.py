import gymnasium as gym
import torch
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from sb3_contrib import CrossQ
import os
import argparse
import time
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
import datetime
import numpy as np
import math
import wandb
from ssl_rl_1v1_continuous import SSL1v1ContinuousEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
torch.set_num_threads(slurm_cpus)

model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps
        
        if progress < 0.04:
            level = 1 
        elif progress < 0.12:
            level = 2
        elif progress < 0.60:
            level = 3
        else:
            level = 4
            

        self.training_env.env_method("set_curriculum_level", level)
        

        self.logger.record("curriculum/level", level)
        return True

def train(sb3_algo, action_type, reward_type, seed, load_path=None):

    log_freq = 50
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
        model = algo_class.load(load_path, env=env, device='auto', tensorboard_log=current_log_dir, custom_objects={'learning_rate': 0.0001})
    else:
        print("Start new Training")

        custom_policy_kwargs = dict(net_arch=[512, 512])
        if sb3_algo == 'CrossQ':
            model = CrossQ('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=current_log_dir, seed=seed,
                            train_freq=24,
                            gradient_steps=24,
                            batch_size=4096,
                            buffer_size=500000,
                            learning_rate=0.0001,
                            ent_coef='auto',
                            target_entropy= -4.0,
                            policy_kwargs=custom_policy_kwargs,
                            gamma=0.99)
        elif sb3_algo == 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=current_log_dir, seed=seed,
                        train_freq=24,
                        gradient_steps=12,
                        batch_size=4096,
                        policy_kwargs=custom_policy_kwargs,
                        buffer_size=500000,
                        learning_rate=0.0001,
                        ent_coef='auto',
                        target_entropy= -4.0,
                        tau=0.01,
                        gamma=0.99
                    )


        elif sb3_algo == 'PPO':
            model = PPO('MlpPolicy', env, verbose=1, device='auto', tensorboard_log=current_log_dir, seed=seed)

        else:
            print(f"Algo {sb3_algo} nicht gefunden")
            return

    TOTAL_STEPS = 19500000

    curriculum_callback = CurriculumCallback(total_timesteps=TOTAL_STEPS)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=model_dir,
        name_prefix=run_name,
        save_replay_buffer=True
    )

    
    callback_list = CallbackList([curriculum_callback, checkpoint_callback])
    
    
    model.learn(
        total_timesteps=TOTAL_STEPS, 
        reset_num_timesteps=False, 
        log_interval=log_freq,
        callback=callback_list
    )
    final_save_path = f"{model_dir}/{run_name}_final"
    model.save(final_save_path)
    print(f"Training done: {final_save_path}")

def test(sb3_algo, action_type, reward_type, path_to_model):
    env = SSL1v1ContinuousEnv(action_type=action_type, reward_type=reward_type, render_mode="human")
    
    algo_class = CrossQ if sb3_algo == 'CrossQ' else globals()[sb3_algo]
    model = algo_class.load(path_to_model, env=env, device='cpu')
    obs, info = env.reset()
    
    print(f"Test Model: {path_to_model}")
    summe = 0.0
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        env.render()
        time.sleep(0.025)
        summe += reward
        print(f"Reward: {reward:.2f}, Gesamt: {summe:.2f} ")

        if done:
            print("\nEpisode done")
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
    args = parser.parse_args()

    if args.train:
        path = ""  # Path to model for continued training
        train(args.sb3_algo, args.action_type, args.reward_type, args.seed, load_path=path if os.path.isfile(path) else None)


    if args.test:
        if os.path.isfile(args.test):
            test(args.sb3_algo, args.action_type, args.reward_type, path_to_model=args.test)
        else:
            print(f'Datei {args.test} wurde nicht gefunden.')
