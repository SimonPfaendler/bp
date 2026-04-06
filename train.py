import gymnasium as gym
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

model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

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
        }
    )


    env = SSL1v1ContinuousEnv(action_type=action_type, reward_type=reward_type)
    env = Monitor(env, info_keywords=("is_success", "match_result", "possession_ratio"))
    print(f"Starte Training: {sb3_algo} | Modus: {action_type} | Reward: {reward_type} | Seed: {seed}")

    if load_path and os.path.exists(load_path):
        print(f"Lade existierendes Modell von {load_path} zum Weitertrainieren...")
        algo_class = CrossQ if sb3_algo == 'CrossQ' else globals()[sb3_algo]
        model = algo_class.load(load_path, env=env, device='auto', tensorboard_log=current_log_dir)
    else:
        print("Start new Training")
        if sb3_algo == 'CrossQ':
            model = CrossQ('MlpPolicy', env, verbose=1, device='auto', tensorboard_log=current_log_dir, seed=seed)
        elif sb3_algo == 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='auto', tensorboard_log=current_log_dir, seed=seed)
        elif sb3_algo == 'PPO':
            model = PPO('MlpPolicy', env, verbose=1, device='auto', tensorboard_log=current_log_dir, seed=seed)

        else:
            print(f"Algo {sb3_algo} nicht gefunden")
            return

    TIMESTEPS = 100000
    iters = 0
    
    
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=log_freq)
        save_path = f"{model_dir}/{run_name}_{TIMESTEPS*iters}"
        model.save(save_path)
        print(f"Modell gespeichert unter: {save_path}")

def test(sb3_algo, action_type, reward_type, path_to_model):
    env = SSL1v1ContinuousEnv(action_type=action_type, reward_type=reward_type, render_mode="human")
    
    algo_class = CrossQ if sb3_algo == 'CrossQ' else globals()[sb3_algo]
    model = algo_class.load(path_to_model, env=env)
    obs, info = env.reset()
    
    print(f"Teste Modell: {path_to_model}")
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