import gym
from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
import os
import argparse
import time
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import math
from ssl_rl_1v1_continuous import SSL1v1ContinuousEnv

model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(sb3_algo, load_path=None):
    env = SSL1v1ContinuousEnv()

    if load_path and os.path.exists(load_path):
        print(f"Lade existierendes Modell von {load_path} zum Weitertrainieren...")
        match sb3_algo:
            case 'PPO':
                model = PPO.load(load_path, env=env, device='cuda', tensorboard_log=log_dir)
            case 'SAC':
                model = SAC.load(load_path, env=env, device='cuda', tensorboard_log=log_dir)
            case 'TD3':
                model = TD3.load(load_path, env=env, device='cuda', tensorboard_log=log_dir)
            case 'A2C':
                model = A2C.load(load_path, env=env, device='cuda', tensorboard_log=log_dir)
            case 'DDPG':
                model = DDPG.load(load_path, env=env, device='cuda', tensorboard_log=log_dir)
            case _:
                print('Algorithmus nicht gefunden')
                return
    else:
        print(f"Starte neues Training mit {sb3_algo}...")
        match sb3_algo:
            case 'PPO':
                model = PPO(
                    'MlpPolicy', 
                    env,       
                    verbose=1, 
                    device='cuda', 
                    tensorboard_log=log_dir

                )
            case 'SAC':
                model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
            case 'TD3':
                n_actions = env.action_space.shape[-1]
                model = TD3(
                    'MlpPolicy', 
                    env, 
                    verbose=1, 
                    device='cuda', 
                    tensorboard_log=log_dir,
                    learning_starts=20000,
                    action_noise= NormalActionNoise(mean=np.zeros(n_actions), sigma=0.4 * np.ones(n_actions))                
                )
            case 'A2C':
                model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
            case 'DDPG':
                model = DDPG(
                    'MlpPolicy', 
                    env, 
                    verbose=1, 
                    device='cuda', 
                    tensorboard_log=log_dir,
                    learning_starts=20000
                 )
            case _:
                print(f'Algorithmus {sb3_algo} nicht unterstützt.')
                return

    TIMESTEPS = 25000
    iters = 0
    
    
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        save_path = f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}"
        model.save(save_path)
        print(f"Modell gespeichert unter: {save_path}")

def test(sb3_algo, path_to_model):
    env = SSL1v1ContinuousEnv()
    
    match sb3_algo:
        case 'PPO':
            model = PPO.load(path_to_model, env=env)
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'TD3':
            model = TD3.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case 'DDPG':
            model = DDPG.load(path_to_model, env=env)
        case _:
            print('Algorithmus nicht gefunden')
            return

    obs = env.reset()
    
    print(f"Teste Modell: {path_to_model}")
    summe = 0.0
    
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        
        env.render()
        time.sleep(0.025)
        summe += reward
        print(f"Reward: {reward:.2f}, Gesamt: {summe:.2f} ")

        if done:
            print("\nEpisode done")
            summe = 0.0
            obs = env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test SSL 1v1 Model.')
    parser.add_argument('sb3_algo', help='RL Algorithmus (PPO, SAC, TD3, A2C, DDPG)')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    if args.train:
        path = "models/SAC_4000000.zip"  # Path to model for continued training
        train(args.sb3_algo, load_path=path if os.path.isfile(path) else None)

    if args.test:
        if os.path.isfile(args.test):
            test(args.sb3_algo, path_to_model=args.test)
        else:
            print(f'Datei {args.test} wurde nicht gefunden.')