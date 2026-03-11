import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import warnings
from gym.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

# 1. Importiere DEINE maßgeschneiderte Umgebung aus der anderen Datei!
from ssl_go_to_ball import SSLGoToBallEnv

warnings.filterwarnings("ignore")

def main():
    print("SSL Trainings-Umgebung...")
    env = SSLGoToBallEnv()

    env = TimeLimit(env, max_episode_steps=500)
    env = Monitor(env)

    # 2. Das physikalische Rauschen (Exploration Noise)
    # Ornstein-Uhlenbeck simuliert eine Art "Trägheit" beim Rauschen,
    # was für Roboter-Motoren viel besser funktioniert als reiner Zufall.
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=float(0.2) * np.ones(n_actions)
    )

    print("Baue DDPG-Agenten (Actor-Critic Netzwerk)...")
    # 3. Der Algorithmus
    model = DDPG(
        "MlpPolicy", # Nutzt klassische Feed-Forward Neural Networks
        env,
        action_noise=action_noise,
        verbose=1,
        tensorboard_log="./ssl_tensorboard/",
        learning_starts=1000,
        batch_size=256,
        buffer_size=100000
    )

    print("Starte Training")
    # 4. Der eigentliche Trainings-Loop
    # 50.000 Schritte sind ein guter erster Testlauf.
    model.learn(total_timesteps=50000, log_interval=10)

    # 5. Gehirn auf der Festplatte speichern
    model.save("ddpg_ssl_goto_ball")
    print("Training beendet! Agent erfolgreich als 'ddpg_ssl_goto_ball.zip' gespeichert.")

    env.close()

if __name__ == "__main__":
    main()