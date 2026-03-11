import time
from stable_baselines3 import DDPG
from ssl_go_to_ball import SSLGoToBallEnv
import warnings
warnings.filterwarnings("ignore")

print("Lade Spielfeld und Agent...")
env = SSLGoToBallEnv()
model = DDPG.load("ddpg_ssl_goto_ball")

# Wir lassen ihn 3 Matches spielen
for episode in range(3):
    obs = env.reset()
    done = False
    schritte = 0
    
    print(f"Match {episode+1} startet!")
    while not done and schritte < 500:
        # deterministic=True zwingt ihn, sein bestes Wissen abzurufen
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        env.render()
        time.sleep(0.03) # Simulation bremsen
        schritte += 1

env.close()