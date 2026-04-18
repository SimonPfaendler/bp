import numpy as np
import math
import time
from gymnasium.spaces import Box
from pynput import keyboard
from rsoccer_gym.Entities import Robot
from ssl_rl_1v1_continuous import SSL1v1ContinuousEnv


class SSL1v1ManualEnv(SSL1v1ContinuousEnv):
    def __init__(self):
        super().__init__()
        self.action_space = Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    def _get_commands(self, actions):
        yellow = self.frame.robots_yellow[0]
        blue_robot_data = self.frame.robots_blue[0]
        v_x_global = actions[0] 
        v_y_global = actions[1] 
        v_theta = actions[2] 
        kick = 6.0 if actions[3] > 0.5 else 0.0
        dribble = True if actions[4] > 0.5 else False

        # Auto-release: mirror of SSL1v1ContinuousEnv._get_commands
        if self.must_release:
            dribble = False
        if not dribble and yellow.infrared and kick == 0.0:
            kick = 3.0

        robot_yellow = Robot(yellow=True, id=0,
                             v_x=v_x_global, v_y=v_y_global, v_theta=v_theta,
                             kick_v_x=kick, dribbler=dribble)

        difficulty = getattr(self, 'difficulty_factor', 0.2)
        b_cmd = np.zeros(5, dtype=np.float32)
        b_angle_rad = np.deg2rad(blue_robot_data.theta)
        bv_x, bv_y, bv_theta = self.convert_actions([b_cmd[0], b_cmd[1], b_cmd[2]], b_angle_rad)
        
        robot_blue = Robot(yellow=False, id=0, 
                           v_x=bv_x, v_y=bv_y, v_theta=bv_theta, 
                           kick_v_x=b_cmd[3], 
                           dribbler=True if b_cmd[4] > 0 else False)

        return [robot_blue, robot_yellow]

current_action = np.zeros(5, dtype=np.float32)
running = True

def on_press(key):
    global current_action, running
    try:
        if key.char == 'w': current_action[0] = 1.0
        elif key.char == 's': current_action[0] = -1.0
        elif key.char == 'a': current_action[1] = 1.0
        elif key.char == 'd': current_action[1] = -1.0
        elif key.char == 'q': current_action[2] = 1.0
        elif key.char == 'e': current_action[2] = -1.0
    except AttributeError:
        if key == keyboard.Key.space: current_action[3] = 1.0
        elif key == keyboard.Key.shift: current_action[4] = 1.0
        elif key == keyboard.Key.esc: running = False

def on_release(key):
    global current_action
    try:
        if key.char in ['w', 's']: current_action[0] = 0.0
        if key.char in ['a', 'd']: current_action[1] = 0.0
        if key.char in ['q', 'e']: current_action[2] = 0.0
    except AttributeError:
        if key == keyboard.Key.space: current_action[3] = 0.0
        elif key == keyboard.Key.shift: current_action[4] = 0.0

def interactive_debug_live():
    global current_action, running
    env = SSL1v1ManualEnv()
    env.render_mode = "human"
    env.reset()
    env.render()

    print("="*60)
    print("Steuerung: W/A/S/D (Fahren), Q/E (Drehen)")
    print("SPACE (Kicken), L-SHIFT (Dribbeln)")
    print("Beenden: ESC")
    print("="*60)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    total_reward = 0.0

    while running:
        step_result = env.step(current_action)
        obs, reward, terminated, truncated, info = step_result
        total_reward += reward
        done = terminated or truncated

        env.render()

        yellow = env.frame.robots_yellow[0]
        ball = env.frame.ball
        ball_speed = math.hypot(ball.v_x, ball.v_y)
        if env.dribble_start_pos is not None:
            dribble_dist = float(np.linalg.norm(
                np.array([ball.x, ball.y]) - env.dribble_start_pos))
        else:
            dribble_dist = 0.0

        print(f"\r Aktion: {current_action} | IR: {yellow.infrared} | must_release: {env.must_release} | dribble_dist: {dribble_dist:4.2f} | Ball Speed: {ball_speed:5.2f} | Y_Pos: ({yellow.x:4.2f}, {yellow.y:4.2f}) ")

        if done:
            print("\n EPISODE DONE\n")
            obs, info = env.reset()
            total_reward = 0.0
        time.sleep(0.025)

    env.close()
    listener.stop()

if __name__ == "__main__":
    interactive_debug_live()