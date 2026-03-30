import gymnasium as gym
import numpy as np
import random
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

class SSLGoToBallEnv(SSLBaseEnv):
    def __init__(self, render_mode=None):
        super().__init__(
            field_type=0, 
            n_robots_blue=1, 
            n_robots_yellow=0, 
            time_step=0.025,
            render_mode=render_mode
        )
        
        
        self.action_space = Box(low=-1.0, high=1.0, shape=(2, ), dtype=np.float32)
        
        
        self.observation_space = Box(
            low=-self.NORM_BOUNDS, 
            high=self.NORM_BOUNDS, 
            shape=(6, ), 
            dtype=np.float32
        )
        
        self.previous_dist = None

    
    def reset(self, seed=None, options=None, **kwargs):
        self.previous_dist = None
        return super().reset(seed=seed, options=options, **kwargs)

    def _frame_to_observations(self):
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        
        return np.array([
            self.norm_pos(ball.x), 
            self.norm_pos(ball.y), 
            self.norm_pos(robot.x), 
            self.norm_pos(robot.y),
            self.norm_v(robot.v_x),
            self.norm_v(robot.v_y)
        ], dtype=np.float32)

    def _get_commands(self, actions):
        
        v_x_real = actions[0] * self.max_v
        v_y_real = actions[1] * self.max_v
        return [Robot(yellow=False, id=0, v_x=v_x_real, v_y=v_y_real, v_theta=0.0)]

    def _calculate_reward_and_done(self):
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        
        dist = np.linalg.norm([ball.x - robot.x, ball.y - robot.y])
        
        reward = 0.0
        done = False
        
        
        half_length = self.field.length / 2.0
        half_width = self.field.width / 2.0
        if abs(robot.x) > half_length or abs(robot.y) > half_width:
            reward = -100.0  
            done = True
            return reward, done
            

        #reward -= 1.0 
        
        
        if self.previous_dist is not None:
            reward += (self.previous_dist - dist) * 50.0
        
        self.previous_dist = dist
        
        
        if dist < 0.18:
            reward += 500.0
            done = True
            
        return reward, done

    def _get_initial_positions_frame(self):
        pos_frame = Frame()
        margin = 0.5
        max_x = (self.field.length / 2) - margin
        max_y = (self.field.width / 2) - margin

        
        random_ball_x = random.uniform(-max_x, max_x)
        random_ball_y = random.uniform(-max_y, max_y)
        pos_frame.ball = Ball(x=random_ball_x, y=random_ball_y)
        
        
        pos_frame.robots_blue[0] = Robot(x=0.0, y=0.0, theta=0.0)
        
        return pos_frame

if __name__ == "__main__":
    import time
    env = SSLGoToBallEnv(render_mode="human")
    
    
    obs, info = env.reset()
    print("SSL Krabbelgruppe gestartet: Omnidirektionaler Roboter sucht Ball.")
    
    for i in range(500):
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        env.render()
        time.sleep(0.03)
        if done:
            print(f"Episode beendet! Reward: {reward}")
            obs, info = env.reset()
            
    env.close()