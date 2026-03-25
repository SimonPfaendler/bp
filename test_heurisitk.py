import gym
import numpy as np
import random
import time
from gym.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from skills import move_to_ball, turn_to_point


def shoot_at_goal_center(env, robot):
    goal_x = env.field.length / 2.0
    goal_y = 0.0
    target_point = np.array([goal_x, goal_y])
    v_theta = turn_to_point(robot, target_point)
    if abs(v_theta) < 0.1:
        # Array: [v_x, v_y, v_theta, kick_v_x, dribbler]
        return np.array([0.0, 0.0, 0.0, 6.0, 0.0])
    else:
        return np.array([0.0, 0.0, v_theta, 0.0, 1.0])

def simple_attacker_heuristic(env, robot):
    if robot.infrared is True:
        return shoot_at_goal_center(env, robot)
    else:
        return move_to_ball(robot, env.frame.ball)


class SSLSingleRobotEnv(SSLBaseEnv):
    def __init__(self):
        super().__init__(field_type=1, n_robots_blue=1, n_robots_yellow=0, time_step=0.025)
        self.action_space = Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)
        self.observation_space = Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)

    def reset(self):
        return super().reset()

    def _frame_to_observations(self):
        return np.zeros(1, dtype=np.float32)

    def _get_commands(self, action):
        blue = self.frame.robots_blue[0]
        

        b_cmd = simple_attacker_heuristic(self, blue)
        
  
        robot_blue = Robot(
            yellow=False, id=0, 
            v_x=b_cmd[0], v_y=b_cmd[1], v_theta=b_cmd[2], 
            kick_v_x=b_cmd[3], 
            dribbler=True if b_cmd[4] > 0 else False
        )
        return [robot_blue]

    def _calculate_reward_and_done(self):
        ball = self.frame.ball
        done = False
        

        half_length = self.field.length / 2.0
        if abs(ball.x) > half_length: 
            done = True
            
        return 0.0, done

    def _get_initial_positions_frame(self):
        pos_frame = Frame()
        

        pos_frame.ball = Ball(x=random.uniform(-1.0, 1.0), y=random.uniform(-1.0, 1.0))
        pos_frame.robots_blue[0] = Robot(x=-2.0, y=0.0, theta=0.0)
        
        return pos_frame

if __name__ == "__main__":
    env = SSLSingleRobotEnv()
    obs = env.reset()    
    while True:
        obs, reward, done, info = env.step(action=0) 
        env.render()
        
        time.sleep(0.025) 
        
        if done:
            obs = env.reset()
            time.sleep(1)