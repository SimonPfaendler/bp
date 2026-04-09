import gymnasium as gym
import numpy as np
import random
import math
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from skills import move_to_ball, shoot_at_point, move_to_point, turn_to_point, dribble_to_point, shoot_at_goal_center




def blue_attacker_heuristic(env, robot):
    if robot.infrared is True:
        return shoot_at_goal_center(env, robot, "blue")
    else:
        return move_to_ball(robot, env.frame.ball, speed=0.8)


class SSL1v1ContinuousEnv(SSLBaseEnv):
    def __init__(self, render_mode=None, action_type="skills", reward_type="dense"):
        super().__init__(field_type=1, n_robots_blue=1, n_robots_yellow=1, time_step=0.025, render_mode=render_mode)
        """
        1v1 Continuous Robot Soccer Environment.
        
        Observation Space (27 dim):
            [0:2]   Ball position (x, y)
            [2:4]   Ball velocity (v_x, v_y)
            [4]     Normalized distance ball to goal
            [5:14]  Yellow robot (x, y, sin(theta), cos(theta), v_x, v_y, v_theta, infrared, dist_to_ball)
            [14:23] Blue robot (x, y, sin(theta), cos(theta), v_x, v_y, v_theta, infrared, dist_to_ball)
            [23:25] Current skill and skill counter
            [25:27] Predicted ball position (x, y) in 0.5s
            
        Action Space:
            - "skills": Box(4,) -> [Skill-Selector, Target X, Target Y, Kick Power]
            - "low_level": Box(6,) -> [v_x, v_y, v_theta, kick_power, kick_trigger, dribble]
        """
        self.action_type = action_type
        self.reward_type = reward_type
        # [0] = Skill-Selector (-1 to 1)
        # [1] = Target X (-1 to 1)
        # [2] = Target Y (-1 to 1)
        # [3] = Kick Power (-1 to 1)
        if self.action_type == "skills":
            self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Low-Level Action Space: [v_x, v_y, v_theta, kick_power, kick_trigger, dribble]
        else:
            self.action_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        
        obs_size = 25 
        
        if self.action_type == "skills":
            obs_size += 2
            
        self.observation_space = Box(
            low=-self.NORM_BOUNDS, 
            high=self.NORM_BOUNDS, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        self.last_dist_robot_ball = None 
        self.last_dist_ball_goal = None
        self.current_step = 0
        self.total_steps = 0
        self.max_steps = 1500
        #self.difficulty_factor = 0.2
        
        self.max_v = 2.0
        self.max_w = 10.0

        self.current_skill = 0
        self.skill_counter = 0
        self.switch_threshold = 10 # Cooldown: allow skill switch only every 10 steps
        self.dribble_start_pos = None
        self.last_possession = None
        self.blue_shot_in_progress = False

        self.match_result = 0
        self.yellow_possession_steps = 0
        self.is_dribbling = False
        self.dribble_start_pos = None
        self.max_dribble_dist = 1.0
        self.robot_ball_contact = 0.12

        
    
    def reset(self, seed=None, **kwargs):
        self.last_dist_robot_ball = None
        self.last_dist_ball_goal = None
        self.current_step = 0
        self.skill_counter = 0
        self.current_skill = 0
        self.dribble_start_pos = None
        self.last_possession = None
        self.blue_shot_in_progress = False
        self.match_result = 0
        self.yellow_possession_steps = 0
        self.is_dribbling = False
        self.dribble_start_pos = None
        return super().reset(seed=seed, **kwargs)
    
    
    def step(self, action):
        self.current_step += 1
        self.total_steps += 1
        #self.difficulty_factor = min(0.8, 0.2 + (self.total_steps / 2000000))
        obs, reward, terminated, truncated, info = super().step(action)

        ball_pos = np.array([self.frame.ball.x, self.frame.ball.y])
        yellow_robot = self.frame.robots_yellow[0]
        robot_pos = np.array([yellow_robot.x, yellow_robot.y])
        dist_robot_ball = np.linalg.norm(robot_pos - ball_pos)
        has_contact = (dist_robot_ball < self.robot_ball_contact) or yellow_robot.infrared

        if has_contact:
            if not self.is_dribbling:
                self.is_dribbling = True
                self.dribble_start_pos = robot_pos.copy()
            else:
                dribble_dist = np.linalg.norm(robot_pos - self.dribble_start_pos)
                if dribble_dist > self.max_dribble_dist:
                    reward -= 1.0
                    truncated = True
                    self.match_result = 0
        else:
            self.is_dribbling = False
            self.dribble_start_pos = None


        
        done = terminated or truncated
        if done:
            info["is_success"] = 1.0 if self.match_result == 1 else 0.0
            info["match_result"] = self.match_result
            info["possession_ratio"] = self.yellow_possession_steps / max(1, self.current_step)

        return obs, reward, terminated, truncated, info


    def set_curriculum_level(self, level):
        self.curriculum_level = level
    
    def convert_actions(self, action_array, angle):
        """Denormalize, clip to absolute max and convert to local"""
        v_x = action_array[0] * self.max_v
        v_y = action_array[1] * self.max_v
        v_theta = action_array[2] * self.max_w

        v_x_local = v_x * np.cos(angle) + v_y * np.sin(angle)
        v_y_local = -v_x * np.sin(angle) + v_y * np.cos(angle)

        v_norm = np.linalg.norm([v_x_local, v_y_local])
        if v_norm > self.max_v:
            c = self.max_v / v_norm
            v_x_local, v_y_local = v_x_local * c, v_y_local * c

        return v_x_local, v_y_local, v_theta

    
    def _frame_to_observations(self):
        ball = self.frame.ball
        yellow = self.frame.robots_yellow[0]
        blue = self.frame.robots_blue[0]
        
        max_x = self.field.length / 2.0
        max_y = self.field.width / 2.0
        max_dist = math.hypot(self.field.length, self.field.width) 
        
        
        future_time = 0.5 
        pred_x = np.clip(ball.x + (ball.v_x * future_time), -max_x, max_x)
        pred_y = np.clip(ball.y + (ball.v_y * future_time), -max_y, max_y)

        
        goal_half_width = 0.5 
        ball_pos = np.array([ball.x, ball.y])
        
        # Define goal vector projection points for distance calculation
        goal_a = np.array([-max_x, goal_half_width])
        goal_b = np.array([-max_x, -goal_half_width])
        goal_vec = goal_b - goal_a
        ball_to_goal_a = ball_pos - goal_a
        
        # Projection Like in Tilmans Thesis
        t = np.dot(ball_to_goal_a, goal_vec) / np.dot(goal_vec, goal_vec)
        t = np.clip(t, 0.0, 1.0)
        closest_goal_point = goal_a + t * goal_vec
        dist_ball_to_goal = np.linalg.norm(ball_pos - closest_goal_point)
        
        # Distance Robot to Ball
        dist_yellow_ball = math.hypot(yellow.x - ball.x, yellow.y - ball.y)
        dist_blue_ball = math.hypot(blue.x - ball.x, blue.y - ball.y)

        obs = [
            # --- BALL ---
            self.norm_pos(ball.x), self.norm_pos(ball.y),
            self.norm_v(ball.v_x), self.norm_v(ball.v_y),
            dist_ball_to_goal / max_dist,        
            
            # --- YELLOW ---
            self.norm_pos(yellow.x), self.norm_pos(yellow.y),
            np.clip(np.sin(np.deg2rad(yellow.theta)), -self.NORM_BOUNDS, self.NORM_BOUNDS),
            np.clip(np.cos(np.deg2rad(yellow.theta)), -self.NORM_BOUNDS, self.NORM_BOUNDS),
            self.norm_v(yellow.v_x), self.norm_v(yellow.v_y), self.norm_w(yellow.v_theta),
            1.0 if yellow.infrared else 0.0,     
            dist_yellow_ball / max_dist,         
            
            # --- BLUE ---
            self.norm_pos(blue.x), self.norm_pos(blue.y),
            np.clip(np.sin(np.deg2rad(blue.theta)), -self.NORM_BOUNDS, self.NORM_BOUNDS),
            np.clip(np.cos(np.deg2rad(blue.theta)), -self.NORM_BOUNDS, self.NORM_BOUNDS),
            self.norm_v(blue.v_x), self.norm_v(blue.v_y), self.norm_w(blue.v_theta),
            1.0 if blue.infrared else 0.0,       
            dist_blue_ball / max_dist,           
            
            
            self.norm_pos(pred_x),
            self.norm_pos(pred_y),
            ]
        if self.action_type == "skills":
            obs.extend([
                self.current_skill / 4.0,
                self.skill_counter / 40.0,
            ])
        
        return np.array(obs, dtype=np.float32)

    
    def _get_commands(self, actions):
        ball = self.frame.ball
        yellow = self.frame.robots_yellow[0]
        blue_robot_data = self.frame.robots_blue[0]
        
        # [v_x, v_y, v_theta, kick, dribble]
        raw_action = np.zeros(5)

        # LOW LEVEL ACTIONS 
        if self.action_type == "low_level":
            # actions: [v_x, v_y, v_theta, raw_kick_power, kick_trigger dribble]
            v_x_global = actions[0]
            v_y_global = actions[1]
            v_theta    = actions[2]

            raw_kick_power = actions[3]
            kick_trigger = actions[4]
            dribbler_trigger = actions[5]
            
            
            if kick_trigger > 0.0:
                kick = 3.0 + ((raw_kick_power + 1.0) / 2.0) * 3.0
            else:
                kick = 0.0
            dribble = True if dribbler_trigger > 0.0 else False
            
            angle_rad = np.deg2rad(yellow.theta)
            v_x_local, v_y_local, v_theta_clipped = self.convert_actions(
                [v_x_global, v_y_global, v_theta], angle_rad
            )
            #print(f"LL Action: v_x={v_x_global:.2f}, v_y={v_y_global:.2f}, v_theta={v_theta:.2f}, kick={kick}, dribble={dribble}   ", end='\r')


        # SKILLS 
        else:
            current_dist_ball = math.hypot(yellow.x - ball.x, yellow.y - ball.y)
            has_ball = (current_dist_ball < 0.15) or (yellow.infrared is True)

            # Target
            max_target_dist = 2.0 
            target_x = np.clip(yellow.x + (actions[1] * max_target_dist), -self.field.length / 2.0, self.field.length / 2.0)
            target_y = np.clip(yellow.y + (actions[2] * max_target_dist), -self.field.width / 2.0, self.field.width / 2.0)
            target_point = np.array([target_x, target_y])

            # Skill Selection
            val = actions[0]
            if val < -0.6:   new_skill = 3 # Turn to Point
            elif val < -0.2: new_skill = 2 # Move to Point
            elif val < 0.2:  new_skill = 0 # Move to Ball
            elif val < 0.6:  new_skill = 4 # Dribble
            else:            new_skill = 1 # Shoot

            if self.skill_counter <= 0:
                self.current_skill = new_skill
                self.skill_counter = 40 if self.current_skill == 4 else self.switch_threshold
            else:
                self.skill_counter -= 1

            # Skill Execution
            if self.current_skill == 4: # Dribble
                raw_action = dribble_to_point(yellow, target_point, speed=0.8)
                # print(f"Skill: Dribble to Point ({target_x:.2f}, {target_y:.2f})", end='\r')
            elif self.current_skill == 0:
                raw_action = move_to_ball(yellow, ball, speed=1.0)
                # print(f"Skill: Move to Ball (Dist: {current_dist_ball:.2f})", end='\r')
            elif self.current_skill == 1:
                raw_action = shoot_at_point(yellow, target_point)
                # print(f"Skill: Shoot at Point ({target_x:.2f}, {target_y:.2f})", end='\r')
            elif self.current_skill == 2:
                v_x, v_y = move_to_point(yellow, target_point, speed=1.0)
                raw_action[0], raw_action[1] = v_x, v_y
                # print(f"Skill: Move to Point ({target_x:.2f}, {target_y:.2f})", end='\r')
            else:
                raw_action[2] = turn_to_point(yellow, target_point)
                # print(f"Skill: Turn to Point ({target_x:.2f}, {target_y:.2f})", end='\r')

            # Final Parameter
            v_x_global, v_y_global, v_theta = raw_action[0], raw_action[1], raw_action[2]
            if raw_action[3] > 0.5:
                kick = 3.0 + ((actions[3] + 1.0) / 2.0) * (6.0 - 3.0)
            else:
                kick = 0.0
                
            dribble = True if raw_action[4] > 0.5 else False

            # Local Conversion
            angle_rad = np.deg2rad(yellow.theta)
            v_x_local, v_y_local, v_theta_clipped = self.convert_actions(
                [v_x_global, v_y_global, v_theta], angle_rad
            )



        robot_yellow = Robot(yellow=True, id=0, 
                             v_x=v_x_local, v_y=v_y_local, v_theta=v_theta_clipped, 
                             kick_v_x=kick, dribbler=dribble)

        # Blue Heuristic
        level = getattr(self, 'curriculum_level', 4)

        if level < 3:
            # LEVEL 1 & 2
            bv_x, bv_y, bv_theta = 0.0, 0.0, 0.0
            blue_kick = 0.0
            blue_dribble = False
        else:
            # LEVEL 3 & 4: Normal Heuristic
            b_cmd = blue_attacker_heuristic(self, blue_robot_data)
            b_angle_rad = np.deg2rad(blue_robot_data.theta)
            bv_x, bv_y, bv_theta = self.convert_actions([b_cmd[0], b_cmd[1], b_cmd[2]], b_angle_rad)
            blue_kick = b_cmd[3]
            blue_dribble = True if b_cmd[4] > 0 else False

        
        robot_blue = Robot(
            yellow=False, 
            id=0, 
            v_x=bv_x, 
            v_y=bv_y, 
            v_theta=bv_theta, 
            kick_v_x=blue_kick, 
            dribbler=blue_dribble
        )
        
        return [robot_blue, robot_yellow]

    
    def _calculate_reward_and_done(self):
        ball = self.frame.ball
        yellow = self.frame.robots_yellow[0]
        blue = self.frame.robots_blue[0]
        
        done = False
        truncated = False
        reward = 0.0
        
        max_x = self.field.length / 2.0
        max_y = self.field.width / 2.0
        goal_half_width = 0.5
        
        
        opponent_goal_pos = np.array([-max_x, 0.0])


        if self.reward_type == "dense":
            reward -= 0.005 


        # SPARSE REWARDS
        if abs(ball.x) > max_x:
            done = True
            if abs(ball.y) <= goal_half_width:
                if ball.x < 0: 
                    reward += 100.0
                    self.match_result = 1 
                else:          
                    reward -= 100.0
                    self.match_result = -1 
            else:
                reward -= 10.0        
            return reward, done
        
        # Ball Out of Bounds
        if abs(ball.y) > max_y:
            done = True
            reward -= 10.0 
            return reward, done

        # Robot Out of Bounds
        if abs(yellow.x) > max_x or abs(yellow.y) > max_y:
            done = True
            reward -= 10.0
            return reward, done
        
        # Timeout
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 5.0
            return reward, done

        # DENSE REWARDS (Potential-Based Shaping)
  
        if self.reward_type == "dense":
            
            # Robot to Ball
            current_dist_robot_ball = math.hypot(yellow.x - ball.x, yellow.y - ball.y)
            if self.last_dist_robot_ball is not None:
                reward += (self.last_dist_robot_ball - current_dist_robot_ball) * 5.0
            self.last_dist_robot_ball = current_dist_robot_ball

            # Ball to Goal
            current_dist_ball_goal = np.linalg.norm(np.array([ball.x, ball.y]) - opponent_goal_pos)
            if self.last_dist_ball_goal is not None:
                reward += (self.last_dist_ball_goal - current_dist_ball_goal) * 10.0
            self.last_dist_ball_goal = current_dist_ball_goal

            # Ballpossession
            if current_dist_robot_ball < 0.12 or yellow.infrared:
                reward += 0.05
                self.yellow_possession_steps += 1
                
        return reward, done

    
    def _get_initial_positions_frame(self):
        pos_frame = Frame()
        
        level = getattr(self, 'curriculum_level', 4)

        if level == 1:
            # LEVEL 1: PENALTY CHALLENGE

            bx = self.np_random.uniform(-2.0, -1.0)
            by = self.np_random.uniform(-0.3, 0.3)
            pos_frame.ball = Ball(x=bx, y=by)

            pos_frame.robots_yellow[0] = Robot(
                x=bx + 0.3, 
                y=by, 
                theta=180.0 
            )

            pos_frame.robots_blue[0] = Robot(x=0.0, y=3.0, theta=0.0)

        elif level == 2:
            # LEVEL 2: FREE BALL

            pos_frame.ball = Ball(
                x=self.np_random.uniform(-1.0, 2.0),
                y=self.np_random.uniform(-1.5, 1.5)
            )

            pos_frame.robots_yellow[0] = Robot(
                x=self.np_random.uniform(2.0, 3.5),
                y=self.np_random.uniform(-2.0, 2.0),
                theta=self.np_random.uniform(-180, 180)
            )

            
            pos_frame.robots_blue[0] = Robot(x=0.0, y=3.0, theta=0.0)

        elif level == 3:
            # LEVEL 3: 1v1 ATTACK
            bx = self.np_random.uniform(-1.0, 2.0)
            by = self.np_random.uniform(-1.5, 1.5)
            pos_frame.ball = Ball(x=bx, y=by)

            pos_frame.robots_yellow[0] = Robot(
                x=bx + self.np_random.uniform(0.3, 1.0), 
                y=by + self.np_random.uniform(-0.5, 0.5), 
                theta=self.np_random.uniform(135.0, 225.0)
            )

            
            pos_frame.robots_blue[0] = Robot(
                x=self.np_random.uniform(-3.5, bx - 0.5),
                y=self.np_random.uniform(-1.5, 1.5),
                theta=self.np_random.uniform(-180, 180)
            )

        else:
            # LEVEL 4:

            scenario_roll = self.np_random.random()

            if scenario_roll < 0.33:
                # Attack
                bx = self.np_random.uniform(-1.0, 2.0)
                by = self.np_random.uniform(-1.5, 1.5)
                pos_frame.ball = Ball(x=bx, y=by)
                pos_frame.robots_yellow[0] = Robot(x=bx+0.5, y=by, theta=180)
                pos_frame.robots_blue[0] = Robot(x=bx-1.0, y=by, theta=0)

            elif scenario_roll < 0.66:
                # Defend
                blue_x = self.np_random.uniform(-1.0, 2.0)
                blue_y = self.np_random.uniform(-2.0, 2.0)
                pos_frame.robots_blue[0] = Robot(x=blue_x, y=blue_y, theta=0)
                pos_frame.ball = Ball(x=blue_x+0.15, y=blue_y)
                pos_frame.robots_yellow[0] = Robot(x=3.0, y=0.0, theta=180)

            else:
                # Chaos
                pos_frame.ball = Ball(x=self.np_random.uniform(-3, 3), y=self.np_random.uniform(-2, 2))
                pos_frame.robots_yellow[0] = Robot(x=self.np_random.uniform(0, 3.5), y=self.np_random.uniform(-2.5, 2.5), theta=self.np_random.uniform(-180, 180))
                pos_frame.robots_blue[0] = Robot(x=self.np_random.uniform(-3.5, 0), y=self.np_random.uniform(-2.5, 2.5), theta=self.np_random.uniform(-180, 180))

        return pos_frame
