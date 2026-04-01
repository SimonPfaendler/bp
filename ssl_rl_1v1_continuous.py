import gymnasium as gym
import numpy as np
import random
import math
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from skills import move_to_ball, shoot_at_goal_center, move_to_point, turn_to_point, dribble_to_point




def blue_attacker_heuristic(env, robot):
    if robot.infrared is True:
        return shoot_at_goal_center(env, robot, "blue")
    else:
        return move_to_ball(robot, env.frame.ball, speed=0.8)


class SSL1v1ContinuousEnv(SSLBaseEnv):
    def __init__(self, render_mode=None, action_type="skills", reward_type="dense"):
        super().__init__(field_type=1, n_robots_blue=1, n_robots_yellow=1, time_step=0.025, render_mode=render_mode)
        
        self.action_type = action_type
        self.reward_type = reward_type
        # [0] = Skill-Selector (-1 to 1)
        # [1] = Target X (-1 to 1)
        # [2] = Target Y (-1 to 1)
        if self.action_type == "skills":
            self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Low-Level Action Space: [v_x, v_y, v_theta, kick, dribble]
        else:
            self.action_space = Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        self.observation_space = Box(low=-self.NORM_BOUNDS, high=self.NORM_BOUNDS, shape=(27, ), dtype=np.float32)

        self.last_dist_ball = None
        self.last_ball_goal_dist = None
        self.current_step = 0
        self.total_steps = 0
        self.max_steps = 1500
        #self.difficulty_factor = 0.2
        
        self.max_v = 2.0
        self.max_w = 10.0

        self.current_skill = 0
        self.skill_counter = 0
        self.switch_threshold = 10 # Alle 10 Steps darf gewechselt werden
        self.dribble_start_pos = None
        self.last_possession = None
        self.blue_shot_in_progress = False

        self.match_result = 0
        self.yellow_possession_steps = 0

        
    
    def reset(self, seed=None, **kwargs):
        self.last_dist_ball = None
        self.last_ball_goal_dist = None
        self.current_step = 0
        self.skill_counter = 0
        self.current_skill = 0
        self.dribble_start_pos = None
        self.last_possession = None
        self.blue_shot_in_progress = False
        self.match_result = 0
        self.yellow_possession_steps = 0
        return super().reset(seed=seed, **kwargs)
    
    
    def step(self, action):
        self.current_step += 1
        self.total_steps += 1
        #self.difficulty_factor = min(0.8, 0.2 + (self.total_steps / 2000000))
        obs, reward, terminated, truncated, info = super().step(action)

        
        done = terminated or truncated
        if done:
            info["is_success"] = 1.0 if self.match_result == 1 else 0.0
            info["match_result"] = self.match_result
            info["possession_ratio"] = self.yellow_possession_steps / max(1, self.current_step)

        return obs, reward, terminated, truncated, info

    
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
            
            
            self.current_skill / 4.0,
            self.skill_counter / 40.0,
            self.norm_pos(pred_x),
            self.norm_pos(pred_y)
        ]
        
        return np.array(obs, dtype=np.float32)

    
    def _get_commands(self, actions):
        ball = self.frame.ball
        yellow = self.frame.robots_yellow[0]
        blue_robot_data = self.frame.robots_blue[0]
        
        #[v_x, v_y, v_theta, kick, dribble]
        raw_action = np.zeros(5)

        # LOW LEVEL ACTIONS 
        if self.action_type == "low_level":
            # Hier nehmen wir die 5 Werte direkt vom Netz
            # actions: [v_x, v_y, v_theta, kick, dribble]
            v_x_global = actions[0]
            v_y_global = actions[1]
            v_theta    = actions[2]
            
            
            kick    = 4.0 if actions[3] > 0.0 else 0.0
            dribble = True if actions[4] > 0.0 else False
            
            angle_rad = np.deg2rad(yellow.theta)
            v_x_local, v_y_local, v_theta_clipped = self.convert_actions(
                [v_x_global, v_y_global, v_theta], angle_rad
            )
            print(f"LL Action: v_x={v_x_global:.2f}, v_y={v_y_global:.2f}, v_theta={v_theta:.2f}, kick={kick}, dribble={dribble}   ", end='\r')


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
            if val < -0.6:   new_skill = 3 # Turn
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
                print(f"Skill: Dribble to Point ({target_x:.2f}, {target_y:.2f})", end='\r')
            elif self.current_skill == 0:
                raw_action = move_to_ball(yellow, ball, speed=1.0)
                print(f"Skill: Move to Ball (Dist: {current_dist_ball:.2f})", end='\r')
            elif self.current_skill == 1:
                raw_action = shoot_at_goal_center(self, yellow, "yellow")
                print(f"Skill: Shoot at Goal Center", end='\r')
            elif self.current_skill == 2:
                v_x, v_y = move_to_point(yellow, target_point, speed=1.0)
                raw_action[0], raw_action[1] = v_x, v_y
                print(f"Skill: Move to Point ({target_x:.2f}, {target_y:.2f})", end='\r')
            else:
                raw_action[2] = turn_to_point(yellow, target_point)
                print(f"Skill: Turn to Point ({target_x:.2f}, {target_y:.2f})", end='\r')

            # Final Parameter
            v_x_global, v_y_global, v_theta = raw_action[0], raw_action[1], raw_action[2]
            kick    = 4.0 if raw_action[3] > 0.5 else 0.0
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
        b_cmd = blue_attacker_heuristic(self, blue_robot_data)
        b_angle_rad = np.deg2rad(blue_robot_data.theta)
        bv_x, bv_y, bv_theta = self.convert_actions([b_cmd[0], b_cmd[1], b_cmd[2]], b_angle_rad)
        
        robot_blue = Robot(yellow=False, id=0, 
                           v_x=bv_x, v_y=bv_y, v_theta=bv_theta, 
                           kick_v_x=b_cmd[3], 
                           dribbler=True if b_cmd[4] > 0 else False)
        
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

        # Ball Possesion
        current_dist_yellow = math.hypot(yellow.x - ball.x, yellow.y - ball.y)
        yellow_has_ball = (current_dist_yellow < 0.12) or yellow.infrared
        
        current_dist_blue = math.hypot(blue.x - ball.x, blue.y - ball.y)
        blue_has_ball = (current_dist_blue < 0.12) or blue.infrared

        if yellow_has_ball:
            self.yellow_possession_steps += 1

        if self.reward_type == "dense":
            reward -= 0.005
            if self.current_skill == 4: # Dribble
                reward -= 0.005
            ball_speed = math.hypot(ball.v_x, ball.v_y)
            if blue_has_ball:
                self.last_possession = 'blue'
                self.blue_shot_in_progress = False 
            elif yellow_has_ball:
                if self.blue_shot_in_progress:
                    reward += 3.0  
                    print(f"*** SCHUSS ABGEFANGEN! *** \033[K", end='\r')
                self.last_possession = 'yellow'
                self.blue_shot_in_progress = False 
            else:
                if self.last_possession == 'blue' and ball_speed > 1.5:
                    self.blue_shot_in_progress = True

            # CRASH AVOIDANCE
            dist_yellow_blue = math.hypot(yellow.x - blue.x, yellow.y - blue.y)
            if dist_yellow_blue < 0.22:
                reward -= 0.05
            if dist_yellow_blue < 0.18:
                reward -= 2.0
                print("CRASH", end='\r')

            # Robot -> Ball
            if hasattr(self, 'last_dist_ball') and self.last_dist_ball is not None:
                progress = self.last_dist_ball - current_dist_yellow
                
                
                is_shot_flying = (ball.v_x < -0.1 and progress < 0)
                
                if not is_shot_flying:
                    reward += progress * 2.0
                
            if current_dist_yellow < 0.12:
                reward += 0.05
                if abs(ball.v_x) > 1.0:
                    reward += 0.1
                
            self.last_dist_ball = current_dist_yellow

            # Ball -> Goal 
            ball_pos = np.array([ball.x, ball.y])
            goal_a = np.array([-max_x, goal_half_width])
            goal_b = np.array([-max_x, -goal_half_width])
            goal_vec = goal_b - goal_a
            ball_to_goal_a = ball_pos - goal_a
            
            # Projection like Tilmans method to find closest point on goal line
            t = np.dot(ball_to_goal_a, goal_vec) / np.dot(goal_vec, goal_vec)
            t = np.clip(t, 0.0, 1.0)
            closest_goal_point = goal_a + t * goal_vec
            current_ball_goal_dist = np.linalg.norm(ball_pos - closest_goal_point)
            
            # Reward for Ball closer to goal
            if hasattr(self, 'last_ball_goal_dist') and self.last_ball_goal_dist is not None:
                progress_ball = self.last_ball_goal_dist - current_ball_goal_dist
                is_good_shot = (ball.v_x < -0.1 and progress_ball > 0)
                
                if yellow_has_ball or is_good_shot:
                    distance_factor = 1.0 + (1.0 / (current_ball_goal_dist + 0.5))
                    reward += progress_ball * 5.0 * distance_factor 
                
            self.last_ball_goal_dist = current_ball_goal_dist

            # Dribbler Bonus
            if ball.v_x < -0.3 and current_dist_yellow < 0.3: 
                reward += abs(ball.v_x) * 0.1

        # END CONDITIONS
        if abs(ball.x) > max_x:
            done = True
            if abs(ball.y) <= goal_half_width:
                if ball.x < 0: 
                    reward += 10.0
                    self.match_result = 1 
                else:          
                    reward -= 10.0
                    self.match_result = -1 
            else:
                reward -= 1.0
            return reward, done
        
        if abs(ball.y) > max_y:
            done = True
            reward -= 1.0 
            return reward, done

        if abs(yellow.x) > max_x or abs(yellow.y) > max_y:
            reward -= 1.0
            done = True
        
        if self.current_step >= self.max_steps:
            truncated = True
            reward -= 0.5 

        return reward, done

    
    def _get_initial_positions_frame(self):
        pos_frame = Frame()
        half_len = self.field.length / 2.0

        is_attack_scenario = self.np_random.random() < 0.25

        if is_attack_scenario:
            pos_frame.ball = Ball(x=self.np_random.uniform(-0.5, 0.5), y=self.np_random.uniform(-0.5, 0.5))

            pos_frame.robots_yellow[0] = Robot(x=1.0, y=0.0, theta=180.0)

            pos_frame.robots_blue[0] = Robot(x=-1.0, y=0.0, theta=0.0)

        else:

            blue_y = self.np_random.uniform(-1.0, 1.0)
            pos_frame.robots_blue[0] = Robot(x=0.0, y=blue_y, theta=0.0)
            pos_frame.ball = Ball(x=0.15, y=blue_y)
            
            yellow_x = self.np_random.uniform(3.7, 3.9)
            yellow_y = self.np_random.uniform(-0.5, 0.5)
            pos_frame.robots_yellow[0] = Robot(x=yellow_x, y=yellow_y, theta=180.0)

        
        return pos_frame