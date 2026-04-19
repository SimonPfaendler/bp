import gymnasium as gym
import numpy as np
import random
import math
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from skills import move_to_ball, shoot_at_point, move_to_point, turn_to_point, dribble_to_point, shoot_at_goal_center



def blue_attacker_heuristic(env, robot):
    ball = env.frame.ball
    yellow = env.frame.robots_yellow[0]
    defend_goal_x = -env.field.length / 2.0

    # Shoot
    if robot.infrared:
        return shoot_at_goal_center(env, robot, team_color="blue")
    
    # Yellow is closer to the ball fall back on the ball on the goal line.
    dist_blue_ball = math.hypot(robot.x - ball.x, robot.y - ball.y)
    dist_yellow_ball = math.hypot(yellow.x - ball.x, yellow.y - ball.y)
    if dist_yellow_ball < dist_blue_ball:
        goal = np.array([defend_goal_x, 0.0])
        ball_pos = np.array([ball.x, ball.y])
        bg = goal - ball_pos
        bg_len = np.linalg.norm(bg)
        if bg_len > 0.01:
            stand = goal - (bg / bg_len) * min(1.0, bg_len * 0.4)
        else:
            stand = np.array([defend_goal_x + 0.3, 0.0])
        v_x, v_y = move_to_point(robot, stand, speed=2.0)
        v_theta = turn_to_point(robot, np.array([ball.x, ball.y]))
        return np.array([v_x, v_y, v_theta, 0.0, 0.0])

    # Intercept
    ball_speed = math.hypot(ball.v_x, ball.v_y)
    if ball_speed > 0.3:
        # Solve |ball + v·t − robot| = s·t for smallest t ≥ 0.
        dx, dy = ball.x - robot.x, ball.y - robot.y
        s = 1.2
        a = ball_speed * ball_speed - s * s
        b = 2.0 * (dx * ball.v_x + dy * ball.v_y)
        c = dx * dx + dy * dy
        t = None
        if abs(a) < 1e-6:
            # ball_speed ≈ pursuit_speed → linear: b·t + c = 0
            if abs(b) > 1e-6:
                cand = -c / b
                if cand > 0.0:
                    t = cand
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sq = math.sqrt(max(disc, 0.0))
                roots = [r for r in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)) if r > 0.0]
                if roots:
                    t = min(roots)
        if t is not None and t < 2.0:
            target = np.array([ball.x + ball.v_x * t, ball.y + ball.v_y * t])
            dx, dy = target[0] - robot.x, target[1] - robot.y
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                v_x, v_y = (dx / dist) * s, (dy / dist) * s
            else:
                v_x, v_y = 0.0, 0.0
            v_theta = turn_to_point(robot, target)
            #print(f"Intercepting ball at t={t:.2f}s  Target ({target[0]:.2f}, {target[1]:.2f})", end='\r')
            return np.array([v_x, v_y, v_theta, 0.0, 1.0])

    return move_to_ball(robot, ball, speed=2.0)

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
        
        
        obs_size = 26
            
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
        self.has_touched_ball = False
        self.blue_personality = "defensive"
        self.must_release = False
        self.min_release_distance = 0.12



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
        self.has_touched_ball = False
        self.must_release = False

        roll = self.np_random.random()
        if roll < 0.6:
            self.blue_personality = "defensive"
        else:
            self.blue_personality = "aggressive"

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
                self.dribble_start_pos = ball_pos.copy()
            else:
                dribble_dist = np.linalg.norm(ball_pos - self.dribble_start_pos)
                if dribble_dist > self.max_dribble_dist:
                    self.must_release = True
                    self.is_dribbling = False
        else:
            if self.must_release:
                # Check if robot has enough distance to ball
                if dist_robot_ball >= self.min_release_distance:
                    self.must_release = False
                    self.dribble_start_pos = None
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
        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        yellow = self.frame.robots_yellow[0]
        blue = self.frame.robots_blue[0]

        observation = []
        max_dist = math.hypot(self.field.length, self.field.width)

        # BALL
        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        # Distance Ball to Blue Goal
        goal_blue_a = np.array([-self.field.length / 2, self.field.goal_width / 2])
        goal_blue_b = np.array([-self.field.length / 2, -self.field.goal_width / 2])
        goal_blue = goal_blue_b - goal_blue_a
        ball_to_goal_blue = ball - goal_blue_a
        t_blue = (ball_to_goal_blue @ goal_blue) / (goal_blue @ goal_blue)
        t_blue = np.clip(t_blue, 0, 1)
        closest_point_blue = goal_blue_a + t_blue * goal_blue
        dist_ball_to_goal_blue = np.linalg.norm(ball - closest_point_blue)

        # Distance Ball to Yellow Goal
        goal_yellow_a = np.array([self.field.length / 2, self.field.goal_width / 2])
        goal_yellow_b = np.array([self.field.length / 2, -self.field.goal_width / 2])
        goal_yellow = goal_yellow_b - goal_yellow_a
        ball_to_goal_yellow = ball - goal_yellow_a
        t_yellow = (ball_to_goal_yellow @ goal_yellow) / (goal_yellow @ goal_yellow)
        t_yellow = np.clip(t_yellow, 0, 1)
        closest_point_yellow = goal_yellow_a + t_yellow * goal_yellow
        dist_ball_to_goal_yellow = np.linalg.norm(ball - closest_point_yellow)

        # Normalisierte Distanzen anhängen
        observation.append(np.clip(dist_ball_to_goal_blue / max_dist, -1.0, 1.0))
        observation.append(np.clip(dist_ball_to_goal_yellow / max_dist, -1.0, 1.0))

        # YELLOW ROBOT
        observation.append(self.norm_pos(yellow.x))
        observation.append(self.norm_pos(yellow.y))
        observation.append(np.sin(np.deg2rad(yellow.theta)))
        observation.append(np.cos(np.deg2rad(yellow.theta)))
        observation.append(self.norm_v(yellow.v_x))
        observation.append(self.norm_v(yellow.v_y))
        observation.append(self.norm_w(yellow.v_theta))
        observation.append(1.0 if yellow.infrared else 0.0)

        # Dist to Ball
        robot_yellow_pos = np.array([yellow.x, yellow.y])
        dist_yellow_ball = np.linalg.norm(ball - robot_yellow_pos)
        observation.append(np.clip(dist_yellow_ball / max_dist, -1.0, 1.0))

        
        dribble_meter = 0.0
        if getattr(self, 'is_dribbling', False) and getattr(self, 'dribble_start_pos', None) is not None:
            current_dribble_dist = np.linalg.norm(np.array([self.frame.ball.x, self.frame.ball.y]) - self.dribble_start_pos)
            dribble_meter = np.clip(current_dribble_dist / self.max_dribble_dist, 0.0, 1.0)
        observation.append(dribble_meter)

        # Release state: 1.0 if must release, else 0.0
        observation.append(1.0 if self.must_release else 0.0)

        # BLUE ROBOT
        observation.append(self.norm_pos(blue.x))
        observation.append(self.norm_pos(blue.y))
        observation.append(np.sin(np.deg2rad(blue.theta)))
        observation.append(np.cos(np.deg2rad(blue.theta)))
        observation.append(self.norm_v(blue.v_x))
        observation.append(self.norm_v(blue.v_y))
        observation.append(self.norm_w(blue.v_theta))
        observation.append(1.0 if blue.infrared else 0.0)

        # Dist to Ball
        robot_blue_pos = np.array([blue.x, blue.y])
        dist_blue_ball = np.linalg.norm(ball - robot_blue_pos)
        observation.append(np.clip(dist_blue_ball / max_dist, -1.0, 1.0))

        return np.array(observation, dtype=np.float32)

    
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
                raw_action = dribble_to_point(yellow, target_point, speed=1.6)
                # print(f"Skill: Dribble to Point ({target_x:.2f}, {target_y:.2f})", end='\r')
            elif self.current_skill == 0:
                raw_action = move_to_ball(yellow, ball, speed=2.0)
                # print(f"Skill: Move to Ball (Dist: {current_dist_ball:.2f})", end='\r')
            elif self.current_skill == 1:
                raw_action = shoot_at_point(yellow, target_point)
                # print(f"Skill: Shoot at Point ({target_x:.2f}, {target_y:.2f})", end='\r')
            elif self.current_skill == 2:
                v_x, v_y = move_to_point(yellow, target_point, speed=2.0)
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



        # If must_release: force the dribbler off
        if self.must_release:
            kick = 0.25

        robot_yellow = Robot(yellow=True, id=0,
                             v_x=v_x_local, v_y=v_y_local, v_theta=v_theta_clipped,
                             kick_v_x=kick, dribbler=dribble)

        # Blue Heuristic
        level = getattr(self, 'curriculum_level', 1)

        if level <= 2:
            # LEVEL 1 & 2: Blue steht still
            bv_x, bv_y, bv_theta = 0.0, 0.0, 0.0
            blue_kick = 0.0
            blue_dribble = False
        elif level == 3:
            # LEVEL 3: Blue bewegt sich langsam zum Ball, kickt nicht
            b_cmd = move_to_ball(blue_robot_data, ball, speed=0.5)
            b_angle_rad = np.deg2rad(blue_robot_data.theta)
            bv_x, bv_y, bv_theta = self.convert_actions([b_cmd[0], b_cmd[1], b_cmd[2]], b_angle_rad)
            blue_kick = 0.0
            blue_dribble = False
        else:
            # LEVEL 4 & 5: Full Heuristic mit Personality
            if self.blue_personality == "aggressive":
                if blue_robot_data.infrared is True:
                    b_cmd = shoot_at_goal_center(self, blue_robot_data, team_color="blue")
                else:
                    b_cmd = move_to_ball(blue_robot_data, ball, speed=1.5)
            else:
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
        goal_half_width = self.field.goal_width / 2.0

        #  Time Penalty (-0.001 to -0.004)
        if self.reward_type == "dense":
            progress = self.current_step / self.max_steps
            reward -= 0.0004 * (1.0 + 2.0 * progress)

        
        
        # End Conditions
        if abs(ball.x) > max_x:
            done = True
            if abs(ball.y) <= goal_half_width:
                if ball.x < 0: # Goal for Yellow
                    reward += 1.0
                    reward += (self.max_steps - self.current_step) * 0.01 
                    self.match_result = 1 
                else: # Goal for Blue (Defeat)
                    reward -= 1.0
                    self.match_result = -1 
            else:
                reward -= 0.5
            return reward, done

        # Ball out of bounds
        if abs(ball.y) > max_y:
            done = True
            reward -= 1.0
            self.match_result = -1 
            return reward, done

        # Robot out of bounds
        if abs(yellow.x) > max_x or abs(yellow.y) > max_y:
            done = True
            reward -= 1.0
            self.match_result = -1
            return reward, done
        
        # Timeout
        if self.current_step >= self.max_steps:
            truncated = True
            done = True
            reward -= 1.0
            self.match_result = -1
            return reward, done

        # DENSE REWARDS
        if self.reward_type == "dense":
            
            dist_robot_ball = math.hypot(yellow.x - ball.x, yellow.y - ball.y)
            if not getattr(self, 'has_touched_ball', False) and (dist_robot_ball < 0.15 or yellow.infrared):
                self.has_touched_ball = True
                reward += 0.1
            
            # Robot to Ball
            dist_robot_ball = math.hypot(yellow.x - ball.x, yellow.y - ball.y)
            if self.last_dist_robot_ball is not None:
                delta_robot_ball = self.last_dist_robot_ball - dist_robot_ball
                reward += np.clip(delta_robot_ball * 5.0, -0.0005, 0.0005)
            self.last_dist_robot_ball = dist_robot_ball

            # Ball to Goal
            goal_a = np.array([-max_x, goal_half_width])
            goal_b = np.array([-max_x, -goal_half_width])
            goal_vec = goal_b - goal_a
            ball_pos = np.array([ball.x, ball.y])
            ball_to_goal_a = ball_pos - goal_a

            t = np.dot(ball_to_goal_a, goal_vec) / np.dot(goal_vec, goal_vec)
            t = np.clip(t, 0.0, 1.0)
            closest_goal_point = goal_a + t * goal_vec
            dist_ball_to_goal = np.linalg.norm(ball_pos - closest_goal_point)

            if self.last_dist_ball_goal is not None:
                delta_ball_goal = self.last_dist_ball_goal - dist_ball_to_goal
                reward += np.clip(delta_ball_goal * 10.0, -0.0006, 0.001)
            self.last_dist_ball_goal = dist_ball_to_goal

            # Ballpossession
            if dist_robot_ball < 0.12 or yellow.infrared:
                reward += 0.0001
                self.yellow_possession_steps += 1

            if ball.v_x < -0.5:
                reward += 0.0002 * min(-ball.v_x, 3.0)
                
        return reward, done

    
    def _get_initial_positions_frame(self):
        pos_frame = Frame()
        
        level = getattr(self, 'curriculum_level', 1)

        if level == 1:
            # LEVEL 1: PENALTY - Ball nah am Tor, Yellow direkt dahinter Richtung Tor
            goal_x = -self.field.length / 2.0
            bx = self.np_random.uniform(goal_x + 0.5, goal_x + 1.5)
            by = self.np_random.uniform(-0.3, 0.3)
            pos_frame.ball = Ball(x=bx, y=by)

            pos_frame.robots_yellow[0] = Robot(
                x=bx + 0.15,
                y=by,
                theta=180.0 + self.np_random.uniform(-10.0, 10.0)
            )

            pos_frame.robots_blue[0] = Robot(x=0.0, y=3.0, theta=0.0)

        elif level == 2:
            # LEVEL 2: FREE BALL + statischer Goalie
            pos_frame.ball = Ball(
                x=self.np_random.uniform(-1.0, 2.0),
                y=self.np_random.uniform(-1.5, 1.5)
            )

            pos_frame.robots_yellow[0] = Robot(
                x=self.np_random.uniform(2.0, 3.5),
                y=self.np_random.uniform(-2.0, 2.0),
                theta=self.np_random.uniform(-180, 180)
            )

            goal_x = -self.field.length / 2.0
            pos_frame.robots_blue[0] = Robot(
                x=goal_x + 0.2,
                y=self.np_random.uniform(-0.3, 0.3),
                theta=0.0
            )

        elif level == 3:
            # LEVEL 3: FREE BALL + langsamer Blue bewegt sich zum Ball
            pos_frame.ball = Ball(
                x=self.np_random.uniform(-1.0, 2.0),
                y=self.np_random.uniform(-1.5, 1.5)
            )

            pos_frame.robots_yellow[0] = Robot(
                x=self.np_random.uniform(2.0, 3.5),
                y=self.np_random.uniform(-2.0, 2.0),
                theta=self.np_random.uniform(-180, 180)
            )

            pos_frame.robots_blue[0] = Robot(
                x=self.np_random.uniform(-3.5, -1.0),
                y=self.np_random.uniform(-1.5, 1.5),
                theta=self.np_random.uniform(-180, 180)
            )

        elif level == 4:
            # LEVEL 4: 1v1 ATTACK
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
            # LEVEL 5:

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
                pos_frame.robots_yellow[0] = Robot(
                    x=blue_x + self.np_random.uniform(0.5, 1.5),
                    y=self.np_random.uniform(-2.0, 2.0),
                    theta=180
                )

            else:
                # Chaos
                pos_frame.ball = Ball(x=self.np_random.uniform(-3, 3), y=self.np_random.uniform(-2, 2))
                pos_frame.robots_yellow[0] = Robot(x=self.np_random.uniform(0.2, 3.5), y=self.np_random.uniform(-2.5, 2.5), theta=self.np_random.uniform(-180, 180))
                pos_frame.robots_blue[0] = Robot(x=self.np_random.uniform(-3.5, -0.2), y=self.np_random.uniform(-2.5, 2.5), theta=self.np_random.uniform(-180, 180))

        return pos_frame
