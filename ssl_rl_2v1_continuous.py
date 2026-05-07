"""
2v1 Robot Soccer Environment with parameter-sharing support.

Layout: 2 yellow attackers (shared policy) vs 1 blue defender (heuristic).
Yellow attacks the -x goal (consistent with the 1v1 env).

This env is NOT a standard gym.Env: step() takes a stacked (2, action_dim)
action and returns a stacked (2, obs_dim) observation plus a per-agent
reward array. Use it via PairVecEnv (see pair_vec_env.py) so SB3 sees
2 * n_pairs parallel "agent slots" backed by n_pairs physics simulators.
"""

import math
import time
from typing import Tuple

import numpy as np
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

from skills import (
    move_to_ball,
    move_to_point,
    shoot_at_goal_center,
    turn_to_point,
)


def blue_defender_heuristic_2v1(env, robot):
    """Lone blue defender vs two yellow attackers.

    Mirrors the structure of the 1v1 blue heuristic but treats the closest
    yellow as the threat-of-record.
    """
    ball = env.frame.ball
    yellows = env.frame.robots_yellow
    closest_yellow = min(
        yellows, key=lambda y: math.hypot(y.x - ball.x, y.y - ball.y)
    )
    defend_goal_x = -env.field.length / 2.0

    if robot.infrared:
        return shoot_at_goal_center(env, robot, team_color="blue")

    dist_blue_ball = math.hypot(robot.x - ball.x, robot.y - ball.y)
    dist_yellow_ball = math.hypot(
        closest_yellow.x - ball.x, closest_yellow.y - ball.y
    )

    if dist_yellow_ball < dist_blue_ball:
        # Stand on the goal-ball line.
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

    ball_speed = math.hypot(ball.v_x, ball.v_y)
    if ball_speed > 0.3:
        dx, dy = ball.x - robot.x, ball.y - robot.y
        s = 1.2
        a = ball_speed * ball_speed - s * s
        b = 2.0 * (dx * ball.v_x + dy * ball.v_y)
        c = dx * dx + dy * dy
        t = None
        if abs(a) < 1e-6:
            if abs(b) > 1e-6:
                cand = -c / b
                if cand > 0.0:
                    t = cand
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sq = math.sqrt(max(disc, 0.0))
                roots = [
                    r
                    for r in (
                        (-b - sq) / (2.0 * a),
                        (-b + sq) / (2.0 * a),
                    )
                    if r > 0.0
                ]
                if roots:
                    t = min(roots)
        if t is not None and t < 2.0:
            target = np.array(
                [ball.x + ball.v_x * t, ball.y + ball.v_y * t]
            )
            dx, dy = target[0] - robot.x, target[1] - robot.y
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                v_x, v_y = (dx / dist) * s, (dy / dist) * s
            else:
                v_x, v_y = 0.0, 0.0
            v_theta = turn_to_point(robot, target)
            return np.array([v_x, v_y, v_theta, 0.0, 1.0])

    return move_to_ball(robot, ball, speed=2.0)


# Observation layout (per agent, all in the agent's local frame):
#   [0:5]   Self:     v_x, v_y, v_theta, infrared, has_ball
#   [5:11]  Ball:     rel_x, rel_y, rel_v_x, rel_v_y, dist, bearing
#   [11:13] Pred ball (0.5s ahead): rel_x, rel_y
#   [13:20] Teammate: rel_x, rel_y, rel_v_x, rel_v_y, dist, sin/cos d_theta
#   [20:25] Opponent: rel_x, rel_y, rel_v_x, rel_v_y, dist
#   [25:28] Attack goal: rel_x, rel_y, dist
#   [28:30] Own goal:    rel_x, rel_y
#   [30:34] Wall distances
#   [34:35] Team possession
SINGLE_OBS_DIM = 35
SINGLE_ACT_DIM = 6  # [v_x, v_y, v_theta, kick_power, kick_trigger, dribble]
N_YELLOW = 2


class SSL2v1SharedEnv(SSLBaseEnv):
    """Pair env: 2 yellow agents share a policy, 1 blue heuristic.

    Step takes actions of shape (2, 6); returns:
      obs:      (2, SINGLE_OBS_DIM) float32
      rewards:  (2,)               float32  (shared team reward, duplicated)
      done:     bool
      truncated:bool
      info:     dict
    """

    def __init__(self, render_mode=None, reward_type="dense"):
        super().__init__(
            field_type=1,
            n_robots_blue=1,
            n_robots_yellow=N_YELLOW,
            time_step=0.025,
            render_mode=render_mode,
        )
        self.reward_type = reward_type

        self.single_observation_space = Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(SINGLE_OBS_DIM,),
            dtype=np.float32,
        )
        self.single_action_space = Box(
            low=-1.0, high=1.0, shape=(SINGLE_ACT_DIM,), dtype=np.float32
        )

        self.observation_space = Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(N_YELLOW, SINGLE_OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(N_YELLOW, SINGLE_ACT_DIM),
            dtype=np.float32,
        )

        self.max_v_cmd = 2.0
        self.max_w_cmd = 10.0

        self.current_step = 0
        self.total_steps = 0
        self.max_steps = 1500

        self.last_min_dist_robot_ball = None
        self.last_dist_ball_goal = None
        self.team_possession_steps = 0
        self.match_result = 0

        # Episode tracking
        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_start_time = time.time()

        self.curriculum_level = 1
        self.blue_personality = "defensive"


    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.last_min_dist_robot_ball = None
        self.last_dist_ball_goal = None
        self.team_possession_steps = 0
        self.match_result = 0
        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_start_time = time.time()

        roll = self.np_random.random() if hasattr(self, "np_random") else 0.0
        self.blue_personality = "defensive" if roll < 0.6 else "aggressive"


        super().reset(seed=seed, options=options)
        return self._stacked_obs(), {}

    def step(self, action_pair):
        """action_pair: ndarray of shape (2, SINGLE_ACT_DIM)."""
        self.current_step += 1
        self.total_steps += 1

        action_pair = np.asarray(action_pair, dtype=np.float32)
        assert action_pair.shape == (N_YELLOW, SINGLE_ACT_DIM), (
            f"expected ({N_YELLOW},{SINGLE_ACT_DIM}), got {action_pair.shape}"
        )

        commands = self._build_commands(action_pair)
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        obs = self._stacked_obs()
        team_r, done, truncated = self._calculate_team_reward_and_done()
        rewards = np.array([team_r, team_r], dtype=np.float32)

        self.ep_reward += float(team_r)
        self.ep_length += 1

        info = {}
        if done or truncated:
            info["is_success"] = 1.0 if self.match_result == 1 else 0.0
            info["match_result"] = self.match_result
            info["possession_ratio"] = self.team_possession_steps / max(
                1, self.current_step
            )
            info["episode"] = {
                "r": self.ep_reward,
                "l": self.ep_length,
                "t": round(time.time() - self.ep_start_time, 4),
            }

        if self.render_mode == "human":
            self.render()

        return obs, rewards, bool(done), bool(truncated), info

    def set_curriculum_level(self, level: int):
        self.curriculum_level = int(level)

    # ---------- internals ----------

    def _stacked_obs(self) -> np.ndarray:
        ya, yb = self.frame.robots_yellow[0], self.frame.robots_yellow[1]
        blue = self.frame.robots_blue[0]
        obs_a = self._egocentric_obs(self_robot=ya, mate=yb, opp=blue)
        obs_b = self._egocentric_obs(self_robot=yb, mate=ya, opp=blue)
        return np.stack([obs_a, obs_b], axis=0).astype(np.float32)

    def _frame_to_observations(self):
        return self._stacked_obs()

    def _egocentric_obs(self, self_robot, mate, opp) -> np.ndarray:
        ball = self.frame.ball
        max_x = self.field.length / 2.0
        max_y = self.field.width / 2.0
        max_dist = math.hypot(self.field.length, self.field.width)

        theta = math.radians(self_robot.theta)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def to_local_pos(x, y):
            dx, dy = x - self_robot.x, y - self_robot.y
            return dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t

        def to_local_vec(vx, vy):
            return vx * cos_t + vy * sin_t, -vx * sin_t + vy * cos_t

        # Self
        own_vx, own_vy = to_local_vec(self_robot.v_x, self_robot.v_y)
        own_w = self_robot.v_theta
        own_ir = 1.0 if self_robot.infrared else 0.0
        self_dist_ball = math.hypot(
            self_robot.x - ball.x, self_robot.y - ball.y
        )
        self_has_ball = (self_dist_ball < 0.12) or self_robot.infrared
        mate_dist_ball = math.hypot(mate.x - ball.x, mate.y - ball.y)
        mate_has_ball = (mate_dist_ball < 0.12) or mate.infrared

        # Ball
        ball_rx, ball_ry = to_local_pos(ball.x, ball.y)
        ball_rvx, ball_rvy = to_local_vec(ball.v_x, ball.v_y)
        ball_dist = math.hypot(ball_rx, ball_ry)
        ball_bearing = math.atan2(ball_ry, ball_rx)

        # Predicted ball (0.5s)
        pred_x = np.clip(ball.x + ball.v_x * 0.5, -max_x, max_x)
        pred_y = np.clip(ball.y + ball.v_y * 0.5, -max_y, max_y)
        pred_rx, pred_ry = to_local_pos(pred_x, pred_y)

        # Teammate
        mate_rx, mate_ry = to_local_pos(mate.x, mate.y)
        mate_rvx, mate_rvy = to_local_vec(mate.v_x, mate.v_y)
        mate_dist = math.hypot(mate_rx, mate_ry)
        d_theta = math.radians(mate.theta) - theta

        # Opponent
        opp_rx, opp_ry = to_local_pos(opp.x, opp.y)
        opp_rvx, opp_rvy = to_local_vec(opp.v_x, opp.v_y)
        opp_dist = math.hypot(opp_rx, opp_ry)

        # Attack goal
        gh = self.field.goal_width / 2.0
        ball_pos = np.array([ball.x, ball.y])
        ga = np.array([-max_x, gh])
        gb = np.array([-max_x, -gh])
        gv = gb - ga
        t = float(np.clip(np.dot(ball_pos - ga, gv) / np.dot(gv, gv), 0.0, 1.0))
        attack_pt = ga + t * gv
        attack_rx, attack_ry = to_local_pos(attack_pt[0], attack_pt[1])
        attack_dist = math.hypot(attack_rx, attack_ry)

        own_rx, own_ry = to_local_pos(max_x, 0.0)

        # Walls
        d_wall_neg_x = (self_robot.x - (-max_x)) / max_x
        d_wall_pos_x = (max_x - self_robot.x) / max_x
        d_wall_neg_y = (self_robot.y - (-max_y)) / max_y
        d_wall_pos_y = (max_y - self_robot.y) / max_y

        team_has_ball = 1.0 if (self_has_ball or mate_has_ball) else 0.0

        obs = np.array(
            [
                # Self
                self.norm_v(own_vx),
                self.norm_v(own_vy),
                self.norm_w(own_w),
                own_ir,
                1.0 if self_has_ball else 0.0,
                # Ball
                ball_rx / max_dist,
                ball_ry / max_dist,
                self.norm_v(ball_rvx),
                self.norm_v(ball_rvy),
                ball_dist / max_dist,
                ball_bearing / math.pi,
                # Pred ball
                pred_rx / max_dist,
                pred_ry / max_dist,
                # Teammate
                mate_rx / max_dist,
                mate_ry / max_dist,
                self.norm_v(mate_rvx),
                self.norm_v(mate_rvy),
                mate_dist / max_dist,
                math.sin(d_theta),
                math.cos(d_theta),
                # Opponent
                opp_rx / max_dist,
                opp_ry / max_dist,
                self.norm_v(opp_rvx),
                self.norm_v(opp_rvy),
                opp_dist / max_dist,
                # Attack goal
                attack_rx / max_dist,
                attack_ry / max_dist,
                attack_dist / max_dist,
                # Own goal
                own_rx / max_dist,
                own_ry / max_dist,
                # Walls
                d_wall_neg_x,
                d_wall_pos_x,
                d_wall_neg_y,
                d_wall_pos_y,
                # Team possession
                team_has_ball,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -self.NORM_BOUNDS, self.NORM_BOUNDS)

    def _yellow_command(self, robot, action) -> Robot:
        """Apply low-level action [v_x, v_y, v_theta, kick_pow, kick_trig, dribble]."""
        v_x_global = float(action[0])
        v_y_global = float(action[1])
        v_theta = float(action[2])
        raw_kick = float(action[3])
        kick_trigger = float(action[4])
        dribble_trigger = float(action[5])

        kick = (3.0 + ((raw_kick + 1.0) / 2.0) * 3.0) if kick_trigger > 0.0 else 0.0
        dribble = dribble_trigger > 0.0

        angle_rad = math.radians(robot.theta)
        v_x = v_x_global * self.max_v_cmd
        v_y = v_y_global * self.max_v_cmd
        w = v_theta * self.max_w_cmd

        v_x_local = v_x * math.cos(angle_rad) + v_y * math.sin(angle_rad)
        v_y_local = -v_x * math.sin(angle_rad) + v_y * math.cos(angle_rad)
        v_norm = math.hypot(v_x_local, v_y_local)
        if v_norm > self.max_v_cmd:
            c = self.max_v_cmd / v_norm
            v_x_local *= c
            v_y_local *= c

        return Robot(
            yellow=True,
            id=robot.id,
            v_x=v_x_local,
            v_y=v_y_local,
            v_theta=w,
            kick_v_x=kick,
            dribbler=dribble,
        )

    def _blue_command(self) -> Robot:
        blue = self.frame.robots_blue[0]
        ball = self.frame.ball
        level = self.curriculum_level

        if level <= 2:
            bv_x = bv_y = bv_w = 0.0
            kick = 0.0
            dribble = False
        elif level == 3:
            cmd = move_to_ball(blue, ball, speed=0.5)
            angle_rad = math.radians(blue.theta)
            bv_x_g, bv_y_g, bv_w = cmd[0], cmd[1], cmd[2]
            bv_x_g *= self.max_v_cmd
            bv_y_g *= self.max_v_cmd
            bv_w *= self.max_w_cmd
            bv_x = bv_x_g * math.cos(angle_rad) + bv_y_g * math.sin(angle_rad)
            bv_y = -bv_x_g * math.sin(angle_rad) + bv_y_g * math.cos(angle_rad)
            kick = 0.0
            dribble = False
        else:
            if self.blue_personality == "aggressive":
                if blue.infrared:
                    cmd = shoot_at_goal_center(self, blue, team_color="blue")
                else:
                    cmd = move_to_ball(blue, ball, speed=1.5)
            else:
                cmd = blue_defender_heuristic_2v1(self, blue)
            angle_rad = math.radians(blue.theta)
            bv_x_g = cmd[0] * self.max_v_cmd
            bv_y_g = cmd[1] * self.max_v_cmd
            bv_w = cmd[2] * self.max_w_cmd
            bv_x = bv_x_g * math.cos(angle_rad) + bv_y_g * math.sin(angle_rad)
            bv_y = -bv_x_g * math.sin(angle_rad) + bv_y_g * math.cos(angle_rad)
            kick = float(cmd[3])
            dribble = bool(cmd[4] > 0)

        return Robot(
            yellow=False,
            id=0,
            v_x=bv_x,
            v_y=bv_y,
            v_theta=bv_w,
            kick_v_x=kick,
            dribbler=dribble,
        )

    def _build_commands(self, action_pair: np.ndarray):
        cmds = []
        cmds.append(
            self._yellow_command(self.frame.robots_yellow[0], action_pair[0])
        )
        cmds.append(
            self._yellow_command(self.frame.robots_yellow[1], action_pair[1])
        )
        cmds.append(self._blue_command())
        return cmds

    def _get_commands(self, action):
        return self._build_commands(np.zeros((N_YELLOW, SINGLE_ACT_DIM), dtype=np.float32))

    def _calculate_reward_and_done(self):
        team_r, done, _ = self._calculate_team_reward_and_done()
        return team_r, done

    def _calculate_team_reward_and_done(self) -> Tuple[float, bool, bool]:
        ball = self.frame.ball
        ya, yb = self.frame.robots_yellow[0], self.frame.robots_yellow[1]

        max_x = self.field.length / 2.0
        max_y = self.field.width / 2.0
        goal_half_width = self.field.goal_width / 2.0

        reward = 0.0
        done = False
        truncated = False

        if self.reward_type == "dense":
            progress = self.current_step / self.max_steps
            if ball.x < 0:
                reward -= 0.02 * (1.0 + 2.0 * progress)
            else:
                reward -= 0.04 * (1.0 + 2.0 * progress)

        # Ball out of pitch: end conditions.
        if abs(ball.x) > max_x:
            done = True
            if abs(ball.y) <= goal_half_width:
                if ball.x < 0:  # Goal for yellow
                    reward += 100.0
                    reward += (self.max_steps - self.current_step) * 0.01
                    self.match_result = 1
                else:  # Goal for blue
                    reward -= 50.0
                    self.match_result = -1
            else:
                reward -= 5.0
            return reward, done, truncated

        if abs(ball.y) > max_y:
            done = True
            reward -= 5.0
            self.match_result = -1
            return reward, done, truncated

        # Yellow OOB
        for y in (ya, yb):
            if abs(y.x) > max_x or abs(y.y) > max_y:
                done = True
                level = self.curriculum_level
                if level <= 2:
                    reward -= 20.0
                elif level == 3:
                    reward -= 50.0
                else:
                    reward -= 200.0
                self.match_result = -1
                return reward, done, truncated

        if self.current_step >= self.max_steps:
            truncated = True
            done = False
            reward -= 10.0
            self.match_result = -1
            return reward, done, truncated

        if self.reward_type == "dense":
            dist_a = math.hypot(ya.x - ball.x, ya.y - ball.y)
            dist_b = math.hypot(yb.x - ball.x, yb.y - ball.y)
            min_dist = min(dist_a, dist_b)
            yellow_has_ball = (
                (dist_a < 0.12) or ya.infrared or (dist_b < 0.12) or yb.infrared
            )

            max_dist = math.hypot(self.field.length, self.field.width)
            # Constant gradient toward ball
            reward += 0.05 * (1.0 - min_dist / max_dist)

            # Penalize standing-still
            for y in (ya, yb):
                speed = math.hypot(y.v_x, y.v_y)
                if speed < 0.1 and not yellow_has_ball:
                    reward -= 0.025

            # Closing distance to ball
            if self.last_min_dist_robot_ball is not None:
                delta = self.last_min_dist_robot_ball - min_dist
                reward += float(np.clip(delta * 5.0, -0.5, 0.5))
            self.last_min_dist_robot_ball = min_dist

            # Ball -> goal progress
            ball_pos = np.array([ball.x, ball.y])
            ga = np.array([-max_x, goal_half_width])
            gb = np.array([-max_x, -goal_half_width])
            gv = gb - ga
            t = float(np.clip(np.dot(ball_pos - ga, gv) / np.dot(gv, gv), 0.0, 1.0))
            closest_goal_pt = ga + t * gv
            dist_ball_goal = float(np.linalg.norm(ball_pos - closest_goal_pt))

            if self.last_dist_ball_goal is not None:
                delta_bg = self.last_dist_ball_goal - dist_ball_goal
                reward += float(np.clip(delta_bg * 10.0, -1.0, 1.5))
            self.last_dist_ball_goal = dist_ball_goal

            if yellow_has_ball:
                reward += 0.01
                self.team_possession_steps += 1

            if ball.v_x < -0.5:
                reward += 0.02 * min(-ball.v_x, 3.0)
            if ball.v_x > 0.5:
                reward -= 0.02 * min(ball.v_x, 3.0)

        return reward, done, truncated

    def _get_initial_positions_frame(self) -> Frame:
        pos = Frame()
        level = self.curriculum_level
        rng = self.np_random

        if level == 1:
            # Free ball 2 yellows nearby
            bx = rng.uniform(-1.0, 2.0)
            by = rng.uniform(-1.5, 1.5)
            pos.ball = Ball(x=bx, y=by)
            pos.robots_yellow[0] = Robot(
                x=bx + rng.uniform(0.4, 1.2),
                y=by + rng.uniform(-0.6, 0.6),
                theta=rng.uniform(135.0, 225.0),
            )
            pos.robots_yellow[1] = Robot(
                x=bx + rng.uniform(0.4, 1.5),
                y=by + rng.uniform(-1.5, 1.5),
                theta=rng.uniform(135.0, 225.0),
            )
            pos.robots_blue[0] = Robot(x=0.0, y=3.0, theta=0.0)

        elif level == 2:
            bx = rng.uniform(-1.0, 2.0)
            by = rng.uniform(-1.5, 1.5)
            pos.ball = Ball(x=bx, y=by)
            pos.robots_yellow[0] = Robot(
                x=rng.uniform(2.0, 3.5),
                y=rng.uniform(-2.0, 2.0),
                theta=rng.uniform(-180, 180),
            )
            pos.robots_yellow[1] = Robot(
                x=rng.uniform(2.0, 3.5),
                y=rng.uniform(-2.0, 2.0),
                theta=rng.uniform(-180, 180),
            )
            goal_x = -self.field.length / 2.0
            pos.robots_blue[0] = Robot(
                x=goal_x + 0.2,
                y=rng.uniform(-0.3, 0.3),
                theta=0.0,
            )

        elif level == 3:
            bx = rng.uniform(-1.0, 2.0)
            by = rng.uniform(-1.5, 1.5)
            pos.ball = Ball(x=bx, y=by)
            pos.robots_yellow[0] = Robot(
                x=rng.uniform(2.0, 3.5),
                y=rng.uniform(-2.0, 2.0),
                theta=rng.uniform(-180, 180),
            )
            pos.robots_yellow[1] = Robot(
                x=rng.uniform(2.0, 3.5),
                y=rng.uniform(-2.0, 2.0),
                theta=rng.uniform(-180, 180),
            )
            pos.robots_blue[0] = Robot(
                x=rng.uniform(-3.5, -1.0),
                y=rng.uniform(-1.5, 1.5),
                theta=rng.uniform(-180, 180),
            )

        elif level == 4:
            bx = rng.uniform(-1.0, 2.0)
            by = rng.uniform(-1.5, 1.5)
            pos.ball = Ball(x=bx, y=by)
            pos.robots_yellow[0] = Robot(
                x=bx + rng.uniform(0.3, 1.2),
                y=by + rng.uniform(-0.5, 0.5),
                theta=rng.uniform(135.0, 225.0),
            )
            pos.robots_yellow[1] = Robot(
                x=rng.uniform(1.0, 3.5),
                y=rng.uniform(-2.0, 2.0),
                theta=rng.uniform(-180, 180),
            )
            pos.robots_blue[0] = Robot(
                x=rng.uniform(-3.5, bx - 0.5),
                y=rng.uniform(-1.5, 1.5),
                theta=rng.uniform(-180, 180),
            )

        else:
            scenario_roll = rng.random()
            if scenario_roll < 0.33:
                # Coordinated attack
                bx = rng.uniform(-1.0, 2.0)
                by = rng.uniform(-1.5, 1.5)
                pos.ball = Ball(x=bx, y=by)
                pos.robots_yellow[0] = Robot(x=bx + 0.5, y=by, theta=180)
                pos.robots_yellow[1] = Robot(x=bx + 1.0, y=by + 1.0, theta=180)
                pos.robots_blue[0] = Robot(x=bx - 1.0, y=by, theta=0)
            elif scenario_roll < 0.66:
                # Defend
                blue_x = rng.uniform(-1.0, 2.0)
                blue_y = rng.uniform(-2.0, 2.0)
                pos.robots_blue[0] = Robot(x=blue_x, y=blue_y, theta=0)
                pos.ball = Ball(x=blue_x + 0.15, y=blue_y)
                pos.robots_yellow[0] = Robot(
                    x=blue_x + rng.uniform(0.5, 1.5),
                    y=blue_y + rng.uniform(-0.3, 0.3),
                    theta=180,
                )
                pos.robots_yellow[1] = Robot(
                    x=blue_x + rng.uniform(1.0, 2.0),
                    y=blue_y + rng.uniform(-1.5, 1.5),
                    theta=180,
                )
            else:
                pos.ball = Ball(
                    x=rng.uniform(-3, 3), y=rng.uniform(-2, 2)
                )
                pos.robots_yellow[0] = Robot(
                    x=rng.uniform(0.2, 3.5),
                    y=rng.uniform(-2.5, 2.5),
                    theta=rng.uniform(-180, 180),
                )
                pos.robots_yellow[1] = Robot(
                    x=rng.uniform(0.2, 3.5),
                    y=rng.uniform(-2.5, 2.5),
                    theta=rng.uniform(-180, 180),
                )
                pos.robots_blue[0] = Robot(
                    x=rng.uniform(-3.5, -0.2),
                    y=rng.uniform(-2.5, 2.5),
                    theta=rng.uniform(-180, 180),
                )

        return pos
