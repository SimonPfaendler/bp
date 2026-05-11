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
    yellows = env.frame.robots_yellow.values()
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
SINGLE_OBS_DIM = 38
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

    def __init__(self, render_mode=None, reward_type="dense", joint_action_mode=False):
        super().__init__(
            field_type=1,
            n_robots_blue=1,
            n_robots_yellow=N_YELLOW,
            time_step=0.025,
            render_mode=render_mode,
        )
        self.reward_type = reward_type
        self.joint_action_mode = joint_action_mode

        self.single_observation_space = Box(
            low=-self.NORM_BOUNDS,
            high=self.NORM_BOUNDS,
            shape=(SINGLE_OBS_DIM,),
            dtype=np.float32,
        )
        self.single_action_space = Box(
            low=-1.0, high=1.0, shape=(SINGLE_ACT_DIM,), dtype=np.float32
        )

        if joint_action_mode:
            # JAL: single agent perspective on joint obs/action.
            self.observation_space = Box(
                low=-self.NORM_BOUNDS,
                high=self.NORM_BOUNDS,
                shape=(N_YELLOW * SINGLE_OBS_DIM,),
                dtype=np.float32,
            )
            self.action_space = Box(
                low=-1.0,
                high=1.0,
                shape=(N_YELLOW * SINGLE_ACT_DIM,),
                dtype=np.float32,
            )
        else:
            # IL with parameter sharing (consumed by PairVecEnv).
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
        self.max_steps = 600  # L1 default; raised by set_curriculum_level for L≥2

        self.last_dist_ball_goal = None
        self.last_dist_to_ball = None
        self.last_ball_pos = None
        self.team_possession_steps = 0
        self.match_result = 0

        # Pass tracking
        self.last_yellow_carrier = None  # 0, 1, or None
        self.blue_touched_since_yellow = False
        self.passes_in_episode = 0
        self.last_action_pair = None
        self.last_yellow_kick_speed = 6.0

        # Per-yellow dribble-distance tracking (SSL "max 1m" rule).
        self.max_dribble_dist = 1.0
        self.min_release_distance = 0.1
        self.robot_ball_contact = 0.12
        self.is_dribbling = [False, False]
        self.dribble_start_pos = [None, None]
        self.must_release = [False, False]

        # Episode tracking
        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_start_time = time.time()

        self.curriculum_level = 1
        self.blue_personality = "defensive"


    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.last_dist_ball_goal = None
        self.last_dist_to_ball = None
        self.last_ball_pos = None
        self.team_possession_steps = 0
        self.match_result = 0
        self.last_yellow_carrier = None
        self.blue_touched_since_yellow = False
        self.passes_in_episode = 0
        self.last_action_pair = None
        self.last_yellow_kick_speed = 6.0
        self.is_dribbling = [False, False]
        self.dribble_start_pos = [None, None]
        self.must_release = [False, False]
        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_start_time = time.time()

        roll = self.np_random.random() if hasattr(self, "np_random") else 0.0
        self.blue_personality = "defensive" if roll < 0.6 else "aggressive"


        super().reset(seed=seed, options=options)
        obs = self._stacked_obs()
        if self.joint_action_mode:
            obs = obs.reshape(-1)
        return obs, {}

    def step(self, action):
        """In IL mode: action shape (2, 6). In JAL mode: action shape (12,)."""
        self.current_step += 1
        self.total_steps += 1

        action = np.asarray(action, dtype=np.float32)
        if self.joint_action_mode:
            action_pair = action.reshape(N_YELLOW, SINGLE_ACT_DIM)
        else:
            assert action.shape == (N_YELLOW, SINGLE_ACT_DIM), (
                f"expected ({N_YELLOW},{SINGLE_ACT_DIM}), got {action.shape}"
            )
            action_pair = action

        commands = self._build_commands(action_pair)
        self.rsim.send_commands(commands)
        self.sent_commands = commands
        self.last_action_pair = action_pair

        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        self._update_dribble_state()
        obs = self._stacked_obs()

        if self.joint_action_mode:
            reward, done, truncated = self._calculate_joint_reward_and_done()
            self.ep_reward += float(reward)
        else:
            reward, done, truncated = self._calculate_team_reward_and_done()
            self.ep_reward += float(reward.mean())
        self.ep_length += 1

        info = {}
        if done or truncated:
            info["is_success"] = 1.0 if self.match_result == 1 else 0.0
            info["match_result"] = self.match_result
            info["possession_ratio"] = self.team_possession_steps / max(
                1, self.current_step
            )
            info["passes"] = self.passes_in_episode
            info["scored_after_pass"] = 1.0 if (
                self.match_result == 1 and self.passes_in_episode > 0
            ) else 0.0
            info["episode"] = {
                "r": self.ep_reward,
                "l": self.ep_length,
                "t": round(time.time() - self.ep_start_time, 4),
            }

        if self.render_mode == "human":
            self.render()

        if self.joint_action_mode:
            return obs.reshape(-1), float(reward), bool(done), bool(truncated), info
        return obs, reward, bool(done), bool(truncated), info

    def set_curriculum_level(self, level: int):
        self.curriculum_level = int(level)
        # Shorter episodes early so more attempts fit per wall-clock and the
        # safe-passive equilibrium can't run out the clock as easily.
        self.max_steps = 600 if self.curriculum_level <= 1 else 1000

    # ---------- internals ----------

    def _stacked_obs(self) -> np.ndarray:
        ya, yb = self.frame.robots_yellow[0], self.frame.robots_yellow[1]
        blue = self.frame.robots_blue[0]
        obs_a = self._egocentric_obs(self_robot=ya, mate=yb, opp=blue, idx=0)
        obs_b = self._egocentric_obs(self_robot=yb, mate=ya, opp=blue, idx=1)
        return np.stack([obs_a, obs_b], axis=0).astype(np.float32)

    def _frame_to_observations(self):
        return self._stacked_obs()

    def _egocentric_obs(self, self_robot, mate, opp, idx) -> np.ndarray:
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
        i_am_closer = 1.0 if self_dist_ball < mate_dist_ball else 0.0

        if self.is_dribbling[idx] and self.dribble_start_pos[idx] is not None:
            start = self.dribble_start_pos[idx]
            cur_d = math.hypot(ball.x - start[0], ball.y - start[1])
            dribble_meter = float(np.clip(cur_d / self.max_dribble_dist, 0.0, 1.0))
        else:
            dribble_meter = 0.0
        must_release_flag = 1.0 if self.must_release[idx] else 0.0

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
                # Role tiebreaker (1 = I am the carrier-candidate)
                i_am_closer,
                # Dribble state (own)
                dribble_meter,
                must_release_flag,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -self.NORM_BOUNDS, self.NORM_BOUNDS)

    def _update_dribble_state(self):
        """Per-yellow SSL max-1m dribble enforcement."""
        ball = self.frame.ball
        yellows = (self.frame.robots_yellow[0], self.frame.robots_yellow[1])
        for i, y in enumerate(yellows):
            dist = math.hypot(y.x - ball.x, y.y - ball.y)
            has_contact = (dist < self.robot_ball_contact) or y.infrared

            if self.must_release[i] and dist >= self.min_release_distance:
                self.must_release[i] = False
                self.is_dribbling[i] = False
                self.dribble_start_pos[i] = None

            if has_contact:
                if not self.is_dribbling[i]:
                    self.is_dribbling[i] = True
                    self.dribble_start_pos[i] = np.array([ball.x, ball.y])
                else:
                    start = self.dribble_start_pos[i]
                    dribble_dist = math.hypot(
                        ball.x - start[0], ball.y - start[1]
                    )
                    if dribble_dist > self.max_dribble_dist:
                        self.must_release[i] = True
                        self.is_dribbling[i] = False
            else:
                if not self.must_release[i]:
                    self.is_dribbling[i] = False
                    self.dribble_start_pos[i] = None

    def convert_actions(self, action_array, angle):
        """Denormalize, clip to absolute max and convert to local."""
        v_x = action_array[0] * self.max_v_cmd
        v_y = action_array[1] * self.max_v_cmd
        v_theta = action_array[2] * self.max_w_cmd

        v_x_local = v_x * math.cos(angle) + v_y * math.sin(angle)
        v_y_local = -v_x * math.sin(angle) + v_y * math.cos(angle)

        v_norm = math.hypot(v_x_local, v_y_local)
        if v_norm > self.max_v_cmd:
            c = self.max_v_cmd / v_norm
            v_x_local *= c
            v_y_local *= c

        return v_x_local, v_y_local, v_theta

    def _yellow_command(self, robot, action, idx) -> Robot:
        """Apply low-level action [v_x, v_y, v_theta, kick_pow, kick_trig, dribble]."""
        raw_kick = float(action[3])
        kick_trigger = float(action[4])
        dribble_trigger = float(action[5])

        kick = max(0.0, raw_kick) * 6.0 if kick_trigger > 0.0 else 0.0
        dribble = dribble_trigger > 0.0

        if self.must_release[idx]:
            kick = 0.01
            dribble = False

        angle_rad = math.radians(robot.theta)
        v_x_local, v_y_local, w = self.convert_actions(
            [float(action[0]), float(action[1]), float(action[2])], angle_rad
        )

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
            bv_x, bv_y, bv_w = self.convert_actions(
                [cmd[0], cmd[1], cmd[2]], angle_rad
            )
            kick = 0.0
            dribble = False
        else:
            if self.blue_personality == "aggressive":
                if blue.infrared:
                    cmd = shoot_at_goal_center(self, blue, team_color="blue")
                else:
                    cmd = move_to_ball(blue, ball, speed=2.0)
            else:
                cmd = blue_defender_heuristic_2v1(self, blue)
            angle_rad = math.radians(blue.theta)
            bv_x, bv_y, bv_w = self.convert_actions(
                [cmd[0], cmd[1], cmd[2]], angle_rad
            )
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
            self._yellow_command(self.frame.robots_yellow[0], action_pair[0], 0)
        )
        cmds.append(
            self._yellow_command(self.frame.robots_yellow[1], action_pair[1], 1)
        )
        cmds.append(self._blue_command())
        return cmds

    def _get_commands(self, action):
        return self._build_commands(np.zeros((N_YELLOW, SINGLE_ACT_DIM), dtype=np.float32))

    def _calculate_reward_and_done(self):
        rewards, done, _ = self._calculate_team_reward_and_done()
        return float(rewards.mean()), done

    def _calculate_team_reward_and_done(self) -> Tuple[np.ndarray, bool, bool]:
        """JAL-style reward (after Ocana et al., 2019, 2v2 free-kick).

        Strictly non-negative per-step shaping (the only sink is a conceded
        goal). Holding the ball or stalling gives 0 — so SAC cannot find a
        passive equilibrium that beats any active strategy.

        Per-agent (each agent gets their own value):
          - Ball-closing: max(0, last_dist_i − dist_i) · 5, capped at 0.5.

        Shared (same to both agents):
          - max( ball-toward-any-agent, ball-toward-goal-mouth ) · 10,
            clipped to [0, 1.5].

        Terminal (shared):
          - Goal scored: +100
          - Goal conceded: -50
          - Ball OOB / yellow OOB / timeout: 0  (just end the episode)
        """
        ball = self.frame.ball
        ya, yb = self.frame.robots_yellow[0], self.frame.robots_yellow[1]
        yellows = (ya, yb)
        blue = self.frame.robots_blue[0]

        max_x = self.field.length / 2.0
        max_y = self.field.width / 2.0
        goal_half_width = self.field.goal_width / 2.0

        rewards = np.zeros(2, dtype=np.float32)
        done = False
        truncated = False

        # --- Terminal: goal / ball OOB ---
        if abs(ball.x) > max_x:
            done = True
            if abs(ball.y) <= goal_half_width:
                if ball.x < 0:  # Goal for yellow
                    rewards += 100.0
                    # Goal-after-pass bonus: pushes the policy to score via
                    # pass rather than solo. Capped at 2 passes.
                    rewards += 50.0 * min(self.passes_in_episode, 2)
                    self.match_result = 1
                else:  # Goal for blue
                    rewards -= 50.0
                    self.match_result = -1
            return rewards, done, truncated

        if abs(ball.y) > max_y:
            done = True
            return rewards, done, truncated

        # --- Yellow OOB: just terminate, no penalty ---
        for y in yellows:
            if abs(y.x) > max_x or abs(y.y) > max_y:
                done = True
                return rewards, done, truncated

        if self.current_step >= self.max_steps:
            truncated = True
            return rewards, done, truncated

        # --- Per-step shaping (≥ 0) ---
        if self.reward_type == "dense":
            dist_a = math.hypot(ya.x - ball.x, ya.y - ball.y)
            dist_b = math.hypot(yb.x - ball.x, yb.y - ball.y)
            dists = (dist_a, dist_b)

            # 1) Per-agent ball-closing (D^B_Ai), positive only.
            if self.last_dist_to_ball is None:
                self.last_dist_to_ball = [dist_a, dist_b]
            for i in range(2):
                delta = self.last_dist_to_ball[i] - dists[i]
                rewards[i] += float(np.clip(delta * 5.0, 0.0, 0.5))
            self.last_dist_to_ball = [dist_a, dist_b]

            # 2) max( ball→agent_i,  ball→goal ).
            ball_pos = np.array([ball.x, ball.y])
            ga = np.array([-max_x, goal_half_width])
            gb_pt = np.array([-max_x, -goal_half_width])
            gv = gb_pt - ga
            t = float(
                np.clip(np.dot(ball_pos - ga, gv) / np.dot(gv, gv), 0.0, 1.0)
            )
            closest_goal_pt = ga + t * gv
            dist_ball_goal = float(np.linalg.norm(ball_pos - closest_goal_pt))

            if self.last_dist_ball_goal is not None:
                goal_delta = self.last_dist_ball_goal - dist_ball_goal
            else:
                goal_delta = 0.0
            self.last_dist_ball_goal = dist_ball_goal

            if self.last_ball_pos is None:
                pass_delta = 0.0
            else:
                prev_bx, prev_by = self.last_ball_pos
                pass_delta = max(
                    math.hypot(prev_bx - ya.x, prev_by - ya.y) - dist_a,
                    math.hypot(prev_bx - yb.x, prev_by - yb.y) - dist_b,
                )
            self.last_ball_pos = (ball.x, ball.y)

            shared = max(pass_delta, goal_delta)
            rewards += float(np.clip(shared * 10.0, 0.0, 1.5))

            # Possession tracking (stats only, no per-step bonus).
            if (dist_a < 0.12) or ya.infrared or (dist_b < 0.12) or yb.infrared:
                self.team_possession_steps += 1

        # --- Pass detection ---
        ya_has = (math.hypot(ya.x - ball.x, ya.y - ball.y) < 0.20) or ya.infrared
        yb_has = (math.hypot(yb.x - ball.x, yb.y - ball.y) < 0.20) or yb.infrared
        blue_has = (
            math.hypot(blue.x - ball.x, blue.y - ball.y) < 0.12
        ) or blue.infrared

        # Pass-attempt shaping (per-agent): kick triggered while owning ball
        # and oriented toward teammate (±30°). Bridges the gradient gap between
        # "no pass" and "successful pass". Also records the kick speed of the
        # last yellow-owned kick, used to modulate the pass-event bonus.
        if self.last_action_pair is not None:
            has = (ya_has, yb_has)
            for i in range(2):
                if not has[i]:
                    continue
                kick_trig = float(self.last_action_pair[i, 4])
                if kick_trig <= 0.0:
                    continue
                y_self = yellows[i]
                y_mate = yellows[1 - i]
                dx = y_mate.x - y_self.x
                dy = y_mate.y - y_self.y
                angle_to_mate = math.atan2(dy, dx)
                self_theta = math.radians(y_self.theta)
                d_angle = abs(
                    ((angle_to_mate - self_theta + math.pi) % (2 * math.pi))
                    - math.pi
                )
                if d_angle <= math.radians(30):
                    rewards[i] += 2.0
                raw_k = max(0.0, float(self.last_action_pair[i, 3]))
                self.last_yellow_kick_speed = raw_k * 6.0

        if blue_has:
            self.blue_touched_since_yellow = True
            self.last_yellow_carrier = None

        current_carrier = None
        if ya_has and not yb_has:
            current_carrier = 0
        elif yb_has and not ya_has:
            current_carrier = 1

        if current_carrier is not None:
            if (
                self.last_yellow_carrier is not None
                and current_carrier != self.last_yellow_carrier
                and not self.blue_touched_since_yellow
            ):
                # Bonus modulated by softness of originating kick.
                # softness in [0,1]: 1 at 0 m/s, 0 at 6 m/s → bonus in [15, 45].
                softness = 1.0 - min(self.last_yellow_kick_speed / 6.0, 1.0)
                rewards += 15.0 + 30.0 * softness
                self.passes_in_episode += 1
            self.last_yellow_carrier = current_carrier
            self.blue_touched_since_yellow = False

        return rewards, done, truncated

    def _calculate_joint_reward_and_done(self) -> Tuple[float, bool, bool]:
        """JAL team reward (Ocana et al. 2019, Eq. 16). Single scalar.

            R = Σᵢ D^B_Aᵢ + max( {D^Aᵢ_B, ∀i}, D^G_B ) + G

        with the same non-negative shaping convention as the IL variant.
        """
        ball = self.frame.ball
        ya, yb = self.frame.robots_yellow[0], self.frame.robots_yellow[1]
        yellows = (ya, yb)
        blue = self.frame.robots_blue[0]

        max_x = self.field.length / 2.0
        max_y = self.field.width / 2.0
        goal_half_width = self.field.goal_width / 2.0

        reward = 0.0
        done = False
        truncated = False

        # --- Terminal: goal / ball OOB ---
        if abs(ball.x) > max_x:
            done = True
            if abs(ball.y) <= goal_half_width:
                if ball.x < 0:
                    reward += 100.0
                    self.match_result = 1
                else:
                    reward -= 50.0
                    self.match_result = -1
            return reward, done, truncated

        if abs(ball.y) > max_y:
            done = True
            return reward, done, truncated

        for y in yellows:
            if abs(y.x) > max_x or abs(y.y) > max_y:
                done = True
                return reward, done, truncated

        if self.current_step >= self.max_steps:
            truncated = True
            return reward, done, truncated

        # --- Per-step shaping (≥ 0) ---
        if self.reward_type == "dense":
            dist_a = math.hypot(ya.x - ball.x, ya.y - ball.y)
            dist_b = math.hypot(yb.x - ball.x, yb.y - ball.y)

            # Σᵢ D^B_Aᵢ — sum of per-agent ball-closing.
            if self.last_dist_to_ball is None:
                self.last_dist_to_ball = [dist_a, dist_b]
            for i, d in enumerate((dist_a, dist_b)):
                delta = self.last_dist_to_ball[i] - d
                reward += float(np.clip(delta * 5.0, 0.0, 0.5))
            self.last_dist_to_ball = [dist_a, dist_b]

            # max( ball→agent_i, ball→goal ) — single shared term, not doubled.
            ball_pos = np.array([ball.x, ball.y])
            ga = np.array([-max_x, goal_half_width])
            gb_pt = np.array([-max_x, -goal_half_width])
            gv = gb_pt - ga
            t = float(
                np.clip(np.dot(ball_pos - ga, gv) / np.dot(gv, gv), 0.0, 1.0)
            )
            closest_goal_pt = ga + t * gv
            dist_ball_goal = float(np.linalg.norm(ball_pos - closest_goal_pt))

            if self.last_dist_ball_goal is not None:
                goal_delta = self.last_dist_ball_goal - dist_ball_goal
            else:
                goal_delta = 0.0
            self.last_dist_ball_goal = dist_ball_goal

            if self.last_ball_pos is None:
                pass_delta = 0.0
            else:
                prev_bx, prev_by = self.last_ball_pos
                pass_delta = max(
                    math.hypot(prev_bx - ya.x, prev_by - ya.y) - dist_a,
                    math.hypot(prev_bx - yb.x, prev_by - yb.y) - dist_b,
                )
            self.last_ball_pos = (ball.x, ball.y)

            shared = max(pass_delta, goal_delta)
            reward += float(np.clip(shared * 10.0, 0.0, 1.5))

            if (dist_a < 0.12) or ya.infrared or (dist_b < 0.12) or yb.infrared:
                self.team_possession_steps += 1

        # --- Pass detection (stats only) ---
        ya_has = (math.hypot(ya.x - ball.x, ya.y - ball.y) < 0.20) or ya.infrared
        yb_has = (math.hypot(yb.x - ball.x, yb.y - ball.y) < 0.20) or yb.infrared
        blue_has = (
            math.hypot(blue.x - ball.x, blue.y - ball.y) < 0.12
        ) or blue.infrared

        if blue_has:
            self.blue_touched_since_yellow = True
            self.last_yellow_carrier = None

        current_carrier = None
        if ya_has and not yb_has:
            current_carrier = 0
        elif yb_has and not ya_has:
            current_carrier = 1

        if current_carrier is not None:
            if (
                self.last_yellow_carrier is not None
                and current_carrier != self.last_yellow_carrier
                and not self.blue_touched_since_yellow
            ):
                self.passes_in_episode += 1
            self.last_yellow_carrier = current_carrier
            self.blue_touched_since_yellow = False

        return reward, done, truncated

    def _get_initial_positions_frame(self) -> Frame:
        pos = Frame()
        level = self.curriculum_level
        rng = self.np_random

        if level == 1:
            sub_roll = rng.random()
            if sub_roll < 0.7:
                # Free ball, 2 yellows nearby (basic ball-chase scenario).
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
            else:
                # Pass drill: Yellow 0 has the ball, Blue is a static blocker
                # on the direct shot line, Yellow 1 stands in a clean
                # receiving slot on the opposite y-side. Mirror randomly so
                # the policy doesn't memorize a fixed pass direction.
                side = 1.0 if rng.random() < 0.5 else -1.0
                ya_x = rng.uniform(1.5, 2.3)
                ya_y = side * rng.uniform(0.4, 0.9)
                yb_x = rng.uniform(0.2, 0.9)
                yb_y = -side * rng.uniform(0.6, 1.1)
                blue_x = rng.uniform(0.3, 0.9)
                blue_y = side * rng.uniform(0.4, 0.9)
                # Yellow 0 faces Yellow 1 (with noise) so a forward kick is
                # already roughly a pass.
                theta_a = math.degrees(
                    math.atan2(yb_y - ya_y, yb_x - ya_x)
                ) + rng.uniform(-15.0, 15.0)
                theta_b = 180.0 + rng.uniform(-25.0, 25.0)
                # Ball just in front of Yellow 0 along its facing direction.
                theta_a_rad = math.radians(theta_a)
                ball_x = ya_x + math.cos(theta_a_rad) * 0.11
                ball_y = ya_y + math.sin(theta_a_rad) * 0.11
                pos.ball = Ball(x=ball_x, y=ball_y)
                pos.robots_yellow[0] = Robot(x=ya_x, y=ya_y, theta=theta_a)
                pos.robots_yellow[1] = Robot(x=yb_x, y=yb_y, theta=theta_b)
                pos.robots_blue[0] = Robot(x=blue_x, y=blue_y, theta=0.0)

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
                pos.ball = Ball(x=blue_x + 0.30, y=blue_y)
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
