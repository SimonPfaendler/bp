"""2v2 SSL self-play env for parameter-sharing IL with a frozen opponent.

Layout: 2 yellow attackers (training side) vs 2 blue defenders. Blue robots
are controlled by a frozen SAC checkpoint loaded from `frozen_path`. Each
team's egocentric observation uses its own attack-goal reference, so a policy
trained on the yellow side acts directly as blue without coordinate flipping.

Observation shape (2, 38) and action shape (2, 6) match the 2v1 IL setup, so
a 2v1-trained model can be loaded as v0 frozen opponent AND as the yellow-side
init for the training policy.

"Closest Opponent" simplification: each agent sees its nearest opponent in
the opp-slot of the obs vector. The second opponent is reflected only
indirectly via the ball / teammate features.
"""

import math
import time
from typing import Tuple

import numpy as np
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from stable_baselines3 import SAC


SINGLE_OBS_DIM_BASE = 38  # 2v1-IL compatible obs without role-index
ROLE_INDEX_DIM = 2
SINGLE_ACT_DIM = 6
N_YELLOW = 2
N_BLUE = 2


class SSL2v2SelfPlayEnv(SSLBaseEnv):
    """2v2 self-play env. Yellow side is exposed to SB3 via PairVecEnv;
    blue side is computed internally from a frozen SAC checkpoint.

    Step takes yellow actions of shape (2, 6) and returns:
      obs:    (2, 38) float32
      reward: (2,)    float32  (yellow team reward, duplicated)
      done:   bool
      truncated: bool
      info:   dict
    """

    def __init__(
        self, render_mode=None, reward_type="dense", frozen_path=None,
        role_index=False, oob_grace_steps=0,
    ):
        super().__init__(
            field_type=1,
            n_robots_blue=N_BLUE,
            n_robots_yellow=N_YELLOW,
            time_step=0.025,
            render_mode=render_mode,
        )
        self.reward_type = reward_type
        self.frozen_path = frozen_path
        self.frozen_model = None  # lazy-load on first step (subproc-safe)
        # Frozen-model input dim (filled on lazy-load). If older than current
        # obs (e.g. v3 trained without role-index), we strip role-index dims
        # before predict so the same policy class can act as blue.
        self.frozen_obs_dim = None

        # Role-index: one-hot agent identity appended to obs. Breaks the
        # permutation symmetry between yellow_a/yellow_b so the shared policy
        # can specialize into roles (carrier/receiver, attacker/defender).
        self.role_index = role_index
        self.single_obs_dim = (
            SINGLE_OBS_DIM_BASE + ROLE_INDEX_DIM if role_index
            else SINGLE_OBS_DIM_BASE
        )

        # OOB curriculum: skip robot-OOB termination during the first
        # `oob_grace_steps` per-env steps so early-stage agents get more
        # productive practice instead of episodes ending the moment a robot
        # wanders off the field.
        self.oob_grace_steps = int(oob_grace_steps)

        self.single_observation_space = Box(
            low=-self.NORM_BOUNDS, high=self.NORM_BOUNDS,
            shape=(self.single_obs_dim,), dtype=np.float32,
        )
        self.single_action_space = Box(
            low=-1.0, high=1.0, shape=(SINGLE_ACT_DIM,), dtype=np.float32,
        )
        self.observation_space = Box(
            low=-self.NORM_BOUNDS, high=self.NORM_BOUNDS,
            shape=(N_YELLOW, self.single_obs_dim), dtype=np.float32,
        )
        self.action_space = Box(
            low=-1.0, high=1.0,
            shape=(N_YELLOW, SINGLE_ACT_DIM), dtype=np.float32,
        )

        self.max_v_cmd = 2.0
        self.max_w_cmd = 10.0

        self.current_step = 0
        self.total_steps = 0
        self.max_steps = 1000

        self.last_dist_ball_goal = None
        self.last_dist_to_ball = None
        self.last_ball_pos = None
        self.team_possession_steps = 0
        self.match_result = 0

        self.last_yellow_carrier = None
        self.blue_touched_since_yellow = False
        self.passes_in_episode = 0
        self.blue_goal_scored = False

        # SSL max-1m dribble enforcement, per team
        self.max_dribble_dist = 1.0
        self.min_release_distance = 0.1
        self.robot_ball_contact = 0.12
        self.is_dribbling_y = [False, False]
        self.dribble_start_pos_y = [None, None]
        self.must_release_y = [False, False]
        self.is_dribbling_b = [False, False]
        self.dribble_start_pos_b = [None, None]
        self.must_release_b = [False, False]

        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_start_time = time.time()

        # Self-play: no curriculum (kept as attribute for callback compat).
        self.curriculum_level = 5

    # ---------- frozen model ----------

    def _maybe_load_frozen(self):
        if self.frozen_model is None and self.frozen_path:
            self.frozen_model = SAC.load(self.frozen_path, device="cpu")
            self.frozen_obs_dim = int(
                self.frozen_model.policy.observation_space.shape[-1]
            )

    # ---------- gym API ----------

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
        self.blue_goal_scored = False
        self.is_dribbling_y = [False, False]
        self.dribble_start_pos_y = [None, None]
        self.must_release_y = [False, False]
        self.is_dribbling_b = [False, False]
        self.dribble_start_pos_b = [None, None]
        self.must_release_b = [False, False]
        self.ep_reward = 0.0
        self.ep_length = 0
        self.ep_start_time = time.time()

        super().reset(seed=seed, options=options)
        self._maybe_load_frozen()
        obs = self._stacked_obs_yellow()
        return obs, {}

    def step(self, yellow_action):
        self.current_step += 1
        self.total_steps += 1
        yellow_action = np.asarray(yellow_action, dtype=np.float32)
        assert yellow_action.shape == (N_YELLOW, SINGLE_ACT_DIM), (
            f"expected ({N_YELLOW},{SINGLE_ACT_DIM}), "
            f"got {yellow_action.shape}"
        )

        blue_action = self._compute_blue_action()
        commands = self._build_commands(yellow_action, blue_action)
        self.rsim.send_commands(commands)
        self.sent_commands = commands

        self.last_frame = self.frame
        self.frame = self.rsim.get_frame()

        self._update_dribble_state()
        obs = self._stacked_obs_yellow()

        reward, done, truncated = self._calculate_team_reward_and_done()
        self.ep_reward += float(reward.mean())
        self.ep_length += 1

        info = {}
        if done or truncated:
            info["is_success"] = 1.0 if self.match_result == 1 else 0.0
            info["blue_goal"] = 1.0 if self.blue_goal_scored else 0.0
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
        return obs, reward, bool(done), bool(truncated), info

    def set_curriculum_level(self, level: int):
        # No-op in self-play, kept for compat with shared CurriculumCallback.
        self.curriculum_level = int(level)

    # ---------- observation ----------

    def _stacked_obs_yellow(self) -> np.ndarray:
        ya, yb = self.frame.robots_yellow[0], self.frame.robots_yellow[1]
        opp = (self.frame.robots_blue[0], self.frame.robots_blue[1])
        obs_a = self._egocentric_obs(
            self_robot=ya, mate=yb, opp_list=opp,
            attack_goal_x=-self.field.length / 2.0,
            is_yellow=True, idx=0,
        )
        obs_b = self._egocentric_obs(
            self_robot=yb, mate=ya, opp_list=opp,
            attack_goal_x=-self.field.length / 2.0,
            is_yellow=True, idx=1,
        )
        return np.stack([obs_a, obs_b], axis=0).astype(np.float32)

    def _stacked_obs_blue(self) -> np.ndarray:
        ba, bb = self.frame.robots_blue[0], self.frame.robots_blue[1]
        opp = (self.frame.robots_yellow[0], self.frame.robots_yellow[1])
        obs_a = self._egocentric_obs(
            self_robot=ba, mate=bb, opp_list=opp,
            attack_goal_x=+self.field.length / 2.0,
            is_yellow=False, idx=0,
        )
        obs_b = self._egocentric_obs(
            self_robot=bb, mate=ba, opp_list=opp,
            attack_goal_x=+self.field.length / 2.0,
            is_yellow=False, idx=1,
        )
        return np.stack([obs_a, obs_b], axis=0).astype(np.float32)

    def _frame_to_observations(self):
        return self._stacked_obs_yellow()

    def _egocentric_obs(
        self, self_robot, mate, opp_list,
        attack_goal_x, is_yellow, idx,
    ) -> np.ndarray:
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

        # Closest opponent only — keeps obs shape compatible with 2v1 IL.
        opp = min(
            opp_list,
            key=lambda o: math.hypot(o.x - self_robot.x, o.y - self_robot.y),
        )

        own_vx, own_vy = to_local_vec(self_robot.v_x, self_robot.v_y)
        own_w = self_robot.v_theta
        own_ir = 1.0 if self_robot.infrared else 0.0
        self_dist_ball = math.hypot(
            self_robot.x - ball.x, self_robot.y - ball.y
        )
        self_has_ball = (self_dist_ball < 0.12) or self_robot.infrared
        mate_dist_ball = math.hypot(mate.x - ball.x, mate.y - ball.y)
        mate_has_ball = (mate_dist_ball < 0.12) or mate.infrared

        ball_rx, ball_ry = to_local_pos(ball.x, ball.y)
        ball_rvx, ball_rvy = to_local_vec(ball.v_x, ball.v_y)
        ball_dist = math.hypot(ball_rx, ball_ry)
        ball_bearing = math.atan2(ball_ry, ball_rx)

        pred_x = np.clip(ball.x + ball.v_x * 0.5, -max_x, max_x)
        pred_y = np.clip(ball.y + ball.v_y * 0.5, -max_y, max_y)
        pred_rx, pred_ry = to_local_pos(pred_x, pred_y)

        mate_rx, mate_ry = to_local_pos(mate.x, mate.y)
        mate_rvx, mate_rvy = to_local_vec(mate.v_x, mate.v_y)
        mate_dist = math.hypot(mate_rx, mate_ry)
        d_theta = math.radians(mate.theta) - theta

        opp_rx, opp_ry = to_local_pos(opp.x, opp.y)
        opp_rvx, opp_rvy = to_local_vec(opp.v_x, opp.v_y)
        opp_dist = math.hypot(opp_rx, opp_ry)

        gh = self.field.goal_width / 2.0
        ball_pos = np.array([ball.x, ball.y])
        ga = np.array([attack_goal_x, gh])
        gb = np.array([attack_goal_x, -gh])
        gv = gb - ga
        t = float(
            np.clip(np.dot(ball_pos - ga, gv) / np.dot(gv, gv), 0.0, 1.0)
        )
        attack_pt = ga + t * gv
        attack_rx, attack_ry = to_local_pos(attack_pt[0], attack_pt[1])
        attack_dist = math.hypot(attack_rx, attack_ry)

        own_goal_x = -attack_goal_x
        own_rx, own_ry = to_local_pos(own_goal_x, 0.0)

        # Wall distances: the x-walls are absolute features in the world,
        # so the slots the policy learned as "near attack wall" / "near own
        # wall" must be team-aware. Swap the two x-features for blue so the
        # semantic stays consistent across teams. y-walls are symmetric.
        d_wall_attack_side_x = (self_robot.x - (-max_x)) / max_x
        d_wall_own_side_x = (max_x - self_robot.x) / max_x
        if not is_yellow:
            d_wall_attack_side_x, d_wall_own_side_x = (
                d_wall_own_side_x, d_wall_attack_side_x,
            )
        d_wall_neg_x = d_wall_attack_side_x
        d_wall_pos_x = d_wall_own_side_x
        d_wall_neg_y = (self_robot.y - (-max_y)) / max_y
        d_wall_pos_y = (max_y - self_robot.y) / max_y

        team_has_ball = 1.0 if (self_has_ball or mate_has_ball) else 0.0
        i_am_closer = 1.0 if self_dist_ball < mate_dist_ball else 0.0

        is_dribbling = (
            self.is_dribbling_y if is_yellow else self.is_dribbling_b
        )
        dribble_start = (
            self.dribble_start_pos_y if is_yellow else self.dribble_start_pos_b
        )
        must_release = (
            self.must_release_y if is_yellow else self.must_release_b
        )

        if is_dribbling[idx] and dribble_start[idx] is not None:
            start = dribble_start[idx]
            cur_d = math.hypot(ball.x - start[0], ball.y - start[1])
            dribble_meter = float(
                np.clip(cur_d / self.max_dribble_dist, 0.0, 1.0)
            )
        else:
            dribble_meter = 0.0
        must_release_flag = 1.0 if must_release[idx] else 0.0

        obs = np.array(
            [
                self.norm_v(own_vx),
                self.norm_v(own_vy),
                self.norm_w(own_w),
                own_ir,
                1.0 if self_has_ball else 0.0,
                ball_rx / max_dist,
                ball_ry / max_dist,
                self.norm_v(ball_rvx),
                self.norm_v(ball_rvy),
                ball_dist / max_dist,
                ball_bearing / math.pi,
                pred_rx / max_dist,
                pred_ry / max_dist,
                mate_rx / max_dist,
                mate_ry / max_dist,
                self.norm_v(mate_rvx),
                self.norm_v(mate_rvy),
                mate_dist / max_dist,
                math.sin(d_theta),
                math.cos(d_theta),
                opp_rx / max_dist,
                opp_ry / max_dist,
                self.norm_v(opp_rvx),
                self.norm_v(opp_rvy),
                opp_dist / max_dist,
                attack_rx / max_dist,
                attack_ry / max_dist,
                attack_dist / max_dist,
                own_rx / max_dist,
                own_ry / max_dist,
                d_wall_neg_x,
                d_wall_pos_x,
                d_wall_neg_y,
                d_wall_pos_y,
                team_has_ball,
                i_am_closer,
                dribble_meter,
                must_release_flag,
            ],
            dtype=np.float32,
        )
        obs = np.clip(obs, -self.NORM_BOUNDS, self.NORM_BOUNDS)
        if not is_yellow:
            # Mirror over y-axis: negate all egocentric-y components, plus
            # the robot's own local side-velocity (slot 1) and angular
            # velocity (slot 2). Together with the x-wall swap and the action
            # mirror in _compute_blue_action, this presents the world to the
            # yellow-trained policy as if blue were yellow.
            for slot in (1, 2, 6, 8, 10, 12, 14, 16, 18, 21, 23, 26, 29):
                obs[slot] = -obs[slot]
        if self.role_index:
            # Append after mirror — role-index is position-independent
            # agent identity, no reflection needed.
            role = np.zeros(ROLE_INDEX_DIM, dtype=np.float32)
            role[idx] = 1.0
            obs = np.concatenate([obs, role]).astype(np.float32)
        return obs

    # ---------- dribble enforcement (both teams) ----------

    def _update_dribble_state(self):
        ball = self.frame.ball
        teams = (
            (
                (self.frame.robots_yellow[0], self.frame.robots_yellow[1]),
                self.is_dribbling_y,
                self.dribble_start_pos_y,
                self.must_release_y,
            ),
            (
                (self.frame.robots_blue[0], self.frame.robots_blue[1]),
                self.is_dribbling_b,
                self.dribble_start_pos_b,
                self.must_release_b,
            ),
        )
        for robots, is_dribbling, dribble_start, must_release in teams:
            for i, r in enumerate(robots):
                dist = math.hypot(r.x - ball.x, r.y - ball.y)
                has_contact = (dist < self.robot_ball_contact) or r.infrared
                if must_release[i] and dist >= self.min_release_distance:
                    must_release[i] = False
                    is_dribbling[i] = False
                    dribble_start[i] = None
                if has_contact:
                    if not is_dribbling[i]:
                        is_dribbling[i] = True
                        dribble_start[i] = np.array([ball.x, ball.y])
                    else:
                        start = dribble_start[i]
                        dd = math.hypot(
                            ball.x - start[0], ball.y - start[1]
                        )
                        if dd > self.max_dribble_dist:
                            must_release[i] = True
                            is_dribbling[i] = False
                else:
                    if not must_release[i]:
                        is_dribbling[i] = False
                        dribble_start[i] = None

    # ---------- command building ----------

    def convert_actions(self, action_array, angle):
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

    def _robot_command(self, robot, action, must_release, yellow: bool):
        raw_kick = float(action[3])
        kick_trigger = float(action[4])
        dribble_trigger = float(action[5])
        kick = max(0.0, raw_kick) * 6.0 if kick_trigger > 0.0 else 0.0
        dribble = dribble_trigger > 0.0
        if must_release:
            kick = 0.01
            dribble = False
        angle_rad = math.radians(robot.theta)
        v_x_local, v_y_local, w = self.convert_actions(
            [float(action[0]), float(action[1]), float(action[2])], angle_rad
        )
        return Robot(
            yellow=yellow,
            id=robot.id,
            v_x=v_x_local,
            v_y=v_y_local,
            v_theta=w,
            kick_v_x=kick,
            dribbler=dribble,
        )

    def _compute_blue_action(self) -> np.ndarray:
        if self.frozen_model is None:
            return np.zeros((N_BLUE, SINGLE_ACT_DIM), dtype=np.float32)
        blue_obs = self._stacked_obs_blue()
        # Frozen blue may have been trained with a smaller obs (e.g. v3 has
        # 38 dims, current env outputs 40 with role-index). Truncate to the
        # frozen model's expected input dim — role-index lives in the trailing
        # ROLE_INDEX_DIM slots, so this drops exactly those.
        if (
            self.frozen_obs_dim is not None
            and self.frozen_obs_dim < blue_obs.shape[-1]
        ):
            blue_obs = blue_obs[..., : self.frozen_obs_dim]
        action, _ = self.frozen_model.predict(blue_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).copy()
        # Action is in world frame (v_x = world x velocity). Policy learned
        # "attack = -x". Mirror over y-axis for blue: negate world-x velocity
        # and world angular velocity; keep world-y velocity and kick/dribble.
        action[:, 0] = -action[:, 0]
        action[:, 2] = -action[:, 2]
        return action

    def _build_commands(self, yellow_action, blue_action):
        cmds = []
        cmds.append(self._robot_command(
            self.frame.robots_yellow[0], yellow_action[0],
            self.must_release_y[0], yellow=True,
        ))
        cmds.append(self._robot_command(
            self.frame.robots_yellow[1], yellow_action[1],
            self.must_release_y[1], yellow=True,
        ))
        cmds.append(self._robot_command(
            self.frame.robots_blue[0], blue_action[0],
            self.must_release_b[0], yellow=False,
        ))
        cmds.append(self._robot_command(
            self.frame.robots_blue[1], blue_action[1],
            self.must_release_b[1], yellow=False,
        ))
        return cmds

    def _get_commands(self, action):
        return self._build_commands(
            np.zeros((N_YELLOW, SINGLE_ACT_DIM), dtype=np.float32),
            np.zeros((N_BLUE, SINGLE_ACT_DIM), dtype=np.float32),
        )

    # ---------- reward (original 2v1-IL reward, no aggressive edits) ----------

    def _calculate_reward_and_done(self):
        rewards, done, _ = self._calculate_team_reward_and_done()
        return float(rewards.mean()), done

    def _calculate_team_reward_and_done(self) -> Tuple[np.ndarray, bool, bool]:
        ball = self.frame.ball
        ya, yb = self.frame.robots_yellow[0], self.frame.robots_yellow[1]
        yellows = (ya, yb)
        blues = (self.frame.robots_blue[0], self.frame.robots_blue[1])

        max_x = self.field.length / 2.0
        max_y = self.field.width / 2.0
        goal_half_width = self.field.goal_width / 2.0
        max_dist = math.hypot(self.field.length, self.field.width)

        rewards = np.zeros(2, dtype=np.float32)
        done = False
        truncated = False

        in_grace = self.total_steps <= self.oob_grace_steps

        # Goal: always terminates. Speed bonus + pass bonus on yellow goal.
        if abs(ball.x) > max_x and abs(ball.y) <= goal_half_width:
            done = True
            if ball.x < 0:  # Yellow goal
                rewards += 100.0
                rewards += (self.max_steps - self.current_step) * 0.01
                rewards += 50.0 * min(self.passes_in_episode, 2)
                self.match_result = 1
            else:  # Blue goal
                rewards -= 50.0
                self.match_result = -1
                self.blue_goal_scored = True
            return rewards, done, truncated

        # Ball OOB without goal: kills the OOB-exit reward hack.
        if (abs(ball.x) > max_x or abs(ball.y) > max_y) and not in_grace:
            done = True
            rewards -= 5.0
            self.match_result = -1
            return rewards, done, truncated

        # Yellow robot OOB: heavy penalty (matches 1v1 level 4-5 scale,
        # softened to -50 because 2v2 has more bodies that bump near edges).
        if not in_grace:
            for r in yellows:
                if abs(r.x) > max_x or abs(r.y) > max_y:
                    done = True
                    rewards -= 50.0
                    self.match_result = -1
                    return rewards, done, truncated

        # Timeout: penalty avoids "wait for episode to end" stalls.
        if self.current_step >= self.max_steps:
            truncated = True
            done = True
            rewards -= 10.0
            self.match_result = -1
            return rewards, done, truncated

        # Per-step shaping
        if self.reward_type == "dense":
            dist_a = math.hypot(ya.x - ball.x, ya.y - ball.y)
            dist_b = math.hypot(yb.x - ball.x, yb.y - ball.y)
            dists = (dist_a, dist_b)

            if self.last_dist_to_ball is None:
                self.last_dist_to_ball = [dist_a, dist_b]

            # Time penalty: ramps with episode progress; heavier when ball
            # is in own half (push it out) than attack half (defending
            # close to opponent goal shouldn't be over-penalized).
            progress = self.current_step / self.max_steps
            if ball.x < 0:  # attack half
                rewards -= 0.02 * (1.0 + 2.0 * progress)
            else:  # own half
                rewards -= 0.04 * (1.0 + 2.0 * progress)

            # Per-agent distance potential: small constant gradient toward
            # the ball regardless of motion.
            for i in range(2):
                rewards[i] += 0.05 * (1.0 - dists[i] / max_dist)

            # Per-agent stand-still penalty: kills "do nothing" strategies.
            for i, agent in enumerate(yellows):
                speed = math.hypot(agent.v_x, agent.v_y)
                has_ball = (dists[i] < 0.12) or agent.infrared
                if speed < 0.1 and not has_ball:
                    rewards[i] -= 0.05

            # Per-agent robot-to-ball delta (signed: penalize moving away).
            for i in range(2):
                delta = self.last_dist_to_ball[i] - dists[i]
                rewards[i] += float(np.clip(delta * 5.0, -0.5, 0.5))
            self.last_dist_to_ball = [dist_a, dist_b]

            # Shared ball-to-goal delta (signed: penalize ball moving away
            # from yellow attack goal).
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
                delta_ball_goal = self.last_dist_ball_goal - dist_ball_goal
                rewards += float(np.clip(delta_ball_goal * 10.0, -1.0, 1.5))
            self.last_dist_ball_goal = dist_ball_goal

            # Shared possession
            if (dist_a < 0.12) or ya.infrared or (dist_b < 0.12) or yb.infrared:
                self.team_possession_steps += 1
                rewards += 0.01

            # Ball direction: ball flying toward yellow attack goal (-x)
            # earns extra; toward blue goal (+x) is penalized.
            if ball.v_x < -0.5:
                rewards += 0.02 * min(-ball.v_x, 3.0)
            elif ball.v_x > 0.5:
                rewards -= 0.02 * min(ball.v_x, 3.0)

        # Pass detection (yellow-side carriers; any blue touch resets)
        ya_has = (
            math.hypot(ya.x - ball.x, ya.y - ball.y) < 0.20
        ) or ya.infrared
        yb_has = (
            math.hypot(yb.x - ball.x, yb.y - ball.y) < 0.20
        ) or yb.infrared
        blue_has = any(
            (math.hypot(b.x - ball.x, b.y - ball.y) < 0.12) or b.infrared
            for b in blues
        )

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
                # Strict pass detection: require the ball to be at least 0.5m
                # from the previous carrier at the moment of transfer. Filters
                # touch-swap artefacts where two yellows are both close to a
                # slow-rolling ball and the carrier flag flips back and forth
                # — that's ping-pong, not a real pass.
                prev = yellows[self.last_yellow_carrier]
                ball_to_prev = math.hypot(ball.x - prev.x, ball.y - prev.y)
                if ball_to_prev > 0.5:
                    self.passes_in_episode += 1
                    if self.reward_type == "dense":
                        rewards += 30.0
            self.last_yellow_carrier = current_carrier
            self.blue_touched_since_yellow = False

        return rewards, done, truncated

    # ---------- initial positions ----------

    def _get_initial_positions_frame(self) -> Frame:
        pos = Frame()
        rng = self.np_random
        pos.ball = Ball(
            x=float(rng.uniform(-3, 3)),
            y=float(rng.uniform(-2, 2)),
        )
        pos.robots_yellow[0] = Robot(
            x=float(rng.uniform(0.2, 3.5)),
            y=float(rng.uniform(-2.5, 2.5)),
            theta=float(rng.uniform(-180, 180)),
        )
        pos.robots_yellow[1] = Robot(
            x=float(rng.uniform(0.2, 3.5)),
            y=float(rng.uniform(-2.5, 2.5)),
            theta=float(rng.uniform(-180, 180)),
        )
        pos.robots_blue[0] = Robot(
            x=float(rng.uniform(-3.5, -0.2)),
            y=float(rng.uniform(-2.5, 2.5)),
            theta=float(rng.uniform(-180, 180)),
        )
        pos.robots_blue[1] = Robot(
            x=float(rng.uniform(-3.5, -0.2)),
            y=float(rng.uniform(-2.5, 2.5)),
            theta=float(rng.uniform(-180, 180)),
        )
        return pos
