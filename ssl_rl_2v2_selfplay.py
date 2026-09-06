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

import json
import math
import os
import time
from typing import Tuple

import numpy as np
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from stable_baselines3 import SAC

from skills import (
    move_to_ball,
    move_to_point,
    shoot_at_goal_center,
    turn_to_point,
)


def blue_attacker_heuristic_2v2(env, robot, yellows) -> np.ndarray:
    """Aggressive blue: drive to ball; on infrared contact shoot at the
    yellow goal. No defensive fallback. Returns 5-dim skill output
    [v_x, v_y, v_theta, kick, dribble].
    """
    if robot.infrared:
        return shoot_at_goal_center(env, robot, team_color="blue")
    return move_to_ball(robot, env.frame.ball, speed=2.0)


def blue_defender_heuristic_2v2(env, robot, yellows) -> np.ndarray:
    """Defensive blue: same structure as the 1v1 blue_attacker_heuristic.
    Shoot on infrared, fall back to the goal-ball defensive line when the
    closest yellow is closer to the ball, intercept fast moving balls,
    otherwise chase ball.
    """
    ball = env.frame.ball
    defend_goal_x = -env.field.length / 2.0

    if robot.infrared:
        return shoot_at_goal_center(env, robot, team_color="blue")

    dist_blue_ball = math.hypot(robot.x - ball.x, robot.y - ball.y)
    closest_yellow = min(
        yellows, key=lambda y: math.hypot(y.x - ball.x, y.y - ball.y)
    )
    dist_yellow_ball = math.hypot(
        closest_yellow.x - ball.x, closest_yellow.y - ball.y
    )
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
                    r for r in (
                        (-b - sq) / (2.0 * a),
                        (-b + sq) / (2.0 * a),
                    ) if r > 0.0
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


SINGLE_OBS_DIM_BASE = 52  # world-frame layout, see _egocentric_obs docstring
ROLE_INDEX_DIM = 2
# Episode-scoped state the REWARD depends on but the 52-dim layout never
# exposed. Verified empirically: mutating passes_in_episode / current_step /
# last_yellow_carrier / blue_touched_since_yellow changed 0 of 104 obs values,
# i.e. the MDP was not Markov w.r.t. the observation — two physically
# identical goal states carried terminal reward 11 or 4 (Gen-11 asymmetric
# payoff) with no observable difference, so the critic could only regress to
# their frequency-weighted mean (~3.3 at 2 pass-goals in 54).
# Appended AFTER the role-index block so the frozen-opponent truncation
# `blue_obs[..., :frozen_obs_dim]` still yields exactly the legacy layout.
EPISODE_STATE_DIM = 4
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
        curriculum_start_level=None,
        curriculum_target_level=5,
        curriculum_threshold=0.9,
        curriculum_window=200,
        blue_heuristic=None,
        pass_scenario_prob=0.0,
        goal_reward=10.0,
        goal_reward_solo=None,
    ):
        super().__init__(
            field_type=1,
            n_robots_blue=N_BLUE,
            n_robots_yellow=N_YELLOW,
            time_step=0.025,
            render_mode=render_mode,
        )
        # Curriculum auto-promotion: each env subprocess tracks its own
        # rolling is_success buffer and bumps itself from start_level to
        # target_level when the rolling mean clears `threshold`. With 24
        # parallel envs they all flip independently but at roughly the same
        # wall time, so the joint training distribution shifts cleanly.
        from collections import deque as _deque
        self.curriculum_target_level = int(curriculum_target_level)
        self.curriculum_threshold = float(curriculum_threshold)
        self._success_buffer = _deque(maxlen=int(curriculum_window))
        self._curriculum_promoted = False
        self.reward_type = reward_type
        self.frozen_path = frozen_path
        self.frozen_model = None  # lazy-load on first step (subproc-safe)
        self.frozen_type = None  # "sb3" or "harl", set on lazy-load
        # Blue heuristic mode: when set (e.g. "attacker"), blue ignores
        # frozen_path and is controlled by the hand-coded heuristic at the
        # _build_commands stage. Mutually exclusive with frozen_path.
        self.blue_heuristic = blue_heuristic

        # Ball speed needs its own scale. norm_v divides by the ROBOT max
        # (4.035 m/s) and clips at NORM_BOUNDS=1.2, while kicks run 3-6 m/s
        # (_robot_command), so every kick above 4.84 m/s produced an
        # identical observation — the top ~40% of the kick range was
        # unobservable, including for the receiver judging an incoming pass.
        self.max_ball_v = 6.5

        # Asymmetric terminal payoff. A goal that FOLLOWS a completed pass is
        # worth `goal_reward`; a solo goal only `goal_reward_solo`. This is a
        # change to the payoff STRUCTURE that defines the equilibrium, not
        # another shaping term — every shaping attempt so far left the solo
        # equilibrium intact because solo scoring stayed exactly as valuable
        # as cooperative scoring.
        # goal_reward_solo=None keeps the symmetric payoff (both = 10), which
        # is also phase 2 of the two-phase test: train asymmetric until the
        # policy passes, then restore symmetry and see whether cooperation
        # survives or erodes back.
        self.goal_reward = float(goal_reward)
        self.goal_reward_solo = (
            float(goal_reward_solo) if goal_reward_solo is not None
            else float(goal_reward)
        )
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
        ) + EPISODE_STATE_DIM

        # OOB curriculum: skip robot-OOB termination during the first
        # `oob_grace_steps` per-env steps so early-stage agents get more
        # productive practice instead of episodes ending the moment a robot
        # wanders off the field.
        self.oob_grace_steps = int(oob_grace_steps)

        # Pass-scenario mix: at level 5, spawn a staged pass situation
        # (carrier cornered by a blocker, mate free in front of the goal)
        # with this probability instead of the chaos spawn. Targets the
        # cooperative-exploration bootstrap problem: passing is the only
        # high-percentage play in this geometry, and the mate already has
        # the finishing skill from level 1.
        self.pass_scenario_prob = float(pass_scenario_prob)
        self._episode_scenario = "chaos"  # set on every spawn

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
        self.last_yellow_pos = None
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

        # Curriculum level the env spawns at. If curriculum_start_level was
        # passed, honour it; otherwise default to the chaotic full-task spawn.
        self.curriculum_level = (
            int(curriculum_start_level)
            if curriculum_start_level is not None else 5
        )

    # ---------- frozen model ----------

    def _maybe_load_frozen(self):
        if self.frozen_model is not None or not self.frozen_path:
            return
        p = self.frozen_path
        if p.endswith(".zip"):
            # SB3 SAC checkpoint (legacy 2v1 / early 2v2 runs).
            self.frozen_type = "sb3"
            self.frozen_model = SAC.load(p, device="cpu")
            self.frozen_obs_dim = int(
                self.frozen_model.policy.observation_space.shape[-1]
            )
            return
        if not os.path.isdir(p):
            raise ValueError(
                f"frozen_path '{p}' is neither a .zip nor an existing directory"
            )
        # HARL HASAC checkpoint — path is either the seed-dir or its models/.
        if os.path.basename(p.rstrip("/")) == "models":
            models_dir = p
            run_dir = os.path.dirname(p.rstrip("/"))
        else:
            run_dir = p
            models_dir = os.path.join(p, "models")
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
        from harl.algorithms.actors.hasac import HASAC
        algo_args = cfg["algo_args"]
        single_obs = Box(
            low=-np.inf, high=np.inf,
            shape=(self.single_obs_dim,), dtype=np.float32,
        )
        single_act = Box(
            low=-1.0, high=1.0, shape=(SINGLE_ACT_DIM,), dtype=np.float32,
        )
        actor_args = {**algo_args["model"], **algo_args["algo"]}
        actor = HASAC(actor_args, single_obs, single_act, device="cpu")
        actor.restore(models_dir, 0)
        actor.turn_off_grad()
        self.frozen_type = "harl"
        self.frozen_model = actor
        self.frozen_obs_dim = self.single_obs_dim

    # ---------- gym API ----------

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.last_dist_ball_goal = None
        self.last_dist_to_ball = None
        self.last_ball_pos = None
        self.last_yellow_pos = None
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
            info["scenario"] = self._episode_scenario
            if self._episode_scenario == "pass":
                info["pass_variant"] = getattr(
                    self, "_episode_pass_variant", "corner"
                )
            # Curriculum auto-promotion: track is_success in a rolling buffer
            # and bump level once the rolling mean clears the threshold.
            # Each env subprocess runs this independently — close enough since
            # all 24 see the same shared policy improving.
            self._success_buffer.append(info["is_success"])
            if (
                not self._curriculum_promoted
                and self.curriculum_level < self.curriculum_target_level
                and len(self._success_buffer) >= self._success_buffer.maxlen
            ):
                sr = sum(self._success_buffer) / len(self._success_buffer)
                if sr >= self.curriculum_threshold:
                    print(
                        f"[env] curriculum: SR={sr:.2f} >= "
                        f"{self.curriculum_threshold} -> "
                        f"L{self.curriculum_level} -> L{self.curriculum_target_level}"
                    )
                    self.set_curriculum_level(self.curriculum_target_level)
                    self._success_buffer.clear()
                    self._curriculum_promoted = True
            info["curriculum_level"] = self.curriculum_level
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

    def set_pass_scenario_prob(self, prob: float):
        # Scheduled by PassScenarioScheduleCallback: anneal the staged-scenario
        # share from high (learn to pass) to low (apply passing in chaos).
        self.pass_scenario_prob = float(prob)

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

    def norm_ball_v(self, v):
        """Ball speed scaled by a BALL max, not the robot max.

        norm_v divides by max_v=4.035 and clips at NORM_BOUNDS=1.2, so every
        kick above 4.84 m/s mapped to the same value while kicks span 3-6 m/s.
        The receiver of a pass could not tell a 5 m/s ball from a 6 m/s one.
        """
        return float(np.clip(
            v / self.max_ball_v, -self.NORM_BOUNDS, self.NORM_BOUNDS
        ))

    def _episode_state_obs(self, is_yellow, idx) -> np.ndarray:
        """The episode-scoped variables the reward function reads.

        Order: has_passed, i_am_last_carrier, blue_touched_since_yellow,
        time_remaining. The first three are yellow-team concepts (the pass
        bookkeeping only tracks yellow), so blue gets zeros for them and
        shares only the clock.
        """
        time_remaining = 1.0 - min(
            1.0, self.current_step / float(self.max_steps)
        )
        if not is_yellow:
            return np.array(
                [0.0, 0.0, 0.0, time_remaining], dtype=np.float32
            )
        return np.array([
            1.0 if self.passes_in_episode > 0 else 0.0,
            1.0 if self.last_yellow_carrier == idx else 0.0,
            1.0 if self.blue_touched_since_yellow else 0.0,
            time_remaining,
        ], dtype=np.float32)

    def _egocentric_obs(
        self, self_robot, mate, opp_list,
        attack_goal_x, is_yellow, idx,
    ) -> np.ndarray:
        """World-frame obs layout, modelled on the 1v1 env (which worked).

        Everything position/velocity-wise is in WORLD frame so the policy's
        world-frame action output naturally aligns with what it sees. The
        only robot-frame quantities are the two relative angles (ball, goal)
        kept as convenience features — they are mirror-anti-invariant
        (negate under y-reflection).

        Layout (52 base dims):
            [ 0:5 ] BALL      pos(x,y), vel(x,y), dist_ball_goal
            [ 5:18] SELF      pos, sin/cos θ, vel, v_theta, infrared,
                              dist_to_ball, rel_angle_ball, rel_angle_goal,
                              dribble_meter, must_release
            [18:27] MATE      pos, sin/cos θ, vel, v_theta, infrared, dist
            [27:37] OPP1      pos, sin/cos θ, vel, v_theta, infrared, dist,
                              (self_dist − opp1_dist)
            [37:46] OPP2      pos, sin/cos θ, vel, v_theta, infrared, dist
            [46:50] PRED      ball_pred (world) + relative to self
            [50:52] TEAM      team_has_ball, i_am_closer
        """
        ball = self.frame.ball
        max_x = self.field.length / 2.0
        max_y = self.field.width / 2.0
        max_dist = math.hypot(self.field.length, self.field.width)
        gh = self.field.goal_width / 2.0

        # Closest point on the attack goal mouth to the ball — for both the
        # dist_ball_goal feature and the rel_angle_goal feature.
        ball_pos = np.array([ball.x, ball.y])
        ga = np.array([attack_goal_x, gh])
        gb = np.array([attack_goal_x, -gh])
        gv = gb - ga
        t = float(
            np.clip(np.dot(ball_pos - ga, gv) / np.dot(gv, gv), 0.0, 1.0)
        )
        closest_goal_pt = ga + t * gv
        dist_ball_goal = float(np.linalg.norm(ball_pos - closest_goal_pt))

        # Self
        theta = math.radians(self_robot.theta)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        self_dist_ball = math.hypot(
            self_robot.x - ball.x, self_robot.y - ball.y
        )
        self_has_ball = (self_dist_ball < 0.12) or self_robot.infrared

        # Relative angle self → ball (in self's frame), in [-π, π].
        ang_to_ball = math.atan2(ball.y - self_robot.y, ball.x - self_robot.x)
        rel_angle_ball = (ang_to_ball - theta + math.pi) % (2 * math.pi) - math.pi
        # Relative angle self → attack-goal point (in self's frame).
        ang_to_goal = math.atan2(
            closest_goal_pt[1] - self_robot.y,
            closest_goal_pt[0] - self_robot.x,
        )
        rel_angle_goal = (ang_to_goal - theta + math.pi) % (2 * math.pi) - math.pi

        # Dribble state for this team / agent.
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

        # Mate
        mate_theta = math.radians(mate.theta)
        mate_dist_ball = math.hypot(mate.x - ball.x, mate.y - ball.y)
        mate_has_ball = (mate_dist_ball < 0.12) or mate.infrared

        # Opponents, sorted closest first.
        opp_sorted = sorted(
            opp_list,
            key=lambda o: math.hypot(o.x - self_robot.x, o.y - self_robot.y),
        )
        opp1, opp2 = opp_sorted[0], opp_sorted[1]
        opp1_theta = math.radians(opp1.theta)
        opp2_theta = math.radians(opp2.theta)
        opp1_dist_ball = math.hypot(opp1.x - ball.x, opp1.y - ball.y)
        opp2_dist_ball = math.hypot(opp2.x - ball.x, opp2.y - ball.y)

        # Ball prediction in world frame.
        pred_x = float(np.clip(ball.x + ball.v_x * 0.5, -max_x, max_x))
        pred_y = float(np.clip(ball.y + ball.v_y * 0.5, -max_y, max_y))

        # Team flags
        team_has_ball = 1.0 if (self_has_ball or mate_has_ball) else 0.0
        i_am_closer = 1.0 if self_dist_ball < mate_dist_ball else 0.0

        obs = np.array(
            [
                # BALL (5)
                self.norm_pos(ball.x),                  # 0
                self.norm_pos(ball.y),                  # 1
                self.norm_ball_v(ball.v_x),             # 2
                self.norm_ball_v(ball.v_y),             # 3
                dist_ball_goal / max_dist,              # 4
                # SELF (13)
                self.norm_pos(self_robot.x),            # 5
                self.norm_pos(self_robot.y),            # 6
                sin_t,                                  # 7
                cos_t,                                  # 8
                self.norm_v(self_robot.v_x),            # 9
                self.norm_v(self_robot.v_y),            # 10
                self.norm_w(self_robot.v_theta),        # 11
                1.0 if self_robot.infrared else 0.0,    # 12
                self_dist_ball / max_dist,              # 13
                rel_angle_ball / math.pi,               # 14
                rel_angle_goal / math.pi,               # 15
                dribble_meter,                          # 16
                must_release_flag,                      # 17
                # MATE (9)
                self.norm_pos(mate.x),                  # 18
                self.norm_pos(mate.y),                  # 19
                math.sin(mate_theta),                   # 20
                math.cos(mate_theta),                   # 21
                self.norm_v(mate.v_x),                  # 22
                self.norm_v(mate.v_y),                  # 23
                self.norm_w(mate.v_theta),              # 24
                1.0 if mate.infrared else 0.0,          # 25
                mate_dist_ball / max_dist,              # 26
                # OPP1 (closest) (10)
                self.norm_pos(opp1.x),                  # 27
                self.norm_pos(opp1.y),                  # 28
                math.sin(opp1_theta),                   # 29
                math.cos(opp1_theta),                   # 30
                self.norm_v(opp1.v_x),                  # 31
                self.norm_v(opp1.v_y),                  # 32
                self.norm_w(opp1.v_theta),              # 33
                1.0 if opp1.infrared else 0.0,          # 34
                opp1_dist_ball / max_dist,              # 35
                (self_dist_ball - opp1_dist_ball) / max_dist,  # 36
                # OPP2 (farther) (9)
                self.norm_pos(opp2.x),                  # 37
                self.norm_pos(opp2.y),                  # 38
                math.sin(opp2_theta),                   # 39
                math.cos(opp2_theta),                   # 40
                self.norm_v(opp2.v_x),                  # 41
                self.norm_v(opp2.v_y),                  # 42
                self.norm_w(opp2.v_theta),              # 43
                1.0 if opp2.infrared else 0.0,          # 44
                opp2_dist_ball / max_dist,              # 45
                # PREDICTION (4)
                self.norm_pos(pred_x),                  # 46
                self.norm_pos(pred_y),                  # 47
                self.norm_pos(pred_x - self_robot.x),   # 48
                self.norm_pos(pred_y - self_robot.y),   # 49
                # TEAM (2)
                team_has_ball,                          # 50
                i_am_closer,                            # 51
            ],
            dtype=np.float32,
        )
        obs = np.clip(obs, -self.NORM_BOUNDS, self.NORM_BOUNDS)

        if not is_yellow:
            # Mirror over y-axis (x → −x). All world-x positions/velocities
            # negate; cos(θ) negates and sin(θ) stays; v_θ negates (angular
            # velocity reverses under reflection); rel_angle_* negate (they
            # flip under left↔right reflection); scalars/distances/flags stay.
            for slot in (
                0, 2,                # ball: x, v_x
                5, 8, 9, 11,         # self: x, cos, v_x, v_theta
                14, 15,              # self: rel_angle_ball, rel_angle_goal
                18, 21, 22, 24,      # mate: x, cos, v_x, v_theta
                27, 30, 31, 33,      # opp1: x, cos, v_x, v_theta
                37, 40, 41, 43,      # opp2: x, cos, v_x, v_theta
                46, 48,              # pred: x, dx
            ):
                obs[slot] = -obs[slot]

        if self.role_index:
            role = np.zeros(ROLE_INDEX_DIM, dtype=np.float32)
            role[idx] = 1.0
            obs = np.concatenate([obs, role]).astype(np.float32)

        # Appended LAST so `blue_obs[..., :frozen_obs_dim]` still hands a
        # legacy 52-dim (or 54-dim role-index) opponent exactly its own
        # layout, and so a warm start can zero-init only the new columns.
        obs = np.concatenate(
            [obs, self._episode_state_obs(is_yellow, idx)]
        ).astype(np.float32)
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
        if kick_trigger > 0.0:
            kick = 3.0 + ((raw_kick + 1.0) / 2.0) * 3.0
        else:
            kick = 0.0
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
        if self.frozen_type == "harl":
            # HASAC actor expects (B, obs_dim), returns torch tensor (B, act_dim).
            out = self.frozen_model.get_actions(blue_obs, stochastic=False)
            action = out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)
        else:
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
        # Level 1 is a STAGED SCORING CHANCE and presupposes passive blues
        # ("parked far from the goal mouth"). An active heuristic turns it
        # into a race that a fresh policy loses (observed: success 0.0 at
        # blue_goal_rate 0.41 after 2.9M steps, curriculum never promotes),
        # so the heuristic only engages from level 2 upward.
        heuristic_active = (
            self.blue_heuristic == "attacker"
            and int(getattr(self, "curriculum_level", 5)) > 1
        )
        if heuristic_active:
            # Blue 0 = aggressive (just chase + shoot), Blue 1 = defensive
            # (chase if closer to ball, else fall back to defensive line).
            cmds.append(self._blue_heuristic_command(
                self.frame.robots_blue[0], personality="aggressive",
            ))
            cmds.append(self._blue_heuristic_command(
                self.frame.robots_blue[1], personality="defensive",
            ))
        else:
            cmds.append(self._robot_command(
                self.frame.robots_blue[0], blue_action[0],
                self.must_release_b[0], yellow=False,
            ))
            cmds.append(self._robot_command(
                self.frame.robots_blue[1], blue_action[1],
                self.must_release_b[1], yellow=False,
            ))
        return cmds

    def _blue_heuristic_command(self, robot, personality="aggressive") -> Robot:
        """Build a blue Robot command from a hand-coded heuristic skill
        output. Bypasses _robot_command's raw_kick/trigger encoding —
        skills return kick magnitude directly. Same shape as 2v1's
        _blue_command pattern.
        """
        yellows = (
            self.frame.robots_yellow[0], self.frame.robots_yellow[1],
        )
        if personality == "defensive":
            cmd = blue_defender_heuristic_2v2(self, robot, yellows)
        else:
            cmd = blue_attacker_heuristic_2v2(self, robot, yellows)
        angle_rad = math.radians(robot.theta)
        bv_x, bv_y, bv_w = self.convert_actions(
            [float(cmd[0]), float(cmd[1]), float(cmd[2])], angle_rad
        )
        kick = float(cmd[3])
        dribble = bool(cmd[4] > 0)
        return Robot(
            yellow=False, id=robot.id,
            v_x=bv_x, v_y=bv_y, v_theta=bv_w,
            kick_v_x=kick, dribbler=dribble,
        )

    def _get_commands(self, action):
        return self._build_commands(
            np.zeros((N_YELLOW, SINGLE_ACT_DIM), dtype=np.float32),
            np.zeros((N_BLUE, SINGLE_ACT_DIM), dtype=np.float32),
        )

    # ---------- reward (original 2v1-IL reward, no aggressive edits) ----------

    def _calculate_reward_and_done(self):
        rewards, done, _ = self._calculate_team_reward_and_done()
        return float(rewards.mean()), done

    def _dist_ball_to_goal(self, bx, by) -> float:
        """Distance from the ball to the yellow attack-goal mouth — the
        projection onto the goal-line segment between the posts
        (Ocana et al. 2019, Eq. 15). Yellow attacks the goal at x=-max_x."""
        goal_x = -self.field.length / 2.0
        gh = self.field.goal_width / 2.0
        if by >= gh:
            return math.hypot(bx - goal_x, by - gh)
        if by <= -gh:
            return math.hypot(bx - goal_x, by + gh)
        return abs(bx - goal_x)

    def _calculate_team_reward_and_done(self) -> Tuple[np.ndarray, bool, bool]:
        ball = self.frame.ball
        ya, yb = self.frame.robots_yellow[0], self.frame.robots_yellow[1]
        yellows = (ya, yb)
        blues = (self.frame.robots_blue[0], self.frame.robots_blue[1])

        max_x = self.field.length / 2.0
        max_y = self.field.width / 2.0
        max_dist = math.hypot(self.field.length, self.field.width)
        goal_half_width = self.field.goal_width / 2.0

        rewards = np.zeros(2, dtype=np.float32)
        done = False
        truncated = False

        in_grace = self.total_steps <= self.oob_grace_steps
        progress = self.current_step / self.max_steps

        # Time penalty (per step, halved while ball is in defensive half so
        # defense isn't punished). Applied first so terminals also pay it.
        # Scaled 10x down together with the terminal rewards so the
        # terminal:shaping ratio matches the original design.
        if self.reward_type == "dense":
            if ball.x < 0:
                rewards -= 0.002 * (1.0 + 2.0 * progress)
            else:
                rewards -= 0.004 * (1.0 + 2.0 * progress)

        if abs(ball.x) > max_x and abs(ball.y) <= goal_half_width:
            done = True
            if ball.x < 0:  # Yellow scored
                # Same condition as info["scored_after_pass"], so the reward
                # and the logged metric can never disagree.
                rewards += (
                    self.goal_reward if self.passes_in_episode > 0
                    else self.goal_reward_solo
                )
                rewards += (self.max_steps - self.current_step) * 0.001
                self.match_result = 1
            else:  # Blue scored
                rewards -= 5.0
                self.match_result = -1
                self.blue_goal_scored = True
            return rewards, done, truncated

        # Ball OOB without a goal: small penalty.
        if (abs(ball.x) > max_x or abs(ball.y) > max_y) and not in_grace:
            done = True
            rewards -= 0.5
            self.match_result = -1
            return rewards, done, truncated
        # Yellow robot OOB: heavy penalty (deters escape). Blue OOB ends
        # the episode without yellow penalty.
        if not in_grace:
            for r in yellows:
                if abs(r.x) > max_x or abs(r.y) > max_y:
                    done = True
                    rewards -= 2.0
                    self.match_result = -1
                    return rewards, done, truncated
            for r in blues:
                if abs(r.x) > max_x or abs(r.y) > max_y:
                    done = True
                    return rewards, done, truncated

        if self.current_step >= self.max_steps:
            truncated = True
            rewards -= 1.0
            self.match_result = -1
            return rewards, done, truncated

        if self.reward_type == "dense":
            dist_a = math.hypot(ya.x - ball.x, ya.y - ball.y)
            dist_b = math.hypot(yb.x - ball.x, yb.y - ball.y)
            dists = (dist_a, dist_b)
            ya_has = (dist_a < 0.12) or ya.infrared
            yb_has = (dist_b < 0.12) or yb.infrared

            # Robot→Ball signed delta — per agent. Only progress signal.
            # Rewards moving toward the ball, penalizes moving away.
            if self.last_dist_to_ball is None:
                self.last_dist_to_ball = [dist_a, dist_b]
            for i in range(2):
                delta = self.last_dist_to_ball[i] - dists[i]
                rewards[i] += float(np.clip(delta * 0.5, -0.05, 0.05))
            self.last_dist_to_ball = [dist_a, dist_b]

            # Ball→Goal signed delta — shared. Rewards ball moving toward
            # opponent goal (progress toward scoring).
            dist_ball_goal = self._dist_ball_to_goal(ball.x, ball.y)
            if self.last_dist_ball_goal is not None:
                goal_delta = self.last_dist_ball_goal - dist_ball_goal
                rewards += float(np.clip(goal_delta * 1.0, -0.1, 0.15))
            self.last_dist_ball_goal = dist_ball_goal

            # Anti-passivity: whenever a Yellow holds the ball, small negative
            # per step. Prevents "hold ball, don't shoot" degenerate policy.
            if ya_has or yb_has:
                rewards -= 0.003
                self.team_possession_steps += 1

            self.last_ball_pos = (ball.x, ball.y)

        # Pass detection: +3 shared event bonus. Receiver must be at true
        # contact distance (0.13 ≈ robot hull + ball) or have infrared —
        # tighter than the old 0.20 fly-by radius.
        ya_has_pass = (
            math.hypot(ya.x - ball.x, ya.y - ball.y) < 0.13
        ) or ya.infrared
        yb_has_pass = (
            math.hypot(yb.x - ball.x, yb.y - ball.y) < 0.13
        ) or yb.infrared
        blue_has = any(
            (math.hypot(b.x - ball.x, b.y - ball.y) < 0.13) or b.infrared
            for b in blues
        )

        if blue_has:
            self.blue_touched_since_yellow = True
            self.last_yellow_carrier = None

        current_carrier = None
        if ya_has_pass and not yb_has_pass:
            current_carrier = 0
        elif yb_has_pass and not ya_has_pass:
            current_carrier = 1
        if current_carrier is not None:
            if (
                self.last_yellow_carrier is not None
                and current_carrier != self.last_yellow_carrier
                and not self.blue_touched_since_yellow
            ):
                prev = yellows[self.last_yellow_carrier]
                ball_to_prev = math.hypot(ball.x - prev.x, ball.y - prev.y)
                if ball_to_prev > 0.5:
                    rewards += 3.0
                    self.passes_in_episode += 1
            self.last_yellow_carrier = current_carrier
            self.blue_touched_since_yellow = False

        return rewards, done, truncated

    # ---------- initial positions ----------

    def _get_initial_positions_frame(self) -> Frame:
        """Curriculum-aware spawn.

        Level 1: easy scoring chance. Ball near the yellow attack goal
        (x ≈ -max_x + 1..2.5), both yellows behind the ball facing the goal,
        both blues parked on yellow's home half — clear shot, no static
        obstacle in the way. This is the 1v1 Level-1 trick adapted to 2v2.
        Level 5: chaotic spawn (original setup) — ball anywhere, yellows on
        their own side, blues between yellows and the attack goal.
        Switch happens externally via set_curriculum_level once the rolling
        success_rate clears the curriculum-callback threshold.
        """
        pos = Frame()
        rng = self.np_random
        max_x = self.field.length / 2.0
        level = int(getattr(self, "curriculum_level", 5))

        if level <= 1:
            # LEVEL 1 — gestellte Torchance.
            self._episode_scenario = "level1"
            bx = float(rng.uniform(-max_x + 1.0, -max_x + 2.5))
            by = float(rng.uniform(-0.6, 0.6))
            pos.ball = Ball(x=bx, y=by)
            # Yellow shooter: close behind ball, roughly facing the goal.
            pos.robots_yellow[0] = Robot(
                x=bx + float(rng.uniform(0.4, 1.0)),
                y=by + float(rng.uniform(-0.5, 0.5)),
                theta=float(rng.uniform(120.0, 240.0)),
            )
            # Yellow supporter: further behind, wider — gives the network a
            # mate-position signal even on easy-shot reps.
            pos.robots_yellow[1] = Robot(
                x=bx + float(rng.uniform(1.0, 2.5)),
                y=by + float(rng.uniform(-1.5, 1.5)),
                theta=float(rng.uniform(120.0, 240.0)),
            )
            # Blues parked far from the goal mouth.
            pos.robots_blue[0] = Robot(
                x=float(rng.uniform(1.0, max_x - 0.5)),
                y=float(rng.uniform(-2.0, 2.0)),
                theta=float(rng.uniform(-180, 180)),
            )
            pos.robots_blue[1] = Robot(
                x=float(rng.uniform(1.0, max_x - 0.5)),
                y=float(rng.uniform(-2.0, 2.0)),
                theta=float(rng.uniform(-180, 180)),
            )
            return pos

        # LEVEL 5 — scenario roll: staged pass situation vs. chaos.
        if rng.random() < self.pass_scenario_prob:
            return self._pass_scenario_frame(pos, rng, max_x)

        self._episode_scenario = "chaos"
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

    def _pass_scenario_frame(self, pos, rng, max_x):
        """Staged pass situation — one of three variants, so the policy has
        to learn the CONCEPT (blocked + free mate -> pass) instead of a
        positional hack tied to a single geometry:

          corner:  carrier wide in the corner, blocker on the shot lane,
                   mate free at the far post. Single pass + finish.
          counter: carrier in OWN half with a chaser behind, mate advanced
                   and free — long forward pass. Closest to the chaos spawn
                   distribution, so the best transfer candidate.
          tiktaka: carrier in front of goal with BOTH blues congesting the
                   shot lanes, mate free BEHIND — back-pass draws the
                   ball-chasing blues, carrier sprints free, return pass,
                   finish. Two-pass give-and-go.
          pressed: chaos geometry with only the blocked shot lane enforced.
                   The three above fix the whole relative topology, so
                   their states barely overlap with chaos spawns; this one
                   trains the concept in open-play geometry and is the
                   bridge the staged->chaos transfer was missing.
        """
        self._episode_scenario = "pass"
        variant = int(rng.integers(0, 4))
        if variant == 0:
            return self._pass_corner_frame(pos, rng, max_x)
        if variant == 1:
            return self._pass_counter_frame(pos, rng, max_x)
        if variant == 2:
            return self._pass_tiktaka_frame(pos, rng, max_x)
        return self._pass_pressed_frame(pos, rng, max_x)

    @staticmethod
    def _ball_in_front(rng, cx, cy, direction):
        """Ball just outside the collision hull (robot ~0.09 + ball ~0.02);
        overlapping spawns get ejected by the physics engine."""
        off = float(rng.uniform(0.14, 0.18))
        return cx + off * float(direction[0]), cy + off * float(direction[1])

    def _clip_field(self, x, y, margin=0.15):
        mx = self.field.length / 2.0 - margin
        my = self.field.width / 2.0 - margin
        return float(np.clip(x, -mx, mx)), float(np.clip(y, -my, my))

    def _carrier_start(self, rng, cx, cy, to_goal,
                       jitter_deg=45.0, loose_prob=0.3):
        """Carrier heading + ball placement for a staged pass spawn.

        The staged variants used to place the ball at the dribbler with the
        carrier facing the goal EXACTLY, every time, while chaos spawns a
        loose ball with randomly oriented robots. Those two state
        distributions barely overlap, which is the most likely reason
        passing learned in the staged scenarios never showed up in chaos.
        Jittering the heading and loosening the ball part of the time
        closes that gap from the staged side — the policy also has to win
        the ball first in a pass situation, exactly as it must in chaos.
        """
        base = math.degrees(math.atan2(to_goal[1], to_goal[0]))
        theta = base + float(rng.uniform(-jitter_deg, jitter_deg))
        if rng.random() < loose_prob:
            ang = float(rng.uniform(-math.pi, math.pi))
            d = float(rng.uniform(0.30, 0.75))
            bx, by = cx + d * math.cos(ang), cy + d * math.sin(ang)
        else:
            bx, by = self._ball_in_front(rng, cx, cy, to_goal)
        return theta, self._clip_field(bx, by)

    def _pass_corner_frame(self, pos, rng, max_x):
        self._episode_pass_variant = "corner"
        side = 1.0 if rng.random() < 0.5 else -1.0
        carrier_idx = int(rng.integers(0, 2))
        mate_idx = 1 - carrier_idx

        cx = float(rng.uniform(-max_x + 1.0, -max_x + 2.0))
        cy = float(side * rng.uniform(1.5, 2.3))

        goal = np.array([-max_x, 0.0])
        to_goal = goal - np.array([cx, cy])
        to_goal = to_goal / np.linalg.norm(to_goal)
        theta_carrier, (bx, by) = self._carrier_start(rng, cx, cy, to_goal)
        pos.ball = Ball(x=bx, y=by)

        yellows = [None, None]
        yellows[carrier_idx] = Robot(x=cx, y=cy, theta=theta_carrier)
        # Mate at the far post (opposite y-side): pass lane diverges from
        # the blocked shot lane.
        mx = float(rng.uniform(-max_x + 0.7, -max_x + 1.4))
        my = float(-side * rng.uniform(0.2, 0.7))
        theta_mate = math.degrees(math.atan2(by - my, bx - mx))
        yellows[mate_idx] = Robot(x=mx, y=my, theta=theta_mate)
        pos.robots_yellow[0] = yellows[0]
        pos.robots_yellow[1] = yellows[1]

        # Blocker on the carrier->goal line, facing the carrier.
        bdist = float(rng.uniform(0.55, 0.85))
        blx = bx + bdist * float(to_goal[0])
        bly = by + bdist * float(to_goal[1])
        theta_blocker = math.degrees(math.atan2(cy - bly, cx - blx))
        pos.robots_blue[0] = Robot(x=blx, y=bly, theta=theta_blocker)
        # Second blue far upfield — arrives late.
        pos.robots_blue[1] = Robot(
            x=float(rng.uniform(0.5, 2.0)),
            y=float(rng.uniform(-1.5, 1.5)),
            theta=float(rng.uniform(-180, 180)),
        )
        return pos

    def _pass_counter_frame(self, pos, rng, max_x):
        """Counter: carrier in OWN half, chaser behind him, mate advanced
        and unmarked — the long forward pass beats the chase."""
        self._episode_pass_variant = "counter"
        carrier_idx = int(rng.integers(0, 2))
        mate_idx = 1 - carrier_idx

        cx = float(rng.uniform(0.8, 1.8))
        cy = float(rng.uniform(-1.5, 1.5))

        goal = np.array([-max_x, 0.0])
        to_goal = goal - np.array([cx, cy])
        to_goal = to_goal / np.linalg.norm(to_goal)
        theta_carrier, (bx, by) = self._carrier_start(rng, cx, cy, to_goal)
        pos.ball = Ball(x=bx, y=by)

        yellows = [None, None]
        yellows[carrier_idx] = Robot(x=cx, y=cy, theta=theta_carrier)
        # Mate advanced in the attacking half, roughly central corridor.
        mx = float(rng.uniform(-2.8, -1.8))
        my = float(np.clip(cy * -0.3 + rng.uniform(-0.8, 0.8), -2.0, 2.0))
        theta_mate = math.degrees(math.atan2(by - my, bx - mx))
        yellows[mate_idx] = Robot(x=mx, y=my, theta=theta_mate)
        pos.robots_yellow[0] = yellows[0]
        pos.robots_yellow[1] = yellows[1]

        # Chaser BEHIND the carrier (pressure, not in the pass lane).
        chx = cx + float(rng.uniform(0.5, 0.9))
        chy = float(np.clip(cy + rng.uniform(-0.3, 0.3), -2.4, 2.4))
        theta_chaser = math.degrees(math.atan2(cy - chy, cx - chx))
        pos.robots_blue[0] = Robot(x=chx, y=chy, theta=theta_chaser)
        # Retreating defender near the goal, displaced off the mate's lane.
        d_side = 1.0 if rng.random() < 0.5 else -1.0
        pos.robots_blue[1] = Robot(
            x=float(rng.uniform(-3.8, -3.0)),
            y=float(np.clip(my + d_side * rng.uniform(1.2, 1.8), -2.4, 2.4)),
            theta=float(rng.uniform(-180, 180)),
        )
        return pos

    def _pass_tiktaka_frame(self, pos, rng, max_x):
        """Give-and-go: both blues congest the carrier's shot lanes in
        front, the mate is free BEHIND the carrier. Intended play:
        back-pass -> blues chase the ball -> carrier sprints goal-ward
        into free space -> return pass -> finish. Two passes."""
        self._episode_pass_variant = "tiktaka"
        carrier_idx = int(rng.integers(0, 2))
        mate_idx = 1 - carrier_idx
        side = 1.0 if rng.random() < 0.5 else -1.0

        cx = float(rng.uniform(-2.4, -1.6))
        cy = float(rng.uniform(-1.2, 1.2))

        goal = np.array([-max_x, 0.0])
        to_goal = goal - np.array([cx, cy])
        to_goal = to_goal / np.linalg.norm(to_goal)
        theta_carrier, (bx, by) = self._carrier_start(rng, cx, cy, to_goal)
        pos.ball = Ball(x=bx, y=by)

        yellows = [None, None]
        yellows[carrier_idx] = Robot(x=cx, y=cy, theta=theta_carrier)
        # Mate free BEHIND the carrier (both blues are in front).
        mx = float(np.clip(cx + rng.uniform(0.9, 1.5), -max_x + 0.3, max_x - 0.3))
        my = float(np.clip(cy + side * rng.uniform(0.3, 0.9), -2.3, 2.3))
        theta_mate = math.degrees(math.atan2(by - my, bx - mx))
        yellows[mate_idx] = Robot(x=mx, y=my, theta=theta_mate)
        pos.robots_yellow[0] = yellows[0]
        pos.robots_yellow[1] = yellows[1]

        # Blue 1: blocker on the shot lane.
        bdist = float(rng.uniform(0.55, 0.85))
        blx = bx + bdist * float(to_goal[0])
        bly = by + bdist * float(to_goal[1])
        theta_blocker = math.degrees(math.atan2(cy - bly, cx - blx))
        pos.robots_blue[0] = Robot(x=blx, y=bly, theta=theta_blocker)
        # Blue 2: congests the second lane in front, laterally displaced.
        b2x = float(np.clip(cx - rng.uniform(0.4, 0.8), -max_x + 0.3, max_x))
        b2y = float(np.clip(cy - side * rng.uniform(0.6, 1.0), -2.4, 2.4))
        pos.robots_blue[1] = Robot(
            x=b2x, y=b2y,
            theta=math.degrees(math.atan2(cy - b2y, cx - b2x)),
        )
        return pos

    def _pass_pressed_frame(self, pos, rng, max_x):
        """Chaos geometry with ONE property enforced: the shot lane is
        blocked while the mate is free.

        corner/counter/tiktaka fix the entire relative topology and only
        jitter positions by roughly half a metre, so their states hardly
        overlap with chaos spawns — which is why passing learned there
        never transferred. This variant keeps chaos-like randomness
        (free positions, random headings, frequently a loose ball) and
        guarantees only what the CONCEPT needs. It is the bridge between
        the staged templates and open play.
        """
        self._episode_pass_variant = "pressed"
        carrier_idx = int(rng.integers(0, 2))
        mate_idx = 1 - carrier_idx

        cx = float(rng.uniform(-max_x + 1.0, 2.0))
        cy = float(rng.uniform(-2.2, 2.2))
        goal = np.array([-max_x, 0.0])
        to_goal = goal - np.array([cx, cy])
        to_goal = to_goal / np.linalg.norm(to_goal)

        # Wider heading jitter and more loose balls than the templates.
        theta_carrier, (bx, by) = self._carrier_start(
            rng, cx, cy, to_goal, jitter_deg=90.0, loose_prob=0.4,
        )
        pos.ball = Ball(x=bx, y=by)

        yellows = [None, None]
        yellows[carrier_idx] = Robot(x=cx, y=cy, theta=theta_carrier)
        # Mate anywhere with a random heading, just not on top of the
        # carrier — "free" is created by the blues, not by placing him.
        mx, my = cx, cy
        for _ in range(20):
            mx = float(rng.uniform(-max_x + 0.5, 2.5))
            my = float(rng.uniform(-2.3, 2.3))
            if math.hypot(mx - cx, my - cy) > 1.0:
                break
        yellows[mate_idx] = Robot(
            x=mx, y=my, theta=float(rng.uniform(-180, 180)),
        )
        pos.robots_yellow[0] = yellows[0]
        pos.robots_yellow[1] = yellows[1]

        # The enforced property: a blue on the ball->goal lane, with a
        # little lateral slop so it is not a perfectly centred wall.
        lane = goal - np.array([bx, by])
        lane_len = float(np.linalg.norm(lane))
        lane = lane / max(lane_len, 1e-6)
        perp = np.array([-lane[1], lane[0]])
        bdist = float(rng.uniform(0.5, max(0.6, min(1.5, lane_len - 0.4))))
        off = float(rng.uniform(-0.25, 0.25))
        blx, bly = self._clip_field(
            bx + bdist * lane[0] + off * perp[0],
            by + bdist * lane[1] + off * perp[1],
            margin=0.3,
        )
        pos.robots_blue[0] = Robot(
            x=blx, y=bly,
            theta=math.degrees(math.atan2(cy - bly, cx - blx)),
        )
        # Second blue: fully random, exactly like a chaos spawn.
        pos.robots_blue[1] = Robot(
            x=float(rng.uniform(-max_x + 0.5, 1.0)),
            y=float(rng.uniform(-2.3, 2.3)),
            theta=float(rng.uniform(-180, 180)),
        )
        return pos
