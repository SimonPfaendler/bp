"""Generate scripted pass->goal demonstration episodes in the staged
pass scenario, for replay-buffer prefill (DQfD-light).

Carrier: secures the ball, turns toward the mate, plays a soft pass.
Mate:    holds position facing the ball; once the ball arrives, controls
         it and shoots at the goal (level-1 finishing pattern).
Blue:    the frozen opponent the next generation will actually train
         against, so demo transitions are on-distribution.

Only episodes that end with scored_after_pass=1 are kept. The script also
doubles as a scenario validator: if the scripted play can't convert, the
spawn geometry is too hard and needs re-staging before burning cluster time.

Usage:
    /home/simon/dev/venv_rl/bin/python generate_pass_demos.py \\
        --frozen models/2v2_selfplay_SAC_dense_seed822_20260706-081229_10560000_steps.zip \\
        --n_success 300 --out_dir pass_demos
"""
import argparse
import math
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from skills import move_to_point, turn_to_point
from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv

# Kick encoding (see SSL2v2SelfPlayEnv._robot_command):
#   trigger > 0  ->  kick_speed = 3.0 + ((raw + 1) / 2) * 3.0   in [3, 6] m/s
PASS_KICK_RAW = -0.85   # ~3.2 m/s — soft pass
SHOT_KICK_RAW = 1.0     # 6.0 m/s — full shot
ALIGN_TOL = 0.10        # |turn_to_point| below this ≈ facing within ~5°


def _norm_move(env, v_x, v_y):
    """World-frame m/s -> normalized action components."""
    return float(np.clip(v_x / env.max_v_cmd, -1, 1)), \
           float(np.clip(v_y / env.max_v_cmd, -1, 1))


def _creep_to_ball(env, robot, ball, max_speed):
    """Approach velocity that keeps pushing until the dribbler mouth makes
    infrared contact. move_to_point() full-stops at 0.09m center distance,
    which is just SHORT of physical contact (~0.11m) — so we roll our own
    with a speed floor instead.
    """
    dx, dy = ball.x - robot.x, ball.y - robot.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, 0.0
    speed = min(max_speed, max(0.25, dist * 1.5))
    return dx / dist * speed, dy / dist * speed


def scripted_yellow_actions(env, passed_flag, carrier_idx):
    """6-dim normalized action per yellow agent for the current frame.

    Returns (actions (2,6), passed_flag). All ball-handling is gated on
    `robot.infrared` (true dribbler-mouth contact) — the kicker only
    connects when the ball is physically at the mouth; distance-based
    gating fires kicks into thin air.
    """
    ball = env.frame.ball
    ya, yb = env.frame.robots_yellow[0], env.frame.robots_yellow[1]
    yellows = [ya, yb]
    actions = np.zeros((2, 6), dtype=np.float32)

    dists = [math.hypot(r.x - ball.x, r.y - ball.y) for r in yellows]
    ball_pos = np.array([ball.x, ball.y])

    for i, robot in enumerate(yellows):
        a = np.zeros(6, dtype=np.float32)
        i_am_closest = dists[i] <= dists[1 - i]
        mate = yellows[1 - i]

        if not passed_flag and i == carrier_idx:
            # --- Carrier: secure ball (infrared), face mate, soft pass. ---
            if not robot.infrared:
                v_x, v_y = _creep_to_ball(env, robot, ball, max_speed=0.8)
                a[0], a[1] = _norm_move(env, v_x, v_y)
                a[2] = turn_to_point(robot, ball_pos)
                a[5] = 1.0  # dribbler on for pickup
            else:
                v_theta = turn_to_point(robot, np.array([mate.x, mate.y]))
                if abs(v_theta) < ALIGN_TOL:
                    # Aligned with ball at the mouth -> the kick connects.
                    a[3] = PASS_KICK_RAW
                    a[4] = 1.0
                    passed_flag = True
                else:
                    # Rotate in place with the ball held by the dribbler.
                    a[2] = v_theta
                    a[5] = 1.0
        elif passed_flag and i_am_closest:
            # --- Receiver: control the incoming ball, then shoot. ---
            if not robot.infrared:
                v_x, v_y = _creep_to_ball(env, robot, ball, max_speed=2.0)
                a[0], a[1] = _norm_move(env, v_x, v_y)
                a[2] = turn_to_point(robot, ball_pos)
                a[5] = 1.0
            else:
                goal = np.array([-env.field.length / 2.0, 0.0])
                v_theta = turn_to_point(robot, goal)
                if abs(v_theta) < ALIGN_TOL:
                    a[3] = SHOT_KICK_RAW
                    a[4] = 1.0
                else:
                    a[2] = v_theta
                    a[5] = 1.0
        else:
            # --- Off-ball: hold position, face the ball. ---
            a[2] = turn_to_point(robot, ball_pos)

        actions[i] = np.clip(a, -1.0, 1.0)

    return actions, passed_flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", required=True,
                        help="Frozen blue checkpoint (= next gen's opponent)")
    parser.add_argument("--n_success", type=int, default=300)
    parser.add_argument("--max_attempts", type=int, default=3000)
    parser.add_argument("--max_ep_steps", type=int, default=400)
    parser.add_argument("--out_dir", default="pass_demos")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = SSL2v2SelfPlayEnv(
        reward_type="dense",
        render_mode="human" if args.render else None,
        frozen_path=args.frozen,
        pass_scenario_prob=1.0,
    )
    env.set_curriculum_level(5)

    n_success = 0
    n_attempts = 0
    t0 = time.time()

    while n_success < args.n_success and n_attempts < args.max_attempts:
        obs, _ = env.reset(seed=args.seed + n_attempts)
        n_attempts += 1
        # Spawn carrier = yellow closest to the ball.
        ball = env.frame.ball
        d0 = math.hypot(env.frame.robots_yellow[0].x - ball.x,
                        env.frame.robots_yellow[0].y - ball.y)
        d1 = math.hypot(env.frame.robots_yellow[1].x - ball.x,
                        env.frame.robots_yellow[1].y - ball.y)
        carrier_idx = 0 if d0 <= d1 else 1

        passed_flag = False
        steps = []
        info = {}
        for _t in range(args.max_ep_steps):
            action, passed_flag = scripted_yellow_actions(
                env, passed_flag, carrier_idx
            )
            next_obs, reward, done, truncated, info = env.step(action)
            steps.append({
                "obs": obs.copy(),
                "action": action.copy(),
                "reward": np.asarray(reward, dtype=np.float32).copy(),
                "next_obs": next_obs.copy(),
                "done": bool(done or truncated),
            })
            obs = next_obs
            if args.render:
                env.render()
                time.sleep(0.025)
            if done or truncated:
                break

        if info.get("scored_after_pass", 0.0) >= 1.0:
            n_success += 1
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            record = {
                "scenario": "pass",
                "frozen_path": args.frozen,
                "n_steps": len(steps),
                "outcome": "scored_after_pass",
                "passes": info.get("passes", 0),
                "steps": steps,
            }
            fname = Path(args.out_dir) / f"pass_demo_{n_success:04d}_{timestamp}.pkl"
            with open(fname, "wb") as f:
                pickle.dump(record, f)

        if n_attempts % 50 == 0:
            rate = n_success / n_attempts
            print(f"attempts={n_attempts}  successes={n_success}  "
                  f"rate={rate:.2%}  ({time.time() - t0:.0f}s)")

    rate = n_success / max(1, n_attempts)
    print("=" * 60)
    print(f"Done: {n_success} demo episodes in {n_attempts} attempts "
          f"(success rate {rate:.2%}) -> {args.out_dir}/")
    if rate < 0.15:
        print("WARNING: scripted success rate is low — the scenario may be "
              "too hard. Consider moving blue2 further away or the blocker "
              "closer to the goal before training on it.")
    env.close()


if __name__ == "__main__":
    main()
