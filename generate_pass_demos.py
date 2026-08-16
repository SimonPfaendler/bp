"""Generate scripted pass->goal demonstration episodes in the staged
pass scenarios, for replay-buffer prefill (DQfD-light).

Variants (chosen by the env spawn, read back via env._episode_pass_variant):
  corner / counter — single pass: carrier secures, turns, passes; the
                     closest yellow to the arriving ball controls & shoots.
  tiktaka          — give-and-go: back-pass draws the ball-chasing blues,
                     the original carrier sprints goal-ward, return pass,
                     finish. Produces passes=2 episodes.

Only episodes ending with scored_after_pass=1 are kept. The script doubles
as a scenario validator: per-variant success rates are reported at the end.

Usage:
    /home/simon/dev/venv_rl/bin/python generate_pass_demos.py \\
        --frozen models/<opponent>.zip --n_success 300 --out_dir pass_demos
"""
import argparse
import math
import os
import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from skills import turn_to_point
from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv

# Kick encoding (see SSL2v2SelfPlayEnv._robot_command):
#   trigger > 0  ->  kick_speed = 3.0 + ((raw + 1) / 2) * 3.0   in [3, 6] m/s
SHOT_KICK_RAW = 1.0     # 6.0 m/s — full shot
ALIGN_TOL = 0.10        # |turn_to_point| below this ≈ facing within ~5°


def _pass_raw(dist, variant="corner"):
    """Distance-adaptive pass power: short give ~3 m/s, long counter ~5."""
    if variant == "counter":
        # Long ball over ~3.5-4.5m: aim for ~1.2 m/s arrival speed
        # (arrival = v0 - 0.6*d) so the receiver can trap the ball instead
        # of it flying past him into the goal.
        desired = float(np.clip(1.2 + 0.6 * dist, 3.0, 4.5))
    else:
        desired = float(np.clip(2.8 + 0.45 * dist, 3.0, 5.4))
    return (desired - 4.5) / 1.5


def _norm_move(env, v_x, v_y):
    return float(np.clip(v_x / env.max_v_cmd, -1, 1)), \
           float(np.clip(v_y / env.max_v_cmd, -1, 1))


def _creep_to_ball(env, robot, ball, max_speed):
    """Approach that keeps pushing until the dribbler mouth makes infrared
    contact (plain move_to_point full-stops just short of contact)."""
    dx, dy = ball.x - robot.x, ball.y - robot.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0, 0.0
    speed = min(max_speed, max(0.25, dist * 1.5))
    return dx / dist * speed, dy / dist * speed


def _move_to(env, robot, target, speed):
    dx, dy = target[0] - robot.x, target[1] - robot.y
    dist = math.hypot(dx, dy)
    if dist < 0.05:
        return 0.0, 0.0
    s = min(speed, max(0.3, dist * 2.0))
    return dx / dist * s, dy / dist * s


def _secure_and_pass(env, robot, ball, target_xy, a, variant="corner"):
    """Shared primitive: creep to infrared contact, rotate to target, kick
    with distance-adaptive power. Returns True on the release frame."""
    if not robot.infrared:
        v_x, v_y = _creep_to_ball(env, robot, ball, max_speed=1.0)
        a[0], a[1] = _norm_move(env, v_x, v_y)
        a[2] = turn_to_point(robot, np.array([ball.x, ball.y]))
        a[5] = 1.0
        return False
    v_theta = turn_to_point(robot, np.asarray(target_xy, dtype=float))
    if abs(v_theta) < ALIGN_TOL:
        dist = math.hypot(target_xy[0] - robot.x, target_xy[1] - robot.y)
        a[3] = _pass_raw(dist, variant)
        a[4] = 1.0
        return True
    a[2] = v_theta
    a[5] = 1.0
    return False


def _control_and_shoot(env, robot, ball, a):
    """Shared primitive: intercept the ball, control, align to goal, shoot."""
    if not robot.infrared:
        v_x, v_y = _creep_to_ball(env, robot, ball, max_speed=2.0)
        a[0], a[1] = _norm_move(env, v_x, v_y)
        a[2] = turn_to_point(robot, np.array([ball.x, ball.y]))
        a[5] = 1.0
        return
    goal = np.array([-env.field.length / 2.0, 0.0])
    v_theta = turn_to_point(robot, goal)
    if abs(v_theta) < ALIGN_TOL:
        a[3] = SHOT_KICK_RAW
        a[4] = 1.0
    else:
        a[2] = v_theta
        a[5] = 1.0


def _hold_and_face_ball(robot, ball, a):
    a[2] = turn_to_point(robot, np.array([ball.x, ball.y]))


def single_pass_actions(env, state):
    """corner / counter: one pass, closest yellow finishes."""
    ball = env.frame.ball
    yellows = [env.frame.robots_yellow[0], env.frame.robots_yellow[1]]
    actions = np.zeros((2, 6), dtype=np.float32)
    dists = [math.hypot(r.x - ball.x, r.y - ball.y) for r in yellows]

    for i, robot in enumerate(yellows):
        a = actions[i]
        mate = yellows[1 - i]
        if state["phase"] == 0 and i == state["carrier_idx"]:
            if _secure_and_pass(env, robot, ball, (mate.x, mate.y), a,
                                variant=state["variant"]):
                state["phase"] = 1
        elif state["phase"] == 1 and dists[i] <= dists[1 - i]:
            _control_and_shoot(env, robot, ball, a)
        else:
            _hold_and_face_ball(robot, ball, a)
        actions[i] = np.clip(a, -1.0, 1.0)
    return actions


def tiktaka_actions(env, state):
    """Give-and-go phase machine.

    0: carrier back-passes to the free mate behind him
    1: carrier sprints to a spot in front of the goal; mate secures
    2: mate returns the ball to the runner (early release under pressure)
    3: runner controls & finishes
    """
    ball = env.frame.ball
    yellows = [env.frame.robots_yellow[0], env.frame.robots_yellow[1]]
    blues = [env.frame.robots_blue[0], env.frame.robots_blue[1]]
    actions = np.zeros((2, 6), dtype=np.float32)
    runner = state["carrier_idx"]
    mate = 1 - runner

    if state["spot"] is None:
        max_x = env.field.length / 2.0
        cy = yellows[runner].y
        state["spot"] = np.array(
            [-max_x + 1.25, float(np.clip(cy * 0.2, -0.5, 0.5))]
        )

    for i, robot in enumerate(yellows):
        a = actions[i]
        if state["phase"] == 0:
            if i == runner:
                other = yellows[mate]
                if _secure_and_pass(env, robot, ball, (other.x, other.y), a):
                    state["phase"] = 1
            else:
                _hold_and_face_ball(robot, ball, a)
        elif state["phase"] == 1:
            if i == runner:
                # Sprint into free space in front of the goal.
                v_x, v_y = _move_to(env, robot, state["spot"], speed=2.0)
                a[0], a[1] = _norm_move(env, v_x, v_y)
                a[2] = turn_to_point(robot, np.array([ball.x, ball.y]))
            else:
                # Secure the incoming back-pass; hold facing the runner.
                if not robot.infrared:
                    v_x, v_y = _creep_to_ball(env, robot, ball, max_speed=2.0)
                    a[0], a[1] = _norm_move(env, v_x, v_y)
                    a[2] = turn_to_point(robot, np.array([ball.x, ball.y]))
                    a[5] = 1.0
                else:
                    a[2] = turn_to_point(
                        robot, np.array([yellows[runner].x, yellows[runner].y])
                    )
                    a[5] = 1.0
                    runner_r = yellows[runner]
                    runner_near = math.hypot(
                        runner_r.x - state["spot"][0],
                        runner_r.y - state["spot"][1],
                    ) < 0.6
                    pressure = min(
                        math.hypot(b.x - robot.x, b.y - robot.y)
                        for b in blues
                    ) < 0.5
                    if runner_near or pressure:
                        state["phase"] = 2
        elif state["phase"] == 2:
            if i == mate:
                target = yellows[runner]
                if _secure_and_pass(env, robot, ball, (target.x, target.y), a):
                    state["phase"] = 3
            else:
                v_x, v_y = _move_to(env, robot, state["spot"], speed=2.0)
                a[0], a[1] = _norm_move(env, v_x, v_y)
                a[2] = turn_to_point(robot, np.array([ball.x, ball.y]))
        else:  # phase 3
            if i == runner:
                _control_and_shoot(env, robot, ball, a)
            else:
                _hold_and_face_ball(robot, ball, a)
        actions[i] = np.clip(a, -1.0, 1.0)
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", default=None,
                        help="Frozen blue checkpoint (= next gen's opponent). "
                             "Omit when using --blue_heuristic.")
    parser.add_argument("--blue_heuristic", default=None, choices=["attacker"],
                        help="Hand-coded blue team as the opponent instead of "
                             "a checkpoint. Demos should be recorded against "
                             "the same opponent the agent will train against.")
    parser.add_argument("--n_success", type=int, default=300)
    parser.add_argument("--max_attempts", type=int, default=3000)
    parser.add_argument("--max_ep_steps", type=int, default=400)
    parser.add_argument("--out_dir", default="pass_demos")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.frozen and not args.blue_heuristic:
        parser.error("give either --frozen <checkpoint> or --blue_heuristic")
    env = SSL2v2SelfPlayEnv(
        reward_type="dense",
        render_mode="human" if args.render else None,
        # The heuristic drives blue at the command stage and ignores any
        # frozen model, so the two are mutually exclusive.
        frozen_path=None if args.blue_heuristic else args.frozen,
        blue_heuristic=args.blue_heuristic,
        pass_scenario_prob=1.0,
    )
    print(
        "Opponent: "
        + (f"heuristic({args.blue_heuristic})" if args.blue_heuristic
           else args.frozen)
    )
    env.set_curriculum_level(5)

    n_success = 0
    n_attempts = 0
    var_attempts = defaultdict(int)
    var_success = defaultdict(int)
    t0 = time.time()

    while n_success < args.n_success and n_attempts < args.max_attempts:
        obs, _ = env.reset(seed=args.seed + n_attempts)
        n_attempts += 1
        variant = getattr(env, "_episode_pass_variant", "corner")
        var_attempts[variant] += 1

        ball = env.frame.ball
        d0 = math.hypot(env.frame.robots_yellow[0].x - ball.x,
                        env.frame.robots_yellow[0].y - ball.y)
        d1 = math.hypot(env.frame.robots_yellow[1].x - ball.x,
                        env.frame.robots_yellow[1].y - ball.y)
        state = {
            "phase": 0,
            "carrier_idx": 0 if d0 <= d1 else 1,
            "spot": None,
            "variant": variant,
        }
        controller = tiktaka_actions if variant == "tiktaka" else single_pass_actions

        steps = []
        info = {}
        for _t in range(args.max_ep_steps):
            action = controller(env, state)
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
            var_success[variant] += 1
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            record = {
                "scenario": "pass",
                "variant": variant,
                "frozen_path": args.frozen or f"heuristic:{args.blue_heuristic}",
                "n_steps": len(steps),
                "outcome": "scored_after_pass",
                "passes": info.get("passes", 0),
                "steps": steps,
            }
            fname = Path(args.out_dir) / f"pass_demo_{variant}_{n_success:04d}_{timestamp}.pkl"
            with open(fname, "wb") as f:
                pickle.dump(record, f)

        if n_attempts % 50 == 0:
            print(f"attempts={n_attempts}  successes={n_success}  "
                  f"({time.time() - t0:.0f}s)  per-variant: " + "  ".join(
                      f"{v}={var_success[v]}/{var_attempts[v]}"
                      for v in sorted(var_attempts)))

    print("=" * 64)
    print(f"Done: {n_success} demo episodes in {n_attempts} attempts "
          f"-> {args.out_dir}/")
    for v in sorted(var_attempts):
        rate = var_success[v] / max(1, var_attempts[v])
        flag = "  <-- LOW, re-stage this variant" if rate < 0.15 else ""
        print(f"  {v:8s}: {var_success[v]}/{var_attempts[v]} "
              f"({rate:.1%}){flag}")
    env.close()


if __name__ == "__main__":
    main()
