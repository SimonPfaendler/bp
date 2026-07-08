"""Replay recorded pass-demo .pkl episodes in the native rSoccer renderer.

No physics re-simulation: world positions/orientations are reconstructed
from the stored observations and stuffed into `env.frame`, which is all the
rSoccer renderer reads. The replay therefore looks exactly like the live
simulator and is exactly what was recorded.

Usage:
    /home/simon/dev/venv_rl/bin/python render_pass_demos.py [--demo_dir pass_demos]
        [--n_eps 10] [--step_sleep 0.025]

Keys:  SPACE pause/resume | N next episode | ESC quit
Kick frames (pass / shot) are printed to the console.
"""
import argparse
import glob
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pygame
from rsoccer_gym.Entities import Ball, Frame, Robot

from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv

# Obs slots (agent-0 row of the joint obs, see _egocentric_obs docstring)
BALL_X, BALL_Y = 0, 1
SELF_X, SELF_Y, SELF_SIN, SELF_COS = 5, 6, 7, 8
MATE_X, MATE_Y, MATE_SIN, MATE_COS = 18, 19, 20, 21
OPP1_X, OPP1_Y, OPP1_SIN, OPP1_COS = 27, 28, 29, 30
OPP2_X, OPP2_Y, OPP2_SIN, OPP2_COS = 37, 38, 39, 40


def obs_to_frame(obs_row, max_pos):
    """Rebuild a renderable Frame from one recorded (normalized) obs row."""
    def pos(ix, iy):
        return float(obs_row[ix]) * max_pos, float(obs_row[iy]) * max_pos

    def theta_deg(isin, icos):
        return math.degrees(
            math.atan2(float(obs_row[isin]), float(obs_row[icos]))
        )

    frame = Frame()
    bx, by = pos(BALL_X, BALL_Y)
    frame.ball = Ball(x=bx, y=by)

    yx, yy = pos(SELF_X, SELF_Y)
    frame.robots_yellow[0] = Robot(
        id=0, yellow=True, x=yx, y=yy, theta=theta_deg(SELF_SIN, SELF_COS)
    )
    mx, my = pos(MATE_X, MATE_Y)
    frame.robots_yellow[1] = Robot(
        id=1, yellow=True, x=mx, y=my, theta=theta_deg(MATE_SIN, MATE_COS)
    )
    # Opponents are distance-sorted in the obs, so blue identity may swap
    # between frames — irrelevant for watching.
    o1x, o1y = pos(OPP1_X, OPP1_Y)
    frame.robots_blue[0] = Robot(
        id=0, yellow=False, x=o1x, y=o1y, theta=theta_deg(OPP1_SIN, OPP1_COS)
    )
    o2x, o2y = pos(OPP2_X, OPP2_Y)
    frame.robots_blue[1] = Robot(
        id=1, yellow=False, x=o2x, y=o2y, theta=theta_deg(OPP2_SIN, OPP2_COS)
    )
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_dir", default="pass_demos")
    parser.add_argument("--n_eps", type=int, default=0, help="0 = all")
    parser.add_argument("--step_sleep", type=float, default=0.025)
    args = parser.parse_args()

    files = sorted(glob.glob(str(Path(args.demo_dir) / "*.pkl")))
    if not files:
        raise SystemExit(f"No .pkl demos in {args.demo_dir}")
    if args.n_eps > 0:
        files = files[: args.n_eps]

    env = SSL2v2SelfPlayEnv(
        reward_type="dense", render_mode="human", frozen_path=None
    )
    env.reset(seed=0)
    max_pos = env.max_pos

    quit_all = False
    for ep_i, fpath in enumerate(files):
        if quit_all:
            break
        with open(fpath, "rb") as f:
            record = pickle.load(f)
        steps = record["steps"]
        print(f"[{ep_i + 1}/{len(files)}] {Path(fpath).name}  "
              f"steps={record['n_steps']}  passes={record.get('passes')}")

        t = 0
        paused = False
        skip_ep = False
        while t < len(steps) and not skip_ep and not quit_all:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_all = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        quit_all = True
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_n:
                        skip_ep = True
            if paused:
                time.sleep(0.05)
                continue

            step = steps[t]
            obs_row = np.asarray(step["obs"])[0]
            env.frame = obs_to_frame(obs_row, max_pos)
            env.render()

            action = step.get("action")
            if action is not None:
                for i in range(2):
                    if float(action[i][4]) > 0.0:
                        print(f"    [kick] step {t} by yellow {i}")

            t += 1
            time.sleep(args.step_sleep)

    env.close()


if __name__ == "__main__":
    main()
