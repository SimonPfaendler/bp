"""Visual check of the staged pass-scenario spawn geometry.

Spawns only pass scenarios (pass_scenario_prob=1.0), renders each for a
couple of seconds with idle robots so you can inspect the layout:
carrier + ball in the corner, blocker on the shot lane, mate at the far
post, second blue far upfield.

Usage:
    /home/simon/dev/venv_rl/bin/python check_pass_scenario.py [--n 8]
"""
import argparse
import time

import numpy as np

from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="Spawns to show")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="Seconds to hold each spawn")
    args = parser.parse_args()

    env = SSL2v2SelfPlayEnv(
        reward_type="dense",
        render_mode="human",
        frozen_path=None,          # blue idle -> pure geometry check
        pass_scenario_prob=1.0,    # only pass scenarios
    )
    env.set_curriculum_level(5)

    idle = np.zeros((2, 6), dtype=np.float32)
    for i in range(args.n):
        env.reset(seed=100 + i)
        variant = getattr(env, "_episode_pass_variant", "-")
        print(f"Spawn {i + 1}/{args.n}  "
              f"(scenario={env._episode_scenario}, variant={variant})")
        t_end = time.time() + args.hold
        while time.time() < t_end:
            env.step(idle)
            env.render()
            time.sleep(0.025)
    env.close()


if __name__ == "__main__":
    main()
