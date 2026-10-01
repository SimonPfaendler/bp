"""Naive self-play on this machine: yellow keeps learning, blue is a FROZEN
copy of the checkpoint yellow starts from (one ladder step, fixed opponent).

    python selfplay_local.py models/c15_s101_L5-6_final.zip --minutes 60

Same algorithm, hyperparameters, rules and vectorisation as the cluster
protocol (24 pairs = 48 slots, 96 gradient steps per 2304 transitions), so
the checkpoint and its replay buffer stay chain-compatible. What differs:

  * the opponent: the frozen start checkpoint instead of the heuristic;
  * the spawn: the open game only (difficulty 1.0, no defensive frames) —
    a learned opponent creates attack and defence by itself;
  * the hardware: no GPU here. The gradient steps run on the CPU (the
    trainer's one-thread limit is lifted for the main process only; the env
    subprocesses stay single-threaded), at roughly 1/8 of the H100's pace.

The rule and reward flags are read from the start checkpoint's sidecar.
The fair-opponent mechanics (blue decides at its training rate and sees its
own pass state) come from the env; see test_blue_fairness.py.
"""
import argparse
import json
import os

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import torch  # noqa: E402  (before the trainer pins the thread env vars)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", help="checkpoint yellow starts from (and blue's copy)")
    p.add_argument("--opponent", default=None, help="frozen blue (default: start)")
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--n_pairs", type=int, default=24)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 6)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--critic_warmup", type=int, default=60000)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    base = args.start[:-4] if args.start.endswith(".zip") else args.start
    with open(f"{base}_replay_buffer.json") as f:
        side = json.load(f)
    name = args.name or f"sp_{os.path.basename(base).replace('_final', '')}"
    os.environ["RUN_NAME"] = name

    import train_2v2_selfplay as T

    torch.set_num_threads(args.threads)      # main process: CPU gradient steps
    print(f"[selfplay_local] run={name} start={args.start} "
          f"opponent={args.opponent or args.start} pairs={args.n_pairs} "
          f"main threads={torch.get_num_threads()} budget={args.minutes} min")

    T.train(
        reward_type="dense", seed=args.seed, n_envs=args.n_pairs,
        frozen_path=args.opponent or args.start, init_path=args.start,
        total_steps=100_000_000, max_minutes=args.minutes, algo="sac",
        start_level=5, target_level=5, load_buffer="off", blue_heuristic=None,
        pass_gate=side["pass_gate"], goal_reward_solo=side["goal_reward_solo"],
        dribble_rule=side["dribble_rule"], shaping=side["shaping"],
        restarts=side["restarts"], foul_restart=side["foul_restart"],
        role_index=bool(side.get("role_index", True)),
        frame_stack=int(side["frame_stack"]), action_repeat=int(side["action_repeat"]),
        difficulty=1.0, defense_frame_prob=0.0,
        critic_warmup_steps=args.critic_warmup,
    )


if __name__ == "__main__":
    # The guard matters: the env subprocesses are started by spawning a
    # fresh interpreter that re-imports this file.
    main()
