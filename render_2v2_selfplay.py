"""Render or evaluate 2v2 episodes. Blue is a frozen checkpoint, the
hand-coded heuristic team, or static.

Two jobs in one script:

  * watching (default): a pygame window, real-time, a few episodes.
  * measuring (--headless): no window, no sleep, hundreds of episodes, and
    a summary with 95 % confidence intervals. This is the frozen-policy
    eval that decides whether passing learned on the reverse curriculum
    transfers to the open game.

The env is built with its DEFAULT rules (pass_gate="loose", restarts="off",
difficulty=None, no defensive frame) unless told otherwise. With
--pass_prob 0 and --level 5 that is the pure chaos spawn under the
Gen-1..12 rules — the one distribution on which the historical ~0.02
passes/episode was measured, and therefore the only fair comparison for a
checkpoint trained on the staged curriculum. --like_training pulls the
training-time rule flags from the checkpoint's sidecar instead, for a
control run that should reproduce the training log.

frame_stack / action_repeat are properties of the checkpoint's I/O, not of
the task, so they are ALWAYS taken from the sidecar (or inferred from the
model's obs dim) unless overridden.

Usage:
    # watch, vs. the hand-coded team
    python render_2v2_selfplay.py models/<yellow>.zip --blue_heuristic attacker

    # THE eval: pure chaos spawn, old rules, 300 episodes, no window
    SDL_VIDEODRIVER=dummy python render_2v2_selfplay.py models/<yellow>.zip \\
        --blue_heuristic attacker --pass_prob 0 --level 5 --n_eps 300 --headless

    # control: the training distribution (should reproduce the training log)
    SDL_VIDEODRIVER=dummy python render_2v2_selfplay.py models/<yellow>.zip \\
        --blue_heuristic attacker --headless --n_eps 300 --like_training \\
        --difficulty 0.398 --defense_frame_prob 0.25
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from stable_baselines3 import SAC

import masac_policy  # noqa: F401 — registers MASACPolicy for checkpoint unpickling
from ssl_rl_2v2_selfplay import (
    SSL2v2SelfPlayEnv,
    single_obs_dim as _single_obs_dim,
)


def _sidecar(model_path):
    """<run>_final_replay_buffer.json written next to every _final.zip.

    Records action_repeat, frame_stack and the reward/rule flags the run
    trained with (see train_2v2_selfplay.py, final save)."""
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    path = f"{base}_replay_buffer.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f), path
    return {}, None


def _infer_layout(model_obs_dim, frame_stack, n_yellow=2):
    """(frame_stack, single_obs_dim, role_index) for a checkpoint of a team
    of n_yellow robots: 56 / 58 dims for two, 65 / 68 for three.

    The two widths share no multiple below their product, so when the
    stack depth is unknown the obs dim identifies it unambiguously."""
    no_role = _single_obs_dim(n_yellow, False)
    with_role = _single_obs_dim(n_yellow, True)
    if frame_stack is None:
        if model_obs_dim % no_role == 0:
            frame_stack = model_obs_dim // no_role
        elif model_obs_dim % with_role == 0:
            frame_stack = model_obs_dim // with_role
        else:
            raise SystemExit(
                f"Cannot infer frame_stack from obs dim {model_obs_dim} "
                f"(not a multiple of {no_role} or {with_role} for a team of "
                f"{n_yellow}); pass --frame_stack / --n_yellow explicitly."
            )
    single = model_obs_dim // frame_stack
    if single not in (no_role, with_role):
        raise SystemExit(
            f"obs dim {model_obs_dim} / frame_stack {frame_stack} = {single}, "
            f"expected {no_role} or {with_role} for a team of {n_yellow}. "
            f"Older checkpoints (52-dim, pre obs-repair) are not supported here."
        )
    return frame_stack, single, single == with_role


class FrameStacker:
    """Mirror of SB3's StackedObservations for one (2, D) observation:
    newest frame LAST, zeros for the missing history at episode start."""

    def __init__(self, k, obs_dim):
        self.k, self.d, self.buf = int(k), int(obs_dim), None

    def reset(self, obs):
        self.buf = np.zeros(obs.shape[:-1] + (self.d * self.k,), np.float32)
        self.buf[..., -self.d:] = obs
        return self.buf.copy()

    def step(self, obs):
        self.buf = np.roll(self.buf, -self.d, axis=-1)
        self.buf[..., -self.d:] = obs
        return self.buf.copy()


def _wilson(k, n, z=1.96):
    """95 % Wilson interval for a rate — sane at 0 and 1, unlike Wald."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _mean_se(xs):
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(xs.size)) if xs.size > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path", nargs="?",
        default="models/2v2_selfplay_SAC_dense_seed822_20260513-112105_final.zip",
        help="Path to the yellow-side SAC checkpoint.",
    )
    parser.add_argument("--blue_path", default=None,
                        help="Blue-side checkpoint. Defaults to model_path "
                             "(self-play vs self).")
    parser.add_argument("--static_blue", action="store_true",
                        help="Blue stands still. Overrides --blue_path.")
    parser.add_argument("--blue_heuristic", default=None,
                        choices=["attacker", "roles"],
                        help="Hand-coded blue team. attacker: blue0 chases+"
                             "shoots, blue1 holds the goal-ball line. roles: "
                             "blue1 is a keeper. Takes precedence over "
                             "--blue_path/--static_blue.")
    parser.add_argument("--blue_kick_speed", type=float, default=None,
                        help="Heuristic shot speed (env default 6.0).")
    parser.add_argument("--pass_prob", type=float, default=0.0,
                        help="Staged pass-scenario probability at level 5 "
                             "(0 = pure chaos spawn). Ignored when "
                             "--difficulty is set (env rolls the curriculum "
                             "frame instead).")
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--n_eps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step_sleep", type=float, default=0.025,
                        help="Wallclock delay per decision (0.025 = real "
                             "time at repeat 1). 0 under --headless.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of the mean.")
    parser.add_argument("--headless", action="store_true",
                        help="No window, no sleep, summary with 95%% CIs.")
    # --- checkpoint I/O layout (sidecar/inferred unless given) ---
    parser.add_argument("--frame_stack", type=int, default=None)
    parser.add_argument("--action_repeat", type=int, default=None)
    parser.add_argument("--n_yellow", type=int, default=None,
                        help="Robots on the yellow side (sidecar, else 2).")
    parser.add_argument("--blue_action_repeat", type=int, default=None,
                        help="Decision rate of a frozen blue checkpoint "
                             "(physics steps per decision). Default: from "
                             "its sidecar, 1 if it has none.")
    # --- task rules (env defaults = the Gen-1..12 rules) ---
    parser.add_argument("--like_training", action="store_true",
                        help="Take pass_gate/restarts/shaping/dribble_rule/"
                             "foul_restart/goal_reward_solo from the "
                             "checkpoint's sidecar (control eval).")
    parser.add_argument("--pass_gate", choices=["loose", "strict"], default=None)
    parser.add_argument("--restarts", choices=["off", "on"], default=None)
    parser.add_argument("--shaping", default=None)
    parser.add_argument("--dribble_rule", choices=["soft", "strict"], default=None)
    parser.add_argument("--foul_restart", choices=["off", "on"], default=None)
    parser.add_argument("--goal_reward_solo", type=float, default=None)
    parser.add_argument("--difficulty", type=float, default=None,
                        help="Level-5 reverse-curriculum position (0..1); "
                             "None = the plain chaos spawn.")
    parser.add_argument("--defense_frame_prob", type=float, default=None)
    parser.add_argument("--defense_difficulty", type=float, default=None)
    parser.add_argument("--json_out", default=None,
                        help="Append the summary as one JSON line here.")
    args = parser.parse_args()

    # ---------------- checkpoint + sidecar ----------------
    yellow_model = SAC.load(args.model_path, device="cpu")
    model_obs_dim = int(yellow_model.policy.observation_space.shape[-1])
    side, side_path = _sidecar(args.model_path)

    frame_stack = args.frame_stack or side.get("frame_stack")
    n_yellow = int(args.n_yellow or side.get("n_yellow") or 2)
    frame_stack, single_obs_dim, role_index = _infer_layout(
        model_obs_dim, frame_stack, n_yellow
    )
    action_repeat = args.action_repeat or side.get("action_repeat") or 1

    # Rule flags: env defaults unless --like_training or an explicit flag.
    rule_keys = ("pass_gate", "restarts", "shaping", "dribble_rule",
                 "foul_restart", "goal_reward_solo")
    rules = {}
    for k in rule_keys:
        v = getattr(args, k)
        if v is None and args.like_training and k in side:
            v = side[k]
        if v is not None:
            rules[k] = v

    # ---------------- opponent ----------------
    if args.blue_heuristic:
        blue_path = None
        blue_desc = f"HEURISTIC ({args.blue_heuristic})"
    elif args.static_blue:
        blue_path = None
        blue_desc = "STATIC (no actions)"
    else:
        blue_path = args.blue_path or args.model_path
        blue_desc = blue_path

    env_kwargs = dict(
        reward_type="dense",
        render_mode=None if args.headless else "human",
        frozen_path=blue_path,
        blue_heuristic=args.blue_heuristic,
        pass_scenario_prob=args.pass_prob,
        role_index=role_index,
        n_yellow=n_yellow,
        action_repeat=int(action_repeat),
        # Pin the level: the env promotes itself to L5 after curriculum_window
        # episodes at >= 90 % success, which turned a 300-episode drill eval
        # into 200 drill episodes plus 100 of the open game.
        curriculum_start_level=args.level, curriculum_target_level=args.level,
        # Likewise freeze the reverse-curriculum difficulty at the value
        # given: an eval must not promote itself.
        difficulty_threshold=1.01,
        **rules,
    )
    for k in ("difficulty", "defense_frame_prob", "defense_difficulty",
              "blue_kick_speed"):
        v = getattr(args, k)
        if v is not None:
            env_kwargs[k] = v
    if args.blue_action_repeat is not None:
        env_kwargs["frozen_action_repeat"] = args.blue_action_repeat
    env = SSL2v2SelfPlayEnv(**env_kwargs)
    env.set_curriculum_level(args.level)
    frozen_blue = blue_path is not None
    if frozen_blue:
        # Resolve the opponent's decision rate now, for the header. A frozen
        # learned blue decides once per k physics steps (k = the rate it
        # trained at) and sees its own pass state, like yellow.
        env._maybe_load_frozen()
        blue_desc += (f"  [frozen policy, 1 decision per "
                      f"{env.frozen_action_repeat} physics steps, own pass state]")
    model = yellow_model
    stacker = FrameStacker(frame_stack, single_obs_dim) if frame_stack > 1 else None
    sleep = 0.0 if args.headless else args.step_sleep

    print(f"Yellow:  {args.model_path}")
    print(f"         obs_dim={model_obs_dim} = {single_obs_dim} x "
          f"frame_stack {frame_stack}, role_index={role_index}, "
          f"n_yellow={n_yellow}, action_repeat={action_repeat}"
          + (f"  [sidecar: {side_path}]" if side_path else "  [no sidecar: inferred]"))
    print(f"Blue:    {blue_desc}")
    task = {k: env_kwargs[k] for k in sorted(env_kwargs)
            if k not in ("reward_type", "render_mode", "frozen_path")}
    print(f"Task:    level {args.level}, " + ", ".join(f"{k}={v}" for k, v in task.items()))
    print(f"Running {args.n_eps} episodes, deterministic={not args.stochastic}, "
          f"seed={args.seed}, headless={args.headless}")
    print()

    # ---------------- rollout ----------------
    keys = ("is_success", "blue_goal", "passes", "passes_strict",
            "scored_after_pass", "scored_after_strict_pass",
            "ball_restarts", "robot_restarts",
            "blue_passes", "blue_passes_strict")
    rows = []
    cap = 1500  # decisions; the env truncates at 1000 physics steps anyway
    for ep in range(args.n_eps):
        obs, _ = env.reset(seed=args.seed + ep if ep == 0 else None)
        if stacker is not None:
            obs = stacker.reset(obs)
        total_r, steps, info = 0.0, 0, {}
        for t in range(cap):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, done, trunc, info = env.step(action)
            if stacker is not None:
                obs = stacker.step(obs)
            if not args.headless:
                env.render()
                if sleep:
                    time.sleep(sleep)
            steps = t + 1
            total_r += float(np.mean(reward))
            if done or trunc:
                break
        row = {k: float(info.get(k, 0.0)) for k in keys}
        row["ep_len"] = float(info.get("episode", {}).get("l", env.current_step))
        scen = info.get("scenario", "?")
        if scen == "pass":
            scen = f"pass/{info.get('pass_variant', '?')}"
        row["scenario"] = scen
        row["return"] = total_r
        row["blue_scored_after_strict_pass"] = float(
            row["blue_goal"] > 0 and row["blue_passes_strict"] > 0)
        rows.append(row)

        if row["is_success"]:
            outcome = "YELLOW GOAL"
        elif row["blue_goal"]:
            outcome = "BLUE GOAL"
        elif trunc:
            outcome = "timeout"
        else:
            outcome = "no goal"
        if not args.headless or (ep + 1) % 25 == 0 or ep < 5:
            print(f"Episode {ep+1:3d}/{args.n_eps}: {outcome:12s} | {scen:14s} "
                  f"| len={int(row['ep_len']):4d} | R={total_r:6.1f} "
                  f"| passes={int(row['passes'])} strict={int(row['passes_strict'])}"
                  + (f" | blue strict={int(row['blue_passes_strict'])}" if frozen_blue else ""))

    # ---------------- summary ----------------
    n = len(rows)
    col = lambda k: np.array([r[k] for r in rows])
    rates = {}
    for k in ("is_success", "blue_goal", "scored_after_pass",
              "scored_after_strict_pass", "blue_scored_after_strict_pass"):
        rates[k] = _wilson(int(col(k).sum()), n)
    means = {k: _mean_se(col(k)) for k in
             ("passes", "passes_strict", "ball_restarts", "robot_restarts",
              "ep_len", "return", "blue_passes", "blue_passes_strict")}

    print()
    print("=" * 72)
    print(f"Summary over {n} episodes (each counted ONCE)   [95 % Wilson CI]")
    print(f"  success_rate (yellow goal)     {rates['is_success'][0]:.3f}  "
          f"[{rates['is_success'][1]:.3f}, {rates['is_success'][2]:.3f}]")
    print(f"  blue_goal_rate                 {rates['blue_goal'][0]:.3f}  "
          f"[{rates['blue_goal'][1]:.3f}, {rates['blue_goal'][2]:.3f}]")
    print(f"  passes / episode               {means['passes'][0]:.3f} ± {means['passes'][1]:.3f}")
    print(f"  passes_strict / episode        {means['passes_strict'][0]:.3f} ± {means['passes_strict'][1]:.3f}")
    print(f"  scored_after_pass_rate         {rates['scored_after_pass'][0]:.3f}  "
          f"[{rates['scored_after_pass'][1]:.3f}, {rates['scored_after_pass'][2]:.3f}]")
    print(f"  scored_after_strict_pass_rate  {rates['scored_after_strict_pass'][0]:.3f}  "
          f"[{rates['scored_after_strict_pass'][1]:.3f}, {rates['scored_after_strict_pass'][2]:.3f}]")
    if frozen_blue:
        print(f"  BLUE passes_strict / episode   {means['blue_passes_strict'][0]:.3f} ± {means['blue_passes_strict'][1]:.3f}"
              f"   (loose {means['blue_passes'][0]:.3f})")
        print(f"  BLUE scored_after_strict_pass  {rates['blue_scored_after_strict_pass'][0]:.3f}  "
              f"[{rates['blue_scored_after_strict_pass'][1]:.3f}, {rates['blue_scored_after_strict_pass'][2]:.3f}]")
    print(f"  ball_oob / ep  {means['ball_restarts'][0]:.2f}   robot_oob / ep  "
          f"{means['robot_restarts'][0]:.2f}   ep_len {means['ep_len'][0]:.0f}   "
          f"return {means['return'][0]:.2f}")
    scen_names = sorted(set(r["scenario"] for r in rows))
    if len(scen_names) > 1:
        print("  by spawn type:")
        for s in scen_names:
            sub = [r for r in rows if r["scenario"] == s]
            m = len(sub)
            print(f"    {s:14s} n={m:3d}  success={np.mean([r['is_success'] for r in sub]):.2f}"
                  f"  passes={np.mean([r['passes'] for r in sub]):.2f}"
                  f"  strict={np.mean([r['passes_strict'] for r in sub]):.2f}"
                  f"  goal_after_strict={np.mean([r['scored_after_strict_pass'] for r in sub]):.2f}")

    if not frozen_blue:
        # Blue's passes are only tracked for a frozen learned opponent; a
        # heuristic's zeros would read as a measurement.
        rates = {k: v for k, v in rates.items() if not k.startswith("blue_scored")}
        means = {k: v for k, v in means.items() if not k.startswith("blue_")}
    summary = dict(
        model=args.model_path, n_eps=n, level=args.level, task=task,
        frame_stack=frame_stack, action_repeat=int(action_repeat),
        blue=blue_desc, seed=args.seed,
        **({"blue_action_repeat": int(env.frozen_action_repeat or 1)} if frozen_blue else {}),
        **{k: v[0] for k, v in rates.items()},
        **{f"{k}_ci": [v[1], v[2]] for k, v in rates.items()},
        **{f"{k}_per_ep": v[0] for k, v in means.items()},
    )
    print()
    print("JSON " + json.dumps(summary))
    if args.json_out:
        with open(args.json_out, "a") as f:
            f.write(json.dumps(summary) + "\n")
    env.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
