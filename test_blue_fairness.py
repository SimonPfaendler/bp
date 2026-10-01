"""Checks for a frozen LEARNED opponent on the blue side (cross-play).

    SDL_VIDEODRIVER=dummy python test_blue_fairness.py [yellow.zip [blue.zip]]

1. shadow    _TeamPassTracker repeats yellow's pass bookkeeping exactly:
             a shadow instance on yellow is compared with the env's own
             variables on every physics step.
2. rate      blue's policy is asked once per k physics steps (k from the
             checkpoint's sidecar), not on every step.
3. mirror    blue's observation of a state equals yellow's observation of
             the mirrored, team-swapped state, in all dims.
4. velocity  the simulator reports robot velocities in the WORLD frame for
             both teams (the mirror negates v_x only, which needs that).

Needs a checkpoint that actually passes; defaults to the seed-101 model.
"""
import math
import sys

import numpy as np
from rsoccer_gym.Entities import Ball, Frame, Robot
from stable_baselines3 import SAC

import masac_policy  # noqa: F401
from ssl_rl_2v2_selfplay import SSL2v2SelfPlayEnv, _TeamPassTracker

YELLOW = sys.argv[1] if len(sys.argv) > 1 else "models/c15_s101_L5-6_final.zip"
BLUE = sys.argv[2] if len(sys.argv) > 2 else YELLOW
TRAIN_RULES = dict(pass_gate="strict", restarts="on", shaping="team_def",
                   dribble_rule="strict", foul_restart="on", goal_reward_solo=2.0,
                   difficulty=1.0)
model = SAC.load(YELLOW, device="cpu")


def make(blue, rules):
    kw = dict(reward_type="dense", render_mode=None, role_index=True,
              action_repeat=4, pass_scenario_prob=0.0, **rules)
    if blue == "heuristic":
        kw.update(blue_heuristic="roles", blue_kick_speed=4.5)
    else:
        kw.update(frozen_path=blue)
    env = SSL2v2SelfPlayEnv(**kw)
    env.set_curriculum_level(5)
    return env


def rollout(env, n_eps, on_physics_step=None, seed=7):
    if on_physics_step is not None:
        orig = env._step_once
        def wrapped(a):
            out = orig(a)
            on_physics_step(env, out)
            return out
        env._step_once = wrapped
    stats = dict(steps=0, yellow_goals=0, blue_goals=0, passes=0, strict=0,
                 blue_passes=0, blue_strict=0)
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        for _ in range(400):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, tr, info = env.step(a)
            if d or tr:
                break
        stats["steps"] += env.current_step
        stats["yellow_goals"] += int(info.get("is_success", 0))
        stats["blue_goals"] += int(info.get("blue_goal", 0))
        stats["passes"] += int(info.get("passes", 0))
        stats["strict"] += int(info.get("passes_strict", 0))
        stats["blue_passes"] += int(info.get("blue_passes", 0))
        stats["blue_strict"] += int(info.get("blue_passes_strict", 0))
    return stats


# ---------------------------------------------------------------- 1. shadow
def test_shadow():
    for blue, rules, name in (("heuristic", {}, "heuristic, old rules"),
                              ("heuristic", TRAIN_RULES, "heuristic, training rules"),
                              (BLUE, {}, "frozen, old rules"),
                              (BLUE, TRAIN_RULES, "frozen, training rules")):
        env = make(blue, rules)
        env._pass_shadow_y = _TeamPassTracker()
        n = [0]
        def check(env, out):
            sh = env._pass_shadow_y
            got = (sh.last_carrier, sh.opp_touched, sh.passes, sh.passes_strict,
                   sh._strict_pending, sh._release)
            want = (env.last_yellow_carrier, env.blue_touched_since_yellow,
                    env.passes_in_episode, env.passes_strict_in_episode,
                    env._strict_pending, env._release)
            assert got == want, f"step {env.current_step}: shadow {got} != env {want}"
            n[0] += 1
        s = rollout(env, 25, check)
        env.close()
        print(f"  shadow ok   {name:27s} {n[0]:6d} physics steps compared, "
              f"{s['passes']} loose / {s['strict']} strict yellow passes"
              + (f", blue {s['blue_passes']} / {s['blue_strict']}" if blue != "heuristic" else ""))
        assert s["strict"] > 0, "no strict pass in the sample — the check saw nothing"


# ------------------------------------------------------------------ 2. rate
def test_rate():
    env = make(BLUE, TRAIN_RULES)
    env.reset(seed=3)
    k = env.frozen_action_repeat
    calls = [0]
    orig = env._frozen_policy_action
    def counted():
        calls[0] += 1
        return orig()
    env._frozen_policy_action = counted
    decisions = 0
    obs, _ = env.reset(seed=3)
    for _ in range(120):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, d, tr, info = env.step(a)
        decisions += 1
        if d or tr:
            break
    physics = env.current_step
    env.close()
    assert k == 4, f"expected the sidecar to say action_repeat 4, got {k}"
    assert calls[0] == math.ceil(physics / k) == decisions, (calls[0], physics, decisions)
    print(f"  rate ok     blue asked {calls[0]}x in {physics} physics steps "
          f"(k={k}), yellow decided {decisions}x")


# ---------------------------------------------------------------- 3. mirror
def _mirror_robot(r, yellow, idx):
    return Robot(yellow=yellow, id=idx, x=-r.x, y=r.y, theta=180.0 - r.theta,
                 v_x=-r.v_x, v_y=r.v_y, v_theta=-r.v_theta, infrared=r.infrared)


def _mirrored_yellow_obs(env):
    """Yellow's observation of the mirrored, team-swapped state."""
    f = env.frame
    keep = dict(frame=env.frame,
                dy=(env.is_dribbling_y, env.dribble_start_pos_y, env.must_release_y),
                db=(env.is_dribbling_b, env.dribble_start_pos_b, env.must_release_b),
                ban=env.dribble_ban_y,
                y=(env.last_yellow_carrier, env.blue_touched_since_yellow,
                   env.passes_in_episode, env.passes_strict_in_episode))
    m = Frame()
    m.ball = Ball(x=-f.ball.x, y=f.ball.y, v_x=-f.ball.v_x, v_y=f.ball.v_y)
    for i in range(2):
        m.robots_yellow[i] = _mirror_robot(f.robots_blue[i], True, i)
        m.robots_blue[i] = _mirror_robot(f.robots_yellow[i], False, i)
    mir = lambda starts: [None if p is None else np.array([-p[0], p[1]]) for p in starts]
    bp = env._blue_pass
    try:
        env.frame = m
        env.is_dribbling_y, env.dribble_start_pos_y, env.must_release_y = (
            list(keep["db"][0]), mir(keep["db"][1]), list(keep["db"][2]))
        env.is_dribbling_b, env.dribble_start_pos_b, env.must_release_b = (
            list(keep["dy"][0]), mir(keep["dy"][1]), list(keep["dy"][2]))
        env.dribble_ban_y = [None, None]          # blue has no ban (known rule gap)
        env.last_yellow_carrier, env.blue_touched_since_yellow = bp.last_carrier, bp.opp_touched
        env.passes_in_episode, env.passes_strict_in_episode = bp.passes, bp.passes_strict
        return env._stacked_obs_yellow()
    finally:
        env.frame = keep["frame"]
        env.is_dribbling_y, env.dribble_start_pos_y, env.must_release_y = keep["dy"]
        env.is_dribbling_b, env.dribble_start_pos_b, env.must_release_b = keep["db"]
        env.dribble_ban_y = keep["ban"]
        (env.last_yellow_carrier, env.blue_touched_since_yellow,
         env.passes_in_episode, env.passes_strict_in_episode) = keep["y"]


def test_mirror():
    for rules, name in (({}, "old rules"), (TRAIN_RULES, "training rules")):
        env = make(BLUE, rules)
        worst = np.zeros(env.single_obs_dim)
        n = [0]
        def check(env, out):
            if env.current_step % 3:
                return
            blue = env._stacked_obs_blue()
            yel = _mirrored_yellow_obs(env)
            np.maximum(worst, np.abs(blue - yel).max(axis=0), out=worst)
            n[0] += 1
        rollout(env, 12, check)
        env.close()
        bad = np.nonzero(worst > 1e-5)[0]
        assert bad.size == 0, (f"{name}: slots {bad.tolist()} differ between blue's obs and "
                               f"yellow's obs of the mirrored state, max {worst[bad].round(4).tolist()}")
        print(f"  mirror ok   {name:15s} {n[0]} states, all {env.single_obs_dim} dims, "
              f"max deviation {worst.max():.1e}")


# -------------------------------------------------------------- 4. velocity
def test_velocity():
    env = make(BLUE, {})
    dt = float(getattr(env, "time_step", 0.025))
    err_world, err_local, prev = [], [], [None]
    def check(env, out):
        cur = env.frame
        if prev[0] is not None and env.current_step > 1:
            for team in ("robots_yellow", "robots_blue"):
                for i in range(2):
                    a, b = getattr(prev[0], team)[i], getattr(cur, team)[i]
                    fd = np.array([(b.x - a.x) / dt, (b.y - a.y) / dt])
                    if np.linalg.norm(fd) < 0.5:
                        continue
                    rep = np.array([0.5 * (a.v_x + b.v_x), 0.5 * (a.v_y + b.v_y)])
                    th = math.radians(b.theta)
                    rot = np.array([rep[0] * math.cos(th) - rep[1] * math.sin(th),
                                    rep[0] * math.sin(th) + rep[1] * math.cos(th)])
                    err_world.append(np.linalg.norm(fd - rep))
                    err_local.append(np.linalg.norm(fd - rot))
        prev[0] = cur
        if out[2] or out[3]:
            prev[0] = None
    rollout(env, 6, check)
    env.close()
    ew, el = float(np.median(err_world)), float(np.median(err_local))
    assert ew < 0.2 and ew < el, (ew, el)
    print(f"  velocity ok reported robot velocity is world-frame: median error "
          f"{ew:.3f} m/s (if it were robot-frame: {el:.3f}), {len(err_world)} samples")


if __name__ == "__main__":
    print(f"yellow = {YELLOW}\nblue   = {BLUE}")
    test_rate()
    test_shadow()
    test_mirror()
    test_velocity()
    print("all checks passed")
