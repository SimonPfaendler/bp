import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat", load_buffer="auto",
    start_level=None, blue_heuristic=None, goal_reward_solo=None,
    target_action_std=None, noise_repeat_s=None, target_level=None,
    pass_gate=None, dribble_rule=None, shaping=None, restarts=None,
    difficulty=None, difficulty_threshold=None, difficulty_step=None,
    difficulty_window=None, max_minutes=None, role_index=False,
    defense_frame_prob=None, foul_restart=None, defense_difficulty=None,
    critic_warmup_steps=None, frame_stack=None, action_repeat=None,
    blue_kick_speed=None, run_name=None, drop_init_buffer=False,
    opponent_pool=None, pool_frac=None, pass_bonus=None, n_yellow=None,
):
    # difficulty="inherit": continue at the difficulty the init checkpoint's
    # segment ENDED at. Resolved here, inside the job, because in a chain
    # the previous segment's sidecar does not exist yet at submit time.
    if difficulty == "inherit":
        import json
        base = init_path[:-4] if init_path and init_path.endswith(".zip") else init_path
        side = {}
        if base and os.path.exists(f"{base}_replay_buffer.json"):
            side = json.load(open(f"{base}_replay_buffer.json"))
        difficulty = side.get("difficulty_end")
        if difficulty is None:
            difficulty = side.get("difficulty_start")
        if difficulty is None:
            raise RuntimeError(
                f"DIFF=inherit: no difficulty recorded next to {init_path}")
        print(f"DIFF=inherit -> {difficulty} (from {base}_replay_buffer.json)")
    init_flag = f"--init_path {init_path} " if init_path else ""
    frozen_flag = f"--frozen_path {frozen_path} " if frozen_path else ""
    demo_flag = f"--demo_dir {demo_dir} " if demo_dir else ""
    # -u: unbuffered stdout — without it the first log tables sit in the
    # 4KB block buffer for minutes and the slurm .out looks dead.
    cmd = (
        f"python -u train_2v2_selfplay.py "
        f"{frozen_flag}{init_flag}{demo_flag}"
        f"--reward_type {reward_type} --seed {seed} "
        f"--n_pairs {n_pairs} --total_steps {total_steps} "
        f"--algo {algo} --net {net} "
        f"--pass_scenario_prob {pass_scenario_prob} "
        f"--load_buffer {load_buffer}"
    )
    if pass_scenario_prob_start is not None:
        cmd += f" --pass_scenario_prob_start {pass_scenario_prob_start}"
    if start_level is not None:
        cmd += f" --start_level {start_level}"
    if blue_heuristic is not None:
        cmd += f" --blue_heuristic {blue_heuristic}"
    if goal_reward_solo is not None:
        cmd += f" --goal_reward_solo {goal_reward_solo}"
    if target_action_std is not None:
        cmd += f" --target_action_std {target_action_std}"
    if noise_repeat_s is not None:
        cmd += f" --noise_repeat_s {noise_repeat_s}"
    if target_level is not None:
        cmd += f" --target_level {target_level}"
    if pass_gate is not None:
        cmd += f" --pass_gate {pass_gate}"
    if dribble_rule is not None:
        cmd += f" --dribble_rule {dribble_rule}"
    if shaping is not None:
        cmd += f" --shaping {shaping}"
    if restarts is not None:
        cmd += f" --restarts {restarts}"
    if difficulty is not None:
        cmd += f" --difficulty {difficulty}"
    if difficulty_threshold is not None:
        cmd += f" --difficulty_threshold {difficulty_threshold}"
    if difficulty_step is not None:
        cmd += f" --difficulty_step {difficulty_step}"
    if difficulty_window is not None:
        cmd += f" --difficulty_window {difficulty_window}"
    if role_index:
        cmd += " --role_index"
    if defense_frame_prob is not None:
        cmd += f" --defense_frame_prob {defense_frame_prob}"
    if foul_restart is not None:
        cmd += f" --foul_restart {foul_restart}"
    if defense_difficulty is not None:
        cmd += f" --defense_difficulty {defense_difficulty}"
    if critic_warmup_steps is not None:
        cmd += f" --critic_warmup_steps {critic_warmup_steps}"
    if frame_stack is not None:
        cmd += f" --frame_stack {frame_stack}"
    if action_repeat is not None:
        cmd += f" --action_repeat {action_repeat}"
    if blue_kick_speed is not None:
        cmd += f" --blue_kick_speed {blue_kick_speed}"
    if max_minutes is not None:
        cmd += f" --max_minutes {max_minutes}"
    if pass_bonus is not None:
        cmd += f" --pass_bonus {pass_bonus}"
    if n_yellow is not None:
        cmd += f" --n_yellow {n_yellow}"
    if opponent_pool:
        cmd += f" --opponent_pool {opponent_pool}"
        if pool_frac is not None:
            cmd += f" --pool_frac {pool_frac}"
    if run_name:
        cmd = f"RUN_NAME={run_name} " + cmd
    import time
    t_start = time.time()
    rc = os.system(cmd)
    # os.system swallows the exit code; without a check a crashed segment
    # looks like success to slurm and the next segment of a chain (afterok)
    # starts on a checkpoint that was never written. With a fixed run name
    # the checkpoint itself is the criterion — the time-bounded seed-822
    # segments all saved correctly yet left a TypeError in stderr, so the
    # exit status alone would be the wrong judge.
    if run_name:
        final = f"models/{run_name}_final.zip"
        fresh = os.path.exists(final) and os.path.getmtime(final) >= t_start
        if not fresh:
            raise RuntimeError(
                f"segment {run_name}: no fresh {final} (exit status {rc})")
        if rc != 0:
            print(f"WARNING: exit status {rc}, but {final} was written — "
                  f"treating the segment as complete")
    elif rc != 0:
        raise RuntimeError(f"training exited with status {rc}")
    if drop_init_buffer and init_path:
        # Chain housekeeping: this segment finished and saved its own
        # buffer, so the predecessor's snapshots (0.5-1 GB each) are dead
        # weight. Policies and sidecars are kept.
        import glob
        base = init_path[:-4] if init_path.endswith(".zip") else init_path
        prefix = base[:-len("_final")] if base.endswith("_final") else base
        for p in glob.glob(f"{prefix}_final_replay_buffer.pkl") + \
                glob.glob(f"{prefix}_replay_buffer_*_steps.pkl"):
            try:
                os.remove(p)
                print(f"removed predecessor buffer {p}")
            except OSError:
                pass


def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    # TIME=HH:MM:SS (default 00:30:00) sets the slurm limit; PARTITION
    # defaults to the 30-min dev queue for <= 30 min and to gpu_h100 beyond
    # (check `sinfo -s` if the long-queue name differs). TIME_MIN (the
    # in-run budget) defaults to TIME minus 6 min: ~3 min of setup before
    # training starts plus the final save of a ~0.5-1 GB replay buffer.
    slurm_time = os.environ.get("TIME", "00:30:00")
    h, m, s = (int(x) for x in slurm_time.split(":"))
    slot_min = h * 60 + m + s / 60.0
    partition = os.environ.get(
        "PARTITION", "dev_gpu_h100" if slot_min <= 30 else "gpu_h100"
    )
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name=os.environ.get("RUN_NAME") or "sp2v2",
        slurm_time=slurm_time,
        slurm_partition=partition,
        slurm_cpus_per_task=24,
        slurm_mem="193300mb",
        slurm_additional_parameters={
            "gres": "gpu:1",
            # AFTER=<jobid>: start only once that job finished successfully
            # (chain stages; see submit_curriculum_chain.py).
            **({"dependency": f"afterok:{os.environ['AFTER']}"}
               if os.environ.get("AFTER") else {}),
        },
    )
    # NOTE: MASAC (JointSubprocPairVecEnv, num_envs=n_pairs) needs ~2x the
    # real simulation steps of independent SAC (SubprocPairVecEnv,
    # num_envs=2*n_pairs) to reach the SAME --total_steps counter, because
    # SB3 counts total_timesteps += num_envs per step_wait(). At the observed
    # ~42 iterations/s, 3.3M steps needs ~55 real minutes for MASAC — the old
    # 30-min dev_gpu_h100 slot cut it off at ~52% of its own pass_scenario
    # schedule, making any SAC-vs-MASAC comparison at that checkpoint an
    # apples-to-oranges mid-schedule-vs-finished comparison.

    # GENERATION 15 — independent SAC on the pass drills, from scratch.
    #
    # The strict pass counter (kick-speed release, aimed at the mate, held by
    # the receiver) showed that no SAC run ever passed: the best checkpoint,
    # 20260811-141655, replayed in the env of its time scores .245 overall and
    # .55 in the staged pass scenario with 3 strict passes and 0 goals after
    # a strict pass in 200 episodes — a soloist. The only thing that produced
    # real passes so far is the drill curriculum (MAPPO: 0 -> .57 strict
    # passes/episode on L2 within 8M steps), and that is a property of the
    # task, not of the algorithm. This generation puts SAC on the same task.
    #
    #   LEVEL=2 (default)  pass drill, from scratch. Terminal = strict pass.
    #   LEVEL=3            pass + finish. Chain it from the L2 result:
    #                      LEVEL=3 INIT_PATH=models/<L2 run>_final.zip
    #   LEVEL=5            CONTROL: the unmodified full game (solo goal pays
    #                      in full, loose +3 on, ball-chasing blues, 35 %
    #                      staged pass scenarios) from the L3 result. L3
    #                      reached .99 strict passes / .93 goals after a
    #                      pass; this run asks whether that survives once
    #                      solo play is possible again, or erodes like every
    #                      L5 run before it. Watch passes_strict_per_episode.
    #   LEVEL=4            the L3 drill with the blue heuristic active — the
    #                      pass has to come off under pressure. From L3.
    #   LEVEL=5 PASS_GATE=strict SOLO=<x>
    #                      the full game with the payoff gated on the STRICT
    #                      counter: goal after a strict pass = 10, solo goal =
    #                      SOLO (default 10 = symmetric), +3 for a strict
    #                      pass instead of a loose one. SOLO=2 is the honest
    #                      rerun of the asymmetric-payoff test, which used
    #                      to pay "goal after a fumble".
    #   DRIBBLE=strict     the excessive-dribbling rule with teeth: after 1 m
    #                      the ball must go and the same robot may not touch
    #                      it until someone else has (foul = episode over).
    #                      Solo dribbling from midfield stops being possible;
    #                      the reason to pass comes from the rules of the
    #                      game, not from the reward.
    #   SHAPING=team       approach term only for the closer yellow and never
    #                      negative (v1 pulled both to the ball and charged
    #                      every kick), off-ball progress toward the goal,
    #                      no anti-passivity charge. Full game only.
    #   RESTARTS=on        ball/robot out of bounds restarts play from rest
    #                      instead of ending the episode. Closes the two exits
    #                      the L4 checkpoint found on L5: clearing the ball
    #                      out (A') and driving the off-ball robot off the
    #                      field (Step 1: 82 % of episodes). Only goals and
    #                      the clock terminate.
    #   Step-1 experiment: LEVEL=5 PASS_GATE=strict SOLO=2 DRIBBLE=strict
    #                      SHAPING=team INIT_PATH=<L4 checkpoint>
    #   Step-1b:           the same plus RESTARTS=on
    #   DIFF=0             reverse curriculum on L5: the spawn interpolates
    #                      between the L4 drill (0) and chaos (1), staged —
    #                      thirds of d move the blues, then the mate, then
    #                      ball + carrier; each env promotes itself by
    #                      DIFF_STEP (0.05) once its rolling rate of goals
    #                      after a strict pass over DIFF_WIN (50) episodes
    #                      clears DIFF_THR (0.6). The old
    #                      staged pass scenarios are off while a difficulty
    #                      is set. When chaining, set DIFF to the value the
    #                      previous run reached (curriculum/difficulty).
    #   Step 2:            LEVEL=5 DIFF=0 PASS_GATE=strict SOLO=2 DRIBBLE=strict
    #                      SHAPING=team RESTARTS=on INIT_PATH=<L4 checkpoint>
    #
    # start_level == target_level pins the run to the drill; without it both
    # the callback and the env promote to L5 at 90 % success, which L3 can
    # reach. Blues are parked and passive on both drills whatever
    # blue_heuristic says; it is set for parity with the L5 runs to come.
    #
    # No warm start from the L5 checkpoints: they are 52-dim, lose ~60 % of
    # their strength under today's ball-speed normalisation (.245 -> .095),
    # and bring the solo habit along. load_buffer="off": an L2 buffer holds
    # L2 rewards and would poison an L3 critic.
    #
    # Compare against the MAPPO runs on rollout/passes_strict_per_episode
    # and selfplay/live_success_rate (success == strict pass on L2,
    # == goal after strict pass on L3).
    init_path = os.environ.get("INIT_PATH")  # None = from scratch
    # Chaining: every unset env var below falls back to the value recorded
    # in the init checkpoint's sidecar JSON (train_2v2_selfplay.py writes it
    # next to every _final.zip). A chained segment therefore keeps the
    # checkpoint's I/O layout (STACK, REPEAT, ROLE) and task flags unless
    # told otherwise; the warm-start assert in the trainer catches the rest.
    # DIFF falls back to the difficulty the previous segment ENDED at.
    # INHERIT=0 switches the fallback off (chain submitter: every flag of
    # a stage is stated by the protocol, nothing may leak in).
    inherited = {}
    if init_path and os.environ.get("INHERIT", "1") != "0":
        base = init_path[:-4] if init_path.endswith(".zip") else init_path
        sidecar_path = f"{base}_replay_buffer.json"
        if os.path.exists(sidecar_path):
            import json
            side = json.load(open(sidecar_path))
            mapping = {  # env var -> sidecar key(s), first present wins
                "PASS_GATE": ("pass_gate",), "DRIBBLE": ("dribble_rule",),
                "SHAPING": ("shaping",), "RESTARTS": ("restarts",),
                "FOUL": ("foul_restart",), "SOLO": ("goal_reward_solo",),
                "STACK": ("frame_stack",), "REPEAT": ("action_repeat",),
                "BLUE_KICK": ("blue_kick_speed",), "ROLE": ("role_index",),
                "DEF_PROB": ("defense_frame_prob",),
                "DEF_DIFF": ("defense_difficulty",), "BLUE": ("blue_heuristic",),
                "DIFF": ("difficulty_end", "difficulty_start"),
                "DIFF_THR": ("difficulty_threshold",),
                "DIFF_STEP": ("difficulty_step",), "DIFF_WIN": ("difficulty_window",),
                "LEVEL": ("curriculum_start_level",),
                "N_YELLOW": ("n_yellow",),
            }
            for var, keys in mapping.items():
                if os.environ.get(var) not in (None, ""):
                    continue
                for k in keys:
                    if side.get(k) is not None:
                        v = side[k]
                        os.environ[var] = str(int(v)) if isinstance(v, bool) else str(v)
                        inherited[var] = os.environ[var]
                        break
            print(f"Inherited from {sidecar_path}: {inherited or 'nothing (all set)'}")
        else:
            print(f"WARNING: no sidecar at {sidecar_path}; STACK/REPEAT/ROLE and "
                  f"the reward flags must be set by hand to match the checkpoint")

    level = int(os.environ.get("LEVEL", "2"))
    assert level in (2, 3, 4, 5), level
    pass_gate = os.environ.get("PASS_GATE")            # None -> loose
    solo = os.environ.get("SOLO")                      # None -> symmetric
    goal_reward_solo = float(solo) if solo is not None else None
    dribble_rule = os.environ.get("DRIBBLE")          # None -> soft
    shaping = os.environ.get("SHAPING")               # None -> v1
    restarts = os.environ.get("RESTARTS")             # None -> off
    diff = os.environ.get("DIFF")                     # None -> curriculum off
    difficulty = (diff if diff == "inherit"
                  else float(diff) if diff is not None else None)
    diff_thr = os.environ.get("DIFF_THR"); diff_thr = float(diff_thr) if diff_thr else None
    diff_step = os.environ.get("DIFF_STEP"); diff_step = float(diff_step) if diff_step else None
    diff_win = os.environ.get("DIFF_WIN"); diff_win = int(diff_win) if diff_win else None
    pass_scenario_prob = 0.35 if level == 5 else 0.0
    reward_type = "dense"
    n_pairs = 24
    seed = int(os.environ.get("SEED", "822"))
    run_name = os.environ.get("RUN_NAME") or None
    # TOTAL_STEPS caps the run; TIME_MIN stops it gracefully after that many
    # minutes and saves (3M steps took 20-24 min depending on the node, so
    # a fixed step count either wastes the slot or gets killed before the
    # final save). On the 30-min dev partition: TOTAL_STEPS=6000000 TIME_MIN=24.
    # (27 was too tight: the 09-26 segment was cut at 22.7 min of TRAINING,
    # after ~3 min of setup, and the final save writes a ~0.5 GB buffer.)
    # The run is time-bound: TIME_MIN defaults to the slot minus 6 min and
    # TOTAL_STEPS to a cap no slot can reach (~92k steps/min for SAC on an
    # H100 -> 3 h ~ 16M). Set TOTAL_STEPS explicitly for a step-bound run.
    time_min = os.environ.get("TIME_MIN")
    max_minutes = float(time_min) if time_min else max(5.0, slot_min - 6.0)
    total_steps = int(os.environ.get("TOTAL_STEPS", "100000000"))
    algo = "sac"
    # BLUE=roles: blue 0 hunts, blue 1 keeps goal and never joins the chase
    # (the "attacker" pair double-chased loose balls). ROLE=1: one-hot agent
    # id appended to the obs so the two yellows can stop doing the same
    # thing; a 56-dim INIT_PATH is widened with zero columns.
    blue_heuristic = os.environ.get("BLUE", "attacker")
    role_index = os.environ.get("ROLE", "0") not in ("", "0", "false", "False")
    # DEF_PROB=0.25: a quarter of the episodes spawn as a blue attack on the
    # yellow goal (the curriculum frame never puts a yellow behind the
    # ball; measured 60-70 % blue goals from open play after a turnover).
    # FOUL=on: the dribbling foul is a blue free kick instead of an -2 exit.
    def_prob = os.environ.get("DEF_PROB"); def_prob = float(def_prob) if def_prob else None
    foul_restart = os.environ.get("FOUL")                # None -> off
    # DEF_DIFF=0: easy defensive frame (hunter 2-3.5 m from the ball,
    # defender within 0.3 m of the shot line); 1 = the original frame.
    # SHAPING=team_def: ball->goal term may go negative under restarts.
    def_diff = os.environ.get("DEF_DIFF"); def_diff = float(def_diff) if def_diff else None
    # CRIT_WARM=250000: actor and alpha frozen for the first 250k env steps
    # of the segment, the critic re-fits first (every chained segment so far
    # dipped hard in its first 500k steps). STACK=2: the last two obs side
    # by side (newest last); the init checkpoint is widened function-identical.
    crit_warm = os.environ.get("CRIT_WARM"); crit_warm = int(crit_warm) if crit_warm else None
    stack = os.environ.get("STACK"); stack = int(stack) if stack else None
    # REPEAT=4: one decision per 4 physics steps (10 Hz). BLUE_KICK=4.5: the
    # blue heuristic shoots at 4.5 m/s instead of 6.0. BUF=auto: load the
    # replay buffer saved next to INIT_PATH (only if its sidecar says the
    # same reward flags and decision rate; a warm buffer skips the 10k
    # random-action steps and the critic keeps its data).
    repeat = os.environ.get("REPEAT"); repeat = int(repeat) if repeat else None
    blue_kick = os.environ.get("BLUE_KICK"); blue_kick = float(blue_kick) if blue_kick else None
    # N_YELLOW=3: three learning robots against the two-robot heuristic
    # (new obs layout, so the chain must start from scratch).
    n_yellow = os.environ.get("N_YELLOW"); n_yellow = int(n_yellow) if n_yellow else None
    # Default "auto": the 09-26 segment ran with BUF unset ("off") and so
    # started with an empty buffer + 10k random actions + critic warm-up on
    # top of a warm policy. "auto" is a no-op without INIT_PATH and refuses
    # a buffer whose sidecar disagrees on reward flags or decision rate, so
    # it cannot silently poison a run. BUF=off for a deliberate cold buffer.
    load_buffer = os.environ.get("BUF", "auto")

    job = executor.submit(
        run_experiment, reward_type, seed, n_pairs,
        None, init_path, total_steps, algo,
        pass_scenario_prob, None,
        None, "flat", load_buffer,
        level, blue_heuristic, goal_reward_solo, None, None, level,
        pass_gate, dribble_rule, shaping, restarts,
        difficulty, diff_thr, diff_step, diff_win, max_minutes, role_index,
        def_prob, foul_restart, def_diff, crit_warm, stack, repeat, blue_kick,
        run_name, os.environ.get("DROP_INIT_BUF") == "1",
        # POOL=a.zip,b.zip: frozen opponents for POOL_FRAC (0.5) of the envs.
        os.environ.get("POOL") or None, os.environ.get("POOL_FRAC") or None,
        # PASS_BONUS=0 with SOLO=10: no payment for passing, only goals.
        os.environ.get("PASS_BONUS") or None,
        n_yellow,
    )
    print(f"Submitted Gen-15 SAC L{level}: job {job.job_id} on {partition} "
          f"for {slurm_time} seed={seed} run_name={run_name or 'auto'} "
          f"after={os.environ.get('AFTER') or '-'} "
          f"[init={init_path or 'scratch'}, start=target={level}, "
          f"pass_gate={pass_gate or 'loose'}, solo={goal_reward_solo}, "
          f"dribble={dribble_rule or 'soft'}, shaping={shaping or 'v1'}, "
          f"restarts={restarts or 'off'}, difficulty={difficulty}, "
          f"blue={blue_heuristic}, role_index={role_index}, "
          f"def_prob={def_prob}, foul_restart={foul_restart or 'off'}, def_diff={def_diff}, "
          f"critic_warmup={crit_warm}, frame_stack={stack}, "
          f"action_repeat={repeat}, blue_kick={blue_kick}, load_buffer={load_buffer}, "
          f"n_yellow={n_yellow or 2}, steps={total_steps}, time_min={max_minutes}]")
    return job.job_id


if __name__ == "__main__":
    main()
