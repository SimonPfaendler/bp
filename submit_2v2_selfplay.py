import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat", load_buffer="auto",
    start_level=None, blue_heuristic=None, goal_reward_solo=None,
):
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
    os.system(cmd)


def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="sp2v2",
        slurm_time="00:30:00",
        slurm_partition="dev_gpu_h100",
        slurm_cpus_per_task=24,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )
    # NOTE: MASAC (JointSubprocPairVecEnv, num_envs=n_pairs) needs ~2x the
    # real simulation steps of independent SAC (SubprocPairVecEnv,
    # num_envs=2*n_pairs) to reach the SAME --total_steps counter, because
    # SB3 counts total_timesteps += num_envs per step_wait(). At the observed
    # ~42 iterations/s, 3.3M steps needs ~55 real minutes for MASAC — the old
    # 30-min dev_gpu_h100 slot cut it off at ~52% of its own pass_scenario
    # schedule, making any SAC-vs-MASAC comparison at that checkpoint an
    # apples-to-oranges mid-schedule-vs-finished comparison.

    # GENERATION 10 — training against the HAND-CODED blue team.
    # Blue 0 chases and shoots, blue 1 holds the goal-ball line
    # (blue_attacker_heuristic_2v2 / blue_defender_heuristic_2v2).
    #
    # Two reasons this differs from every previous generation:
    #  1) Fixed yardstick. Every frozen opponent so far was a different
    #     checkpoint, so success rates were never comparable across
    #     generations. Against a fixed heuristic they are, exactly like the
    #     1v1 project's "72% vs heuristic baseline".
    #  2) A defender that HOLDS POSITION. In self-play both blues chase the
    #     ball (mirrored solo equilibrium), leaving the goal open, which is
    #     why the solo route stayed profitable. A dedicated defender blocks
    #     the direct shot lane in ordinary chaos episodes, not just in the
    #     staged pass scenarios. That is the structural "make solo
    #     unattractive" lever, for free from the opponent design.
    #
    # Caveat for the thesis (see BP 5.4): a static heuristic caps the
    # strategic ceiling and invites exploitation. This is a CONDITION and an
    # evaluator, not a replacement for self-play.
    #
    # SAC (not MASAC): the stronger and better-understood arm, and it makes
    # the comparison against the earlier SAC-vs-frozen curves meaningful.
    # TWO ARMS, because warm-vs-scratch is itself the test of the attractor
    # hypothesis (Gen 7 erosion + Gen 8 solo-solves-pass-scenarios):
    #   10a warm    — can an existing solo player be re-educated by a
    #                 permanently blocked shot lane? Answer in one chunk.
    #   10b scratch — does cooperation emerge when the policy learns from
    #                 the start in a world where solo does not reliably pay?
    #                 Needs ~3 chunks (5a scratch took 3.3M for success .30),
    #                 so it is the long pole — start it now, chain it via
    #                 init = newest _final with start_level=None.
    # Everything else identical between arms, so the difference is the init.
    # 140727 = Gen-7 chunk 2: the strongest player overall (chaos success
    # 0.40, best ever) and simultaneously the most pass-eroded one
    # (scenario_pass passes 0.16, scored_after_pass 0). That makes it the
    # sharpest possible warm test: if a permanently blocked shot lane can
    # re-educate THIS policy, the effect is real.
    # Chunk 2 of the warm arm. Chunk 1 (20260809-220118) was still improving
    # at cutoff (success .12 -> .14 inside the log window, Q healthy at
    # q_mean 2.0), so this continues it rather than restarting.
    # load_buffer="auto" now: chunk 1's buffer was collected against THIS
    # heuristic, so it is valid (chunk 1 had to drop the old one, which came
    # from a different opponent). That also enables learning_starts=0.
    warm_chain = "models/2v2_selfplay_SAC_vsheur_dense_seed822_20260809-220118_final.zip"
    champion_solo = "models/2v2_selfplay_SAC_dense_seed822_20260801-140727_final.zip"
    # 10b restart: the first attempt could not promote because the heuristic
    # blues were active at level 1, which is designed around PASSIVE blues
    # (success 0.0 at blue_goal_rate 0.41 after 2.9M). The env now engages
    # the heuristic only from level 2 up. The Q blow-up (critic_loss 2.7e3)
    # was downstream of that: with no grounded success signal the fresh
    # critic had nothing to anchor against while 25% demo states pulled it
    # off-distribution. Scratch MASAC on level 1 with passive blues was
    # stable, so the level fix should be enough without target clipping.
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 3_000_000
    algo = "sac"
    pass_scenario_prob = 0.35         # fixed — no anneal (erosion lesson)
    pass_scenario_prob_start = None
    blue_heuristic = "attacker"

    # GENERATION 11 — asymmetric terminal payoff (phase 1 of the two-phase
    # test). Goal after a completed pass +10, solo goal +3. Every earlier
    # intervention left solo scoring exactly as valuable as cooperative
    # scoring, so the solo equilibrium stayed optimal; this changes the
    # payoff structure that DEFINES the equilibrium.
    # 11a continues the heuristic chain (blocked lane + payoff), 11b isolates
    # the payoff against the same heuristic from the best solo champion, so
    # the two differ only in their starting point.
    # PHASE 2 (later): drop goal_reward_solo -> symmetric payoff again, and
    # check whether the learned passing survives or erodes back.
    goal_reward_solo = 3.0

    runs = [
        # (label, init_path, start_level, load_buffer)
        ("11a_payoff_warm_chain", warm_chain, 5, "auto"),
        ("11b_payoff_from_champion", champion_solo, 5, "off"),
    ]

    jobs = []
    for label, init_path, start_level, load_buffer in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            None, init_path, total_steps, algo,
            pass_scenario_prob, "pass_demos",
            pass_scenario_prob_start, "flat", load_buffer,
            start_level, blue_heuristic, goal_reward_solo,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-11 job(s) submitted [asymmetric payoff: "
          f"pass-goal +10 vs solo-goal +{goal_reward_solo}, "
          f"opponent=heuristic, pass_scenario_prob={pass_scenario_prob}]")


if __name__ == "__main__":
    main()
