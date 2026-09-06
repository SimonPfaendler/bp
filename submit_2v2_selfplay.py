import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat", load_buffer="auto",
    start_level=None, blue_heuristic=None, goal_reward_solo=None,
    target_action_std=None, noise_repeat_s=None,
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
    if target_action_std is not None:
        cmd += f" --target_action_std {target_action_std}"
    if noise_repeat_s is not None:
        cmd += f" --noise_repeat_s {noise_repeat_s}"
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

    # GENERATION 13 — Gen 11 repeated, this time with the incentive it was
    # supposed to test actually observable.
    #
    # Gen 11 gave a goal after a pass +10 and a solo goal +3, and concluded
    # "the bottleneck is not incentive" because ~54 goals per window still
    # contained only 2 after a pass. That conclusion does not follow: the
    # payoff is gated on `passes_in_episode > 0`, which appeared in NO obs
    # slot. Verified by mutating passes_in_episode / current_step /
    # last_yellow_carrier / blue_touched_since_yellow on a live env: 0 of 104
    # obs values changed. Two physically identical goal states carried
    # terminal reward 11 or 4 with no observable difference, so the critic
    # could only regress to their frequency-weighted mean (~3.3). The
    # incentive was never representable, hence never tested.
    #
    # Repaired in the env (obs 52 -> 56, appended last):
    #   has_passed, i_am_last_carrier, blue_touched_since_yellow  -> the
    #     +10/+3 goal split and the +3 pass bonus become functions of
    #     observed state
    #   time_remaining -> the time penalty scales with current_step and
    #     truncation costs -1, in a finite-horizon MDP with no clock feature
    #   plus norm_ball_v: ball speed was divided by the ROBOT max (4.035)
    #     and clipped at 1.2, so every kick above 4.84 m/s looked identical
    #     while kicks span 3-6 m/s — the receiver could not judge an
    #     incoming pass in the top 40% of the kick range
    #
    # Warm start survives the layout change: _fit_param keeps the old weight
    # columns and zero-inits the appended ones, so the net is function-
    # identical at init. The replay buffer cannot migrate (the new dims are
    # not recoverable from stored 52-dim obs) — one slot of refill is the
    # unavoidable price, hence load_buffer="off".
    #
    # 13a isolates the repair; 13b adds the asymmetric payoff on top, so the
    # pair answers "was Gen 11's null result an artefact of unobservability?"
    warm_chain = "models/2v2_selfplay_SAC_vsheur_dense_seed822_20260809-220118_final.zip"
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 3_000_000
    algo = "sac"
    pass_scenario_prob = 0.35
    pass_scenario_prob_start = None
    blue_heuristic = "attacker"
    demo_dir = "pass_demos_v3"        # regenerated at the 56-dim layout

    runs = [
        # (label, goal_reward_solo)
        ("13a_obsfix_symmetric", None),
        ("13b_obsfix_asymmetric", 3.0),
    ]

    jobs = []
    for label, solo in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            None, warm_chain, total_steps, algo,
            pass_scenario_prob, demo_dir,
            pass_scenario_prob_start, "flat", "off",
            5, blue_heuristic, solo, None, None,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-13 job(s) submitted [obs repair 52->56; "
          f"b adds the asymmetric payoff, now representable; "
          f"init={warm_chain}]")


if __name__ == "__main__":
    main()
