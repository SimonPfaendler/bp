import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat", load_buffer="auto",
    start_level=None, blue_heuristic=None,
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
    init_from = "models/2v2_selfplay_SAC_dense_seed822_20260720-171222_final.zip"
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 3_000_000
    algo = "sac"
    pass_scenario_prob = 0.35         # fixed — no anneal (erosion lesson)
    pass_scenario_prob_start = None
    load_buffer = "off"               # old buffer is vs. a different opponent
    start_level = 5                   # warm init already plays level 5
    blue_heuristic = "attacker"

    runs = [
        # (label, net)
        ("10_sac_vs_heuristic", "flat"),
    ]

    jobs = []
    for label, net in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            None, init_from, total_steps, algo,
            pass_scenario_prob, "pass_demos",
            pass_scenario_prob_start, net, load_buffer,
            start_level, blue_heuristic,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-10 job(s) submitted [opponent=heuristic team, "
          f"init={init_from}, start_level={start_level}, "
          f"fixed pass_scenario_prob={pass_scenario_prob}]")


if __name__ == "__main__":
    main()
