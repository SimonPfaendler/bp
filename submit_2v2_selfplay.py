import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat", load_buffer="auto",
    start_level=None,
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

    # GENERATION 9 — MASAC restart with the full stack and a FAIR budget.
    # The old MASAC run was handicapped three ways: half the sim budget
    # (its counter logs 24 timesteps/sim-tick vs SAC's 48, so 3.3M counted
    # steps = only half the episodes), cut off mid-anneal, and missing the
    # BC loss (was SAC-only). Fixed here:
    #   total_steps 1.65M       == 3.3M SAC sim-equivalent, fits 30-min dev
    #   pass_scenario_prob 0.35 fixed, no anneal (erosion lesson)
    #   BC loss now MASAC-capable (joint demo batches unstacked per-agent)
    #   critic_warmup 10000     fresh centralized critic, warm actor
    #   start_level=1           L1 tap-ins calibrate the fresh joint critic
    #       on dense value targets; the competent actor promotes quickly
    #   init = 171222 (most pass-capable actor; critic can't transfer)
    #   frozen = Gen-2 champion (comparability with all main curves)
    #   load_buffer=off (SAC per-agent buffer is joint-incompatible anyway)
    gen2_champion = "models/2v2_selfplay_SAC_dense_seed822_20260706-111029_final.zip"
    best_passer = "models/2v2_selfplay_SAC_dense_seed822_20260720-171222_final.zip"
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 1_650_000
    algo = "masac"
    pass_scenario_prob = 0.35         # fixed — no anneal
    pass_scenario_prob_start = None
    load_buffer = "off"
    start_level = 1

    runs = [
        # (label, net)
        ("9_masac_full_stack", "flat"),
    ]

    jobs = []
    for label, net in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            gen2_champion, best_passer, total_steps, algo,
            pass_scenario_prob, "pass_demos",
            pass_scenario_prob_start, net, load_buffer,
            start_level,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-9 job(s) submitted [MASAC full stack, "
          f"1.65M (=3.3M SAC-equiv), start_level={start_level}, "
          f"fixed pass_scenario_prob={pass_scenario_prob}]")


if __name__ == "__main__":
    main()
