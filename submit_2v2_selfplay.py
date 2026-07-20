import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None,
):
    init_flag = f"--init_path {init_path} " if init_path else ""
    frozen_flag = f"--frozen_path {frozen_path} " if frozen_path else ""
    demo_flag = f"--demo_dir {demo_dir} " if demo_dir else ""
    cmd = (
        f"python train_2v2_selfplay.py "
        f"{frozen_flag}{init_flag}{demo_flag}"
        f"--reward_type {reward_type} --seed {seed} "
        f"--n_pairs {n_pairs} --total_steps {total_steps} "
        f"--algo {algo} "
        f"--pass_scenario_prob {pass_scenario_prob}"
    )
    if pass_scenario_prob_start is not None:
        cmd += f" --pass_scenario_prob_start {pass_scenario_prob_start}"
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

    # GENERATION 4 — MASAC (centralized critic) with the new stack:
    #   demo mixing + staged->chaos schedule (BC is SAC-only, auto-off here).
    # init = best SAC actor (Gen-3 3b champion) -> MASAC transfers the actor,
    # the centralized critic starts fresh (critic_warmup handles it). The SAC
    # replay buffer is per-agent, incompatible with MASAC's joint buffer, so
    # the reload is auto-skipped and MASAC fills a fresh joint buffer.
    gen2_champion = "models/2v2_selfplay_SAC_dense_seed822_20260706-111029_final.zip"
    init_actor = "models/2v2_selfplay_SAC_dense_seed822_20260720-171222_final.zip"
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 3_300_000
    algo = "masac"
    # Pass-scenario schedule: start heavily staged (learn to pass), anneal to
    # mostly chaos (apply passing in unstructured play).
    pass_scenario_prob = 0.2          # end value
    pass_scenario_prob_start = 0.8    # start value

    runs = [
        # (label, demo_dir)
        ("4_masac_demos_schedule", "pass_demos"),
    ]

    jobs = []
    for label, demo_dir in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            gen2_champion, init_actor, total_steps, algo,
            pass_scenario_prob, demo_dir,
            pass_scenario_prob_start,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-3 job(s) submitted [algo={algo}, "
          f"pass_scenario_prob={pass_scenario_prob_start}->{pass_scenario_prob}]")


if __name__ == "__main__":
    main()
