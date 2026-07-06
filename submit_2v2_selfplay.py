import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
):
    init_flag = f"--init_path {init_path} " if init_path else ""
    frozen_flag = f"--frozen_path {frozen_path} " if frozen_path else ""
    cmd = (
        f"python train_2v2_selfplay.py "
        f"{frozen_flag}{init_flag}"
        f"--reward_type {reward_type} --seed {seed} "
        f"--n_pairs {n_pairs} --total_steps {total_steps} "
        f"--algo {algo}"
    )
    os.system(cmd)


def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="sp2v2",
        slurm_time="01:30:00",
        slurm_partition="gpu_h100",
        slurm_cpus_per_task=24,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )

    # RUN A — build a competent attacker from scratch with the clean stack
    # (MASAC + LayerNorm + thin reward). Blue stands still (frozen_path=None),
    checkpoint = "models/2v2_selfplay_SAC_dense_seed822_20260706-081229_10560000_steps.zip"
    frozen_path = checkpoint
    init_path = checkpoint
    reward_type = "dense"
    n_pairs = 24
    seeds = [822]
    total_steps = 11_500_000
    algo = "sac"  # "sac" for the Independent-SAC diagnostic

    jobs = []
    for seed in seeds:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            frozen_path, init_path, total_steps, algo,
        )
        jobs.append(job)
    print(f"Submitted {len(jobs)} 2v2 self-play job(s) [algo={algo}]")


if __name__ == "__main__":
    main()
