import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path, init_path=None,
    total_steps=5_000_000,
):
    init_flag = f"--init_path {init_path} " if init_path else ""
    cmd = (
        f"python train_2v2_selfplay.py "
        f"--frozen_path {frozen_path} {init_flag}"
        f"--reward_type {reward_type} --seed {seed} "
        f"--n_pairs {n_pairs} --total_steps {total_steps}"
    )
    os.system(cmd)


def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="sp2v2",
        slurm_time="00:30:00",
        slurm_partition="dev_gpu_h100",
        slurm_cpus_per_task=48,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )

    # v4 = latest self-play champion (role-index off, 38-dim obs).
    # Env auto-strips role-index dims before feeding to this frozen model.
    frozen_path = (
        "models/2v2_selfplay_SAC_dense_seed822_20260513-112105_final.zip"
    )
    init_path = None  # None → reuses frozen_path as yellow init (parity start).
    reward_type = "dense"
    n_pairs = 24
    seeds = [822]
    total_steps = 5_000_000

    jobs = []
    for seed in seeds:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            frozen_path, init_path, total_steps,
        )
        jobs.append(job)
    print(f"Submitted {len(jobs)} 2v2 self-play job(s)")


if __name__ == "__main__":
    main()
