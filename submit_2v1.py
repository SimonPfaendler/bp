import os

import submitit


def run_experiment(algo, reward_type, seed, n_pairs, start_level=1, joint=False):
    joint_flag = "--joint " if joint else ""
    cmd = (
        f"python train_2v1.py {algo} -t {joint_flag}"
        f"--reward_type {reward_type} --seed {seed} "
        f"--n_pairs {n_pairs} --start_level {start_level}"
    )
    os.system(cmd)


def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)

    # Testing: dev_gpu_h100, 30 min.
    # Production: switch to slurm_partition="gpu_h100" and bump slurm_time.
    executor.update_parameters(
        slurm_job_name="sac2v1",
        slurm_time="00:30:00",
        slurm_partition="dev_gpu_h100",
        slurm_cpus_per_task=48,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )

    algo = "SAC"
    reward_type = "dense"
    n_pairs = 24
    start_level = 5
    seeds = [822]
    joint = False  # JAL: joint-action learner (single policy, joint obs/action)

    jobs = []
    for seed in seeds:
        job = executor.submit(
            run_experiment, algo, reward_type, seed, n_pairs, start_level, joint
        )
        jobs.append(job)
    print(f"Submitted {len(jobs)} job(s) (joint={joint})")


if __name__ == "__main__":
    main()
