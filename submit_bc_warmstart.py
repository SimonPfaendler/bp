"""30-minute dev_gpu_h100 smoke test for BC-warm-started RL training.

Loads models/bc_pretrained_v2.zip and continues SAC training under the same
curriculum used for the original 1v1 champion run. Single seed, no self-play,
no anchors -- the goal is to verify the BC handover works on cluster and to
get a first signal whether BC helps the early curriculum stages.
"""
import os
import submitit


def run_experiment(algo, action_type, reward_type, seed, load_path, start_level=5):
    cmd = (
        f"python train.py {algo} -t "
        f"--action_type {action_type} --reward_type {reward_type} "
        f"--seed {seed} --start_level {start_level} "
        f"--load_path {load_path}"
    )
    os.system(cmd)


def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="bc_warm",
        slurm_time="00:30:00",
        slurm_partition="dev_gpu_h100",
        slurm_cpus_per_task=48,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )

    algo = "SAC"
    action_type = "low_level"
    reward_type = "dense"
    load_path = "models/bc_pretrained_v2.zip"
    start_level = 5
    seed = 820

    job = executor.submit(
        run_experiment,
        algo, action_type, reward_type, seed, load_path, start_level,
    )
    print(f"Submitted job: {job.job_id}")


if __name__ == "__main__":
    main()
