import submitit
import os

def run_experiment(algo, action_type, reward_type, seed, start_level=1):
    cmd = f"python train.py {algo} -t --action_type {action_type} --reward_type {reward_type} --seed {seed} --start_level {start_level}"
    os.system(cmd)

def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)


    executor = submitit.AutoExecutor(folder=log_folder)


    executor.update_parameters(
        slurm_job_name="SAC_low",
        slurm_time="04:30:00",
        slurm_partition="gpu_h100_short",
        slurm_cpus_per_task=48,
        slurm_mem="193300mb",

        slurm_additional_parameters={
            "gres": "gpu:1"
        }
    )

    
    algos = ["SAC"]
    action_types = ["low_level"]
    reward_types = ["dense"]
    seeds = [820]
    start_level = 5

    jobs = []

    with executor.batch():
        for algo in algos:
            for action_type in action_types:
                for reward_type in reward_types:
                    for seed in seeds:
                        job = executor.submit(run_experiment, algo, action_type, reward_type, seed, start_level)
                        jobs.append(job)
    print(len(jobs))

if __name__ == "__main__":
    main()
