import submitit
import os

def run_experiment(algo, action_type, reward_type, seed):
    cmd = f"python train.py {algo} -t --action_type {action_type} --reward_type {reward_type} --seed {seed}"
    os.system(cmd)

def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)


    executor = submitit.AutoExecutor(folder=log_folder)


    executor.update_parameters(
        slurm_job_name="Test",
        slurm_time="00:30:00",
        slurm_partition="dev_gpu_h100",
        slurm_cpus_per_task=48,
        slurm_mem="193300MB",

        slurm_additional_parameters={
            "gres": "gpu:1"
        }
    )

    
    algos = ["SAC"]
    action_types = ["low_level", "skills"]
    reward_types = ["dense"]
    seeds = [700]

    jobs = []

    with executor.batch():
        for algo in algos:
            for action_type in action_types:
                for reward_type in reward_types:
                    for seed in seeds:
                        job = executor.submit(run_experiment, algo, action_type, reward_type, seed)
                        jobs.append(job)
    print(len(jobs))

if __name__ == "__main__":
    main()
