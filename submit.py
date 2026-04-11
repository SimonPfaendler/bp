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
        slurm_job_name="sac_finetune",
        slurm_time="02:00:00",
        slurm_partition="gpu_h100_short",
        slurm_cpus_per_task=24,
        slurm_mem="193GB",

        slurm_additional_parameters={
            "gres": "gpu:1"
        }
    )

    
    algos = ["SAC"]
    action_type = "low_level"
    reward_type = "dense"
    seeds = [200]

    jobs = []

    with executor.batch():
        for algo in algos:
            for seed in seeds:
                job = executor.submit(run_experiment, algo, action_type, reward_type, seed)
                jobs.append(job)
    print(len(jobs))

if __name__ == "__main__":
    main()
