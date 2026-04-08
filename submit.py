import submitit
import os

def run_experiment(algo, action_type, reward_type, seed):
    cmd = f"python train.py {algo} -t --action_type {action_type} --reward_type {reward_type} --seed {seed}"
    print(f"Führe aus: {cmd}")
    os.system(cmd)

def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)


    executor = submitit.AutoExecutor(folder=log_folder)


    executor.update_parameters(
        slurm_job_name="ssl_phase1",
        slurm_time="24:00:00",          
        slurm_partition="cpu",                     
        slurm_cpus_per_task=4,          
        slurm_mem="8GB",               
    )

    
    algo = "SAC"
    action_type = "low_level"
    reward_type = "dense"
    seeds = range(8)

    jobs = []

    with executor.batch():
        for seed in seeds:
            job = executor.submit(run_experiment, algo, action_type, reward_type, seed)
            jobs.append(job)
    print(len(jobs))

if __name__ == "__main__":
    main()
