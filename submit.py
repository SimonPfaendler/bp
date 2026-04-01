import submitit
import os

def run_experiment(algo, action_type, reward_type, seed):
    """
    Diese Funktion wird auf dem zugewiesenen Compute-Node (der GPU) ausgeführt.
    Wir bauen hier einfach den Terminal-Befehl zusammen und feuern ihn ab.
    """
    cmd = f"python train.py {algo} -t --action_type {action_type} --reward_type {reward_type} --seed {seed}"
    print(f"Führe aus: {cmd}")
    os.system(cmd)

def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    
    executor = submitit.AutoExecutor(folder=log_folder)


    executor.update_parameters(
        slurm_job_name="ssl_phase1",
        slurm_time="2:00:00",          
        slurm_partition="cpu",                     
        slurm_cpus_per_task=4,          
        slurm_mem="8GB",               
    )

    
    algos = ["SAC", "CrossQ"]
    action_types = ["skills", "low_level"]
    reward_types = ["dense", "sparse"]
    seeds = [0, 1, 2]

    jobs = []

    print("Generiere Jobs und sende sie an den Slurm-Scheduler...")


    with executor.batch():
        for algo in algos:
            for action_type in action_types:
                for reward_type in reward_types:
                    for seed in seeds:
                        job = executor.submit(run_experiment, algo, action_type, reward_type, seed)
                        jobs.append(job)

    print(f"Erfolg! {len(jobs)} Jobs wurden in die Warteschlange eingereiht.")
    print("Nutze 'squeue -u <dein_kürzel>' im Terminal, um den Status zu überprüfen.")

if __name__ == "__main__":
    main()