import os

import submitit


def run_experiment(
    algo, action_type, reward_type, seed, opponent_path,
    start_level=5, pool_snapshot_freq=200_000,
    anchor_heuristic_prob=0.3, anchor_v0_prob=0.2,
):
    cmd = (
        f"python train.py {algo} -t "
        f"--action_type {action_type} --reward_type {reward_type} "
        f"--seed {seed} --start_level {start_level} "
        f"--selfplay --opponent {opponent_path} "
        f"--pool_snapshot_freq {pool_snapshot_freq} "
        f"--anchor_heuristic_prob {anchor_heuristic_prob} "
        f"--anchor_v0_prob {anchor_v0_prob}"
    )
    os.system(cmd)


def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)

    # Smoke-test config: 30min on dev_gpu_h100 to confirm the selfplay
    # pipeline boots on the cluster. Bump slurm_time + partition for the
    # full ~3.8M-step run after this passes.
    executor.update_parameters(
        slurm_job_name="sp1v1",
        slurm_time="00:30:00",
        slurm_partition="dev_gpu_h100",
        slurm_cpus_per_task=48,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )

    algo = "SAC"
    action_type = "low_level"
    reward_type = "dense"
    opponent_path = "models/SAC_low_level_dense_seed820_20260422-113429_final.zip"
    start_level = 5
    pool_snapshot_freq = 200_000
    # Anchored self-play: 0.3 heuristic + 0.2 frozen-v0 + 0.5 pool sample.
    anchor_heuristic_prob = 0.3
    anchor_v0_prob = 0.2
    seeds = [820, 821, 822]

    jobs = []
    with executor.batch():
        for seed in seeds:
            job = executor.submit(
                run_experiment,
                algo, action_type, reward_type, seed, opponent_path,
                start_level, pool_snapshot_freq,
                anchor_heuristic_prob, anchor_v0_prob,
            )
            jobs.append(job)
    print(f"Submitted {len(jobs)} 1v1 self-play job(s) "
          f"(anchors: heuristic={anchor_heuristic_prob}, v0={anchor_v0_prob})")


if __name__ == "__main__":
    main()
