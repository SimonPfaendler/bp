import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
):
    init_flag = f"--init_path {init_path} " if init_path else ""
    frozen_flag = f"--frozen_path {frozen_path} " if frozen_path else ""
    demo_flag = f"--demo_dir {demo_dir} " if demo_dir else ""
    cmd = (
        f"python train_2v2_selfplay.py "
        f"{frozen_flag}{init_flag}{demo_flag}"
        f"--reward_type {reward_type} --seed {seed} "
        f"--n_pairs {n_pairs} --total_steps {total_steps} "
        f"--algo {algo} "
        f"--pass_scenario_prob {pass_scenario_prob}"
    )
    os.system(cmd)


def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="sp2v2",
        slurm_time="04:00:00",
        slurm_partition="gpu_h100",
        slurm_cpus_per_task=24,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )

    # GENERATION 3 — cooperation induction A/B:
    #   Run 3a: pass scenario only          (attribution baseline)
    #   Run 3b: pass scenario + demo prefill (full package)
    # frozen/init = Gen-2 champion; demos must be generated against the SAME
    # checkpoint (generate_pass_demos.py --frozen <gen2>) and uploaded to
    # demo_dir before submitting 3b.
    gen2_champion = "models/<GEN2-CHAMPION>.zip"  # TODO: set before submit
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 12_000_000
    algo = "sac"
    pass_scenario_prob = 0.3

    runs = [
        # (label, demo_dir)
        ("3a_scenario_only", None),
        ("3b_scenario_plus_demos", "pass_demos"),
    ]

    jobs = []
    for label, demo_dir in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            gen2_champion, gen2_champion, total_steps, algo,
            pass_scenario_prob, demo_dir,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-3 job(s) submitted [algo={algo}, "
          f"pass_scenario_prob={pass_scenario_prob}]")


if __name__ == "__main__":
    main()
