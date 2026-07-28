import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat",
):
    init_flag = f"--init_path {init_path} " if init_path else ""
    frozen_flag = f"--frozen_path {frozen_path} " if frozen_path else ""
    demo_flag = f"--demo_dir {demo_dir} " if demo_dir else ""
    cmd = (
        f"python train_2v2_selfplay.py "
        f"{frozen_flag}{init_flag}{demo_flag}"
        f"--reward_type {reward_type} --seed {seed} "
        f"--n_pairs {n_pairs} --total_steps {total_steps} "
        f"--algo {algo} --net {net} "
        f"--pass_scenario_prob {pass_scenario_prob}"
    )
    if pass_scenario_prob_start is not None:
        cmd += f" --pass_scenario_prob_start {pass_scenario_prob_start}"
    os.system(cmd)


def main():
    log_folder = "slurm_logs"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="sp2v2",
        slurm_time="02:00:00",
        slurm_partition="gpu_h100",
        slurm_cpus_per_task=24,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )
    # NOTE: MASAC (JointSubprocPairVecEnv, num_envs=n_pairs) needs ~2x the
    # real simulation steps of independent SAC (SubprocPairVecEnv,
    # num_envs=2*n_pairs) to reach the SAME --total_steps counter, because
    # SB3 counts total_timesteps += num_envs per step_wait(). At the observed
    # ~42 iterations/s, 3.3M steps needs ~55 real minutes for MASAC — the old
    # 30-min dev_gpu_h100 slot cut it off at ~52% of its own pass_scenario
    # schedule, making any SAC-vs-MASAC comparison at that checkpoint an
    # apples-to-oranges mid-schedule-vs-finished comparison.

    # GENERATION 6 — pooling ablation inside the Deep-Sets idea.
    # Gen 5 showed deepsets matching flat in STAGED scenarios (pass sr 0.34 vs
    # 0.35) but collapsing in CHAOS (sr 0.04 vs 0.23, passes 0.0 vs 0.013):
    # mean pooling alone cannot separate "one opponent on me, one far" from
    # "both mid-range" at N=2. meanmax adds per-feature max pooling (the
    # "most threatening opponent" signal) at equal permutation invariance.
    #
    # Same seed/config as Gen 5, so 6a is directly comparable to the existing
    # 5a (flat) and 5b (mean-pooled) curves. 6b re-runs mean-only under the
    # current extractor (mate encoder removed) for a clean pooling ablation —
    # drop it if GPU time is tight and use 5b as the reference instead.
    gen2_champion = "models/2v2_selfplay_SAC_dense_seed822_20260706-111029_final.zip"
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 3_300_000
    algo = "sac"
    # Pass-scenario schedule: start heavily staged (learn to pass), anneal to
    # mostly chaos (apply passing in unstructured play).
    pass_scenario_prob = 0.2          # end value
    pass_scenario_prob_start = 0.8    # start value

    runs = [
        # (label, net)
        ("6a_deepsets_meanmax", "deepsets"),
        ("6b_deepsets_mean", "deepsets_mean"),
    ]

    jobs = []
    for label, net in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            gen2_champion, None, total_steps, algo,
            pass_scenario_prob, "pass_demos",
            pass_scenario_prob_start, net,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-6 job(s) submitted [pooling ablation, "
          f"algo={algo}, from scratch, "
          f"pass_scenario_prob={pass_scenario_prob_start}->{pass_scenario_prob}]")


if __name__ == "__main__":
    main()
