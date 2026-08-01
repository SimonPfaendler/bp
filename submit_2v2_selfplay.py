import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat", load_buffer="auto",
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
        f"--pass_scenario_prob {pass_scenario_prob} "
        f"--load_buffer {load_buffer}"
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
        slurm_time="00:30:00",
        slurm_partition="dev_gpu_h100",
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

    # GENERATION 8 — ladder step with role-split checkpoints.
    # Chain verdict (Gen 7, 6M cumulative): pass skill ERODES under
    # continued training at prob 0.2 (scenario passes 0.26->0.16,
    # scored_after_pass 0.033->0) while solo skill climbs to best-ever
    # (chaos success 0.40). Two consequences, both applied here:
    #   frozen := chunk-2 final (140727) — the strongest solo presser we
    #       have; pressure is what should make the solo path unprofitable.
    #   init   := 171222 — the most pass-capable champion.
    #   pass_scenario_prob FIXED at 0.35, no anneal (the erosion lesson).
    #   load_buffer=off — 171222's buffer holds experience against the OLD
    #       frozen; stale dynamics under the new opponent.
    strongest_presser = "models/2v2_selfplay_SAC_dense_seed822_20260801-140727_final.zip"
    best_passer = "models/2v2_selfplay_SAC_dense_seed822_20260720-171222_final.zip"
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 3_000_000
    algo = "sac"
    pass_scenario_prob = 0.35         # fixed — no anneal
    pass_scenario_prob_start = None
    load_buffer = "off"

    runs = [
        # (label, net)
        ("8_ladder_passer_vs_presser", "flat"),
    ]

    jobs = []
    for label, net in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            strongest_presser, best_passer, total_steps, algo,
            pass_scenario_prob, "pass_demos",
            pass_scenario_prob_start, net, load_buffer,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-8 job(s) submitted [frozen=strongest presser, "
          f"init=best passer, fixed pass_scenario_prob={pass_scenario_prob}]")


if __name__ == "__main__":
    main()
