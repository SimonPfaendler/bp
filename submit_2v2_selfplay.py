import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat", load_buffer="auto",
    start_level=None,
):
    init_flag = f"--init_path {init_path} " if init_path else ""
    frozen_flag = f"--frozen_path {frozen_path} " if frozen_path else ""
    demo_flag = f"--demo_dir {demo_dir} " if demo_dir else ""
    # -u: unbuffered stdout — without it the first log tables sit in the
    # 4KB block buffer for minutes and the slurm .out looks dead.
    cmd = (
        f"python -u train_2v2_selfplay.py "
        f"{frozen_flag}{init_flag}{demo_flag}"
        f"--reward_type {reward_type} --seed {seed} "
        f"--n_pairs {n_pairs} --total_steps {total_steps} "
        f"--algo {algo} --net {net} "
        f"--pass_scenario_prob {pass_scenario_prob} "
        f"--load_buffer {load_buffer}"
    )
    if pass_scenario_prob_start is not None:
        cmd += f" --pass_scenario_prob_start {pass_scenario_prob_start}"
    if start_level is not None:
        cmd += f" --start_level {start_level}"
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

    # GENERATION 9 — MASAC FROM SCRATCH with the full cooperation stack.
    # Rationale: the erosion results show the solo equilibrium is an
    # attractor — warm-starting from a solo champion drops the policy inside
    # its basin. From scratch, demos + BC steer early learning toward
    # passing BEFORE solo habits form, and the centralized critic assigns
    # team credit from step 0.
    #   frozen = Gen-2 champion (curve comparability; L1 shields the early
    #       phase — blues spawn parked far away at level 1)
    #   pass_scenario_prob 0.35 fixed, no anneal (erosion lesson)
    #   BC loss MASAC-capable   (joint demo batches unstacked per-agent)
    #   critic warmup + TD target clip +-30 (fresh centralized critic + 25%
    #       demo states per batch diverged to q_mean 1.6e6 without them)
    #
    # CHAINING: 1.4M is the empirical 30-min-slot max incl. startup, so the
    # run continues in chunks. Per chunk, set `chain_from` to the newest
    # *_final.zip; everything else adapts automatically.
    #   chunk 1 (20260803-174912): scratch, L1, reached success 0.63
    #   chunk 2 (20260803-193423): PROMOTED to L5 mid-chunk; arrived there
    #       at success 0.02 (curriculum cliff: L1 tap-ins -> full play).
    #       Q stayed healthy (q_mean 5.7, critic_loss 0.14) — fixes hold.
    #   chunk 3+: start_level PINNED TO 5 — the chain already promoted, and
    #       the callback would otherwise restart the env on L1 tap-ins.
    gen2_champion = "models/2v2_selfplay_SAC_dense_seed822_20260706-111029_final.zip"
    chain_from = "models/2v2_selfplay_MASAC_dense_seed822_20260803-193423_final.zip"
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 1_400_000
    algo = "masac"
    pass_scenario_prob = 0.35         # fixed — no anneal
    pass_scenario_prob_start = None
    load_buffer = "auto"              # chunk 1 used "off" (nothing to load)
    start_level = 5                   # chain promoted during chunk 2

    runs = [
        # (label, net)
        ("9_masac_chain3", "flat"),
    ]

    jobs = []
    for label, net in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            gen2_champion, chain_from, total_steps, algo,
            pass_scenario_prob, "pass_demos",
            pass_scenario_prob_start, net, load_buffer,
            start_level,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-9 job(s) submitted [MASAC chain, "
          f"init={chain_from}, start_level={start_level}, "
          f"fixed pass_scenario_prob={pass_scenario_prob}]")


if __name__ == "__main__":
    main()
