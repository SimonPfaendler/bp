import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat", load_buffer="auto",
    start_level=None, blue_heuristic=None, goal_reward_solo=None,
    target_action_std=None, noise_repeat_s=None,
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
    if blue_heuristic is not None:
        cmd += f" --blue_heuristic {blue_heuristic}"
    if goal_reward_solo is not None:
        cmd += f" --goal_reward_solo {goal_reward_solo}"
    if target_action_std is not None:
        cmd += f" --target_action_std {target_action_std}"
    if noise_repeat_s is not None:
        cmd += f" --noise_repeat_s {noise_repeat_s}"
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

    # GENERATION 14 — demo ablation. ONE variable, isolated.
    #
    # Reading all ten 2v2-vsheur runs end to end, the best policy is four
    # weeks old: 20260811-132942 finished at success .214 / scenario_pass
    # .508 / chaos .078. Everything after it is worse, monotonically, down
    # to Gen 13 at .068 / .186 / .017. Three things changed in between:
    #   1. asymmetric payoff       (47b40a8, 11.08) — isolated by 161843,
    #      which alone dropped success .214 -> .113
    #   2. scenario redesign       (9c6204a, 16.08) — heading jitter +-45deg,
    #      loose ball 30%, new `pressed` variant at +-90deg / 40%
    #   3. demo mixing + BC loss   (9c6204a, 16.08) — 25% of every batch
    # 2 and 3 landed in the SAME commit and have never been separated.
    #
    # Gen 13a already restored the symmetric payoff (undoing 1) and still
    # only reached .068, so the residual regression is in 2 and/or 3.
    #
    # Why demos are the prime suspect: 132942 is the ONLY run without them.
    # bc_loss sits at ~.015 in every later run, i.e. the BC term converged
    # long ago and teaches nothing new, while 25% of every batch keeps being
    # drawn from a fixed, never-evicted distribution. For an off-policy
    # critic that is a permanent distribution shift, not a warmup aid.
    # Consistent with that, actor_loss (~ alpha*log_pi - Q) degraded from
    # -2.42 in 132942 to -0.35 in Gen 13a: the actor finds lower-Q actions.
    #
    # 14a = no demos, 14b = demos. Everything else identical, so the pair
    # measures the demo effect alone. 14a doubles as the control against
    # 132942: if it returns to ~.2 the regression is the demos, if it stays
    # at ~.07 it is the scenario redesign, which is then the next ablation.
    #
    # Symmetric payoff in BOTH arms (goal_reward_solo=None): Gen 13 showed
    # the asymmetric variant is strictly worse (.050 vs .068) even once it
    # became representable, so it stays out until the regression is found.
    #
    # load_buffer="off" in both: the 56-dim layout cannot load any stored
    # 52-dim buffer anyway, and keeping it equal across arms matters more
    # than the refill cost.
    warm_chain = "models/2v2_selfplay_SAC_vsheur_dense_seed822_20260809-220118_final.zip"
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 3_000_000
    algo = "sac"
    pass_scenario_prob = 0.35
    pass_scenario_prob_start = None
    blue_heuristic = "attacker"

    runs = [
        # (label, demo_dir)
        ("14a_nodemos", None),
        ("14b_demos", "pass_demos_v3"),
    ]

    jobs = []
    for label, demos in runs:
        job = executor.submit(
            run_experiment, reward_type, seed, n_pairs,
            None, warm_chain, total_steps, algo,
            pass_scenario_prob, demos,
            pass_scenario_prob_start, "flat", "off",
            5, blue_heuristic, None, None, None,
        )
        jobs.append((label, job))
        print(f"Submitted {label}: job {job.job_id}")
    print(f"{len(jobs)} Gen-14 job(s) submitted [demo ablation: a=none, "
          f"b=pass_demos_v3; symmetric payoff both; init={warm_chain}]")


if __name__ == "__main__":
    main()
