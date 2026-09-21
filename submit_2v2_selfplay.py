import os

import submitit


def run_experiment(
    reward_type, seed, n_pairs, frozen_path=None, init_path=None,
    total_steps=5_000_000, algo="masac",
    pass_scenario_prob=0.0, demo_dir=None,
    pass_scenario_prob_start=None, net="flat", load_buffer="auto",
    start_level=None, blue_heuristic=None, goal_reward_solo=None,
    target_action_std=None, noise_repeat_s=None, target_level=None,
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
    if target_level is not None:
        cmd += f" --target_level {target_level}"
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

    # GENERATION 15 — independent SAC on the pass drills, from scratch.
    #
    # The strict pass counter (kick-speed release, aimed at the mate, held by
    # the receiver) showed that no SAC run ever passed: the best checkpoint,
    # 20260811-141655, replayed in the env of its time scores .245 overall and
    # .55 in the staged pass scenario with 3 strict passes and 0 goals after
    # a strict pass in 200 episodes — a soloist. The only thing that produced
    # real passes so far is the drill curriculum (MAPPO: 0 -> .57 strict
    # passes/episode on L2 within 8M steps), and that is a property of the
    # task, not of the algorithm. This generation puts SAC on the same task.
    #
    #   LEVEL=2 (default)  pass drill, from scratch. Terminal = strict pass.
    #   LEVEL=3            pass + finish. Chain it from the L2 result:
    #                      LEVEL=3 INIT_PATH=models/<L2 run>_final.zip
    #
    # start_level == target_level pins the run to the drill; without it both
    # the callback and the env promote to L5 at 90 % success, which L3 can
    # reach. Blues are parked and passive on both drills whatever
    # blue_heuristic says; it is set for parity with the L5 runs to come.
    #
    # No warm start from the L5 checkpoints: they are 52-dim, lose ~60 % of
    # their strength under today's ball-speed normalisation (.245 -> .095),
    # and bring the solo habit along. load_buffer="off": an L2 buffer holds
    # L2 rewards and would poison an L3 critic.
    #
    # Compare against the MAPPO runs on rollout/passes_strict_per_episode
    # and selfplay/live_success_rate (success == strict pass on L2,
    # == goal after strict pass on L3).
    level = int(os.environ.get("LEVEL", "2"))
    assert level in (2, 3), level
    init_path = os.environ.get("INIT_PATH")  # None = from scratch
    reward_type = "dense"
    n_pairs = 24
    seed = 822
    total_steps = 3_000_000
    algo = "sac"
    blue_heuristic = "attacker"

    job = executor.submit(
        run_experiment, reward_type, seed, n_pairs,
        None, init_path, total_steps, algo,
        0.0, None,
        None, "flat", "off",
        level, blue_heuristic, None, None, None, level,
    )
    print(f"Submitted Gen-15 SAC drill L{level}: job {job.job_id} "
          f"[init={init_path or 'scratch'}, start=target={level}]")


if __name__ == "__main__":
    main()
