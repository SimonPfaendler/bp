"""Submit HARL (HASAC / MAPPO continuous) 2v2 SSL training to SLURM.

Defaults: bp/, HARL/, venv_harl/ all siblings in the same workspace dir.
Override via env vars BP_DIR / HARL_DIR / VENV_PY if your layout differs:
  BP_DIR=/pfs/.../bp HARL_DIR=/pfs/.../HARL VENV_PY=/pfs/.../venv_harl/bin/python \\
    python submit_harl_2v2.py
"""
import os

import submitit


BP_DIR = os.environ.get(
    "BP_DIR", os.path.dirname(os.path.abspath(__file__))
)
WORKSPACE = os.path.dirname(BP_DIR)
HARL_DIR = os.environ.get("HARL_DIR", f"{WORKSPACE}/HARL")


def _find_venv_py():
    if "VENV_PY" in os.environ:
        return os.environ["VENV_PY"]
    candidates = [
        f"{WORKSPACE}/venv_harl/bin/python",
        f"{BP_DIR}/miniforge3/envs/harl_env/bin/python",
        f"{WORKSPACE}/miniforge3/envs/harl_env/bin/python",
        os.path.expanduser("~/miniforge3/envs/harl_env/bin/python"),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise SystemExit(
        f"No HARL python found. Set VENV_PY=... explicitly. Tried: {candidates}"
    )


VENV_PY = _find_venv_py()


def run_experiment(
    algo, seed, n_rollout_threads, n_eval_rollout_threads,
    num_env_steps, curriculum_level, frozen_path, exp_name,
    model_dir=None, warmup_steps=None,
):
    """Invoke HARL train.py with the ssl_2v2 env via CLI overrides."""
    # HARL's update_args matches CLI args by *leaf* key only — dot-notation
    # like --env_args.curriculum_level is silently ignored. Pass the leaf
    # names directly (curriculum_level / frozen_path are unique leaves
    # inside env_args, so no ambiguity).
    frozen_flag = (
        f"--frozen_path {frozen_path} " if frozen_path else ""
    )
    # When chaining 30-min runs, --model_dir restores actor + critic +
    # value_normalizer from a previous run's models/ folder. The replay
    # buffer doesn't get saved, so we still need a small warmup to fill
    # the buffer to batch_size before training updates kick in.
    model_dir_flag = (
        f"--model_dir {model_dir} " if model_dir else ""
    )
    warmup_flag = (
        f"--warmup_steps {warmup_steps} " if warmup_steps is not None else ""
    )
    # PYTHONUNBUFFERED=1 + python -u: submitit redirects stdout/stderr to
    # files (no tty), which makes CPython block-buffer prints. Without this
    # the job looks hung for minutes while warmup output sits in the kernel
    # buffer; with it, every print() lands in the log file immediately.
    #
    # --load_config pulls tuned_configs/ssl_2v2/<algo>.json which sets the
    # important non-default knobs: share_param=True (parameter-shared actor
    # over both homogeneous yellows), auto_alpha=True + alpha=0.2 (adaptive
    # entropy, breaks the alpha=0.001 mode-collapse from yaml defaults),
    # hidden_sizes=[512,512,512], gamma=0.99, batch=1024, use_valuenorm=True
    # (normalises reward targets — important given our +100/-50 outcomes),
    # use_huber_loss=True (robust critic loss for those goal/concede spikes).
    # CLI flags after --load_config still override (see train.py update_args).
    tuned_cfg = f"{HARL_DIR}/tuned_configs/ssl_2v2/{algo}.json"
    wandb_env = ""
    for var in ("WANDB_API_KEY", "WANDB_PROJECT", "WANDB_RUN_NAME",
                "WANDB_MODE", "WANDB_ENTITY"):
        if var in os.environ:
            wandb_env += f"{var}={os.environ[var]} "
    cmd = (
        f"cd {HARL_DIR} && "
        f"PYTHONUNBUFFERED=1 BP_DIR={BP_DIR} {wandb_env}{VENV_PY} -u examples/train.py "
        f"--load_config {tuned_cfg} "
        f"--algo {algo} --env ssl_2v2 --exp_name {exp_name} "
        f"--seed {seed} "
        f"--n_rollout_threads {n_rollout_threads} "
        f"--n_eval_rollout_threads {n_eval_rollout_threads} "
        f"--num_env_steps {num_env_steps} "
        f"--update_per_train 1 "
        f"--curriculum_level {curriculum_level} "
        f"{warmup_flag}{model_dir_flag}{frozen_flag}"
    )
    os.system(cmd)


def main():
    log_folder = "slurm_logs_harl"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="harl2v2",
        slurm_time="00:30:00",
        slurm_partition="dev_gpu_h100",   # may need adjusting per cluster
        slurm_cpus_per_task=48,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )

    # HASAC with tuned config (share_param + auto_alpha + valuenorm + huber).
    # Run-name distinguishes it from the earlier yaml-default run so the
    # results dir doesn't collide.
    algo = "hasac"
    seeds = [822]
    # 36 train + 8 eval = 44 + main + torch ≈ 48 cores (matches SLURM alloc).
    # update_per_train=2 compensates for the larger n_rollout_threads keeping
    # the env-steps-per-gradient-update ratio similar to the 24-thread config.
    n_rollout_threads = 36
    n_eval_rollout_threads = 8
    num_env_steps = 4_000_000
    curriculum_level = 5
    frozen_path = (
        "/pfs/work9/workspace/scratch/fr_sp329-ssl_rl_project/bp/models/"
        "2v2_selfplay_SAC_dense_seed822_20260512-154059_final.zip"
    )
    exp_name = "ssl2v2_hasac_lvl5_h512_vs_sac0512"
    # Chain-from-checkpoint: when MODEL_DIR is set, the run loads actor+
    # critic+value_norm from that path (must be a HARL run's models/ dir),
    # forces curriculum_level=5 (since the loaded policy is already
    # L1-competent — no need to redo curriculum), and shrinks warmup to
    # just refill the replay buffer to batch_size.
    model_dir = os.environ.get("MODEL_DIR")
    warmup_steps = None
    if model_dir:
        curriculum_level = 5
        warmup_steps = 1500
        # Tag the chained run with the source so checkpoint chain stays readable.
        src = os.path.basename(os.path.dirname(model_dir.rstrip("/")))
        exp_name = f"ssl2v2_hasac_l5_continue_from_{src}"
        print(f"Chaining from MODEL_DIR={model_dir}")
        print(f"  curriculum_level forced to 5, warmup_steps={warmup_steps}")
        print(f"  exp_name={exp_name}")

    jobs = []
    for seed in seeds:
        job = executor.submit(
            run_experiment, algo, seed, n_rollout_threads,
            n_eval_rollout_threads, num_env_steps,
            curriculum_level, frozen_path, exp_name,
            model_dir, warmup_steps,
        )
        jobs.append(job)
    print(f"Submitted {len(jobs)} HARL job(s) [algo={algo}, exp={exp_name}]")


if __name__ == "__main__":
    main()
