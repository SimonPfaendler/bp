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
VENV_PY = os.environ.get("VENV_PY", f"{WORKSPACE}/venv_harl/bin/python")


def run_experiment(
    algo, seed, n_rollout_threads, num_env_steps,
    curriculum_level, frozen_path, exp_name,
):
    """Invoke HARL train.py with the ssl_2v2 env via CLI overrides."""
    frozen_flag = (
        f"--env_args.frozen_path {frozen_path} " if frozen_path else ""
    )
    cmd = (
        f"cd {HARL_DIR} && "
        f"BP_DIR={BP_DIR} {VENV_PY} examples/train.py "
        f"--algo {algo} --env ssl_2v2 --exp_name {exp_name} "
        f"--seed {seed} "
        f"--n_rollout_threads {n_rollout_threads} "
        f"--num_env_steps {num_env_steps} "
        f"--env_args.curriculum_level {curriculum_level} "
        f"{frozen_flag}"
    )
    os.system(cmd)


def main():
    log_folder = "slurm_logs_harl"
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        slurm_job_name="harl2v2",
        slurm_time="04:00:00",
        slurm_partition="gpu_h100",   # may need adjusting per cluster
        slurm_cpus_per_task=48,
        slurm_mem="193300mb",
        slurm_additional_parameters={"gres": "gpu:1"},
    )

    # First serious run: HASAC + parameter sharing on Level 1 + static blue
    # (no frozen opponent). 4M env steps, 24 parallel rollout envs.
    algo = "hasac"
    seeds = [822]
    n_rollout_threads = 24
    num_env_steps = 4_000_000
    curriculum_level = 1
    frozen_path = None
    exp_name = "ssl2v2_hasac_lvl1_static"

    jobs = []
    for seed in seeds:
        job = executor.submit(
            run_experiment, algo, seed, n_rollout_threads,
            num_env_steps, curriculum_level, frozen_path, exp_name,
        )
        jobs.append(job)
    print(f"Submitted {len(jobs)} HARL job(s) [algo={algo}, exp={exp_name}]")


if __name__ == "__main__":
    main()
