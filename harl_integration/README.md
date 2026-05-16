# HARL Integration for SSL 2v2

Drop-in to plug `SSL2v2SelfPlayEnv` from `bp/` into the
[HARL](https://github.com/PKU-MARL/HARL) MARL framework so we can train
**HASAC / MAPPO continuous / HAPPO / MATD3** on the same env we used for
the MASAC experiments.

## What's in here

```
harl_integration/
├── new_files/                                # Files we ADD to upstream HARL
│   ├── harl/envs/ssl_2v2/                    # Env wrapper + logger
│   ├── harl/configs/envs_cfgs/ssl_2v2.yaml   # Env defaults
│   ├── tuned_configs/ssl_2v2/hasac.json      # Pre-tuned HASAC algo config
│   └── test_ssl_2v2_wrapper.py               # Wrapper unit smoke test
├── patches/harl.patch                        # Modifications to upstream HARL
│                                             #   (5 files: envs/__init__.py,
│                                             #   envs_tools.py, configs_tools.py,
│                                             #   off_policy_base_runner.py,
│                                             #   examples/train.py)
├── setup_harl_cluster.sh                     # Idempotent one-shot installer
└── README.md
```

## Cluster bring-up

On a fresh cluster login node (or any new machine):

```bash
# 1. Get the bp repo
git clone https://github.com/SimonPfaendler/bp.git ~/dev/bp
cd ~/dev/bp
git checkout 2v1                              # or whichever branch has this folder

# 2. Run the setup script (clones HARL, makes venv, installs, applies patches)
bash harl_integration/setup_harl_cluster.sh

# 3. Submit the first training run
~/dev/venv_harl/bin/python ~/dev/bp/submit_harl_2v2.py
```

`setup_harl_cluster.sh` is re-runnable — it skips already-done steps. After
a `git pull` that updates `harl_integration/`, re-running it will rsync the
updated `new_files/` into HARL and (no-op) re-check the patch.

## How a training run is structured

```
WSL (local)                  Cluster login            SLURM compute node
=============                =================        ====================
edit code in bp/      ──►    git pull          ──►    submitit spawns job
+ harl_integration/          submit_harl_2v2.py       runs train.py with
git commit / push            (uses submitit)          --algo hasac --env ssl_2v2
                                                      writes to ./results/
```

The compute node uses **the same NFS-mounted home dir** as the login node,
so `~/dev/HARL` + `~/dev/venv_harl` + `~/dev/bp` are all visible from inside
the job.

## What submit_harl_2v2.py runs

By default:
- `algo=hasac`, `env=ssl_2v2`
- 4M env steps, 24 parallel rollout threads, 1 H100, 48 CPUs, 4h walltime
- Curriculum Level 1 (easy spawn), static Blue (`frozen_path=None`)
- Edits at the top of [`submit_harl_2v2.py`](../submit_harl_2v2.py) — change
  `algo` / `seed` / `frozen_path` / `curriculum_level` there.

## Updating HARL upstream

If we ever need to bump HARL past the pinned commit
`b1af98b0dbab72a2eee9d160751cd09aedbb8ce2`:

1. Update `HARL_COMMIT` in `setup_harl_cluster.sh`.
2. `cd ~/dev/HARL && git fetch && git checkout <new_commit>` locally.
3. Re-apply our patches manually, fix any conflicts.
4. `cd ~/dev/HARL && git diff > ~/dev/bp/harl_integration/patches/harl.patch`
   to refresh the patch file.
5. Commit + push.
