#!/usr/bin/env bash
# One-shot HARL setup for a fresh cluster login node.
#
# What it does:
#   1. Clones HARL @ pinned commit into $HARL_DIR (default ~/dev/HARL).
#   2. Creates a fresh venv at $VENV_DIR (default ~/dev/venv_harl) using the
#      Python at $PY (default python3.10).
#   3. Installs torch 2.10.0+cu128 + HARL + tensorboard + rsoccer-gym (from
#      the local /home/simon/dev/rSoccer clone — assumes that exists) +
#      pygame + stable-baselines3.
#   4. Applies our patches (harl.patch) and drops in our new files
#      (envs/ssl_2v2/, configs/envs_cfgs/ssl_2v2.yaml, tuned_configs/ssl_2v2/,
#      test_ssl_2v2_wrapper.py).
#   5. Runs the wrapper smoke test to confirm everything imports.
#
# Re-runnable: skips clone if HARL_DIR exists, skips venv if VENV_DIR exists.
# To re-apply only the patches+files (after pulling fresh bp), delete those
# steps' guards or just run them by hand.

set -euo pipefail

HARL_DIR="${HARL_DIR:-$HOME/dev/HARL}"
VENV_DIR="${VENV_DIR:-$HOME/dev/venv_harl}"
BP_DIR="${BP_DIR:-$HOME/dev/bp}"
RSOCCER_DIR="${RSOCCER_DIR:-$HOME/dev/rSoccer}"
PY="${PY:-python3.10}"
HARL_COMMIT="b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"

INTEG="$BP_DIR/harl_integration"

echo "=== 1/5  HARL clone ==="
if [[ ! -d "$HARL_DIR" ]]; then
    git clone https://github.com/PKU-MARL/HARL.git "$HARL_DIR"
fi
cd "$HARL_DIR"
git fetch --quiet origin
git checkout --quiet "$HARL_COMMIT"
echo "HARL @ $(git log -1 --format='%h %s')"

echo "=== 2/5  venv ==="
if [[ ! -d "$VENV_DIR" ]]; then
    "$PY" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip wheel
fi
PIP="$VENV_DIR/bin/pip"

echo "=== 3/5  dependencies ==="
# Torch first (large, ~3GB of cu12 libs + 750MB torch wheel).
$PIP install --quiet torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# HARL editable + its declared deps (tensorboard, sacred, setproctitle, …).
$PIP install --quiet -e "$HARL_DIR"
# rSoccer from local clone — PyPI ships an unrelated 1.4 fork that is API-incompatible.
if [[ ! -d "$RSOCCER_DIR" ]]; then
    git clone https://github.com/robocin/rSoccer.git "$RSOCCER_DIR"
fi
$PIP install --quiet "$RSOCCER_DIR"
$PIP install --quiet pygame stable-baselines3

echo "=== 4/5  apply ssl_2v2 integration ==="
# New files: rsync mirrors $INTEG/new_files into $HARL_DIR.
rsync -a "$INTEG/new_files/" "$HARL_DIR/"
# Patches against upstream HARL (1 file with 5 hunks). Re-apply is a no-op via
# `git apply --reverse --check` then `--reverse` then `--forward`; easier to
# just check whether one of the patched markers is already present.
cd "$HARL_DIR"
if ! grep -q '"ssl_2v2"' examples/train.py; then
    git apply "$INTEG/patches/harl.patch"
    echo "patches applied"
else
    echo "patches already present, skipping"
fi

echo "=== 5/5  wrapper smoke ==="
BP_DIR="$BP_DIR" "$VENV_DIR/bin/python" "$HARL_DIR/test_ssl_2v2_wrapper.py"

echo
echo "Done. Submit with:"
echo "  $VENV_DIR/bin/python $BP_DIR/submit_harl_2v2.py"
