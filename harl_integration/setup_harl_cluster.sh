#!/usr/bin/env bash
# One-shot HARL setup. Idempotent — skips already-done steps.
#
# Defaults: everything lives as siblings of the bp/ checkout that contains
# this script, and Python is auto-detected (miniforge3 in or next to bp,
# else system python3.10/3). Override any of these via env vars:
#   PY            python interpreter (must be 3.10+)
#   BP_DIR        path to bp/ checkout
#   HARL_DIR      where HARL clone goes
#   VENV_DIR      where venv_harl gets built
#   RSOCCER_DIR   where rSoccer clone goes
#
# Layout the defaults produce:
#   <workspace>/bp/         (you are here, contains this script)
#   <workspace>/HARL/       (cloned)
#   <workspace>/venv_harl/  (created)
#   <workspace>/rSoccer/    (cloned)

set -euo pipefail

# Resolve BP_DIR from this script's location (works regardless of cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BP_DIR="${BP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE="$(dirname "$BP_DIR")"

HARL_DIR="${HARL_DIR:-$WORKSPACE/HARL}"
VENV_DIR="${VENV_DIR:-$WORKSPACE/venv_harl}"
RSOCCER_DIR="${RSOCCER_DIR:-$WORKSPACE/rSoccer}"

# Auto-detect Python: miniforge3 inside or next to bp/, then system options.
if [[ -z "${PY:-}" ]]; then
    for candidate in \
        "$BP_DIR/miniforge3/bin/python" \
        "$WORKSPACE/miniforge3/bin/python" \
        "$HOME/miniforge3/bin/python" \
        "$(command -v python3.11 || true)" \
        "$(command -v python3.10 || true)" \
        "$(command -v python3 || true)"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            ver="$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
            major="${ver%.*}"; minor="${ver#*.}"
            if [[ "$major" -ge 3 && "$minor" -ge 10 ]]; then
                PY="$candidate"
                break
            fi
        fi
    done
fi
if [[ -z "${PY:-}" ]]; then
    echo "ERROR: no Python >= 3.10 found. Set PY=/path/to/python and re-run."
    exit 1
fi

HARL_COMMIT="b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
INTEG="$BP_DIR/harl_integration"

echo "BP_DIR=$BP_DIR"
echo "HARL_DIR=$HARL_DIR"
echo "VENV_DIR=$VENV_DIR"
echo "RSOCCER_DIR=$RSOCCER_DIR"
echo "PY=$PY ($("$PY" --version 2>&1))"
echo

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
# Torch first (large; cu12 nvidia libs ~2GB + torch wheel ~750MB).
$PIP install --quiet torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# HARL editable + its declared deps (tensorboard, sacred, setproctitle, …).
$PIP install --quiet -e "$HARL_DIR"
# rSoccer from upstream — PyPI ships an unrelated 1.4 fork that's API-incompatible.
if [[ ! -d "$RSOCCER_DIR" ]]; then
    git clone https://github.com/robocin/rSoccer.git "$RSOCCER_DIR"
fi
$PIP install --quiet "$RSOCCER_DIR"
$PIP install --quiet pygame stable-baselines3

echo "=== 4/5  apply ssl_2v2 integration ==="
rsync -a "$INTEG/new_files/" "$HARL_DIR/"
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
echo "  BP_DIR=$BP_DIR HARL_DIR=$HARL_DIR VENV_PY=$VENV_DIR/bin/python \\"
echo "    $VENV_DIR/bin/python $BP_DIR/submit_harl_2v2.py"
