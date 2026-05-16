#!/usr/bin/env bash
# One-shot HARL setup. Idempotent — skips already-done steps.
#
# Two modes, auto-selected:
#   A) CLONE mode (preferred on the cluster): if conda is found AND a base env
#      named $CLONE_FROM (default: rl_env) exists, we clone it into a new
#      conda env named $HARL_ENV_NAME (default: harl_env). The base env is
#      expected to already have torch, rsoccer-gym + rc-robosim, gymnasium,
#      pygame, sb3 etc. — we only add HARL on top.
#   B) VENV mode (local / no rl_env): we build a fresh venv at $VENV_DIR with
#      Python from $PY, then install torch 2.10+cu128, rsoccer (from git),
#      pygame, sb3, HARL editable.
#
# Override anything via env vars: PY, BP_DIR, HARL_DIR, VENV_DIR, RSOCCER_DIR,
# CLONE_FROM, HARL_ENV_NAME, FORCE_VENV_MODE=1 (to skip clone-mode probe).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BP_DIR="${BP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORKSPACE="$(dirname "$BP_DIR")"

HARL_DIR="${HARL_DIR:-$WORKSPACE/HARL}"
VENV_DIR="${VENV_DIR:-$WORKSPACE/venv_harl}"
RSOCCER_DIR="${RSOCCER_DIR:-$WORKSPACE/rSoccer}"
CLONE_FROM="${CLONE_FROM:-rl_env}"
HARL_ENV_NAME="${HARL_ENV_NAME:-harl_env}"
HARL_COMMIT="b1af98b0dbab72a2eee9d160751cd09aedbb8ce2"
INTEG="$BP_DIR/harl_integration"

# --- locate conda (used for clone-mode and for finding ODE in venv-mode) ---
CONDA_BIN=""
for c in \
    "$BP_DIR/miniforge3/bin/conda" \
    "$WORKSPACE/miniforge3/bin/conda" \
    "$HOME/miniforge3/bin/conda" \
    "$(command -v conda || true)"; do
    if [[ -n "$c" && -x "$c" ]]; then
        CONDA_BIN="$c"; break
    fi
done

# --- mode selection ---
USE_CLONE=0
if [[ -z "${FORCE_VENV_MODE:-}" && -n "$CONDA_BIN" ]]; then
    if "$CONDA_BIN" env list 2>/dev/null | awk '{print $1}' | grep -qx "$CLONE_FROM"; then
        USE_CLONE=1
    fi
fi

echo "BP_DIR=$BP_DIR"
echo "HARL_DIR=$HARL_DIR"
if [[ "$USE_CLONE" -eq 1 ]]; then
    echo "MODE=clone (base=$CLONE_FROM -> $HARL_ENV_NAME, conda=$CONDA_BIN)"
else
    echo "MODE=venv (VENV_DIR=$VENV_DIR)"
fi
echo

# ============================================================
echo "=== 1/5  HARL clone ==="
if [[ ! -d "$HARL_DIR" ]]; then
    git clone https://github.com/PKU-MARL/HARL.git "$HARL_DIR"
fi
cd "$HARL_DIR"
git fetch --quiet origin
git checkout --quiet "$HARL_COMMIT"
echo "HARL @ $(git log -1 --format='%h %s')"

# ============================================================
echo "=== 2/5  python env ==="
if [[ "$USE_CLONE" -eq 1 ]]; then
    if ! "$CONDA_BIN" env list 2>/dev/null | awk '{print $1}' | grep -qx "$HARL_ENV_NAME"; then
        echo "Cloning $CLONE_FROM -> $HARL_ENV_NAME ..."
        "$CONDA_BIN" create --yes --name "$HARL_ENV_NAME" --clone "$CLONE_FROM"
    else
        echo "$HARL_ENV_NAME already exists, reusing"
    fi
    ENV_PREFIX="$("$CONDA_BIN" env list | awk -v n="$HARL_ENV_NAME" '$1==n {print $NF}')"
    PY="$ENV_PREFIX/bin/python"
    PIP="$ENV_PREFIX/bin/pip"
else
    # Auto-detect Python for venv mode.
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
                    PY="$candidate"; break
                fi
            fi
        done
    fi
    if [[ -z "${PY:-}" ]]; then
        echo "ERROR: no Python >= 3.10 found. Set PY=/path/to/python and re-run."
        exit 1
    fi
    if [[ ! -d "$VENV_DIR" ]]; then
        "$PY" -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install --quiet --upgrade pip wheel
    fi
    PY="$VENV_DIR/bin/python"
    PIP="$VENV_DIR/bin/pip"
fi
echo "PY=$PY ($("$PY" --version 2>&1))"

# ============================================================
echo "=== 3/5  dependencies ==="
if [[ "$USE_CLONE" -eq 1 ]]; then
    # Cloned env already has torch + rsoccer + pygame + sb3 from rl_env.
    # Only HARL editable is missing; pip pulls absl-py, setproctitle, tensorboardX, etc.
    $PIP install --quiet -e "$HARL_DIR"
else
    # Fresh venv: pull torch (cu128) + HARL + rsoccer (built from git) + extras.
    $PIP install --quiet torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    $PIP install --quiet -e "$HARL_DIR"
    if [[ ! -d "$RSOCCER_DIR" ]]; then
        git clone https://github.com/robocin/rSoccer.git "$RSOCCER_DIR"
    fi
    # rc-robosim's pybind11 needs ODE + CMake legacy policy. Try conda-forge ode,
    # fall back to expecting libode-dev / module load ode.
    if [[ -n "$CONDA_BIN" ]]; then
        CONDA_PREFIX_DIR="$("$CONDA_BIN" info --base)"
        if [[ ! -f "$CONDA_PREFIX_DIR/include/ode/ode.h" ]]; then
            "$CONDA_BIN" install -y -c conda-forge ode >/dev/null 2>&1 || true
        fi
        if [[ -f "$CONDA_PREFIX_DIR/include/ode/ode.h" ]]; then
            export CMAKE_PREFIX_PATH="$CONDA_PREFIX_DIR${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
            export CPATH="$CONDA_PREFIX_DIR/include${CPATH:+:$CPATH}"
            export LIBRARY_PATH="$CONDA_PREFIX_DIR/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
        fi
    fi
    CMAKE_POLICY_VERSION_MINIMUM=3.5 $PIP install --quiet "$RSOCCER_DIR"
    $PIP install --quiet pygame stable-baselines3
fi

# ============================================================
echo "=== 4/5  apply ssl_2v2 integration ==="
rsync -a "$INTEG/new_files/" "$HARL_DIR/"
cd "$HARL_DIR"
if ! grep -q '"ssl_2v2"' examples/train.py; then
    git apply "$INTEG/patches/harl.patch"
    echo "patches applied"
else
    echo "patches already present, skipping"
fi

# ============================================================
echo "=== 5/5  wrapper smoke ==="
BP_DIR="$BP_DIR" "$PY" "$HARL_DIR/test_ssl_2v2_wrapper.py"

echo
echo "Done. Submit with:"
echo "  BP_DIR=$BP_DIR HARL_DIR=$HARL_DIR VENV_PY=$PY \\"
echo "    $PY $BP_DIR/submit_harl_2v2.py"
