"""Reconstruct the lineage of a chained run from the slurm logs.

Run ON THE CLUSTER, in the bp/ directory:

    python trace_chain.py 2v2_selfplay_SAC_vsheur_dense_seed822_20260926-131614

Walks back from that run through every "Transferring policy weights from
<checkpoint>" line to the from-scratch ancestor and prints, per segment,
the configuration header the trainer printed, where it warm-started from,
whether a replay buffer was loaded, and the last logged metrics. The
result is the protocol the seed-822 chain actually followed — the env
vars of those submissions were never recorded anywhere else.

Writes the same text to chain_trace_<run>.txt. Standard library only.
"""
import glob
import os
import re
import sys

LOG_GLOB = "slurm_logs/*_log.out"
HEAD_BYTES = 200_000       # config header + first table
TAIL_BYTES = 400_000       # last tables + final save lines
NOISE = ("pygame", "Hello from", "wandb", "warn", "Warning", "Detected CPUs",
         "FutureWarning", "torch.load", "submitit")
KEYS = ("total_timesteps", "time_elapsed", "level", "difficulty",
        "success_rate", "live_success_rate", "blue_goal_rate",
        "passes_per_episode", "passes_strict_per_episode",
        "scored_after_pass_rate", "scored_after_strict_pass_rate",
        "ep_len_mean", "ep_rew_mean", "ent_coef", "critic_loss")


def head(path):
    with open(path, "rb") as f:
        return f.read(HEAD_BYTES).decode("utf-8", "replace")


def tail(path):
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - TAIL_BYTES))
        return f.read().decode("utf-8", "replace")


def build_index():
    """run name -> list of slurm logs that trained it."""
    idx = {}
    for path in glob.glob(LOG_GLOB):
        m = re.search(r"Logging to logs/([^/\s]+)", head(path))
        if m:
            idx.setdefault(m.group(1), []).append(path)
    return idx


def last_value(text, key):
    hits = re.findall(r"\|\s+%s\s+\|\s+([-+0-9.eE]+)\s+\|" % re.escape(key), text)
    return hits[-1] if hits else None


def describe(run, path, out):
    h, t = head(path), tail(path)
    pre = h.split("Logging to logs/")[0]
    cfg = [ln.strip() for ln in pre.splitlines()
           if ln.strip() and not any(n in ln for n in NOISE)]
    out.append(f"=== {run}")
    out.append(f"    log: {path}  ({os.path.getsize(path) / 1e6:.0f} MB)")
    for ln in cfg[-40:]:
        out.append(f"    | {ln}")
    metrics = {k: last_value(t, k) for k in KEYS}
    out.append("    end: " + "  ".join(f"{k}={v}" for k, v in metrics.items() if v is not None))
    first_promos = list(dict.fromkeys(re.findall(r"\[Curriculum\][^\n]*", h + t)))
    for ln in first_promos[:6]:
        out.append(f"    {ln.strip()}")
    for pat in (r"Time budget[^\n]*", r"Saved [^\n]*", r"Traceback[^\n]*",
                r"CANCELLED[^\n]*", r"DUE TO TIME LIMIT[^\n]*"):
        for ln in re.findall(pat, t)[-1:]:
            out.append(f"    {ln.strip()}")
    err = path[:-4] + ".err"
    if os.path.exists(err):
        et = tail(err)
        for pat in (r"CANCELLED[^\n]*", r"Error[^\n]*", r"Killed[^\n]*"):
            for ln in re.findall(pat, et)[-1:]:
                out.append(f"    err: {ln.strip()}")
    m = re.search(r"Transferring policy weights from (\S+)", h)
    return m.group(1) if m else None


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = os.path.basename(sys.argv[1])
    run = re.sub(r"(_final|_\d+_steps|_best)?(\.zip)?$", "", run)
    idx = build_index()
    out, seen = [f"Chain trace, newest first. {len(idx)} runs indexed."], set()
    first = run
    while run and run not in seen:
        seen.add(run)
        paths = idx.get(run)
        if not paths:
            out.append(f"=== {run}\n    (no slurm log found — trained elsewhere or log deleted)")
            break
        init = describe(run, sorted(paths)[-1], out)
        if not init:
            out.append("    init: FROM SCRATCH")
            break
        out.append(f"    init: {init}")
        run = re.sub(r"(_final|_\d+_steps|_best)?(\.zip)?$", "", os.path.basename(init))
    text = "\n".join(out)
    print(text)
    with open(f"chain_trace_{first}.txt", "w") as f:
        f.write(text + "\n")
    print(f"\nwritten to chain_trace_{first}.txt")


if __name__ == "__main__":
    main()
