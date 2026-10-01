"""Training curves of a curriculum chain over all seeds.

    python plot_training_curves.py                       # tag c15, seeds 101-105
    python plot_training_curves.py --tag c15-DEF_PROB0 --stages L5

Reads logs/<tag>_s<seed>_<stage>-<k>/SAC_*/ (TensorBoard), puts the segments
of a seed end to end on one axis of cumulative decisions, and writes

    figures/<tag>_training_curriculum.{pdf,png}   whole chain, 3 panels
    figures/<tag>_training_l5.{pdf,png}           full-game stage, 4 panels
    results/<tag>_training_curves.csv             the plotted numbers (table view)
    results/<tag>_segment_ends.csv                last 10 % of every segment

One thin line per seed (colour = seed, the same in every figure) and the mean
over seeds in ink. Every logged value is already a rolling mean over the last
300 episodes; here it is only averaged into bins of BIN decisions.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

STAGES = [("L2", 1), ("L3", 1), ("L4", 1), ("L5", 6)]     # (name, segments)
SEG_STEPS = 2_000_000
BIN = 40_000
STAGE_LABEL = {"L2": "L2\npass drill", "L3": "L3\npass + finish",
               "L4": "L4\n+ opponents", "L5": "L5\nfull game, start positions randomised step by step"}
# Categorical slots 1-5 of the reference palette, fixed per seed.
SEED_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"

METRICS = {
    "strict":   ("rollout/passes_strict_per_episode", "Strict passes per episode"),
    "sasp":     ("rollout/scored_after_strict_pass_rate", "Episodes with a goal after a strict pass"),
    "success":  ("selfplay/live_success_rate", "Success rate (objective of the stage)"),
    "diff":     ("curriculum/difficulty", "Curriculum difficulty $d$"),
    "def_bg":   ("scenario_defense/blue_goal_rate", "Defensive frame: opponent scores"),
    "def_succ": ("scenario_defense/success_rate", "Defensive frame: yellow scores"),
    "cur_sasp": ("scenario_curriculum/scored_after_strict_pass_rate", "Curriculum frame: goal after a strict pass"),
    "blue":     ("selfplay/blue_goal_rate", "Opponent goals, all episodes"),
}


def load_segment(run):
    dirs = sorted(glob.glob(f"logs/{run}/SAC_*"))
    if not dirs:
        return None
    ea = EventAccumulator(dirs[-1], size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    out = {}
    for key, (tag, _) in METRICS.items():
        if tag in have:
            ev = ea.Scalars(tag)
            out[key] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def binned(steps, vals, offset, grid):
    """Mean of vals in each BIN-wide bin of the global grid (NaN if empty)."""
    idx = np.floor((steps + offset) / BIN).astype(int)
    out = np.full(len(grid), np.nan)
    for i in np.unique(idx):
        if 0 <= i < len(grid):
            out[i] = vals[idx == i].mean()
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    ap.add_argument("--stages", nargs="+", default=[s for s, _ in STAGES])
    args = ap.parse_args()
    stages = [(s, n) for s, n in STAGES if s in args.stages]
    segs = [(s, k) for s, n in stages for k in range(1, n + 1)]
    total = len(segs) * SEG_STEPS
    grid = (np.arange(total // BIN) + 0.5) * BIN
    data = {k: np.full((len(args.seeds), len(grid)), np.nan) for k in METRICS}
    ends = []
    for si, seed in enumerate(args.seeds):
        for j, (stage, k) in enumerate(segs):
            run = f"{args.tag}_s{seed}_{stage}-{k}"
            seg = load_segment(run)
            if seg is None:
                print(f"missing: logs/{run}")
                continue
            for key, (st, v) in seg.items():
                b = binned(st, v, j * SEG_STEPS, grid)
                m = ~np.isnan(b)
                data[key][si, m] = b[m]
                tail = v[st >= 0.9 * st.max()]
                ends.append((seed, f"{stage}-{k}", key, float(tail.mean())))
        print(f"seed {seed} loaded")

    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.tag}_training_curves.csv", "w") as f:
        f.write("metric,decisions," + ",".join(f"seed{s}" for s in args.seeds) + ",mean\n")
        for key in METRICS:
            mean = np.nanmean(data[key], axis=0) if np.isfinite(data[key]).any() else None
            for i, x in enumerate(grid):
                row = data[key][:, i]
                if np.isfinite(row).any():
                    f.write(f"{key},{int(x)}," + ",".join("" if np.isnan(v) else f"{v:.4f}" for v in row)
                            + f",{np.nanmean(row):.4f}\n")
    with open(f"results/{args.tag}_segment_ends.csv", "w") as f:
        f.write("seed,segment,metric,value_last_10pct\n")
        for row in ends:
            f.write("%d,%s,%s,%.4f\n" % row)

    x = grid / 1e6
    bounds, acc = [], 0
    for s, n in stages:
        bounds.append((s, acc * SEG_STEPS / 1e6, (acc + n) * SEG_STEPS / 1e6, n))
        acc += n

    def panel(ax, key, ymax=None, x0=0.0, legend=False):
        style(ax)
        for si, seed in enumerate(args.seeds):
            ax.plot(x - x0, data[key][si], color=SEED_COLOR[si % 5], lw=1.0, alpha=0.9,
                    label=f"seed {seed}", solid_capstyle="round")
        with np.errstate(all="ignore"):
            mean = np.nanmean(data[key], axis=0)
        ax.plot(x - x0, mean, color=INK, lw=2.0, label="mean of 5 seeds", solid_capstyle="round")
        ax.set_title(METRICS[key][1], loc="left", fontsize=8.5, color=INK, pad=4)
        if ymax is not None:
            ax.set_ylim(0, ymax)
        return ax

    def stage_lines(ax, x0=0.0, labels=False, only=None):
        lo, hi = ax.get_ylim()
        for s, a, b, n in bounds:
            if only and s != only:
                continue
            if a - x0 > 0:
                ax.axvline(a - x0, color=INK2, lw=0.8)
            for k in range(1, n):                       # segment restarts inside a stage
                ax.axvline(a - x0 + k * SEG_STEPS / 1e6, color=INK2, lw=0.6, ls=(0, (1, 3)))
            if labels:
                # a row of its own above the panel title
                ax.annotate(STAGE_LABEL[s], xy=((a + b) / 2 - x0, 1.17), xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=7.5, color=INK2, linespacing=1.2,
                            annotation_clip=False)

    # ---------------- figure 1: the whole chain ----------------
    if len(stages) > 1:
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.6), sharex=True)
        for ax, key, ymax in zip(axes, ("strict", "sasp", "success"), (1.15, 1.0, 1.0)):
            panel(ax, key, ymax)
        for i, ax in enumerate(axes):
            stage_lines(ax, labels=(i == 0))
            ax.set_xlim(0, total / 1e6)
        axes[-1].set_xlabel("Decisions (millions, 10 Hz), all stages end to end. Dotted lines: restart of a 30-minute segment.",
                            fontsize=8, color=INK2)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7.5, handlelength=1.6,
                   columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(left=0.07, right=0.985, top=0.885, bottom=0.115, hspace=0.32)
        for ext in ("pdf", "png"):
            fig.savefig(f"figures/{args.tag}_training_curriculum.{ext}", dpi=200)
        plt.close(fig)

    # ---------------- figure 2: the full-game stage ----------------
    l5 = [b for b in bounds if b[0] == "L5"]
    if l5:
        _, a, b, n = l5[0]
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharex=True)
        for ax, key, ymax in zip(axes.ravel(), ("diff", "cur_sasp", "def_bg", "def_succ"), (1.02, 1.0, 1.02, 1.0)):
            panel(ax, key, ymax, x0=a)
            stage_lines(ax, x0=a, only="L5")
            ax.set_xlim(0, b - a)
        fig.text(0.525, 0.085, "Decisions in L5 (millions). Dotted lines: restart of a 30-minute segment.",
                 ha="center", va="center", fontsize=8, color=INK2)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7.5, handlelength=1.6,
                   columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(left=0.065, right=0.985, top=0.94, bottom=0.165, hspace=0.30, wspace=0.14)
        for ext in ("pdf", "png"):
            fig.savefig(f"figures/{args.tag}_training_l5.{ext}", dpi=200)
        plt.close(fig)
    print("written: figures/, results/")


if __name__ == "__main__":
    main()
