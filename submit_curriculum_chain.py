"""Send several seeds through the whole reverse curriculum, one slurm job
per stage, each stage starting when the previous one finished (afterok).

    # print the plan, submit nothing
    SEEDS="101 102 103 104 105" python submit_curriculum_chain.py

    # submit
    GO=1 SEEDS="101 102 103 104 105" python submit_curriculum_chain.py

Every seed runs the SAME protocol below with the CURRENT code — unlike the
seed-822 chain, which was assembled over ten days while rules, shaping and
the decision rate were still changing. 822 is therefore the pilot, not
sample #1; the seeds submitted here are the sample the thesis reports.

Run names are fixed (<TAG>_s<seed>_<stage>), so stage k+1 knows its
INIT_PATH before stage k has started. Nothing is inherited from sidecars
and nothing leaks in from the shell: each stage's environment is exactly
COMMON + the stage dict.

ABLATE="REPEAT=1" (or "STACK=2", "ROLE=0", ...) overrides COMMON for every
stage and appends the override to the tag, e.g. c15-REPEAT1_s101_L5 — one
changed factor, same protocol, same seeds.
"""
import os

import submit_2v2_selfplay as sub

# Env vars submit_2v2_selfplay.py reads. All are cleared before a stage is
# submitted so that the stage's environment is fully specified here.
MANAGED = (
    "LEVEL PASS_GATE SOLO DRIBBLE SHAPING RESTARTS DIFF DIFF_THR DIFF_STEP "
    "DIFF_WIN INIT_PATH TOTAL_STEPS TIME_MIN BLUE ROLE DEF_PROB FOUL DEF_DIFF "
    "CRIT_WARM STACK REPEAT BLUE_KICK BUF TIME PARTITION SEED RUN_NAME AFTER "
    "INHERIT"
).split()

# I/O layout and opponent: identical on every stage, so no stage boundary
# ever widens an input layer or changes the decision rate.
COMMON = dict(ROLE="1", REPEAT="4", STACK="1", BLUE="roles", BLUE_KICK="4.5")

# The protocol. BUF=off between stages: a buffer from another level holds
# that level's rewards and terminals. TOTAL_STEPS is the budget (decisions
# at REPEAT=4, ~87k/min on an H100); TIME is the slurm limit with slack.
#
# >>> The step budgets are PROVISIONAL until they are aligned with what
# >>> the seed-822 chain actually used (python trace_chain.py <run> on the
# >>> cluster). The L5 budget stays below the 4.5M steps at which the
# >>> 3-hour segment's critic diverged.
STAGES = [
    dict(name="L2", LEVEL="2", TOTAL_STEPS="3000000", TIME="00:50:00"),
    dict(name="L3", LEVEL="3", TOTAL_STEPS="3000000", TIME="00:50:00"),
    dict(name="L4", LEVEL="4", TOTAL_STEPS="3000000", TIME="00:50:00"),
    dict(name="L5", LEVEL="5", TOTAL_STEPS="4000000", TIME="01:05:00",
         DIFF="0", DIFF_THR="0.6", PASS_GATE="strict", SOLO="2",
         DRIBBLE="strict", SHAPING="team_def", RESTARTS="on", FOUL="on",
         DEF_PROB="0.25", DEF_DIFF="0", CRIT_WARM="60000"),
]


def main():
    seeds = [int(s) for s in os.environ.get("SEEDS", "").split()]
    if not seeds:
        raise SystemExit(__doc__)
    go = os.environ.get("GO") == "1"
    partition = os.environ.get("CHAIN_PARTITION", "gpu_h100")
    ablate = dict(kv.split("=", 1) for kv in os.environ.get("ABLATE", "").split())
    unknown = set(ablate) - set(MANAGED)
    if unknown:
        raise SystemExit(f"ABLATE: unknown variable(s) {sorted(unknown)}")
    tag = os.environ.get("TAG", "c15") + "".join(
        f"-{k}{v}" for k, v in sorted(ablate.items())
    )
    only = os.environ.get("STAGES")           # e.g. STAGES="L4 L5" to resume
    stages = [s for s in STAGES if not only or s["name"] in only.split()]

    print(f"{'SUBMITTING' if go else 'DRY RUN (GO=1 to submit)'}: "
          f"{len(seeds)} seeds x {len(stages)} stages, tag={tag}, "
          f"partition={partition}, ablate={ablate or '-'}")
    for seed in seeds:
        prev_job, prev_name = None, None
        first = STAGES.index(stages[0])
        if first > 0:   # resuming: the earlier stage's checkpoint must exist
            prev_name = f"{tag}_s{seed}_{STAGES[first - 1]['name']}"
        for st in stages:
            name = f"{tag}_s{seed}_{st['name']}"
            env = dict(COMMON)
            env.update({k: v for k, v in st.items() if k != "name"})
            env.update(ablate)
            env.update(SEED=str(seed), RUN_NAME=name, PARTITION=partition,
                       BUF="off", INHERIT="0")
            if prev_name:
                env["INIT_PATH"] = f"models/{prev_name}_final.zip"
            if prev_job:
                env["AFTER"] = str(prev_job)
            line = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
            if not go:
                print(f"  {name}: {line}")
                prev_job, prev_name = f"<job:{name}>", name
                continue
            for k in MANAGED:
                os.environ.pop(k, None)
            os.environ.update(env)
            prev_job, prev_name = sub.main(), name
    if not go:
        print("\nNothing submitted. Re-run with GO=1.")


if __name__ == "__main__":
    main()
