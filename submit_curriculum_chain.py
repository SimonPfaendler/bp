"""Send several seeds through the whole reverse curriculum on the 30-min
dev queue: every stage is a fixed number of 30-min SEGMENTS, every segment
one slurm job that starts when its predecessor finished (afterok).

    # print the plan, submit nothing
    SEEDS="101 102 103 104 105" python submit_curriculum_chain.py

    # submit
    GO=1 SEEDS="101 102 103 104 105" python submit_curriculum_chain.py

    # resume one seed from a stage (its predecessor's checkpoint must exist)
    GO=1 SEEDS="103" STAGES="L4 L5" python submit_curriculum_chain.py

Every seed runs the SAME protocol below with the CURRENT code — unlike the
seed-822 chain, which was assembled over ten days while rules, shaping and
the decision rate were still changing. 822 is the pilot, not sample #1.

Names are fixed: <TAG>_s<seed>_<stage>-<segment>, e.g. c15_s101_L5-2, so
each job knows its INIT_PATH before its predecessor has started. Nothing
is inherited from sidecars at submit time and nothing leaks in from the
shell: a segment's environment is exactly COMMON + stage + segment rules.

Within a stage, segment k+1 warm-starts from segment k WITH its replay
buffer (BUF=auto) and, on L5, at the difficulty segment k ended at
(DIFF=inherit). Across stages the buffer is dropped (BUF=off): it holds
the previous level's rewards and terminals.

ABLATE="REPEAT=1" (or "STACK=2", "ROLE=0", ...) overrides COMMON on every
segment and extends the tag (c15-REPEAT1_s101_L5-2): one changed factor,
same protocol, same seeds.
"""
import os

import submit_2v2_selfplay as sub

MANAGED = (
    "LEVEL PASS_GATE SOLO DRIBBLE SHAPING RESTARTS DIFF DIFF_THR DIFF_STEP "
    "DIFF_WIN INIT_PATH TOTAL_STEPS TIME_MIN BLUE ROLE DEF_PROB FOUL DEF_DIFF "
    "CRIT_WARM STACK REPEAT BLUE_KICK BUF TIME PARTITION SEED RUN_NAME AFTER "
    "INHERIT DROP_INIT_BUF"
).split()

# I/O layout and opponent: identical on every segment, so no boundary ever
# widens an input layer or changes the decision rate.
COMMON = dict(ROLE="1", REPEAT="4", STACK="1", BLUE="roles", BLUE_KICK="4.5")

# One segment = one 30-min dev slot. The step budget is what ends it
# (~1450 decisions/s on an H100 at REPEAT=4 -> ~21 min); TIME_MIN is only
# the safety net that still leaves room for setup and the final save.
SEGMENT = dict(TIME="00:30:00", TIME_MIN="24", TOTAL_STEPS="1800000")

# >>> Segment counts are PROVISIONAL until aligned with what the seed-822
# >>> chain actually used (python trace_chain.py <run> on the cluster).
# >>> L5 stays at 2 segments (3.6M steps): the 3-hour L5 segment's critic
# >>> diverged at 4.5M.
STAGES = [
    dict(name="L2", segs=2, LEVEL="2"),
    dict(name="L3", segs=2, LEVEL="3"),
    dict(name="L4", segs=2, LEVEL="4"),
    dict(name="L5", segs=2, LEVEL="5", DIFF="0", DIFF_THR="0.6",
         PASS_GATE="strict", SOLO="2", DRIBBLE="strict", SHAPING="team_def",
         RESTARTS="on", FOUL="on", DEF_PROB="0.25", DEF_DIFF="0",
         CRIT_WARM="60000"),
]


def main():
    seeds = [int(s) for s in os.environ.get("SEEDS", "").split()]
    if not seeds:
        raise SystemExit(__doc__)
    go = os.environ.get("GO") == "1"
    partition = os.environ.get("CHAIN_PARTITION", "dev_gpu_h100")
    ablate = dict(kv.split("=", 1) for kv in os.environ.get("ABLATE", "").split())
    unknown = set(ablate) - set(MANAGED)
    if unknown:
        raise SystemExit(f"ABLATE: unknown variable(s) {sorted(unknown)}")
    tag = os.environ.get("TAG", "c15") + "".join(
        f"-{k}{v}" for k, v in sorted(ablate.items())
    )
    only = os.environ.get("STAGES")
    stages = [s for s in STAGES if not only or s["name"] in only.split()]
    n_jobs = len(seeds) * sum(s["segs"] for s in stages)

    print(f"{'SUBMITTING' if go else 'DRY RUN (GO=1 to submit)'}: "
          f"{len(seeds)} seeds x {sum(s['segs'] for s in stages)} segments "
          f"= {n_jobs} jobs of 30 min, tag={tag}, partition={partition}, "
          f"ablate={ablate or '-'}")
    for seed in seeds:
        prev_job, prev_name = None, None
        first = STAGES.index(stages[0])
        if first > 0:   # resuming: last segment of the stage before
            p = STAGES[first - 1]
            prev_name = f"{tag}_s{seed}_{p['name']}-{p['segs']}"
        for st in stages:
            for k in range(1, st["segs"] + 1):
                name = f"{tag}_s{seed}_{st['name']}-{k}"
                env = dict(COMMON)
                env.update(SEGMENT)
                env.update({a: b for a, b in st.items() if a not in ("name", "segs")})
                env.update(ablate)
                env.update(SEED=str(seed), RUN_NAME=name, PARTITION=partition,
                           INHERIT="0", BUF="off")
                if k > 1:
                    # same stage: keep the buffer, continue the difficulty,
                    # and clean up the predecessor's buffer afterwards
                    env.update(BUF="auto", DROP_INIT_BUF="1")
                    if "DIFF" in st:
                        env["DIFF"] = "inherit"
                if prev_name:
                    env["INIT_PATH"] = f"models/{prev_name}_final.zip"
                if prev_job:
                    env["AFTER"] = str(prev_job)
                if not go:
                    show = {a: b for a, b in env.items()
                            if a not in COMMON and a not in SEGMENT
                            and a not in ("PARTITION", "INHERIT", "RUN_NAME", "SEED")}
                    print(f"  {name}: " + " ".join(f"{a}={b}" for a, b in sorted(show.items())))
                    prev_job, prev_name = f"<{name}>", name
                    continue
                for a in MANAGED:
                    os.environ.pop(a, None)
                os.environ.update(env)
                prev_job, prev_name = sub.main(), name
    if not go:
        print(f"\n  every segment also has: "
              + " ".join(f"{a}={b}" for a, b in sorted({**COMMON, **SEGMENT, **ablate}.items())))
        print("Nothing submitted. Re-run with GO=1.")


if __name__ == "__main__":
    main()
