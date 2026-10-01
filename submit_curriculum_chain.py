"""Send several seeds through the whole reverse curriculum on the 30-min
dev queue: every stage is a fixed number of 30-min SEGMENTS, one slurm job
each, fed into the queue as the submit limit allows.

    # show the plan and the current state; submit nothing
    SEEDS="101 102 103 104 105" python submit_curriculum_chain.py

    # submit what fits now, then keep feeding until everything is done
    GO=1 LOOP=1 SEEDS="101 102 103 104 105" nohup python submit_curriculum_chain.py > chain_feed.log 2>&1 &

The dev queue accepts only LIMIT (default 4) queued jobs per user, so the
whole chain cannot be submitted at once. The script is a FEEDER: each pass
it looks at what exists and submits the next segments that fit. It is
stateless apart from chain_state_<TAG>.json (job ids and attempt counts),
so it can be stopped and restarted at any time, and started again with
more seeds.

A segment is
    done      if models/<name>_final.zip exists,
    queued    if its recorded job id is still in squeue,
    failed    if it was submitted, is gone from the queue, and left no
              checkpoint — retried up to MAX_ATTEMPTS (2), then its seed
              is dropped and reported,
    pending   otherwise.
A segment is submitted once its predecessor is done, or queued (then with
afterok on it). A job stuck on a failed predecessor is cancelled.

ADOPT="<name>=<jobid> ..." records jobs that were submitted before this
state file existed.

Every seed runs the SAME protocol below with the CURRENT code — unlike the
seed-822 chain, which was assembled over ten days while rules, shaping and
the decision rate were still changing. 822 is the pilot, not sample #1.

Names are fixed: <TAG>_s<seed>_<stage>-<segment>, e.g. c15_s101_L5-2.
Nothing is inherited from sidecars at submit time and nothing leaks in
from the shell: a segment's environment is exactly COMMON + SEGMENT +
stage + segment rules.

Within a stage, segment k+1 warm-starts from segment k WITH its replay
buffer (BUF=auto) and, on L5, at the difficulty segment k ended at
(DIFF=inherit). Across stages the buffer is dropped (BUF=off): it holds
the previous level's rewards and terminals.

ABLATE="REPEAT=1" (or "STACK=2", "ROLE=0", ...) overrides COMMON on every
segment and extends the tag (c15-REPEAT1_s101_L5-2): one changed factor,
same protocol, same seeds.
"""
import json
import os
import subprocess
import time

import submit_2v2_selfplay as sub

MANAGED = (
    "LEVEL PASS_GATE SOLO DRIBBLE SHAPING RESTARTS DIFF DIFF_THR DIFF_STEP "
    "DIFF_WIN INIT_PATH TOTAL_STEPS TIME_MIN BLUE ROLE DEF_PROB FOUL DEF_DIFF "
    "CRIT_WARM STACK REPEAT BLUE_KICK BUF TIME PARTITION SEED RUN_NAME AFTER "
    "INHERIT DROP_INIT_BUF POOL POOL_FRAC"
).split()

# I/O layout and opponent: identical on every segment, so no boundary ever
# widens an input layer or changes the decision rate.
COMMON = dict(ROLE="1", REPEAT="4", STACK="1", BLUE="roles", BLUE_KICK="4.5")

# One segment = one 30-min dev slot. The step budget is what ends it
# (1464 decisions/s on an H100 at REPEAT=4 -> 22.8 min); TIME_MIN is the
# safety net (the seed-822 segments ran 27 min and still saved in time).
SEGMENT = dict(TIME="00:30:00", TIME_MIN="25", TOTAL_STEPS="2000000")

# Segment counts, from the seed-822 lineage (python trace_chain.py, 2026-09-30):
#
#   stage  822 segments          822 budget            822 result at stage end
#   L2     1 (REPEAT=1)          3.0M decisions        .79 strict-pass success
#   L3     1 (REPEAT=1)          3.0M                  .93 goal after strict pass
#   L4     1 (REPEAT=1)          3.0M                  .85
#   L5     9 (REPEAT=1) + 4 (4)  34.5M + 9.6M          d .03 -> .40, goal after
#                                (= 73M physics steps)  strict pass .73 -> .42
#
# 822's L5 rules were still changing during those 13 segments (roles and
# role index at #5, defensive frames and foul restart at #7, team_def at
# #8, one symmetric-payoff segment at #9, REPEAT=4 at #10), and the
# difficulty sat at .25 through five of them. The protocol here runs the
# FINAL rule set and layout from the first step, so it needs fewer:
# one segment per drill (2.0M decisions = 8M physics steps at REPEAT=4)
# and L5_SEGS (default 6) L5 segments = 12M decisions = 48M physics steps.
# No single L5 segment comes near the 4.5M steps at which the 3-hour
# segment's critic diverged; every segment boundary resets the optimizer
# and re-fits the critic first (CRIT_WARM), as in the pilot.
DRILL_SEGS = int(os.environ.get("DRILL_SEGS", "1"))
L5_SEGS = int(os.environ.get("L5_SEGS", "6"))
STAGES = [
    dict(name="L2", segs=DRILL_SEGS, LEVEL="2"),
    dict(name="L3", segs=DRILL_SEGS, LEVEL="3"),
    dict(name="L4", segs=DRILL_SEGS, LEVEL="4"),
    dict(name="L5", segs=L5_SEGS, LEVEL="5", DIFF="0", DIFF_THR="0.6",
         PASS_GATE="strict", SOLO="2", DRIBBLE="strict", SHAPING="team_def",
         RESTARTS="on", FOUL="on", DEF_PROB="0.25", DEF_DIFF="0",
         CRIT_WARM="60000"),
]

# Self-play stage, off by default. SP_SEGS=2 appends it after L5:
#   * every seed keeps learning from its own last L5 checkpoint, WITH that
#     segment's replay buffer and at the difficulty it reached;
#   * POOL_FRAC (0.5) of the envs play frozen opponents — the last L5
#     checkpoints of POOL_SEEDS (default: all five), the same pool for every
#     learner — on the open game; the other envs go on exactly as in L5
#     (heuristic, curriculum frame, defensive frames). The heuristic share
#     is the anchor: without it, self-play against one frozen copy lost the
#     heuristic within 600k decisions (success 0.78 -> 0.20).
SP_SEGS = int(os.environ.get("SP_SEGS", "0"))
if SP_SEGS > 0:
    STAGES.append({**STAGES[-1], "name": "SP", "segs": SP_SEGS,
                   "DIFF": "inherit", "keep_buffer": True, "pool": True})
POOL = None   # set in main(): comma-separated checkpoint paths


def squeue():
    """{job_id: reason} for all of this user's jobs (any partition)."""
    out = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%P|%r"],
        capture_output=True, text=True, check=True,
    ).stdout
    jobs = {}
    for ln in out.splitlines():
        jid, part, reason = (ln.split("|") + ["", ""])[:3]
        jobs[jid.strip()] = (part.strip(), reason.strip())
    return jobs


def segments_for(seed, tag, stages, ablate, partition, prev=None):
    """Ordered [(name, env, prev_name)] for one seed. `prev` is the run the
    first segment warm-starts from (BRANCH), None for a chain from scratch."""
    segs = []
    for st in stages:
        for k in range(1, st["segs"] + 1):
            name = f"{tag}_s{seed}_{st['name']}-{k}"
            env = dict(COMMON)
            env.update(SEGMENT)
            env.update({a: b for a, b in st.items()
                        if a not in ("name", "segs", "keep_buffer", "pool")})
            env.update(ablate)
            env.update(SEED=str(seed), RUN_NAME=name, PARTITION=partition,
                       INHERIT="0", BUF="off")
            if k > 1:
                # same stage: keep the buffer, continue the difficulty, and
                # clean up the predecessor's buffer afterwards
                env.update(BUF="auto", DROP_INIT_BUF="1")
                if "DIFF" in st:
                    env["DIFF"] = "inherit"
            if k == 1 and st.get("keep_buffer") and prev:
                env["BUF"] = "auto"     # same level and rules: keep the data
            if st.get("pool"):
                env.update(POOL=POOL, POOL_FRAC=os.environ.get("POOL_FRAC", "0.5"))
            if prev:
                env["INIT_PATH"] = f"models/{prev}_final.zip"
            segs.append((name, env, prev))
            prev = name
    return segs


def feed_once(chains, state, go, limit, partition, max_attempts, log):
    """One pass. Returns True while there is still something to wait for."""
    jobs = squeue() if go else {}
    in_queue = sum(1 for p, _ in jobs.values() if p == partition)
    done = lambda n: os.path.exists(f"models/{n}_final.zip")
    queued = lambda n: str(state.get(n, {}).get("job")) in jobs
    hold = set()   # submitted, gone from the queue, no checkpoint — seen once

    def cancel(name, why):
        nonlocal in_queue
        rec = state[name]
        jid = str(rec["job"])
        log(f"  {name}: job {jid} {why} -> scancel")
        subprocess.run(["scancel", jid])
        if jobs.pop(jid, (None,))[0] == partition:
            in_queue -= 1
        rec["job"] = None
        rec["attempts"] = max(0, rec.get("attempts", 1) - 1)   # not its fault

    # Per seed, in order: the first segment that is neither done nor queued
    # breaks the chain — everything queued behind it waits on a job that
    # will never succeed (slurm leaves such jobs in the queue forever,
    # holding a submit slot), so it is cancelled and resubmitted later.
    for seed, segs in chains.items():
        broken = False
        for name, _, _ in segs:
            if done(name):
                continue
            rec = state.get(name)
            if queued(name):
                if broken:
                    cancel(name, "waits on a broken chain")
                continue
            if rec and rec.get("job"):
                # Gone without a checkpoint. Believe it only on the second
                # pass: the checkpoint may not be visible on this node yet.
                rec["gone"] = rec.get("gone", 0) + 1
                if rec["gone"] < 2:
                    hold.add(name)
                    break
                log(f"  {name}: job {rec['job']} ended without a checkpoint "
                    f"(attempt {rec.get('attempts', 1)} failed)")
                rec["job"], rec["gone"] = None, 0
            broken = True

    waiting, full = bool(hold), False
    progressed = True
    while progressed:
        progressed = False
        # Fair order: the seed with the fewest jobs in the queue goes first,
        # one submission per round. Free slots therefore spread over the
        # seeds (which can then run side by side) instead of one seed
        # chaining all four slots behind its own running job.
        order = sorted(chains, key=lambda s: (
            sum(queued(n) for n, _, _ in chains[s]), s))
        for seed in order:
            segs = chains[seed]
            nxt = next(((n, e, p) for n, e, p in segs
                        if not (done(n) or queued(n))), None)
            if nxt is None:
                waiting |= any(queued(n) for n, _, _ in segs)
                continue
            name, env, prev = nxt
            rec = state.setdefault(name, {"job": None, "attempts": 0})
            if rec["attempts"] >= max_attempts:
                if not rec.get("reported"):
                    log(f"  seed {seed}: {name} failed {rec['attempts']}x — seed "
                        f"dropped (see slurm_logs; delete its entry in the "
                        f"state file to retry)")
                    rec["reported"] = True
                continue
            waiting = True
            if name in hold or (prev and not (done(prev) or queued(prev))):
                continue
            if full or in_queue >= limit:
                full = True
                continue
            env = dict(env)
            if prev and not done(prev):
                env["AFTER"] = str(state[prev]["job"])
            if not go:
                log(f"  would submit {name}"
                    + (f" after {prev}" if "AFTER" in env else ""))
                jid = f"dry:{name}"
            else:
                for a in MANAGED:
                    os.environ.pop(a, None)
                os.environ.update(env)
                try:
                    jid = str(sub.main())
                except Exception as e:      # QOSMaxSubmitJobPerUserLimit etc.
                    msg = (str(e).strip().splitlines() or [repr(e)])[0]
                    log(f"  {name}: submit refused ({msg}) — will retry")
                    full = True
                    continue
                log(f"  submitted {name} as job {jid}"
                    + (f" after {env['AFTER']}" if "AFTER" in env else "")
                    + f" (attempt {rec['attempts'] + 1})")
            rec.update(job=jid, attempts=rec["attempts"] + 1, gone=0)
            jobs[jid] = (partition, "")
            in_queue += 1
            progressed = True
            break
    return waiting


def main():
    seeds = [int(s) for s in os.environ.get("SEEDS", "").split()]
    if not seeds:
        raise SystemExit(__doc__)
    go = os.environ.get("GO") == "1"
    loop = os.environ.get("LOOP") == "1"
    limit = int(os.environ.get("LIMIT", "4"))
    poll = int(os.environ.get("POLL", "120"))
    max_attempts = int(os.environ.get("MAX_ATTEMPTS", "2"))
    partition = os.environ.get("CHAIN_PARTITION", "dev_gpu_h100")
    ablate = dict(kv.split("=", 1) for kv in os.environ.get("ABLATE", "").split())
    unknown = set(ablate) - set(MANAGED)
    if unknown:
        raise SystemExit(f"ABLATE: unknown variable(s) {sorted(unknown)}")
    tag = os.environ.get("TAG", "c15") + "".join(
        f"-{k}{v}" for k, v in sorted(ablate.items())
    )
    # BRANCH=L5: start at that stage, from the BASE_TAG run's last segment
    # of the stage before, instead of repeating identical earlier stages.
    # For ablations that only change L5 (e.g. ABLATE="DEF_PROB=0"): every
    # seed keeps its own drill checkpoint, so the arms differ in exactly
    # the ablated factor and not in the luck of a second drill run.
    branch = os.environ.get("BRANCH")
    base_tag = os.environ.get("BASE_TAG", os.environ.get("TAG", "c15"))
    stages, first_prev = STAGES, (lambda seed: None)
    if branch:
        names = [s["name"] for s in STAGES]
        if branch not in names or names.index(branch) == 0:
            raise SystemExit(f"BRANCH must be one of {names[1:]}")
        if tag == base_tag:
            raise SystemExit("BRANCH needs ABLATE or a TAG other than BASE_TAG, "
                             "otherwise it would overwrite the baseline runs")
        i = names.index(branch)
        stages, p = STAGES[i:], STAGES[i - 1]
        first_prev = lambda seed: f"{base_tag}_s{seed}_{p['name']}-{p['segs']}"
    if SP_SEGS > 0:
        global POOL
        pool_seeds = os.environ.get("POOL_SEEDS", "101 102 103 104 105").split()
        POOL = ",".join(f"models/{base_tag}_s{ps}_L5-{L5_SEGS}_final.zip"
                        for ps in pool_seeds)
        log_pool = f"pool = last L5 checkpoint of seeds {' '.join(pool_seeds)}"
    chains = {s: segments_for(s, tag, stages, ablate, partition, first_prev(s))
              for s in seeds}
    state_path = f"chain_state_{tag}.json"
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    for kv in os.environ.get("ADOPT", "").split():
        n, jid = kv.split("=", 1)
        state[n] = {"job": jid, "attempts": 1}

    def log(msg):
        print(f"{time.strftime('%m-%d %H:%M')} {msg}", flush=True)

    def save():
        if go:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=1)

    n_seg = len(next(iter(chains.values())))
    log(f"{'FEEDING' if go else 'DRY RUN (GO=1 to submit)'}: {len(seeds)} seeds "
        f"x {n_seg} segments, tag={tag}, partition={partition}, limit={limit}, "
        f"ablate={ablate or '-'}")
    if SP_SEGS > 0:
        log(f"  self-play stage: {SP_SEGS} segments, {log_pool}, "
            f"{os.environ.get('POOL_FRAC', '0.5')} of the envs")
    if not go:
        for n, env, _ in next(iter(chains.values())):
            show = {a: b for a, b in env.items()
                    if a not in COMMON and a not in SEGMENT
                    and a not in ("PARTITION", "INHERIT", "RUN_NAME", "SEED")}
            print(f"  {n}: " + " ".join(f"{a}={b}" for a, b in sorted(show.items())))
        print("  every segment also has: " + " ".join(
            f"{a}={b}" for a, b in sorted({**COMMON, **SEGMENT, **ablate}.items())))
    last_done = -1
    while True:
        waiting = feed_once(chains, state, go, limit, partition, max_attempts, log)
        save()
        n_done = sum(os.path.exists(f"models/{n}_final.zip")
                     for segs in chains.values() for n, _, _ in segs)
        if n_done != last_done:
            log(f"  {n_done}/{len(seeds) * n_seg} segments done")
            last_done = n_done
        if not (go and loop and waiting):
            break
        time.sleep(poll)
    if go and not loop and waiting:
        log("queue filled; run again (or with LOOP=1) to feed the rest")
    if not go:
        print("Nothing submitted. Re-run with GO=1 (add LOOP=1 to keep feeding).")


if __name__ == "__main__":
    main()
