#!/usr/bin/env python3
"""Compare two llama-bench binaries on throughput, interleaved, with a CI.

Written for FINDINGS F24, which used it to measure a one-line ggml change at
+1.95% [+1.59, +2.35] on decode -- the first throughput result this project has
ever certified.

Two rules are baked in because breaking either produced a wrong answer once:

1. **Both binaries must be uninstrumented.** A traced token carries the
   recording cost on the token being measured. Every throughput number in this
   repo comes from a `GGML_TOKENSCOPE=OFF` build for that reason.

2. **Both binaries must be built from the same tree, in the same session.**
   F24's first run compared a binary saved three days earlier against a fresh
   one and reported +2.71% where the truth was +1.95% -- the older arm was
   missing an unrelated patch that had landed in between. Nothing looked wrong;
   the ranges separated cleanly, which made it more convincing rather than less.
   **An A/B where one arm is a binary you saved earlier is not an A/B.**
   This script cannot check that for you. Check binary timestamps against
   `git -C ../llama.cpp status` before believing the output.

Arms alternate within one session, and which arm leads flips each round, so
neither is systematically favoured by cache or thermal state.

    python tools/ab_throughput.py --a bench-stock.exe --b bench-patched.exe \\
        -m ../models/mid.gguf -t 16 -n 20

Give `--control` a workload the change cannot affect, and the run becomes much
harder to fool: F24 used prefill, which sits above the chunking threshold in
both arms. If the control certifies too, something is wrong with the comparison
rather than right with the change.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import statistics
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from bench_overhead import bootstrap_ratio_ci  # noqa: E402


def run_once(exe, model, n_prompt, n_gen, threads, reps):
    """One llama-bench invocation; returns {test_name: tokens_per_second}."""
    cmd = [exe, "-m", model, "-p", str(n_prompt), "-n", str(n_gen),
           "-t", str(threads), "-r", str(reps), "-o", "csv"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    if not rows:
        sys.exit("no CSV from:\n  %s\nstdout: %s\nstderr: %s"
                 % (" ".join(cmd), p.stdout[:400], p.stderr[:800]))
    out = {}
    for r in rows:
        name = ("pp%s" % r["n_prompt"]) if int(r["n_prompt"]) > 0 \
            else ("tg%s" % r["n_gen"])
        out[name] = float(r["avg_ts"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="interleaved throughput A/B between two llama-bench binaries")
    ap.add_argument("--a", required=True, help="baseline binary")
    ap.add_argument("--b", required=True, help="candidate binary")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-t", "--threads", type=int, default=8)
    ap.add_argument("-n", "--rounds", type=int, default=20,
                    help="interleaved rounds per arm (default 20)")
    ap.add_argument("--reps", type=int, default=3,
                    help="llama-bench -r within each round (default 3)")
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--n-prompt", type=int, default=64,
                    help="0 to skip the prefill control")
    args = ap.parse_args()

    for path in (args.a, args.b):
        if not os.path.exists(path):
            sys.exit("no such binary: %s" % path)
    if os.path.getsize(args.a) == os.path.getsize(args.b):
        print("note: the two binaries are the same size. If they are the same "
              "file, everything below is measuring noise.\n")

    res = collections.defaultdict(lambda: collections.defaultdict(list))
    arms = {"a": args.a, "b": args.b}

    for i in range(args.rounds):
        order = ["a", "b"] if i % 2 == 0 else ["b", "a"]
        for name in order:
            got = run_once(arms[name], args.model, args.n_prompt,
                           args.n_gen, args.threads, args.reps)
            for test, ts in got.items():
                res[test][name].append(ts)
        print("  round %d/%d" % (i + 1, args.rounds), flush=True)

    print()
    print("throughput, tok/s, %d interleaved rounds per arm, %d threads"
          % (args.rounds, args.threads))
    print("  a = %s" % args.a)
    print("  b = %s" % args.b)
    print()

    for test in sorted(res, key=lambda k: (not k.startswith("tg"), k)):
        a, b = res[test]["a"], res[test]["b"]
        if not a or not b:
            continue
        ma, mb = statistics.median(a), statistics.median(b)
        # bootstrap_ratio_ci returns the % by which `test` is FASTER than `base`
        # when called (base=b, test=a); call it so positive means b beat a.
        pt, lo, hi = bootstrap_ratio_ci(b, a)
        print("  %s" % test)
        print("    a  median %9.2f   min %9.2f   max %9.2f" % (ma, min(a), max(a)))
        print("    b  median %9.2f   min %9.2f   max %9.2f" % (mb, min(b), max(b)))
        print("    b vs a  %+.2f%%  [%+.2f, %+.2f]  95%% CI" % (pt, lo, hi))
        print("    %s" % ("CERTIFIED: the interval excludes zero" if lo > 0 or hi < 0
                          else "not certified: the interval contains zero"))
        print()

    print("  A certified control workload means the comparison is wrong, not that")
    print("  the change is good. Check the binaries differ only as intended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
