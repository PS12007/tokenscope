#!/usr/bin/env python3
"""Isolate the one huge after-arrival barrier that level-3 traces contain.

FINDINGS P28. `--barriers` reports that a single barrier can hold most of the
after-arrival time in a trace, and explains it as thread-pool spin-up. This
runs the experiment that tells that explanation apart from the alternative --
that it is tokenscope's own lazy buffer allocation, paid by every worker at
once on the first token inside the capture window, and landing in the barrier
because `TS_NODE_WORK_END` reads its clock before it emits.

The discriminator is where the capture window opens:

    python tools/spinup_probe.py -m ../models/mid.gguf -t 8 \
        --windows 1:2,40:41 -n 8

  thread-pool spin-up  -> the 40:41 spike is much smaller; the pool is warm
  first-touch alloc    -> the two are the same size, wherever the window opens

Reports the largest single barrier's after-arrival time, the node it precedes,
and the trace-wide imbalance/release split, as medians and ranges over N runs
-- because one trace is one draw (F23), and this quantity is a maximum, which
is the least stable statistic there is.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import collections
import os
import statistics
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from trace_analyze import Trace, _barrier_groups  # noqa: E402


def trace_stats(path: str):
    """(biggest after-arrival barrier us, its node name, total imbalance us,
    total after-arrival us, n barriers) for one trace."""
    tr = Trace(path)
    _, _, nw = tr.split_totals(only_decode=bool(tr.decode))
    slices = tr.decode if tr.decode else tr.prefill
    tot_wait = tot_imb = 0.0
    biggest = (0.0, "", -1)
    n = 0
    for t in slices:
        tok = t["args"]["tok"]
        for nd, ba in _barrier_groups(tr, tok, nw):
            if not ba:
                continue
            n += 1
            last = max(e["ts"] for e in ba)
            wait = sum(e.get("dur", 0.0) for e in ba)
            imb = sum(last - e["ts"] for e in ba)
            tot_wait += wait
            tot_imb += imb
            if wait - imb > biggest[0]:
                biggest = (wait - imb, nd[0]["name"], tok)
    return biggest[0], biggest[1], biggest[2], tot_imb, tot_wait - tot_imb, n


def one_run(binary, model, nth, window, tmpdir, i):
    out = os.path.join(tmpdir, "sp-%s-%d-%d.trace.json"
                       % (window.replace(":", "_"), nth, i))
    env = dict(os.environ, TOKENSCOPE_LEVEL="3",
               TOKENSCOPE_TOKENS=window, TOKENSCOPE_OUT=out)
    # -n has to comfortably exceed the window, or the tokens are never reached.
    hi = int(window.split(":")[-1].split("-")[-1])
    cmd = [binary, "-m", model, "-p", "0", "-n", str(max(16, hi + 6)),
           "-t", str(nth), "-r", "1"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if not os.path.exists(out):
        sys.exit("no trace written by:\n  %s\n%s" % (" ".join(cmd), r.stderr[-2000:]))
    try:
        return trace_stats(out)
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


def summarize(label, values, unit="ms"):
    if not values:
        print("    %-22s (no data)" % label)
        return
    lo, hi = min(values), max(values)
    print("    %-22s median %8.3f %s   min %8.3f   max %8.3f   spread %5.1fx"
          % (label, statistics.median(values), unit, lo, hi,
             (hi / lo) if lo > 0 else float("inf")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-b", "--bench", default=None)
    ap.add_argument("-t", "--threads", default="8",
                    help="comma-separated thread counts (default 8)")
    ap.add_argument("--windows", default="1:2,40:41",
                    help="comma-separated TOKENSCOPE_TOKENS windows")
    ap.add_argument("-n", "--runs", type=int, default=8)
    args = ap.parse_args()

    bench = args.bench or os.path.join(
        _HERE, "..", "..", "llama.cpp", "build-ts-on", "bin",
        "llama-bench.exe" if os.name == "nt" else "llama-bench")
    bench = os.path.abspath(bench)
    if not os.path.exists(bench):
        sys.exit("no llama-bench at %s -- pass --bench" % bench)

    windows = [w.strip() for w in args.windows.split(",") if w.strip()]
    nth_list = [int(x) for x in args.threads.split(",") if x.strip()]

    big = collections.defaultdict(list)
    imb = collections.defaultdict(list)
    rel = collections.defaultdict(list)
    where = collections.defaultdict(collections.Counter)

    with tempfile.TemporaryDirectory(prefix="ts-spin-") as tmpdir:
        for i in range(args.runs):
            for nth in nth_list:
                for w in windows:
                    b_us, node, tok, t_imb, t_rel, n = one_run(
                        bench, args.model, nth, w, tmpdir, i)
                    key = (nth, w)
                    big[key].append(b_us / 1000.0)
                    imb[key].append(t_imb / 1000.0)
                    rel[key].append(t_rel / 1000.0)
                    where[key][(node, tok)] += 1
            print("  run %d/%d done" % (i + 1, args.runs), flush=True)

    print()
    print("largest single after-arrival barrier, %d runs per cell" % args.runs)
    print()
    for nth in nth_list:
        for w in windows:
            key = (nth, w)
            print("  t=%d  window=%s" % (nth, w))
            summarize("biggest barrier", big[key])
            summarize("trace imbalance", imb[key])
            summarize("trace after-arrival", rel[key])
            share = [100.0 * b / r for b, r in zip(big[key], rel[key]) if r > 0]
            if share:
                print("    that barrier is %.1f%% (median) of all after-arrival time"
                      % statistics.median(share))
            for (node, tok), c in where[key].most_common(3):
                print("    %2d/%d runs: before \"%s\" on token %d"
                      % (c, args.runs, node, tok))
            print()

    print("  A spike that does not shrink when the window opens later is not")
    print("  thread-pool spin-up. See FINDINGS P28.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
