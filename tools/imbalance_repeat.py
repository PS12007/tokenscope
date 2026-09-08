#!/usr/bin/env python3
"""Per-node arrival imbalance over N identical runs, because one trace is one draw.

FINDINGS F23 is the reason this exists. A single trace put `ffn_up`/`ffn_down`
arrival imbalance at 0.554; a second trace of the same model on the same machine
with the same binary put it at 1.086 — a difference large enough to invert the
conclusion being drawn from it. Six runs gave 0.330. Twelve gave 0.557. Along the
way a six-run median invented a "28-thread anomaly" that twelve runs showed did
not exist, after a prediction had already been written and committed to explain
it.

Per-node imbalance spreads by up to **8.5x** between identical runs at 8 threads
on the machine F23 was measured on. It is far better behaved at 16 (1.3-1.5x).
Neither is a number you can read off one trace.

`bench_overhead.py` has interleaved arms and bootstrap confidence intervals since
session 1, and the project applied that discipline to throughput and not to
numbers read out of traces — which got treated as exact because the *tracing* is
exact. The tracing is exact. The machine underneath it is not. This is the same
discipline for trace-derived per-node quantities.

    python tools/imbalance_repeat.py -m model.gguf -t 8,16 -n 12
    python tools/imbalance_repeat.py -m model.gguf -t 28 -n 12 --nodes ffn_up,ffn_out
    python tools/imbalance_repeat.py -m model.gguf -t 20 -n 12 -C 0x0FFF5555
    python tools/imbalance_repeat.py -m model.gguf -t 8,16 -n 12 --ratio ffn_up/ffn_out

`--ratio A/B` is the form to prefer for a claim. A ratio of two nodes from the
same trace divides out the machine being globally fast or slow that minute, which
is most of what moves between runs -- and F23's whole result is a ratio, because
its P23.1 (a level, not a ratio) held and meant almost nothing.

Requires an instrumented `llama-bench` (GGML_TOKENSCOPE=ON). It runs the binary
N x len(threads) times, so it is minutes, not seconds.
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

from trace_analyze import Trace, _barrier_groups, _node_base  # noqa: E402

DEFAULT_NODES = ("ffn_up", "ffn_gate", "ffn_out", "attn_out", "Qcur", "Kcur")


def node_stats(path: str):
    """work, arrival imbalance and total barrier wait per node base name.

    Uses trace_analyze's own barrier matching, so these are the same numbers
    `--barriers` reports rather than a second implementation of them. Release
    latency -- the part of the wait that happens after the last thread has
    arrived -- is `wait - imbalance`, exactly as `--barriers` computes it.

    The synthetic node `ALL` aggregates every matched barrier in the trace, so
    a whole-trace figure can be repeated across runs the same way a per-node
    one can. P27.2 predicts a whole-trace quantity.
    """
    tr = Trace(path)
    _, _, nw = tr.split_totals(only_decode=bool(tr.decode))
    slices = tr.decode if tr.decode else tr.prefill
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    for t in slices:
        for nd, ba in _barrier_groups(tr, t["args"]["tok"], nw):
            if not ba:
                continue
            last = max(e["ts"] for e in ba)
            work = sum(e.get("dur", 0.0) for e in nd)
            imb  = sum(last - e["ts"] for e in ba)
            wait = sum(e.get("dur", 0.0) for e in ba)
            for key in (_node_base(nd[0]["name"]), "ALL"):
                a = agg[key]
                a[0] += work
                a[1] += imb
                a[2] += wait
    return agg


# metric -> (human name, function of the [work, imbal, wait] triple)
METRICS = {
    "imbalance": ("arrival imbalance / work", lambda a: a[1] / a[0]),
    "release":   ("release latency / work",   lambda a: (a[2] - a[1]) / a[0]),
    "wait":      ("total barrier wait / work", lambda a: a[2] / a[0]),
}


def one_run(binary, model, nth, tokens, mask, tmpdir, i):
    out = os.path.join(tmpdir, "rep-%d-%d.trace.json" % (nth, i))
    env = dict(os.environ,
               TOKENSCOPE_LEVEL="3",
               TOKENSCOPE_TOKENS=tokens,
               TOKENSCOPE_OUT=out)
    cmd = [binary, "-m", model, "-p", "0", "-n", "16", "-t", str(nth), "-r", "1"]
    if mask:
        cmd += ["-C", mask, "--cpu-strict", "1"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if not os.path.exists(out):
        sys.exit("no trace written by:\n  %s\n%s" % (" ".join(cmd), r.stderr[-2000:]))
    try:
        return node_stats(out)
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


def summarize(label, values):
    if not values:
        return
    lo, hi = min(values), max(values)
    spread = (hi / lo) if lo > 0 else float("inf")
    print("    %-12s median %6.3f   min %6.3f   max %6.3f   spread %5.1fx"
          % (label, statistics.median(values), lo, hi, spread))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="per-node arrival imbalance over N identical runs")
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-b", "--bench", default=None,
                    help="path to an instrumented llama-bench "
                         "(default: ../llama.cpp/build-ts-on/bin/llama-bench.exe)")
    ap.add_argument("-t", "--threads", default="8",
                    help="comma-separated thread counts (default 8)")
    ap.add_argument("-n", "--runs", type=int, default=12,
                    help="repetitions per thread count (default 12; F23 found "
                         "6 too few to tell a 3x effect from noise)")
    ap.add_argument("--nodes", default=",".join(DEFAULT_NODES))
    ap.add_argument("--ratio", default=None, metavar="A/B",
                    help="also report node A's imbalance/work divided by node "
                         "B's -- the form to prefer for a claim")
    ap.add_argument("--metric", default="imbalance", choices=sorted(METRICS),
                    help="which half of the barrier decomposition to repeat: "
                         "imbalance (default, F23's quantity), release (the "
                         "part after the last arrival, which is what a barrier "
                         "IMPLEMENTATION changes -- P27.2), or wait (both)")
    ap.add_argument("--tokens", default="10:11",
                    help="TOKENSCOPE_TOKENS capture window (default 10:11)")
    ap.add_argument("-C", "--cpu-mask", default=None,
                    help="hex affinity mask, passed to llama-bench with "
                         "--cpu-strict 1")
    args = ap.parse_args()

    bench = args.bench or os.path.join(
        _HERE, "..", "..", "llama.cpp", "build-ts-on", "bin",
        "llama-bench.exe" if os.name == "nt" else "llama-bench")
    bench = os.path.abspath(bench)
    if not os.path.exists(bench):
        sys.exit("no llama-bench at %s -- pass --bench" % bench)

    nth_list = [int(x) for x in args.threads.split(",") if x.strip()]
    nodes = [x.strip() for x in args.nodes.split(",") if x.strip()]
    ratio = args.ratio.split("/") if args.ratio else None
    if ratio and len(ratio) != 2:
        sys.exit("--ratio takes the form A/B")
    metric_name, metric = METRICS[args.metric]

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    ratios = collections.defaultdict(list)

    with tempfile.TemporaryDirectory(prefix="ts-imb-") as tmpdir:
        for i in range(args.runs):
            for nth in nth_list:
                a = one_run(bench, args.model, nth, args.tokens,
                            args.cpu_mask, tmpdir, i)
                for n in nodes:
                    if n in a and a[n][0] > 0:
                        per[nth][n].append(metric(a[n]))
                if ratio and all(r in a and a[r][0] > 0 for r in ratio):
                    den = metric(a[ratio[1]])
                    if den > 0:
                        ratios[nth].append(metric(a[ratio[0]]) / den)
            print("  run %d/%d done" % (i + 1, args.runs), flush=True)

    print()
    print("%s, %d identical runs per thread count" % (metric_name, args.runs))
    if args.cpu_mask:
        print("affinity mask %s, --cpu-strict 1" % args.cpu_mask)
    print()
    for nth in nth_list:
        print("  t=%d" % nth)
        for n in nodes:
            summarize(n, per[nth][n])
        if ratio and ratios[nth]:
            summarize("%s/%s" % (ratio[0], ratio[1]), ratios[nth])
        print()

    print("  Report the range, not just the median. F23's ranges are wide enough")
    print("  that two arms overlapping is the normal case, and non-overlap is the")
    print("  claim worth making.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
