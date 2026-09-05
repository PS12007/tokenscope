#!/usr/bin/env python3
"""
bench_overhead.py - measure what tokenscope costs, honestly.

The README claims two things: zero overhead when compiled out, and under 2%
when active. Neither claim is worth anything unmeasured, and a single
before/after pair is not a measurement.

Three configurations, because conflating them is how profilers end up quoting a
fictional number (docs/01 section 8, "Risk 3"):

  A  compiled out            TOKENSCOPE_ENABLED=OFF          true baseline
  B  compiled in, level 0    ON, TOKENSCOPE_LEVEL=0          cost of the residual branch
  C  active                  ON, TOKENSCOPE_LEVEL=1|2|3      real profiling overhead

A vs B tests "zero overhead when disabled". B vs C is the number people care
about.

Method:
  - arms are INTERLEAVED (ABCABC...), never blocked (AAABBBCCC), so thermal
    drift and background load hit every arm equally
  - the first repetition of each arm is discarded as warm-up
  - medians and IQR, not means and stdev: tok/s has a hard ceiling and a long
    slow tail, so the mean tracks outliers that are not the effect
  - the overhead figure is a bootstrap confidence interval, so the claim reads
    "1.2% [0.9, 1.6]" rather than "1.2%"
  - if the baseline arm's own spread is wider than the effect, the honest
    output is "this machine cannot resolve it", not a number

Usage:
  bench_overhead.py --model M.gguf --bin-off DIR --bin-on DIR [-n 20] [-t 8]

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time


# ---------------------------------------------------------------------------

def run_bench(exe: str, model: str, n_gen: int, n_prompt: int, threads: int,
              env_extra: dict[str, str], timeout: int = 900) -> dict[str, float]:
    """One llama-bench invocation. Returns {'pp': tok/s, 'tg': tok/s}."""
    env = dict(os.environ)
    env.update(env_extra)
    env.setdefault("TOKENSCOPE_LEVEL", "0")

    cmd = [exe, "-m", model, "-p", str(n_prompt), "-n", str(n_gen),
           "-t", str(threads), "-r", "1", "-o", "json"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{os.path.basename(exe)} failed ({p.returncode}):\n"
                           f"{p.stderr[-2000:]}")

    out: dict[str, float] = {}
    try:
        # llama-bench -o json emits a JSON array of result objects
        m = re.search(r"\[.*\]", p.stdout, re.S)
        rows = json.loads(m.group(0)) if m else []
        for r in rows:
            tps = float(r.get("avg_ts", 0.0))
            if int(r.get("n_gen", 0)) > 0:
                out["tg"] = tps
            elif int(r.get("n_prompt", 0)) > 0:
                out["pp"] = tps
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        raise RuntimeError(f"could not parse llama-bench output: {e}\n{p.stdout[-1000:]}")
    if not out:
        raise RuntimeError(f"no results parsed from:\n{p.stdout[-1000:]}")
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def iqr(xs: list[float]) -> float:
    if len(xs) < 4:
        return 0.0
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[2] - q[0]


def bootstrap_ratio_ci(base: list[float], test: list[float],
                       iters: int = 20000, conf: float = 0.95,
                       seed: int = 0xC0FFEE) -> tuple[float, float, float]:
    """CI for the percentage slowdown of `test` relative to `base`, resampling
    each arm independently. Returns (point, lo, hi) in percent."""
    rng = random.Random(seed)
    point = 100.0 * (statistics.median(base) / statistics.median(test) - 1.0)
    if not base or not test:
        return point, float("nan"), float("nan")

    samples = []
    for _ in range(iters):
        b = statistics.median([rng.choice(base) for _ in base])
        t = statistics.median([rng.choice(test) for _ in test])
        if t > 0:
            samples.append(100.0 * (b / t - 1.0))
    samples.sort()
    lo = samples[int((1 - conf) / 2 * len(samples))]
    hi = samples[int((1 + conf) / 2 * len(samples)) - 1]
    return point, lo, hi


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="measure tokenscope overhead",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin-off", required=True,
                    help="bin dir of the TOKENSCOPE_ENABLED=OFF build")
    ap.add_argument("--bin-on", required=True,
                    help="bin dir of the TOKENSCOPE_ENABLED=ON build")
    ap.add_argument("-n", "--reps", type=int, default=20,
                    help="repetitions per arm (default 20; the first is discarded)")
    ap.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--n-prompt", type=int, default=256)
    ap.add_argument("--levels", default="1,2,3",
                    help="active levels to measure (default 1,2,3)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    exe = "llama-bench.exe" if os.name == "nt" else "llama-bench"
    off = os.path.join(args.bin_off, exe)
    on = os.path.join(args.bin_on, exe)
    for p in (off, on, args.model):
        if not os.path.exists(p):
            print(f"error: not found: {p}", file=sys.stderr)
            return 1

    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    arms: list[tuple[str, str, dict[str, str]]] = [
        ("A: compiled out", off, {"TOKENSCOPE_LEVEL": "0"}),
        ("B: in, level 0", on, {"TOKENSCOPE_LEVEL": "0"}),
    ]
    for lv in levels:
        arms.append((f"C{lv}: active level {lv}", on,
                     {"TOKENSCOPE_LEVEL": str(lv), "TOKENSCOPE_OUT": ""}))

    print(f"model    {os.path.basename(args.model)}")
    print(f"threads  {args.threads}")
    print(f"workload pp{args.n_prompt} / tg{args.n_gen}")
    print(f"reps     {args.reps} per arm, interleaved, first discarded")
    print(f"arms     {len(arms)}\n")

    results: dict[str, dict[str, list[float]]] = {a[0]: {"pp": [], "tg": []} for a in arms}

    t_start = time.time()
    for rep in range(args.reps):
        for label, binary, env in arms:
            try:
                r = run_bench(binary, args.model, args.n_gen, args.n_prompt,
                              args.threads, env)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"\nerror in arm {label}, rep {rep}: {e}", file=sys.stderr)
                return 1
            if rep > 0:                                  # discard warm-up
                results[label]["pp"].append(r.get("pp", 0.0))
                results[label]["tg"].append(r.get("tg", 0.0))
            print(f"\r  rep {rep + 1}/{args.reps}  {label:<22} "
                  f"tg={r.get('tg', 0):7.2f} tok/s   ", end="", flush=True)
    print(f"\n\nelapsed {time.time() - t_start:.0f}s\n")

    # -- report -------------------------------------------------------------
    base_label = arms[0][0]
    for phase, pname in (("tg", "decode (tg)"), ("pp", "prefill (pp)")):
        base = results[base_label][phase]
        if not base or statistics.median(base) <= 0:
            continue

        med_b = statistics.median(base)
        spread = 100.0 * iqr(base) / med_b

        print(f"{pname}")
        print(f"  {'arm':<24}{'median tok/s':>14}{'IQR':>8}   overhead vs A")
        print("  " + "-" * 68)
        for label, _, _ in arms:
            xs = results[label][phase]
            if not xs:
                continue
            m = statistics.median(xs)
            spread_i = 100.0 * iqr(xs) / m if m else 0.0
            if label == base_label:
                print(f"  {label:<24}{m:>14.2f}{spread_i:>7.1f}%   {'-':>16}")
                continue
            pt, lo, hi = bootstrap_ratio_ci(base, xs)
            print(f"  {label:<24}{m:>14.2f}{spread_i:>7.1f}%   "
                  f"{pt:+6.2f}%  [{lo:+.2f}, {hi:+.2f}]")

        print(f"\n  baseline IQR is {spread:.2f}% of median.")
        if spread > 2.0:
            print("  NOTE: that is wider than the 2% budget being tested.")
            print("  This machine cannot resolve a 2% effect right now. Close")
            print("  background work, pin threads, and rerun before quoting a number.")
        print()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=2)
        print(f"raw results -> {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
