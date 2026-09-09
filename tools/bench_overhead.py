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
  - and the order ROTATES each round (ABC, BCA, CAB, ...), so no arm is
    permanently first. A fixed order leaves every sub-round-length transient
    landing on the same arms; F30 caught one doing exactly that
  - the first repetition of each arm is discarded as warm-up
  - medians and IQR, not means and stdev: tok/s has a hard ceiling and a long
    slow tail, so the mean tracks outliers that are not the effect
  - the overhead figure is a bootstrap confidence interval, so the claim reads
    "1.2% [0.9, 1.6]" rather than "1.2%". Read that interval as RESOLVED
    WITHIN THIS RUN, never as reproducible: it resamples one invocation's
    runs, drift makes them correlated, and a naive bootstrap on correlated
    samples reads narrower than the truth. This project called such intervals
    "certified" for five sessions before F31 put two non-overlapping ones for
    the same quantity side by side
  - if the baseline arm's own spread is wider than the effect, the honest
    output is "this machine cannot resolve it", not a number
  - and with --blocks N the whole round-robin runs N times, reporting the
    spread of each block's point estimate. The bootstrap resamples INSIDE one
    invocation and knows nothing about the next one; F31 found two of its
    intervals, for one quantity on unrebuilt binaries, that did not overlap

N pairs, one invocation (F25). A "pair" is an off/on build directory couple --
one baseline and one instrumented binary built the same way. Passing more than
one pair puts *every* arm of *every* pair in the same round-robin, which is the
only way to compare two builds' overheads against each other.

F25 is why this exists. The shared and static overheads were measured as two
separate invocations thirteen minutes apart; the machine's baseline IQR moved
by a factor of two between them, and the comparison drowned. The harness had
interleaved arms since session 1 -- but the arms it interleaved were the ones
inside a single run, and the comparison being asked for spanned two runs. The
lesson generalises past this tool: **the arms are whatever you are comparing,
and if your comparison spans two invocations you have not interleaved it.**

A note on the bootstrap. Arms are resampled independently, not paired by round,
even though the round-robin does pair them. Paired resampling would credit the
design for the drift it cancels and give tighter intervals; independent
resampling is the conservative choice and is what every number in FINDINGS was
computed with, so it stays the default. An interval here is wider than the
design earns, never narrower.

Usage:
  # one pair, as before
  bench_overhead.py --model M.gguf --bin-off DIR --bin-on DIR [-n 20] [-t 8]

  # two pairs, interleaved together -- answers "does the shared build cost more?"
  bench_overhead.py --model M.gguf -n 20 -t 8 --levels 3 --no-level0 \\
      --pair static=../llama.cpp/build-ts-off/bin,../llama.cpp/build-ts-on/bin \\
      --pair shared=../llama.cpp/build-ts-shared-off/bin/Release,../llama.cpp/build-ts-shared/bin/Release

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


def free_ram_bytes():
    """Best-effort available physical memory, or None if it cannot be read."""
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MS()
            st.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
            return None
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def preflight_ram(model_path: str, force: bool = False, margin: float = 1.5) -> None:
    """Refuse to start a measurement that the machine cannot hold.

    F34 was run with 870 MB free against an 840 MB model and produced a table
    in which the *instrumented* build was faster than the compiled-out one in
    all three pairs -- physically impossible, and one pair's between-block
    interval excluded zero, so it looked resolved. The launch command printed
    the free-memory figure and started the run in the same breath, which is no
    check at all. Decode is bandwidth-bound (F14): once the weights do not stay
    resident, every number is about paging.
    """
    try:
        need = os.path.getsize(model_path)
    except OSError:
        return
    free = free_ram_bytes()
    if free is None:
        print("note: could not read free memory; skipping the RAM pre-flight\n")
        return
    gb = 1024.0 ** 3
    print(f"free RAM  {free / gb:.2f} GiB against a {need / gb:.2f} GiB model")
    if free >= need * margin:
        return
    msg = (f"free memory ({free / gb:.2f} GiB) is under {margin:g}x the model "
           f"({need / gb:.2f} GiB). The weights will not stay resident and "
           f"every number will be about paging, not about the code.")
    if force:
        print(f"WARNING: {msg}\n  --force given, continuing anyway.\n")
        return
    raise SystemExit(f"error: {msg}\n"
                     f"       Close what is using the memory, or pass --force.")

# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def iqr(xs: list[float]) -> float:
    if len(xs) < 4:
        return 0.0
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[2] - q[0]


# Student's t, two-sided 95%, by degrees of freedom. Small-sample values matter
# here because B is 3 or 5, never 30, and using 1.96 at B=3 would understate the
# interval by more than a factor of two -- which is the exact failure this whole
# addition exists to stop repeating.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    if df > 30:
        return 1.960
    return _T95[max(k for k in _T95 if k < df)]


def between_block_ci(points: list[float], conf: float = 0.95
                     ) -> tuple[float, float, float, float]:
    """Interval on the mean of per-block point estimates. Returns
    (mean, lo, hi, spread).

    This is the interval to quote. The bootstrap resamples inside ONE block and
    therefore cannot see drift between blocks; F31 found two of its intervals,
    for one quantity on unrebuilt binaries, that did not overlap. Each block is
    a fresh pass over every arm, so the spread of their point estimates carries
    the between-pass variation the bootstrap is blind to -- and it carries the
    within-block noise too, since each block's estimate contains it.

    With B blocks the interval is mean +- t(B-1) * s / sqrt(B). At B=3 that t is
    4.303, which is deliberately unflattering: three passes do not pin a number
    down, and an interval that pretends otherwise is how this project got here.
    """
    good = [x for x in points if x == x]                # drop NaN
    n = len(good)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = statistics.mean(good)
    spread = max(good) - min(good)
    if n < 2:
        return mean, float("nan"), float("nan"), spread
    half = t95(n - 1) * statistics.stdev(good) / (n ** 0.5)
    return mean, mean - half, mean + half, spread

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


def bootstrap_overhead_diff_ci(base_x: list[float], test_x: list[float],
                               base_y: list[float], test_y: list[float],
                               iters: int = 20000, conf: float = 0.95,
                               seed: int = 0xC0FFEE
                               ) -> tuple[float, float, float]:
    """CI for (overhead of Y) - (overhead of X), in percentage points.

    Each of the four arms is resampled independently, the two overheads are
    formed inside each iteration, and the difference is what gets the interval.
    Differencing two published point estimates would not give a CI at all;
    this is the comparison F25's P25.1 needed and could not make, because its
    four arms were spread over two invocations.
    """
    rng = random.Random(seed)

    def overhead(b: list[float], t: list[float]) -> float:
        mt = statistics.median(t)
        return 100.0 * (statistics.median(b) / mt - 1.0) if mt > 0 else float("nan")

    point = overhead(base_y, test_y) - overhead(base_x, test_x)
    if not all((base_x, test_x, base_y, test_y)):
        return point, float("nan"), float("nan")

    samples = []
    for _ in range(iters):
        bx = statistics.median([rng.choice(base_x) for _ in base_x])
        tx = statistics.median([rng.choice(test_x) for _ in test_x])
        by = statistics.median([rng.choice(base_y) for _ in base_y])
        ty = statistics.median([rng.choice(test_y) for _ in test_y])
        if tx > 0 and ty > 0:
            samples.append(100.0 * (by / ty - 1.0) - 100.0 * (bx / tx - 1.0))
    samples.sort()
    lo = samples[int((1 - conf) / 2 * len(samples))]
    hi = samples[int((1 + conf) / 2 * len(samples)) - 1]
    return point, lo, hi


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

class Arm:
    """One measured configuration: a binary plus the environment it runs under.

    `pair` names the off/on build couple it belongs to and `kind` is "A", "B"
    or "C<level>", so the report can find each pair's own baseline without
    parsing labels back apart.
    """

    def __init__(self, label: str, pair: str, kind: str, binary: str,
                 env: dict[str, str]):
        self.label = label
        self.pair = pair
        self.kind = kind
        self.binary = binary
        self.env = env


def parse_pairs(args, exe: str) -> list[tuple[str, str, str]]:
    """Return [(name, off_exe, on_exe)]. Accepts --pair and the older
    --bin-off/--bin-on, which is kept because docs/02 and bootstrap.py both
    print that invocation."""
    pairs: list[tuple[str, str, str]] = []

    if args.bin_off or args.bin_on:
        if not (args.bin_off and args.bin_on):
            raise SystemExit("error: --bin-off and --bin-on go together")
        pairs.append(("", os.path.join(args.bin_off, exe),
                      os.path.join(args.bin_on, exe)))

    for spec in args.pair or []:
        name, _, dirs = spec.partition("=")
        if not dirs:
            raise SystemExit(f"error: --pair wants NAME=OFF_DIR,ON_DIR, got {spec!r}")
        # comma, not colon: Windows build dirs are absolute and contain "C:"
        off_dir, sep, on_dir = dirs.partition(",")
        if not sep:
            raise SystemExit(f"error: --pair wants NAME=OFF_DIR,ON_DIR, got {spec!r}")
        pairs.append((name.strip(), os.path.join(off_dir.strip(), exe),
                      os.path.join(on_dir.strip(), exe)))

    if not pairs:
        raise SystemExit("error: give --bin-off/--bin-on, or one or more --pair")
    names = [p[0] for p in pairs]
    if len(set(names)) != len(names):
        raise SystemExit(f"error: duplicate pair names: {names}")
    return pairs


def build_arms(pairs: list[tuple[str, str, str]], levels: list[int],
               want_level0: bool) -> list[Arm]:
    """Round-robin order is the order of this list, so pairs are interleaved
    arm-by-arm rather than block-by-block: A_x, C_x, A_y, C_y, A_x, ..."""
    arms: list[Arm] = []
    for name, off, on in pairs:
        tag = f" [{name}]" if name else ""
        arms.append(Arm(f"A: compiled out{tag}", name, "A", off,
                        {"TOKENSCOPE_LEVEL": "0"}))
        if want_level0:
            arms.append(Arm(f"B: in, level 0{tag}", name, "B", on,
                            {"TOKENSCOPE_LEVEL": "0"}))
        for lv in levels:
            arms.append(Arm(f"C{lv}: active level {lv}{tag}", name, f"C{lv}", on,
                            {"TOKENSCOPE_LEVEL": str(lv), "TOKENSCOPE_OUT": ""}))
    return arms


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="measure tokenscope overhead",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin-off", help="bin dir of the TOKENSCOPE_ENABLED=OFF build")
    ap.add_argument("--bin-on", help="bin dir of the TOKENSCOPE_ENABLED=ON build")
    ap.add_argument("--pair", action="append", metavar="NAME=OFF_DIR,ON_DIR",
                    help="an off/on build couple to measure; repeatable. Every "
                         "arm of every pair is interleaved together")
    ap.add_argument("-n", "--reps", type=int, default=20,
                    help="repetitions per arm (default 20; the first is discarded)")
    ap.add_argument("-t", "--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--n-prompt", type=int, default=256)
    ap.add_argument("--levels", default="1,2,3",
                    help="active levels to measure (default 1,2,3)")
    ap.add_argument("--no-level0", action="store_true",
                    help="drop the B arm. Arms multiply by pair -- two pairs "
                         "at all three levels is ten runs a round; this trades "
                         "the residual-branch control for finishing")
    ap.add_argument("--blocks", type=int, default=1, metavar="N",
                    help="run the whole round-robin N times as separate blocks "
                         "and report the spread of each block's point estimate "
                         "next to the bootstrap CI. F31: the bootstrap is a "
                         "WITHIN-invocation interval, and two of them for one "
                         "quantity came out non-overlapping. Costs N x the time")
    ap.add_argument("--no-rotate", action="store_true",
                    help="keep one fixed arm order every round (the pre-F30 "
                         "protocol). Rotation is on by default; see F30")
    ap.add_argument("--force", action="store_true",
                    help="run even if free memory is under 1.5x the model size. "
                         "F34 was run at 870 MB free against an 840 MB model and "
                         "reported the instrumented build as FASTER in all three "
                         "pairs")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    exe = "llama-bench.exe" if os.name == "nt" else "llama-bench"
    pairs = parse_pairs(args, exe)

    missing = [p for p in [args.model] + [b for _, o, n in pairs for b in (o, n)]
               if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"error: not found: {p}", file=sys.stderr)
        return 1

    preflight_ram(args.model, args.force)

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    arms = build_arms(pairs, levels, not args.no_level0)

    print(f"model    {os.path.basename(args.model)}")
    print(f"threads  {args.threads}")
    print(f"workload pp{args.n_prompt} / tg{args.n_gen}")
    print(f"reps     {args.reps} per arm, interleaved, first discarded")
    print(f"arms     {len(arms)} across {len(pairs)} pair(s)\n")

    # per-block, so the between-block spread can be reported; `results` is the
    # pooled view every existing report path already expects.
    blocks: list[dict[str, dict[str, list[float]]]] = []
    results: dict[str, dict[str, list[float]]] = {
        a.label: {"pp": [], "tg": []} for a in arms}

    t_start = time.time()
    for block in range(args.blocks):
        cur: dict[str, dict[str, list[float]]] = {
            a.label: {"pp": [], "tg": []} for a in arms}
        blocks.append(cur)
        for rep in range(args.reps):
            # Rotate the order each round (F30). Interleaving stops drift
            # landing on one arm, but with a FIXED order every transient
            # shorter than a round lands on the same arms every time -- in
            # F30's own run one extra warm-up round hit all three arms of the
            # pair that ran first, and cost that pair its certification. An
            # interleave with a fixed order is a Latin square with one row.
            k = (rep + block) % len(arms)
            order = arms if args.no_rotate else arms[k:] + arms[:k]
            for arm in order:
                try:
                    r = run_bench(arm.binary, args.model, args.n_gen,
                                  args.n_prompt, args.threads, arm.env)
                except (RuntimeError, subprocess.TimeoutExpired) as e:
                    print(f"\nerror in arm {arm.label}, block {block}, "
                          f"rep {rep}: {e}", file=sys.stderr)
                    return 1
                if rep > 0:                              # discard warm-up
                    for ph in ("pp", "tg"):
                        cur[arm.label][ph].append(r.get(ph, 0.0))
                        results[arm.label][ph].append(r.get(ph, 0.0))
                tag = f"block {block + 1}/{args.blocks}  " if args.blocks > 1 else ""
                print(f"\r  {tag}rep {rep + 1}/{args.reps}  {arm.label:<30} "
                      f"tg={r.get('tg', 0):7.2f} tok/s   ", end="", flush=True)
    print(f"\n\nelapsed {time.time() - t_start:.0f}s\n")

    # -- report -------------------------------------------------------------
    # Each pair is scored against its OWN compiled-out arm; a pair's overhead
    # is only meaningful relative to the same build without tokenscope in it.
    base_of = {a.pair: a.label for a in arms if a.kind == "A"}

    for phase, pname in (("tg", "decode (tg)"), ("pp", "prefill (pp)")):
        print(f"{pname}")
        print(f"  {'arm':<32}{'median tok/s':>14}{'IQR':>8}   overhead vs A")
        print("  " + "-" * 76)

        floors: dict[str, float] = {}
        pooled_floors: dict[str, float] = {}
        for name, _, _ in pairs:
            base = results[base_of[name]][phase]
            if not base or statistics.median(base) <= 0:
                continue
            med_b = statistics.median(base)
            pooled_floors[name] = 100.0 * iqr(base) / med_b
            # F39/D9: gate on the WITHIN-block spread, not the pooled one. The
            # gate asks "is the baseline's own noise wider than the effect",
            # which is a within-invocation question. Pooling N blocks folds the
            # between-block drift into it as well -- and that drift is already
            # reported, by the t interval that exists to carry it. Gating on the
            # pooled figure therefore charges a --blocks run twice for the same
            # variance and can print "this machine cannot resolve it" about a
            # run whose interval already says so honestly. F39 measured the gap:
            # 2.02% pooled against 1.46% within-block on the same 60 samples.
            # The worst block is used, not the median, so the gate stays
            # conservative. With --blocks 1 this is identical to the old path.
            per_block = []
            for blk in blocks:
                xs = blk[base_of[name]][phase]
                if xs and len(xs) >= 4 and statistics.median(xs) > 0:
                    per_block.append(100.0 * iqr(xs) / statistics.median(xs))
            floors[name] = max(per_block) if per_block else pooled_floors[name]

            for arm in (a for a in arms if a.pair == name):
                xs = results[arm.label][phase]
                if not xs:
                    continue
                m = statistics.median(xs)
                spread_i = 100.0 * iqr(xs) / m if m else 0.0
                if arm.kind == "A":
                    print(f"  {arm.label:<32}{m:>14.2f}{spread_i:>7.1f}%   {'-':>16}")
                    continue
                pt, lo, hi = bootstrap_ratio_ci(base, xs)
                print(f"  {arm.label:<32}{m:>14.2f}{spread_i:>7.1f}%   "
                      f"{pt:+6.2f}%  [{lo:+.2f}, {hi:+.2f}]")
            if len(pairs) > 1:
                print()

        for name, spread in floors.items():
            tag = f" [{name}]" if name else ""
            if len(blocks) > 1:
                print(f"  baseline IQR{tag} is {spread:.2f}% of median "
                      f"(worst block; {pooled_floors[name]:.2f}% pooled, which "
                      f"also carries between-block drift -- see F39).")
            else:
                print(f"  baseline IQR{tag} is {spread:.2f}% of median.")
        gate_failed = any(s > 2.0 for s in floors.values())
        if gate_failed:
            print("  NOTE: that is wider than the 2% budget being tested.")
            print("  This machine cannot resolve a 2% effect right now. Close")
            print("  background work, pin threads, and rerun before quoting a number.")
        elif len(blocks) < 2:
            # F31. An interval that clears the gate is RESOLVED WITHIN THIS RUN
            # and nothing more. Two of them, for one quantity on unrebuilt
            # binaries two hours apart, came out non-overlapping. The project
            # called such intervals "certified" for five sessions and the word
            # promised reproducibility the method never tested.
            print("  Intervals above are RESOLVED WITHIN THIS RUN. That is not")
            print("  reproducibility: the bootstrap resamples these runs only, and")
            print("  drift makes them correlated, so it reads narrower than the")
            print("  truth. Re-run with --blocks 3 before quoting an interval.")
        print()

        # -- between-block spread ------------------------------------------
        # F31: the bootstrap resamples inside one invocation, so it answers
        # "if I redrew these runs from the same afternoon". Two such intervals
        # for one quantity, on unrebuilt binaries two hours apart, came out
        # NON-OVERLAPPING. Blocks estimate the part the bootstrap cannot see.
        if len(blocks) > 1:
            print("  each block's own point estimate, and the interval to quote")
            print(f"  {'arm':<28}" + "".join(f"{'blk ' + str(i + 1):>8}"
                                             for i in range(len(blocks)))
                  + f"{'spread':>9}{'mean':>8}{'95% CI (t)':>20}")
            print("  " + "-" * 88)
            worst = 0.0
            impossible = 0
            for arm in arms:
                if arm.kind == "A":
                    continue
                base_lbl = base_of[arm.pair]
                pts = []
                for blk in blocks:
                    b, t = blk[base_lbl][phase], blk[arm.label][phase]
                    mt = statistics.median(t) if t else 0.0
                    pts.append(100.0 * (statistics.median(b) / mt - 1.0)
                               if b and mt > 0 else float("nan"))
                mean, lo, hi, sp = between_block_ci(pts)
                worst = max(worst, sp if sp == sp else 0.0)
                if hi == hi and hi < 0:
                    impossible += 1
                ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo == lo else "--"
                print(f"  {arm.label:<28}"
                      + "".join(f"{x:>+8.2f}" for x in pts)
                      + f"{sp:>8.2f}{mean:>+8.2f}{ci:>20}")
            print(f"\n  widest between-block spread: {worst:.2f}pp.")
            if gate_failed:
                # F34. Blocks defend against drift BETWEEN passes. They do
                # nothing about contamination spanning every pass: three
                # blocks inside one bad window agree with each other and are
                # consistently wrong, which reads exactly like a resolved
                # result. The baseline-IQR gate is the defence, and it fired.
                print("  DO NOT QUOTE ANY OF THE ABOVE. The baseline gate failed,")
                print("  so all three blocks sit inside one bad window and agree")
                print("  with each other while being wrong together. Blocks see")
                print("  drift BETWEEN passes, not contamination across all of")
                print("  them. Fix the machine and rerun.")
            else:
                print("  QUOTE THE t INTERVAL, not the bootstrap one above it.")
                print("  The bootstrap resamples within a block and cannot see")
                print("  drift between them; this column can. At 3 blocks t is")
                print("  4.303, so the interval is wide on purpose.")
            if impossible:
                print(f"  IMPLAUSIBLE: {impossible} arm(s) came out FASTER than")
                print("  their own compiled-out baseline, interval excluding zero.")
                print("  Instrumentation does not speed code up. Something else moved.")
            print()
        # -- pair vs pair ---------------------------------------------------
        # The point of one invocation: these four arms shared a round-robin,
        # so this difference is not a between-run comparison the way F25's was.
        if len(pairs) > 1:
            ref = pairs[0][0]
            kinds = [k for k in (a.kind for a in arms if a.pair == ref) if k != "A"]
            print(f"  overhead difference vs pair [{ref}], percentage points")
            print(f"  {'comparison':<32}{'difference':>24}")
            print("  " + "-" * 76)
            for name, _, _ in pairs[1:]:
                for kind in kinds:
                    lx = next((a.label for a in arms
                               if a.pair == ref and a.kind == kind), None)
                    ly = next((a.label for a in arms
                               if a.pair == name and a.kind == kind), None)
                    if not lx or not ly:
                        continue
                    bx, tx = results[base_of[ref]][phase], results[lx][phase]
                    by, ty = results[base_of[name]][phase], results[ly][phase]
                    if not all((bx, tx, by, ty)):
                        continue
                    pt, lo, hi = bootstrap_overhead_diff_ci(bx, tx, by, ty)
                    print(f"  {kind + ': ' + name + ' - ' + ref:<32}"
                          f"{pt:+9.2f}pp  [{lo:+.2f}, {hi:+.2f}]")

            # The uninstrumented builds differ only in how they were linked, so
            # this one is a straight throughput comparison and not an overhead.
            print(f"\n  {'compiled-out arms only':<32}{'slower than ' + (ref or 'first'):>24}")
            print("  " + "-" * 76)
            for name, _, _ in pairs[1:]:
                bx = results[base_of[ref]][phase]
                by = results[base_of[name]][phase]
                if not bx or not by:
                    continue
                pt, lo, hi = bootstrap_ratio_ci(bx, by)
                print(f"  {'A: ' + name + ' vs ' + ref:<32}"
                      f"{pt:+9.2f}%   [{lo:+.2f}, {hi:+.2f}]")
            print()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "results": results,
                       "blocks": blocks}, f, indent=2)
        print(f"raw results -> {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
