#!/usr/bin/env python3
"""
tokenscope-analyze - turn a Chrome Trace Event file into answers.

A trace file alone is a toy. This is the part that makes it a tool.

    trace_analyze.py run.trace.json                  summary
    trace_analyze.py run.trace.json --tokens         per-token table
    trace_analyze.py run.trace.json --outliers 10    slowest tokens, with cause
    trace_analyze.py base.json --diff after.json     did my change help, and where

Python standard library only. No dependencies, same as the C++ side.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

class Trace:
    def __init__(self, path: str):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):                 # object form, "traceEvents" key
            raw = raw.get("traceEvents", [])

        self.events = [e for e in raw if e.get("ph") == "X"]
        self.instants = [e for e in raw if e.get("ph") == "i"]
        self.meta = {}
        for e in raw:
            if e.get("ph") == "M" and e.get("name") == "tokenscope":
                self.meta = e.get("args", {})

        # token slices: the synthetic track written by ts_flush
        self.tokens = [e for e in self.events if e.get("cat") in ("decode", "prefill")]
        self.tokens.sort(key=lambda e: e["ts"])
        self.decode = [e for e in self.tokens if e["cat"] == "decode"]
        self.prefill = [e for e in self.tokens if e["cat"] == "prefill"]

        # everything else, bucketed by the token ordinal recorded in args
        self.by_token = defaultdict(list)
        for e in self.events:
            if e.get("cat") in ("decode", "prefill"):
                continue
            self.by_token[e.get("args", {}).get("tok", -1)].append(e)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def category_totals(self, only_decode: bool = True) -> dict[str, float]:
        """Total microseconds per category. Nested scopes on the same thread are
        NOT double counted: a scope's self-time is its duration minus the time
        covered by its direct children."""
        keep = {t["args"]["tok"] for t in self.decode} if only_decode else None
        totals: dict[str, float] = defaultdict(float)
        for tok, evs in self.by_token.items():
            if keep is not None and tok not in keep:
                continue
            for cat, dur in self_times(evs):
                totals[cat] += dur
        return dict(totals)


def self_times(events: list[dict]):
    """Yield (category, self_microseconds) for a set of events, subtracting
    time attributable to nested children so categories sum to wall time
    rather than to something larger than the run."""
    by_thread = defaultdict(list)
    for e in events:
        by_thread[e.get("tid", 0)].append(e)

    for evs in by_thread.values():
        evs.sort(key=lambda e: (e["ts"], -e.get("dur", 0.0)))
        stack: list[tuple[float, float, str, list[float]]] = []
        for e in evs:
            t0 = e["ts"]
            dur = e.get("dur", 0.0)
            t1 = t0 + dur
            cat = e.get("cat", "other")

            while stack and stack[-1][1] <= t0:
                st0, st1, scat, kids = stack.pop()
                yield scat, max(0.0, (st1 - st0) - sum(kids))

            if stack:
                stack[-1][3].append(dur)
            stack.append((t0, t1, cat, []))

        while stack:
            st0, st1, scat, kids = stack.pop()
            yield scat, max(0.0, (st1 - st0) - sum(kids))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def fmt_us(us: float) -> str:
    if us >= 1000.0:
        return f"{us / 1000.0:8.2f} ms"
    return f"{us:8.1f} us"


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def report_summary(tr: Trace) -> None:
    print(f"\n=== {tr.name} ===")
    if tr.meta:
        m = tr.meta
        print(f"    clock={m.get('clock')}  level={m.get('level')}  "
              f"threads={m.get('threads')}  dropped={m.get('dropped')}")
        if m.get("dropped", 0):
            print("    WARNING: events were dropped. Totals below are a lower bound.")

    if tr.prefill:
        total = sum(e["dur"] for e in tr.prefill)
        print(f"\nprefill: {len(tr.prefill)} batch(es), {total / 1000.0:.1f} ms")

    d = [e["dur"] for e in tr.decode]
    if not d:
        print("\nno decode tokens in this trace.")
        return

    total_ms = sum(d) / 1000.0
    per_tok = sum(d) / len(d) / 1000.0
    print(f"\ndecode: {len(d)} tokens, {total_ms:.1f} ms total, "
          f"{per_tok:.2f} ms/tok ({1000.0 / per_tok:.1f} tok/s)")

    cats = tr.category_totals()
    if cats:
        grand = sum(cats.values())
        print(f"\n  {'category':<18}{'total':>12}  {'%':>6}   {'per-tok':>10}")
        print("  " + "-" * 50)
        for cat, us in sorted(cats.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:<18}{fmt_us(us):>12}  {100.0 * us / grand:5.1f}%   "
                  f"{us / len(d) / 1000.0:8.3f} ms")
        accounted = 100.0 * grand / sum(d)
        print(f"\n  {accounted:.1f}% of decode wall time is attributed to a scope.")
        if accounted < 90.0:
            print("  The remainder is time inside llama_decode that no scope covers yet.")

    print(f"\n  p50 {pct(d, 50) / 1000.0:6.2f} ms   p95 {pct(d, 95) / 1000.0:6.2f} ms   "
          f"p99 {pct(d, 99) / 1000.0:6.2f} ms   max {max(d) / 1000.0:6.2f} ms")


def report_tokens(tr: Trace, limit: int) -> None:
    print(f"\n{'token':>6}  {'ms':>8}  breakdown")
    print("-" * 60)
    for e in tr.decode[:limit]:
        tok = e["args"]["tok"]
        cats = defaultdict(float)
        for cat, dur in self_times(tr.by_token.get(tok, [])):
            cats[cat] += dur
        top = ", ".join(f"{c} {v / 1000.0:.2f}ms"
                        for c, v in sorted(cats.items(), key=lambda kv: -kv[1])[:3])
        print(f"{tok:>6}  {e['dur'] / 1000.0:8.2f}  {top}")


def report_outliers(tr: Trace, n: int) -> None:
    d = [e["dur"] for e in tr.decode]
    if not d:
        print("no decode tokens.")
        return
    med = pct(d, 50)
    worst = sorted(tr.decode, key=lambda e: -e["dur"])[:n]

    print(f"\nslowest {len(worst)} decode tokens (median is {med / 1000.0:.2f} ms)")
    print(f"\n{'token':>6}  {'ms':>8}  {'x median':>9}  dominant excess")
    print("-" * 66)
    for e in worst:
        tok = e["args"]["tok"]
        cats = defaultdict(float)
        for cat, dur in self_times(tr.by_token.get(tok, [])):
            cats[cat] += dur

        # Attribute the excess over median, not the absolute time: on a slow
        # token every category is large, and the interesting question is which
        # one is large *for this token*.
        base = defaultdict(list)
        for other in tr.decode:
            if other is e:
                continue
            for cat, dur in self_times(tr.by_token.get(other["args"]["tok"], [])):
                base[cat].append(dur)
        excess = {c: v - pct(base.get(c, [0.0]), 50) for c, v in cats.items()}
        cause = max(excess.items(), key=lambda kv: kv[1]) if excess else ("?", 0.0)

        print(f"{tok:>6}  {e['dur'] / 1000.0:8.2f}  {e['dur'] / med:8.2f}x  "
              f"{cause[0]} (+{cause[1] / 1000.0:.2f} ms)")


def report_diff(a: Trace, b: Trace) -> None:
    """The question during optimization work is always: did my change help,
    and where. Absolute totals answer neither if the runs differ in length."""
    da = [e["dur"] for e in a.decode]
    db = [e["dur"] for e in b.decode]
    if not da or not db:
        print("both traces need decode tokens to diff.")
        return

    pa, pb = sum(da) / len(da), sum(db) / len(db)
    print(f"\n=== {a.name}  ->  {b.name} ===\n")
    print(f"  tokens        {len(da):>10}   {len(db):>10}")
    print(f"  ms/token      {pa / 1000.0:>10.3f}   {pb / 1000.0:>10.3f}   "
          f"{100.0 * (pb - pa) / pa:+7.2f}%")
    print(f"  tok/s         {1e6 / pa:>10.2f}   {1e6 / pb:>10.2f}")
    for p in (50, 95, 99):
        qa, qb = pct(da, p), pct(db, p)
        print(f"  p{p:<12}{qa / 1000.0:>10.3f}   {qb / 1000.0:>10.3f}   "
              f"{100.0 * (qb - qa) / qa:+7.2f}%")

    ca = {k: v / len(da) for k, v in a.category_totals().items()}
    cb = {k: v / len(db) for k, v in b.category_totals().items()}
    keys = sorted(set(ca) | set(cb), key=lambda k: -(abs(cb.get(k, 0) - ca.get(k, 0))))
    if not keys:
        return

    print(f"\n  per-token, by category (us)\n")
    print(f"  {'category':<18}{'before':>10}{'after':>10}{'delta':>10}{'':>4}")
    print("  " + "-" * 52)
    for k in keys:
        va, vb = ca.get(k, 0.0), cb.get(k, 0.0)
        d = vb - va
        mark = "  <--" if abs(d) > 0.05 * max(pa, pb) else ""
        print(f"  {k:<18}{va:>10.1f}{vb:>10.1f}{d:>+10.1f}{mark}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="analyze a tokenscope Chrome Trace file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--diff", metavar="OTHER",
                    help="compare against a second trace")
    ap.add_argument("--tokens", nargs="?", type=int, const=40, default=None,
                    metavar="N", help="per-token table (default 40 rows)")
    ap.add_argument("--outliers", nargs="?", type=int, const=10, default=None,
                    metavar="N", help="slowest N tokens with attributed cause")
    args = ap.parse_args()

    try:
        tr = Trace(args.trace)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: {args.trace}: {e}", file=sys.stderr)
        return 1

    if args.diff:
        report_diff(tr, Trace(args.diff))
        return 0

    report_summary(tr)
    if args.tokens is not None:
        report_tokens(tr, args.tokens)
    if args.outliers is not None:
        report_outliers(tr, args.outliers)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
