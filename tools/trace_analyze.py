#!/usr/bin/env python3
"""
tokenscope-analyze - turn a Chrome Trace Event file into answers.

A trace file alone is a toy. This is the part that makes it a tool.

    trace_analyze.py run.trace.json                  summary
    trace_analyze.py run.trace.json --tokens         per-token table
    trace_analyze.py run.trace.json --outliers 10    slowest tokens, with cause
    trace_analyze.py run.trace.json --layers          per-layer thread time
    trace_analyze.py run.trace.json --barriers        barrier wait: imbalance vs release
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

# Host scopes are emitted with their own scope name as the category, so the set
# is closed and known. Everything else in a trace is a graph node or a barrier.
HOST_CATS = frozenset({
    "batch-init", "sched-reserve", "kv.update", "kv.slot-search",
    "output-reserve", "ubatch", "graph-build", "graph-alloc", "set-inputs",
    "kv.find-slot",
    "graph-compute", "logits-readback", "sample", "tok.encode", "tok.decode",
})


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

        # per-token work that happens outside the token's own slice, filled
        # in by split_totals
        self.outside: dict[str, float] = {}

        # everything else, bucketed by the token ordinal recorded in args
        self.by_token = defaultdict(list)
        for e in self.events:
            if e.get("cat") in ("decode", "prefill"):
                continue
            self.by_token[e.get("args", {}).get("tok", -1)].append(e)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def split_totals(self, only_decode: bool = True):
        """Host scopes measure wall time on one thread; node scopes measure
        thread time across N workers. Returns (host, node, n_workers) so the
        report can put them against the right denominators instead of summing
        two different quantities into one meaningless percentage.

        `host` counts only scopes that lie *inside* their token's slice. Work
        that belongs to a token but happens outside `llama_context::decode` --
        sampling and detokenization, which run after decode returns -- is in
        `self.outside` instead. Charging those against the decode slice makes
        attribution exceed 100%, which is how this was found."""
        keep = {t["args"]["tok"] for t in self.decode} if only_decode else None
        host: dict[str, float] = defaultdict(float)
        outside: dict[str, float] = defaultdict(float)
        node: dict[str, float] = defaultdict(float)
        worker_tids: set = set()

        bounds = {t["args"]["tok"]: (t["ts"], t["ts"] + t["dur"])
                  for t in self.tokens}

        for tok, evs in self.by_token.items():
            if keep is not None and tok not in keep:
                continue
            lo_hi = bounds.get(tok)
            host_evs, out_evs = [], []
            node_evs = []
            for e in evs:
                if e.get("cat") not in HOST_CATS:
                    node_evs.append(e)
                elif lo_hi is not None and not (
                        lo_hi[0] - _EPS_US <= e["ts"]
                        and e["ts"] + e.get("dur", 0.0) <= lo_hi[1] + _EPS_US):
                    out_evs.append(e)
                else:
                    host_evs.append(e)

            for cat, dur in self_times(host_evs):
                host[cat] += dur
            for cat, dur in self_times(out_evs):
                outside[cat] += dur
            for e in node_evs:
                # node scopes never nest, so self-time is duration
                node[e.get("cat", "other")] += e.get("dur", 0.0)
                worker_tids.add(e.get("tid", 0))

        self.outside = dict(outside)
        return dict(host), dict(node), max(1, len(worker_tids))

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


# Two scopes that touch share an instant: the parent's end and the child's
# start are the same microsecond. In floating point they are not, and a
# comparison that is wrong by 1e-9 reparents a sibling as a child, which makes
# self-time attribution silently wrong. tokenscope therefore records nesting
# depth explicitly; this tolerance is only for traces that lack it.
_EPS_US = 1e-6


def self_times(events: list[dict]):
    """Yield (category, self_microseconds) for a set of events, subtracting
    time attributable to nested children so categories sum to wall time
    rather than to something larger than the run.

    Nesting comes from the recorded `args.depth` when present, and falls back
    to timestamp containment otherwise, so traces from other producers still
    work."""
    by_thread = defaultdict(list)
    for e in events:
        by_thread[e.get("tid", 0)].append(e)

    for evs in by_thread.values():
        evs.sort(key=lambda e: (e["ts"], -e.get("dur", 0.0)))
        stack: list[tuple[float, float, str, list[float], int]] = []
        for e in evs:
            t0 = e["ts"]
            dur = e.get("dur", 0.0)
            t1 = t0 + dur
            cat = e.get("cat", "other")
            depth = e.get("args", {}).get("depth")

            if depth is None:
                # fall back to timestamp containment, with a tolerance
                while stack and stack[-1][1] <= t0 + _EPS_US:
                    st0, st1, scat, kids, _ = stack.pop()
                    yield scat, max(0.0, (st1 - st0) - sum(kids))
            else:
                # a scope at depth d closes every open scope at depth >= d
                while stack and stack[-1][4] >= depth:
                    st0, st1, scat, kids, _ = stack.pop()
                    yield scat, max(0.0, (st1 - st0) - sum(kids))

            if stack:
                stack[-1][3].append(dur)
            stack.append((t0, t1, cat, [], depth if depth is not None else -1))

        while stack:
            st0, st1, scat, kids, _ = stack.pop()
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
        # Printed unconditionally, including "unrecorded", because the whole
        # point of F26 is that the reader was assuming a value. A field you
        # only print when it is interesting is a field nobody checks.
        print(f"    threading={m.get('threading', 'unrecorded')}  "
              f"compute_linkage={m.get('compute_linkage', 'unrecorded')}")
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

    # Host scopes run on the calling thread and measure WALL time. Node scopes
    # run on N workers in parallel and measure THREAD time. Adding them
    # together produces a percentage of several hundred, which is not a bug in
    # the trace -- it is a category error in the report. So they are reported
    # against different denominators, and labelled.
    host, node, n_workers = tr.split_totals()
    wall = sum(d)

    if host:
        grand = sum(host.values())
        print(f"\n  host scopes -- wall time on the calling thread\n")
        print(f"  {'category':<18}{'total':>12}  {'%':>6}   {'per-tok':>10}")
        print("  " + "-" * 50)
        for cat, us in sorted(host.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:<18}{fmt_us(us):>12}  {100.0 * us / wall:5.1f}%   "
                  f"{us / len(d) / 1000.0:8.3f} ms")
        acc = 100.0 * grand / wall
        print(f"\n  {acc:.1f}% of decode wall time is attributed to a host scope.")
        # A capture window (TOKENSCOPE_TOKENS) records scopes for only some
        # tokens while the token slices themselves are still recorded for the
        # whole run. Then a low percentage means "most tokens were not
        # captured", not "most of decode is uninstrumented", and saying the
        # latter would send a reader hunting for a gap that is not there.
        covered = sum(1 for t in tr.decode
                      if any(e.get("cat") in HOST_CATS
                             for e in tr.by_token.get(t["args"]["tok"], [])))
        if covered < len(d):
            print("  {} of {} decode tokens carry host scopes -- the rest were"
                  " outside".format(covered, len(d)))
            print("  the TOKENSCOPE_TOKENS capture window, so the percentage"
                  " above is of the")
            print("  whole run, not of the captured tokens.")
        elif acc < 90.0:
            print("  The remainder is time inside llama_decode that no scope"
                  " covers yet.")

    # Per-token work that runs outside llama_context::decode -- sampling and
    # detokenization happen after decode returns. It is real per-token cost,
    # but it is not part of the decode slice, and charging it there pushed
    # attribution over 100% when the sampling scopes were first added.
    if tr.outside:
        tot = sum(tr.outside.values())
        print("\n  per-token work OUTSIDE the decode slice"
              " (sampling, detokenization)\n")
        print("  {:<18}{:>12}  {:>6}   {:>10}".format(
            "category", "total", "%", "per-tok"))
        print("  " + "-" * 50)
        for cat, us in sorted(tr.outside.items(), key=lambda kv: -kv[1]):
            print("  {:<18}{:>12}  {:5.1f}%   {:8.3f} ms".format(
                cat, fmt_us(us), 100.0 * us / wall, us / len(d) / 1000.0))
        print("\n  {} on top of decode, i.e. {:.1f}% more wall time per token."
              .format(fmt_us(tot).strip(), 100.0 * tot / wall))
        print("  These run between decode calls, so they are NOT included in the")
        print("  percentages above -- llama-bench never calls them at all.")

    if node:
        grand = sum(node.values())
        budget = wall * max(1, n_workers)
        print(f"\n  graph nodes -- thread time across {n_workers} worker"
              f"{'' if n_workers == 1 else 's'}"
              f" ({grand / 1000.0:.1f} ms busy of {budget / 1000.0:.1f} ms available)\n")
        print(f"  {'category':<18}{'total':>12}  {'%':>6}   {'per-tok':>10}")
        print("  " + "-" * 50)
        for cat, us in sorted(node.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:<18}{fmt_us(us):>12}  {100.0 * us / grand:5.1f}%   "
                  f"{us / len(d) / 1000.0:8.3f} ms")
        wait = sum(v for k, v in node.items() if k == "barrier")
        if wait:
            print(f"\n  {100.0 * wait / grand:.1f}% of worker thread time is barrier wait,"
                  f" not compute.")
        print(f"  {100.0 * grand / budget:.1f}% of available thread time is inside a"
              f" node scope.")
        print("  Categories prefixed \"~\" are inferred from graph position, not"
              " from a node name.")

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

    has_subscopes = any(tr.by_token.get(e["args"]["tok"]) for e in worst)

    print(f"\nslowest {len(worst)} decode tokens (median is {med / 1000.0:.2f} ms)")
    if not has_subscopes:
        print("\n  This trace has no scopes nested inside the token slices, so the")
        print("  cause column cannot be filled in. That is a property of the trace,")
        print("  not of the tokens: re-run at a higher TOKENSCOPE_LEVEL, or with")
        print("  more instrumentation sites enabled, to attribute the excess.")
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
        if excess:
            cause = max(excess.items(), key=lambda kv: kv[1])
            why = f"{cause[0]} (+{cause[1] / 1000.0:.2f} ms)"
        else:
            why = "unattributed (no nested scopes)"

        print(f"{tok:>6}  {e['dur'] / 1000.0:8.2f}  {e['dur'] / med:8.2f}x  {why}")


# ---------------------------------------------------------------------------
# barrier decomposition
# ---------------------------------------------------------------------------

# On every worker thread a node scope and the barrier that follows it strictly
# alternate, and every thread walks the same node list in the same order. So
# the k-th barrier on each thread is the *same* barrier and the k-th node is
# the same node. Attribution here is exact, not inferred -- and the checks in
# _barrier_groups refuse the analysis rather than guess if that ever stops
# holding.

def _node_base(name: str) -> str:
    """Strip the trailing layer index that graph_get_cb appends."""
    head, sep, tail = name.rpartition("-")
    return head if sep and tail.isdigit() else name


def _barrier_groups(tr: "Trace", tok: int, n_threads: int):
    """Yield (nodes, barriers) per barrier for one token, each a list indexed
    by thread. Yields nothing at all if the trace lacks the strict
    node/barrier alternation this analysis depends on."""
    per = defaultdict(list)
    for e in tr.by_token.get(tok, []):
        if e.get("tid", -1) >= 0 and e.get("cat") not in HOST_CATS:
            per[e["tid"]].append(e)
    if len(per) != n_threads:
        return
    for tid in per:
        per[tid].sort(key=lambda e: e["ts"])

    nodes, bars = {}, {}
    for tid, evs in per.items():
        nodes[tid] = [e for e in evs if e.get("cat") != "barrier"]
        bars[tid] = [e for e in evs if e.get("cat") == "barrier"]
        # strict alternation: node, barrier, node, ..., ending on a node
        if len(nodes[tid]) != len(bars[tid]) + 1:
            return

    tids = sorted(per)
    ref = [e["name"] for e in nodes[tids[0]]]
    for t in tids:
        # if the threads disagree on the node list, "barrier k" is not one
        # barrier and every number below would be nonsense
        if [e["name"] for e in nodes[t]] != ref:
            return

    for k in range(len(bars[tids[0]])):
        yield [nodes[t][k] for t in tids], [bars[t][k] for t in tids]


def report_barriers(tr: "Trace", top: int) -> None:
    """Split barrier wait into the part caused by threads arriving at
    different times and the part spent after the last one arrived.

    FINDINGS F6 measured that 11.2% of worker thread time is barrier wait and
    deliberately declined to say how much of it was recoverable. This is the
    measurement that answers that question."""
    # A prefill-only trace (llama-bench -p N) has no decode slices, and the
    # graph there is the same graph with more rows per tensor -- which is
    # exactly the comparison F9 rests on. So fall back to prefill rather
    # than reporting nothing.
    slices = tr.decode if tr.decode else tr.prefill
    kind = "decode" if tr.decode else "prefill"
    _, node, n_threads = tr.split_totals(only_decode=bool(tr.decode))
    if not any(k == "barrier" for k in node):
        print("\n  no barrier scopes in this trace (needs TOKENSCOPE_LEVEL=3).")
        return

    tot_wait = tot_imb = 0.0
    by_cat = defaultdict(lambda: [0.0, 0.0, 0, 0])    # work, imbal, n, serial
    by_node = defaultdict(lambda: [0.0, 0.0, 0, 0.0])  # work, imbal, n, active
    biggest = (0.0, None)
    n_bar = 0
    toks = []

    for t in slices:
        tok = t["args"]["tok"]
        for nd, ba in _barrier_groups(tr, tok, n_threads):
            n_bar += 1
            if tok not in toks:
                toks.append(tok)

            arrive = [e["ts"] for e in ba]
            last = max(arrive)
            wait = sum(e.get("dur", 0.0) for e in ba)
            # every thread that arrived before the last one was idle for the
            # difference; that idleness is what better partitioning could buy
            imb = sum(last - a for a in arrive)
            tot_wait += wait
            tot_imb += imb
            if wait - imb > biggest[0]:
                biggest = (wait - imb, (tok, nd[0]["name"]))

            durs = [e.get("dur", 0.0) for e in nd]
            mx = max(durs)
            # a thread counts as busy if it took a real share of the busiest
            active = sum(1 for d in durs if d > 0.25 * mx and d > 0.5)
            work = sum(durs)

            c = by_cat[nd[0].get("cat", "other")]
            c[0] += work
            c[1] += imb
            c[2] += 1
            if active * 4 <= n_threads:
                c[3] += 1

            b = by_node[_node_base(nd[0]["name"])]
            b[0] += work
            b[1] += imb
            b[2] += 1
            b[3] += active

    if not n_bar:
        print("\n  barrier scopes are present, but node and barrier scopes do not")
        print("  strictly alternate on every thread, so barriers cannot be matched")
        print("  across threads. No decomposition is reported rather than a guess.")
        return

    over = tot_wait - tot_imb
    plural = "" if len(toks) == 1 else "s"
    print("\n  barrier decomposition -- {} threads, {} barriers over {} {}"
          " token{}\n".format(n_threads, n_bar, len(toks), kind, plural))
    print("  total barrier wait   {}   thread-time".format(fmt_us(tot_wait)))
    print("    arrival imbalance  {}   {:5.1f}%   threads idle, waiting for the"
          " last".format(fmt_us(tot_imb), 100.0 * tot_imb / tot_wait))
    print("    after last arrival {}   {:5.1f}%   release latency and spin-up"
          .format(fmt_us(over), 100.0 * over / tot_wait))

    if biggest[1] and over > 0 and biggest[0] > 0.25 * over:
        tok, nm = biggest[1]
        print("\n  A single barrier accounts for {} of the after-arrival time:"
              " token {},".format(fmt_us(biggest[0]), tok))
        print("  before \"{}\". On the FIRST token of the capture window that is"
              " tokenscope's".format(nm))
        print("  own first-touch buffer allocation, not ggml -- see FINDINGS"
              " F28, which")
        print("  measured it and moved it out of the node loop. A trace from"
              " before that")
        print("  fix carries it; a trace from after should not."
              " Excluding it, the split is")
        rest = tot_wait - biggest[0]
        print("  {:.1f}% imbalance / {:.1f}% release latency."
              .format(100.0 * tot_imb / rest,
                      100.0 * (over - biggest[0]) / rest))

    print("\n  imbalance by phase\n")
    print("  {:<14}{:>11}{:>12}{:>11}{:>7}{:>8}".format(
        "category", "work", "imbalance", "wait/work", "nodes", "serial"))
    print("  " + "-" * 63)
    for cat, (w, i, n, sr) in sorted(by_cat.items(), key=lambda kv: -kv[1][1]):
        ratio = "{:8.0f}%".format(100.0 * i / w) if w > 1e-9 else "       --"
        print("  {:<14}{:>11}{:>12}{:>11}{:>7}{:>8}".format(
            cat, fmt_us(w), fmt_us(i), ratio, n, sr))

    # The headline. Nodes whose barrier costs more than their own arithmetic
    # are not a partitioning problem you can tune away -- they are work that
    # does not divide, followed by a barrier that is paid anyway.
    tot_work = sum(w for w, _, _, _ in by_node.values())
    cheap = {k: v for k, v in by_node.items() if v[1] > v[0]}
    cheap_i = sum(v[1] for v in cheap.values())
    cheap_w = sum(v[0] for v in cheap.values())
    if cheap and tot_work > 0:
        names = sorted(cheap, key=lambda k: -cheap[k][1])[:6]
        print("\n  {} node types cost more in other threads' waiting"
              " than in their own work:".format(len(cheap)))
        print("  " + ", ".join(names) + ".")
        print("  Together {} of compute causes {} of waiting -- {:.0f}% of"
              " all imbalance".format(fmt_us(cheap_w), fmt_us(cheap_i),
                                      100.0 * cheap_i / tot_imb))
        print("  from {:.2f}% of the work.".format(100.0 * cheap_w / tot_work))

    print("\n  worst nodes by imbalance\n")
    print("  {:<16}{:>11}{:>12}{:>6}{:>14}".format(
        "node", "work", "imbalance", "n", "threads busy"))
    print("  " + "-" * 61)
    for nm, (w, i, n, act) in sorted(by_node.items(), key=lambda kv: -kv[1][1])[:top]:
        print("  {:<16}{:>11}{:>12}{:>6}{:>13.1f}".format(
            nm, fmt_us(w), fmt_us(i), n, act / n))
    print("\n  \"threads busy\" counts threads taking more than 25% of the busiest")
    print("  thread's time on that node. A value near 1 means the node is")
    print("  effectively serial and the other threads pay a barrier for nothing.")

    # Cross-check. tot_wait is summed from barriers matched across threads;
    # node["barrier"] is summed independently by split_totals over every
    # barrier event in the trace. If the two disagree, some barrier was not
    # matched and every percentage above is computed on a subset.
    claimed = node.get("barrier", 0.0)
    if claimed > 0:
        miss = abs(claimed - tot_wait) / claimed
        if miss > 0.005:
            print("\n  WARNING: matched {} of {} of barrier wait"
                  " ({:.1f}% unmatched)."
                  .format(fmt_us(tot_wait), fmt_us(claimed), 100.0 * miss))
            print("  The decomposition above covers only the matched part.")


def report_layers(tr: Trace, top: int) -> None:
    """Per-layer thread time. The layer index is carried in every node name
    (`<role>-<layer>`), so this is a grouping, not an extra measurement."""
    keep = {t["args"]["tok"] for t in tr.decode}
    n_tok = max(1, len(tr.decode))

    per_layer: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for tok, evs in tr.by_token.items():
        if tok not in keep:
            continue
        for e in evs:
            cat = e.get("cat", "other")
            if cat in HOST_CATS:
                continue
            L = e.get("args", {}).get("L")
            if L is None:
                continue
            per_layer[L][cat] += e.get("dur", 0.0)

    if not per_layer:
        print("\nno per-layer data in this trace.")
        print("Layer indices come from graph node names, which only appear at")
        print("TOKENSCOPE_LEVEL=2 or 3. Re-run at a higher level.")
        return

    cats = sorted({c for v in per_layer.values() for c in v},
                  key=lambda c: -sum(v.get(c, 0.0) for v in per_layer.values()))[:top]

    print(f"\nper-layer thread time, us/token ({n_tok} decode tokens)\n")
    head = "  " + "layer".rjust(5) + "".join(c.rjust(12) for c in cats) + "total".rjust(12)
    print(head)
    print("  " + "-" * (len(head) - 2))

    totals = []
    for L in sorted(per_layer):
        row = per_layer[L]
        tot = sum(row.values()) / n_tok
        totals.append((L, tot))
        line = f"  {L:>5}"
        for c in cats:
            line += f"{row.get(c, 0.0) / n_tok:12.1f}"
        line += f"{tot:12.1f}"
        print(line)

    # The point of a per-layer view is spotting the layer that is not like the
    # others. Say it outright instead of leaving it in the table.
    body = [(L, t) for L, t in totals if L >= 0]
    if len(body) >= 3:
        vals = sorted(t for _, t in body)
        med = pct(vals, 50)
        worst = max(body, key=lambda kv: kv[1])
        best = min(body, key=lambda kv: kv[1])
        print(f"\n  median layer {med:.1f} us/tok"
              f"   slowest L{worst[0]} {worst[1]:.1f} ({worst[1] / med:.2f}x)"
              f"   fastest L{best[0]} {best[1]:.1f} ({best[1] / med:.2f}x)")
        if worst[1] / med < 1.10:
            print("  Layers are uniform to within 10%, which is what an evenly")
            print("  partitioned homogeneous stack should look like.")


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

    # Host scopes measure wall time on the calling thread; node scopes measure
    # thread time summed across workers. The summary report puts them against
    # different denominators for that reason, and this table would invite the
    # same category error by listing them together unlabelled -- each row is
    # still compared only against itself, but "graph-compute 6808" sitting
    # above "ffn 29057" reads as a contradiction unless the units are stated.
    print("\n  per-token change, by category\n")
    print("  {:<18}{:>10}{:>10}{:>10}{:>10}".format(
        "category", "before", "after", "delta", "units"))
    print("  " + "-" * 62)
    for k in keys:
        va, vb = ca.get(k, 0.0), cb.get(k, 0.0)
        d = vb - va
        mark = "  <--" if abs(d) > 0.05 * max(pa, pb) else ""
        unit = "wall us" if k in HOST_CATS else "thread us"
        print("  {:<18}{:>10.1f}{:>10.1f}{:>+10.1f}{:>10}{}".format(
            k, va, vb, d, unit, mark))
    print("\n  wall us is time on the calling thread; thread us is summed across"
          " workers.")
    print("  They are not comparable to each other, only to themselves before"
          " and after.")


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
    ap.add_argument("--layers", nargs="?", type=int, const=6, default=None,
                    metavar="N", help="per-layer thread time, top N categories")
    ap.add_argument("--barriers", nargs="?", type=int, const=12, default=None,
                    metavar="N", help="barrier wait split into imbalance vs"
                                      " release, worst N nodes")
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
    if args.layers is not None:
        report_layers(tr, args.layers)
    if args.barriers is not None:
        report_barriers(tr, args.barriers)
    if args.outliers is not None:
        report_outliers(tr, args.outliers)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
