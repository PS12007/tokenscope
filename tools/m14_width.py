#!/usr/bin/env python3
"""M14 on committed data: does a 3-block interval's width follow a baseline shift?

FINDINGS P54 defines the sample and the three predictions; this script computes
them. No measurement is taken -- it reads ab_throughput.py JSON that carries a
`timeline` (every run since F44).

    python tools/m14_width.py            # from the repo root
"""
import itertools
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_overhead import between_block_ci, iqr  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "overhead")
NATIVE = ["f45-layout-void", "f45-null-void", "f47-layout", "f47-null"]   # 3 x 20
SIX = ["f49-layout", "f49-null", "f50-layout48", "f50-null"]              # 6 x 10
TEST = "tg64"


def load(name):
    with open(os.path.join(DATA, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def window(d, blocks, label):
    pts = [d["block_points"][TEST][i] for i in blocks]
    mean, lo, hi, _ = between_block_ci(pts)
    meds = [statistics.median(d["blocks"][i][TEST]["a"]) for i in blocks]
    shift = 100.0 * (max(meds) - min(meds)) / statistics.mean(meds)
    gate = max(100.0 * iqr(d["blocks"][i][TEST]["a"])
               / statistics.median(d["blocks"][i][TEST]["a"]) for i in blocks)
    return {"label": label, "width": hi - lo, "mean": mean,
            "shift": shift, "gate": gate, "a_med": statistics.mean(meds)}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def perm_p(x, y, rho, iters=20000, seed=0xC0FFEE):
    """one-sided permutation p for rho >= observed"""
    import random
    rng = random.Random(seed)
    y = list(y)
    hits = 0
    for _ in range(iters):
        rng.shuffle(y)
        if spearman(x, y) >= rho - 1e-12:
            hits += 1
    return hits / iters


def main():
    w = []
    for n in NATIVE:
        w.append(window(load(n), [0, 1, 2], n))
    halves = {}
    for n in SIX:
        d = load(n)
        h1 = window(d, [0, 1, 2], n + "[0-2]")
        h2 = window(d, [3, 4, 5], n + "[3-5]")
        w += [h1, h2]
        halves[n] = (h1, h2)

    print("%-24s %8s %8s %8s %8s %8s" % ("window", "a tok/s", "shift%", "gate%",
                                          "mean pp", "width pp"))
    for r in w:
        print("%-24s %8.2f %8.2f %8.2f %+8.2f %8.2f"
              % (r["label"], r["a_med"], r["shift"], r["gate"], r["mean"], r["width"]))

    W = [r["width"] for r in w]
    S = [r["shift"] for r in w]
    G = [r["gate"] for r in w]
    rs, rg = spearman(S, W), spearman(G, W)
    print()
    print("P54.1  rho(shift, width) = %+.3f   one-sided perm p = %.3f   (holds if >= +0.5)"
          % (rs, perm_p(S, W, rs)))
    print("P54.2  rho(gate,  width) = %+.3f   one-sided perm p = %.3f   (holds if > rho(shift))"
          % (rg, perm_p(G, W, rg)))
    print("       max/min width over all 12 windows: %.1fx" % (max(W) / min(W)))

    print()
    print("P54.3  half-vs-half width ratio per 6x10 run (i.i.d. null: P(>3x) = 0.20 each)")
    over = 0
    for n, (h1, h2) in halves.items():
        ratio = max(h1["width"], h2["width"]) / min(h1["width"], h2["width"])
        over += ratio > 3
        print("       %-14s %.2f / %.2f  -> %.1fx" % (n, h1["width"], h2["width"], ratio))
    print("       %d of 4 exceed 3x   (holds if >= 1; null P(>=1) = 0.59)" % over)

    # descriptive, not a prediction: every 3-of-6 subset of each 6x10 run
    print()
    print("descriptive: width range over all 20 three-block subsets of each 6x10 run")
    for n in SIX:
        d = load(n)
        ws = [window(d, list(c), "")["width"] for c in itertools.combinations(range(6), 3)]
        print("       %-14s min %.2f  median %.2f  max %.2f  -> %.1fx"
              % (n, min(ws), statistics.median(ws), max(ws), max(ws) / min(ws)))


if __name__ == "__main__":
    main()
