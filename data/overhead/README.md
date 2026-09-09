# Raw measurement data, sessions 6-7

Every number in `FINDINGS.md` F30–F36 is computed from one of these files. They
are committed because the findings argue *about the intervals themselves*, and
that argument cannot be checked from a summary table — F31's whole point is that
two runs whose point estimates agreed had intervals that did not overlap.

Written by `tools/bench_overhead.py --json-out` (and `tools/ab_throughput.py
--json-out`, added in session 6 precisely because F33's first run could not be
diagnosed afterwards).

| file | finding | what it is | usable? |
|---|---|---|---|
| `f30.json` | **F30** | 6 arms, 2 pairs, 20 reps, **fixed arm order** (pre-`9ec7c82`) | yes, with the ordering caveat |
| `f31.json` | **F31** | 9 arms, 3 pairs, 20 reps, rotating order | yes |
| `f33b.json` | **F33** | `ab_throughput` A/B of F24's patch, 3 blocks × 20 rounds. The clean run; run A predates `--json-out` | yes |
| `f34.json` | **F34** | **VOID.** 870 MB free against an 840 MB model; every instrumented arm came out *faster* than its baseline | as a failure specimen only |
| `f35.json` | **F35** | **REFUSED.** Baseline IQR 3.5–4.7%, gate failed | as a failure specimen only |
| `f36.json` | **F36** | 6 arms, 3 pairs, 15 reps × **3 blocks**. The settled overhead numbers | **yes — this is the one to quote** |
| `f39a.json` | **F39** run A | `ab_throughput` A/B of the **null control**, 3 blocks × 20 rounds. Two binaries differing in one code byte, in a function this model never enters | yes |
| `f39b.json` | **F39** run B | the same pair, same protocol, a second independent three-block run. Pooled with `f39a` to six blocks — this is the pair that gives the false-positive rate | **yes — pool both** |
| `f40a.json` / `f40b.json` | **F40** | the **same null pair at 8 threads**, two three-block runs. Pool both. This is the floor to read F36 against, and it is half as wide as F39's 16-thread one | **yes — pool both** |
| `f43a-void.json` | **F43** | **VOID.** Layout arm, first attempt; A-arm 40.37 tok/s in F44's slow regime, gate 2.65% | as a failure specimen only |
| `f44-recovery.csv` | **F44** | Thirteen identical probes 55s apart. The regime switch, caught in the act | yes |
| `f49-layout.json` / `f49-null.json` | **F49** | **The M6 answer.** Layout arm **-0.08% [-0.16, +0.00]** and matched null **-0.08% [-0.20, +0.04]**, both **passing the gate** at 6 blocks x 10 rounds. Identical to two decimals | **yes — these are the ones to quote for M6** |
| `f47-layout.json` / `f47-null.json` | **F47** | The layout arm's best pair: **-0.00% [-0.18, +0.18]** against a matched null of -0.11%. Gate marginal (2.21% / 2.31%), so not certified — but it bounds the layout effect nine times below F24's +1.62% | **the ones to read for M6** |
| `f45-layout-void.json` / `f45-null-void.json` | **F45** | **VOID.** Layout arm and matched null; gates 9.37% and 5.61%. **First files carrying `timeline`** — the layout run's 50-second fast excursion is visible in it | as failure specimens, and as the `timeline` example |
| `f41a.json` / `f41b.json` | **F41** | F27's `GGML_OPENMP=OFF` vs default at 8 threads, two three-block runs. Also the specimen for **M14**: run A's blocks agree to 0.05pp and run B's to 0.37pp, giving `t` intervals 7× different in width for one quantity | **yes — pool both** |

## Structure

```
{"config": {...},                 # the exact invocation
 "results": {"<arm label>": {"tg": [...], "pp": [...]}},   # pooled, in order
 "blocks":  [ {same shape}, ... ]}                          # per block
```

`results` lists measurements in the order they were taken, first rep already
discarded. With `--blocks`, `results` is the concatenation of `blocks`.

**Since F44**, `ab_throughput.py` also writes `timeline`: one entry per
measurement with `t` (seconds from run start), `block`, `round`, `arm`, `test`
and `ts`. It exists because this machine holds two throughput regimes 12.6%
apart and switches between them unprompted, so a 25-minute run can span a switch
with nothing in the pooled medians to show it. To check a run:

```python
tl = [e for e in d["timeline"] if e["test"] == "tg64"]
fast = [e for e in tl if e["ts"] >= 44]
print(len(fast), "of", len(tl), "in the fast regime")
```

## Re-deriving a published interval

```python
import json, statistics, sys
sys.path.insert(0, "tools")
from bench_overhead import bootstrap_ratio_ci, between_block_ci

d = json.load(open("data/overhead/f36.json"))
A = "A: compiled out [static]"; C = "C3: active level 3 [static]"
pts = [100*(statistics.median(b[A]["tg"])/statistics.median(b[C]["tg"])-1)
       for b in d["blocks"]]
print(between_block_ci(pts))      # -> the +0.56% [-0.05, +1.16] in F36
```

**Quote `between_block_ci`, not `bootstrap_ratio_ci`.** The bootstrap resamples
inside one invocation and cannot see drift between them; see the banner at the
top of `FINDINGS.md`.

## Re-deriving F39's false-positive rate

```python
import json, statistics, sys
sys.path.insert(0, "tools")
from bench_overhead import between_block_ci

A = json.load(open("data/overhead/f39a.json"))
B = json.load(open("data/overhead/f39b.json"))
for test in ("tg64", "pp64"):
    pts = A["block_points"][test] + B["block_points"][test]
    print(test, between_block_ci(pts))
# tg64 -> -0.04% [-0.49, +0.42]   clean
# pp64 -> +0.59% [+0.01, +1.16]   RESOLVED, and a false positive by construction
```

Swap in `f40a`/`f40b` for the **8-thread** floor, which is the one to use for
anything measured at 8 threads (F36's overheads, all of them):

```
# tg64 -> +0.02% [-0.21, +0.26]
# pp64 -> -0.01% [-0.12, +0.11]
```

**Use the floor at your own thread count.** F39 compared F36's 8-thread numbers
against the 16-thread floor and drew a conclusion F40 refuted.

Note the shape difference: `ab_throughput.py` writes `pooled`/`blocks`/
`block_points` keyed by test name, where `bench_overhead.py` writes
`results`/`blocks` keyed by arm label.

## What is *not* here

The binaries. `bench-stock.exe`, `bench-patched.exe` and `bench-layoutctl.exe`
lived in a temp directory and are gitignored anyway. Rebuild them from
`patches/03-mulmat-chunk-threshold.patch` and `patches/04-layout-control.patch`
against `build-ts-off`, both arms in one session — see `HANDOFF.md` §3 on the
stale-binary trap, which this project has now fallen into twice.
