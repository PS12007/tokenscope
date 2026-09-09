# Raw measurement data, session 6

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

## Structure

```
{"config": {...},                 # the exact invocation
 "results": {"<arm label>": {"tg": [...], "pp": [...]}},   # pooled, in order
 "blocks":  [ {same shape}, ... ]}                          # per block
```

`results` lists measurements in the order they were taken, first rep already
discarded. With `--blocks`, `results` is the concatenation of `blocks`.

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

## What is *not* here

The binaries. `bench-stock.exe`, `bench-patched.exe` and `bench-layoutctl.exe`
lived in a temp directory and are gitignored anyway. Rebuild them from
`patches/03-mulmat-chunk-threshold.patch` and `patches/04-layout-control.patch`
against `build-ts-off`, both arms in one session — see `HANDOFF.md` §3 on the
stale-binary trap, which this project has now fallen into twice.
