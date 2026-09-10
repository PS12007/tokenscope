# 09 — Machine 2 (Windows/GCC): what to change, what to keep in mind, and all the data

Companion to [`08-windows-gcc-bringup.md`](08-windows-gcc-bringup.md). 08 is the
narrative of the session; **this file is the actionable part** — the defects to
fix, the things that will bite the next machine, the complete dataset, and the
analysis of what the numbers mean.

Raw artifacts live in [`../results/08-windows-gcc/`](../results/08-windows-gcc/):

| file | what it is |
|---|---|
| `summary.json` | every headline number, structured, with machine-1 references alongside for cross-machine comparison |
| `scaling.csv` | all 20 thread-scaling points, both threading paths, all three sweeps |
| `overhead-t8.json`, `overhead-t2.json` | full `bench_overhead.py` output, every rep of every block |
| `f27-t8.json`, `f27-t12.json` | full `ab_throughput.py` output |

**Nothing in `src/`, `patches/`, `tools/` or `scripts/` was modified this
session.** Everything below is a proposal, not a change already made.

> **Status, session 9 (machine 1, 2026-09-10): A1-A6 are all fixed.** Each was
> reproduced on machine 1 before being changed. A1: `bootstrap.py` applies only
> 01+02, lists the arms as skipped, takes `--arm NAME` to apply one on purpose,
> and **refuses** a tree where `nth * 4` does not occur twice -- tested on a
> fresh local clone, including one deliberately contaminated with 03. A2/A4/A6:
> doc 07 carries the threshold grep, the `\bts_` predicate and `python-yaml`.
> A3: `bench_overhead.py` accepts `-m`, and doc 07 drops the redundant `-n 20`.
> A5: `#ifndef NOMINMAX`; a GCC rebuild on machine 1 emits zero warnings and
> selftest + dlltest pass.

---

## Part 1 — Things to CHANGE (defects, in priority order)

### A1 — `scripts/bootstrap.py` applies experiment patches to the baseline

**Severity: highest. This silently corrupts every measurement taken from a
freshly bootstrapped tree.**

`apply_patches()` globs `patches/*.patch` and applies all of them. Patches 03-06
are hand-applied experiment arms by their own headers — 03 says outright *"NOT
APPLIED to the tree by bootstrap.py"*. On a clean clone the result is a tree with
F24's `nth * 2` treatment compiled into what everything downstream treats as
stock, plus two "do not merge" layout arms, followed by a hard `SystemExit` when
05 and 06 conflict (they are alternative arms on the same region, so they can
never coexist).

**Why it survived seven sessions:** machine 1 has had `../llama.cpp` since before
03-06 existed, and `apply_patches` skips what is already applied. A fresh clone
is the only path to the bug.

**Suggested shape of a fix** (not applied):

- Give bootstrap an explicit baseline set — `BASELINE_PATCHES = ["01-instrument.patch",
  "02-name-attn-output.patch"]` — and apply only those.
- Move 03-06 to `patches/arms/`, or have bootstrap skip any patch whose header
  contains `do not merge` / `NOT APPLIED`.
- Add `--arm <name>` for deliberately applying one.

The 8-modified-file count in HANDOFF is the arithmetic check that 01+02 is the
intended set: 01 touches 7 files, 02 adds `src/llama-graph.cpp`, plus the
untracked `ggml/src/tokenscope/` = the documented 9 entries.

### A2 — the documented tree check cannot detect A1

`git status --short` returns **9 entries for the contaminated tree too**, because
03/04/05 modify files 01 already modified. Doc 07 step 2 relies on that count
alone.

**Fix:** put HANDOFF's threshold grep into doc 07's step 2, and treat it as the
real gate:

```bash
grep -c 'nchunk0 \* nchunk1 < nth \* 4' ../llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c
# MUST print 2. Anything else means an experiment arm is in your baseline.
```

### A3 — doc 07 section 4(a)'s overhead command cannot run

```bash
python tools/bench_overhead.py -m ../models/mid.gguf -n 20 -t 8 ... --reps 10
```

- **`-m` does not exist** on `bench_overhead.py`; the option is `--model` (`-m`
  is `ab_throughput.py`'s spelling). The command exits having measured nothing.
- **`-n 20` and `--reps 10` are the same option** (`-n, --reps`), so `-n 20` is
  silently discarded.

**Fix:** `--model`, and drop the redundant `-n 20`. Optionally give
`bench_overhead.py` an `-m` alias so the two tools agree.

### A4 — the zero-overhead check is MSVC-specific

```bash
nm -C build-ts-off/bin/llama-bench | grep -i tokenscope   # docs say: expect NOTHING
nm -C build-ts-on/bin/llama-bench  | grep -i tokenscope   # docs say: expect symbols
```

On GCC **both return nothing**, because the symbols are named `ts_*`. A reader
following doc 07 concludes the instrumented build failed. `nm` sees 37,920
symbols in the ON arm; 84 of them match `ts_`, and 0 in the OFF arm.

**Fix:** use a predicate that works on both toolchains:

```bash
nm -C build-ts-on/bin/llama-bench  | grep -cE '\bts_'   # expect > 0
nm -C build-ts-off/bin/llama-bench | grep -cE '\bts_'   # expect 0
# portable fallback that needs no nm at all:
strings -a build-ts-off/bin/llama-bench | grep -ic tokenscope   # expect 0
```

### A5 — `NOMINMAX` redefinition warning under MinGW

```
src/tokenscope.cpp:27:11: warning: 'NOMINMAX' redefined
```

MinGW's `bits/os_defines.h` defines it already; the guard at `tokenscope.cpp:27`
does not test for it. Cosmetic — a `#ifndef NOMINMAX` would close it. **Left
alone deliberately this session.**

### A6 — doc 07's Arch prerequisites are incomplete

`gguf-py` imports `yaml`. Doc 07 installs `python python-numpy`, so step 3 dies
with `ModuleNotFoundError: No module named 'yaml'`. Add `python-yaml`.

---

## Part 2 — Things to KEEP IN MIND (traps that are not bugs)

### K1 — the calibration constants in the docs are machine-1 constants

`--min-baseline 46`, "40.9 / 43.5 / 46.0 tok/s", "decode wanders 38.6-46.0" are
all **machine 1**. This machine decodes at **8-24 tok/s** depending on thread
count and threading path. Using the documented values either refuses to start or
never triggers. Probe first, then set `--min-baseline` from the probe.

### K2 — thread count is not a free parameter here; it changes the noise floor

The baseline IQR is a strong function of `-t`, because past the scaling peak the
measurement is mostly barrier wait:

| threads | A-arm decode IQR (worst block) | pooled |
|---|---|---|
| 8 | 7.50% | 5.34% |
| 2 | **4.02%** | **1.88%** |

**Measure at the scaling peak, not at a fixed `-t 8`.** Machine 1's habit of
using 8 threads is right for machine 1 because 8 is near its peak; on this
machine 8 is deep in the collapse.

### K3 — any scaling or barrier claim must name its threading path

After W8 (below), "this machine scales to 1.21x" is meaningless without
"on libgomp". The same silicon reaches 1.77x on the ggml threadpool.

### K4 — do not keep the working tree inside OneDrive

The first clone landed in a synced folder. A bootstrapped `../llama.cpp` and an
840 MB model there get uploaded and rescanned mid-run — a contamination source
the "nothing else may run" rule cannot see. Everything here lives under `C:\cs\`.

### K5 — Windows cannot see this chassis' temperature either

`MSAcpi_ThermalZoneTemperature` returns *Not supported* on the Inspiron 15 3520,
exactly as on machine 1. M10 is a **Windows/WMI pattern, not one bad laptop**.
Do not expect another Windows box to fix it.

### K6 — the git trap still applies, and bit differently here

HANDOFF warns never to `git checkout <file>` or `git apply -R` inside
`../llama.cpp`. Rebuilding the tree after A1 needs a *deliberate* full reset —
`git reset --hard` back to the pin, re-copy sources, re-apply 01 and 02 — not a
per-file undo.

---

## Part 3 — The data

### 3.1 Environment

```
Dell Inspiron 15 3520 | i7-1255U | 10 cores / 12 logical | 2 P (SMT) + 8 E | 15.7 GB
Windows 11 Home 10.0.26200 | GCC 16.1.0 UCRT64 (libgomp) | CMake 4.4.3 | Ninja 1.13.2
Python 3.13.14 | numpy 2.4.3 | llama.cpp 4d9176092 + patches 01,02
model: mid.gguf, 840.3 MB, 220.24 M params, 24L / 768e / 12h / 4kv / 8192v
```

### 3.2 Self-test

| | machine 2 | machine 1 |
|---|---|---|
| per-scope cost | **62.8 ns** | 52.8 ns |
| selftest | PASS, 205,143 events, 0 dropped | PASS |
| dlltest (F22 guard) | **PASS — first GCC build anywhere** | PASS (MSVC) |

62.8 / 52.8 = **1.19×**, in line with a 15 W part against a 55 W one. Since the
README puts the clock read at ~90% of a scope's cost, this is mostly a slower
`steady_clock` read.

### 3.3 Thread scaling — full data

`llama-bench -p 0 -n 64 -r 5`, tg64 decode. Full raw in `scaling.csv`.

**OpenMP / libgomp** (`build-ts-off`), three sweeps:

| threads | A (asc) | B (desc) | C (paired) | mean | vs t=1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.09 ± 1.04 | 18.90 ± 0.36 | 18.99 ± 1.08 | 19.33 | 1.00× |
| **2** | 23.78 ± 0.35 | 23.07 ± 0.66 | 23.05 ± 0.51 | **23.30** | **1.21× peak** |
| 4 | 16.03 ± 0.26 | 15.79 ± 0.87 | 15.57 ± 0.63 | 15.80 | 0.82× |
| 8 | 10.67 ± 1.46 | 10.01 ± 0.41 | 10.28 ± 0.75 | 10.32 | 0.53× |
| 12 | 8.67 ± 0.36 | 8.83 ± 0.37 | 8.97 ± 0.50 | 8.82 | 0.46× |

**ggml threadpool** (`build-noomp-off`), paired with sweep C:

| threads | tok/s | vs t=1 | vs libgomp at same t |
|---:|---:|---:|---:|
| 1 | 18.37 ± 0.53 | 1.00× | **1.00× — agree** |
| 2 | 29.95 ± 0.19 | 1.63× | 1.30× |
| 4 | 26.09 ± 0.40 | 1.42× | 1.68× |
| **8** | **32.44 ± 0.46** | **1.77× peak** | **3.16×** |
| 12 | 21.29 ± 7.61 | 1.16× | 2.37× |

### 3.4 F27 — `GGML_OPENMP=OFF`, full data

Arms verified distinct: `ldd` shows `libgomp-1.dll` on A and nothing on B.
Positive = B (no OpenMP) faster.

| | threads | A median | B median | **quoted** | blocks | spread |
|---|---:|---:|---:|---|---|---:|
| decode | 8 | 11.54 | 35.19 | **+204.87% [+200.16, +209.58]** | 206.69 / 202.91 / 205.00 | 3.78 pp |
| decode | 12 | 10.32 | 32.74 | **+216.71% [+175.17, +258.26]** | 234.12 / 200.77 / 215.24 | 33.35 pp |
| prefill | 8 | 278.02 | 586.19 | +113.67% [+95.86, +131.49] | 121.95 / 109.59 / 109.47 | 12.48 pp |
| prefill | 12 | 276.69 | 647.09 | +133.79% [+128.00, +139.59] | 135.15 / 131.10 / 135.13 | 4.05 pp |

All four resolved (between-block interval excludes zero). One artifact left in
rather than trimmed: the t=12 prefill B arm has a `min` of 9.66 against a median
of 647 — a single collapsed run.

**Machine 1 for the same flag: -2.15% at 8 threads, -54.17% at 28.**

### 3.5 Overhead — NOT QUOTABLE, gate failed at both thread counts

The harness rejected both runs:

> `baseline IQR [static] is 7.50% of median` … `wider than the 2% budget being
> tested` … **`DO NOT QUOTE ANY OF THE ABOVE`**

Recorded for calibration only:

| | t=8 | t=2 |
|---|---|---|
| workload | pp256 / tg128 | pp256 / tg128 |
| elapsed | 2325 s | 1313 s |
| A-arm decode median | 11.19 | 23.93 |
| baseline IQR worst / pooled | 7.50% / 5.34% | **4.02% / 1.88%** |
| decode level 0 | +1.28% [-0.48, +2.66] | +0.05% [-0.37, +0.44] |
| decode level 3 | +1.50% [-0.03, +3.14] | **+0.28% [-0.16, +0.60]** |
| decode level 3, per block | +1.56 [-0.34, +3.46] | **-0.02 [-0.76, +0.73]** |
| block spread | 4.97 pp | 2.00 pp |
| prefill level 3 | +0.58% [-0.75, +1.18] | -0.35% [-1.22, +0.48] |
| prefill IQR worst / pooled | 5.90% / 3.03% | 11.98% / 3.12% |

Per-block estimates, decode level 3:

```
t=8:  +1.74  +0.75  +1.80  +3.45  -1.52  +3.15
t=2:  -0.10  +0.34  -0.17  +0.45  +0.69  -1.31
```

**Machine 1 for reference: +0.56% [-0.05, +1.16] at 8 threads.**

---

## Part 4 — Analysis

### 4.1 F27's reversal is the real result, and the t=1 control is what makes it one

A 200%+ effect on a noisy laptop invites the obvious objection: the binaries
differ in more than the flag. Three things answer it.

- **At one thread the two binaries are the same speed** — 18.99 vs 18.37, inside
  each other's error bars. With one thread there is no barrier to take, so if
  `GGML_OPENMP` were dragging in a codegen or link difference, it would show
  here. It does not.
- **The gap grows monotonically with thread count** to 3.16× at t=8. That is the
  shape a barrier-cost story predicts; a codegen story predicts a roughly
  constant ratio.
- **The 8-thread A/B is extremely tight** — 3.78 pp of between-block spread on a
  205% effect. The machine's noise is ~5%; the effect is forty times that.

So the finding is real *on this machine*: **libgomp's barrier is catastrophically
expensive here, and ggml's own threadpool is not.**

### 4.2 What it does *not* isolate

Machine 1 is MSVC/`vcomp` on 8P+12E. This is GCC/`libgomp` on 2P+8E. **Two
variables moved.** The reversal could be:

- **libgomp vs vcomp** — a runtime difference, in which case Linux/Arch on this
  same laptop reproduces it, or
- **2P+8E vs 8P+12E** — a topology difference, in which case it is about having
  only two fast cores and eight slow ones, or
- an interaction of the two.

The Arch side of *this laptop* holds CPU and OpenMP runtime constant and changes
only the OS, so it splits "libgomp is bad here" from "Windows schedules libgomp
badly here". **That is now the highest-value single measurement in the project.**

### 4.3 W8 reframes F10/F14, and I got this wrong first

Doc 07's prediction — ceiling *sooner and lower* on a 2:8 part — **held**: 1.21×
at t=2 against machine 1's 2.2× near t=4. The tempting next step, which this
session took and then had to retract, is that **the ceiling tracks the P-core
count** (2 P-cores → peak at 2). It looked clean. It does not survive the
threadpool comparison: same silicon, same 2 P-cores, peak moves to **t=8** at
**1.77×**.

So core heterogeneity is **confounded with the threading runtime, and on this
machine the runtime is the larger term.** F14's mechanism is not refuted — it may
still be the whole story on machine 1, which was measured on OpenMP throughout —
but no scaling number from either machine means anything now without naming its
barrier.

### 4.4 The collapse is decode-only, which supports the barrier reading

The overhead runs give this for free, same binary and harness, only `-t` changing:

| workload | t=2 | t=8 | ratio |
|---|---:|---:|---:|
| prefill (pp256) | 335.44 | 580.83 | **1.73× — scales** |
| decode (tg128) | 23.93 | 11.19 | **0.47× — collapses** |

**Prefill parallelises normally on the exact configuration where decode falls
apart.** That is what a per-barrier fixed cost predicts: prefill amortises one
barrier over a 256-token batch, decode pays it per token on a batch of one. It
also rules out a whole class of alternative explanations — thermal, memory
bandwidth, a bad `-t` path — since all of those would hurt prefill too.

### 4.5 Why the overhead question could not be answered here

The gate failed twice, and the t=2 run shows why it will keep failing on this
machine under Windows. Pooled IQR at t=2 is **1.88%** — inside the 2% budget. The
gate uses the **worst block**, which is **4.02%**. So the run is not uniformly
noisy; it is quiet with **one bad block**. Both runs show it (decode block 6 at
t=2 is -1.31 against five estimates within ±0.7; prefill's worst block hits
11.98% IQR).

One-bad-block-per-run is the signature of an occasional throttle excursion, and
K5 means Windows will not confirm it. This makes
`cpupower frequency-set -g performance` on the Arch side **a requirement for
getting G1's headline, not a nicety** — it is aimed exactly at the failure mode
observed.

The point estimates, unquotable but not meaningless, sit at **+0.28%
[-0.16, +0.60]** decode at level 3, comfortably inside machine 1's **+0.56%
[-0.05, +1.16]**. That is weakly consistent with "under 1% on GCC" without being
evidence for it.

### 4.6 Where the effects sit relative to each other

Worth keeping in view, since this project's history is of ~1% effects fought over
for sessions:

| effect | size |
|---|---|
| **F27 here (OpenMP off, t=8)** | **+205%** |
| libgomp scaling collapse, t=1 → t=12 | -55% |
| machine-1 F27 at 28 threads | -54% |
| this machine's throughput noise | ~5% |
| F24's scheduler change | +1.62% |
| tokenscope level-3 overhead | < 1% |

F27 is two orders of magnitude larger than the overhead question the profiler was
built to answer, and larger than every other effect in the project combined. If
it reproduces on Arch, **it is the most consequential thing tokenscope has
found**, and it is a result about llama.cpp's threading rather than about
tokenscope.

---

## Part 5 — Suggested order for the Arch session

1. **Fix A1 first**, or measure a contaminated baseline (30 minutes, and it
   unblocks everything).
2. **Pin the governor**, then re-run overhead at this machine's scaling peak
   (K2), not at `-t 8`. This is G1's headline and 4.5 says the failure mode is
   addressable.
3. **Re-run F27 on both threading paths at t = 1, 2, 4, 8, 12.** Same silicon,
   different OS. Section 4.2 says this is the measurement that splits runtime
   from topology, and it is cheap.
4. **Re-run the scaling sweep on both paths** and record which barrier every
   number came from (K3).
5. Only then revisit M6, M10 and the layout arms — with `sensors` available, M10
   is finally answerable.

---

## Part 6 — Starting the Arch session

**This supersedes doc 07 section 8.** That prompt tells the session to run
`scripts/bootstrap.py` and verify 9 entries, which A1 and A2 show is exactly the
path that produces a contaminated tree and a check that cannot see it.

### Prerequisites

```bash
sudo pacman -S --needed base-devel cmake ninja git \
                        python python-numpy python-yaml \
                        cpupower lm_sensors
```

`python-yaml` is A6. `cpupower` is not optional — section 4.5 says the overhead
gate fails on throttle excursions, and pinning the governor is the fix aimed at
exactly that.

```bash
sudo cpupower frequency-set -g performance
sudo sensors-detect --auto && sensors     # M10: Linux will actually answer
```

### The prompt — copy from here

I want to bring up tokenscope on the Arch side of this laptop. It is a
deterministic scope-timing profiler compiled into llama.cpp, plus eight sessions
of measurements about where CPU inference time goes.

**Read `docs/09-actions-and-data.md` first** — it is the actionable handoff and
it supersedes `docs/07-linux-bringup.md` section 8. Then read `docs/08` for the
narrative of the previous session, `docs/06-overview.md` for what the project is,
and `docs/04-project-audit.md` before quoting any number.

This is the **same laptop** as session 8 — Dell Inspiron 15 3520, i7-1255U,
2 P-cores + 8 E-cores, 12 logical — but booted into **Arch instead of Windows**.
Session 8 measured the Windows/GCC side. That means **CPU and compiler family are
held constant and only the OS changes**, which is the comparison the project has
never been able to make.

**`A1` is fixed (session 9)** — `scripts/bootstrap.py` now applies only 01 and
02 and refuses a contaminated tree. Pull first so you have the fix, and still
verify the tree with the grep in A2 — **not** with the 9-entry `git status`
check, which passes on a contaminated tree.

**Then work section 5's order:** governor-pinned overhead at the scaling peak
(not at `-t 8` — see K2), then F27 on **both** threading paths at t = 1, 2, 4, 8,
12, then the scaling sweep on both paths.

**The headline to test:** session 8 measured `GGML_OPENMP=OFF` at **+204.87%
[+200.16, +209.58]** on 8 threads, where machine 1 measured **-2.15%**. The sign
reversed. **Session 9 (FINDINGS F51) has since split runtime from topology on
machine 1**: the same GCC reverses F27 there too, so it is the runtime — the
mingw port of libgomp has a sleep-only barrier (also upstream as #26200).
Linux libgomp uses a different, spinning barrier, so **score P51.8 first**: it
predicts `GGML_OPENMP=OFF` lands within ±20% of OpenMP on decode at 8 threads
here. Run F27 on both paths at t = 1, 2, 4, 8, 12 with `tools/runtime_sweep.py`
and a formal `ab_throughput.py` at 8, and record the outcome in FINDINGS
whichever way it goes.

**Measurement discipline, all of it learned the hard way:**

- **pin the governor before anything**, and record `sensors` alongside each run —
  Windows could not see throttling and section 4.5 says that is what broke the
  overhead gate
- **measure at this machine's scaling peak**, and know that the peak differs by
  threading path (t=2 on libgomp, t=8 on the threadpool)
- probe first and set `--min-baseline` from the probe; the values in the docs
  are machine-1 values and are meaningless here (K1)
- gate first, physical plausibility second, interval third — and if the gate
  fails, say so and do not quote the number
- **name the threading path in every scaling claim** (K3)
- close background work; do not run anything while a measurement is in flight
- **write predictions down and commit them before the run that tests them**,
  then score them honestly including the failures. Session 8 got W8 wrong first
  and had to retract it; that retraction is in `docs/09` section 4.3

Commit regularly with real commit messages, and push. Put raw output under
`results/` with `--json-out` as session 8 did, so the runs can be compared
across machines without re-reading prose.

Ask me before downloading anything large. **Do not write any upstream issue, PR
or comment text** — llama.cpp's `AGENTS.md` marks that non-overridable and the
penalty is a contributor ban; `docs/05-f24-filing-kit.md` explains what is
allowed instead.

### Copy to here
