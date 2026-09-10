# 08 — Second machine, Windows/GCC: bring-up log and findings

**Machine 2, Windows side.** Everything in this project before this document was
MSVC on one Windows laptop (an 8P+12E, 28-thread part). This is a **different
CPU** — the dual-boot Dell Inspiron of [`07-linux-bringup.md`](07-linux-bringup.md)
section 2b — measured on its **Windows** side with **GCC/MinGW-UCRT**, not on the
Arch side that document was written for.

**For the actionable version of this document — the defects to fix, the traps to
carry forward, the complete dataset and the analysis — see
[`09-actions-and-data.md`](09-actions-and-data.md).** This file is the narrative
of the session; that one is what the next machine needs.

That makes this an odd but useful cell of the design:

| | machine 1 | **this run** | machine 2, Arch (future) |
|---|---|---|---|
| CPU | 8P+12E, 28 threads | **2P+8E, 12 threads** | 2P+8E, 12 threads |
| OS | Windows | **Windows** | Linux |
| toolchain | MSVC | **GCC 16.1 (UCRT64)** | GCC |
| OpenMP runtime | `vcomp` | **libgomp** | libgomp |

So against machine 1 this varies CPU *and* toolchain together, and against the
future Arch run it isolates **OS** with CPU and compiler family held constant.
Neither is a clean single-variable comparison on its own; together they bracket
one. **G1 is not closed by this** — G1 asks for Linux, and this is not Linux —
but the *GCC/libgomp half* of G1 now has data where it had none.

---

## 1. Environment

```
Dell Inspiron 15 3520
12th Gen Intel Core i7-1255U   10 cores / 12 logical   (2 P-cores SMT + 8 E-cores)
15.7 GB RAM
Windows 11 Home, build 10.0.26200
GCC 16.1.0 (Rev5, MSYS2) — UCRT64, libgomp
CMake 4.4.3, Ninja 1.13.2, Python 3.13.14, numpy 2.4.3
llama.cpp pinned commit 4d9176092
```

Two environment notes that cost time and are not in doc 07:

- **`gguf-py` needs `pyyaml`.** Doc 07's Arch prerequisite line installs
  `python python-numpy` only, so the Arch bring-up will hit the same
  `ModuleNotFoundError: No module named 'yaml'` at step 3.
- **Do not keep the working tree inside OneDrive.** This repo was first cloned
  into a synced folder; a bootstrapped `../llama.cpp` and an 840 MB model there
  would be uploaded and rescanned by the sync client mid-run, which is a
  contamination source the "nothing else may run" rule cannot see. Everything
  here lives under `C:\cs\`.

---

## 2. Finding W1 — `bootstrap.py` produces a contaminated tree on a fresh clone

**This is the headline of the Windows-side bring-up, and it is a real defect.**

`bootstrap.py::apply_patches()` globs `patches/*.patch` and applies **every**
one, then `raise SystemExit` on the first that does not apply. But patches 03-06
are documented, in their own headers, as hand-applied experiment arms:

> `03-mulmat-chunk-threshold.patch`: "Measured in FINDINGS F24. **NOT APPLIED to
> the tree by bootstrap.py** ... Apply it by hand to reproduce F24, and revert it
> before regenerating 01"

> `04`, `05`, `06`: "This is not a proposal ... it exists to be measured and then
> reverted"

What actually happens on a clean checkout:

```
01-instrument.patch:          applied
02-name-attn-output.patch:    applied
03-mulmat-chunk-threshold:    applied     <- F24's treatment, into the baseline
04-layout-control.patch:      applied     <- "do not merge" control arm
05-layout-arm.patch:          applied     <- "do not merge" layout arm
06-layout-arm-48.patch:       DOES NOT APPLY -> SystemExit
```

06 fails because **05 and 06 are mutually exclusive by construction** — they are
alternative layout arms editing the same region of `ggml-cpu.c`. So bootstrap can
never complete on a clean tree once both exist.

**The damage is worse than the error message.** The tree it leaves behind has
F24's scheduler change compiled into what every downstream step treats as the
*stock baseline*:

```
$ grep -n 'nchunk0 \* nchunk1 <' ggml/src/ggml-cpu/ggml-cpu.c
1422:    if (nchunk0 * nchunk1 < nth * 2 || ggml_is_numa()) {
1711:        if (nchunk0 * nchunk1 < nth * 2 || disable_chunking) {
```

`nth * 2` is F24. HANDOFF's cold-start checklist says stock is **two `nth * 4`
lines**. Any overhead or A/B number taken from that tree is measuring against a
baseline that already contains the change under test.

**Why machine 1 cannot see this.** Machine 1 has had `../llama.cpp` since before
patches 03-06 existed, and `apply_patches` has an "already applied, skipping"
branch. A fresh clone is the only way to reach the failure — which is precisely
what a second machine is for.

### W1b — doc 07's step 2 check cannot detect W1

Doc 07 step 2 says to verify:

```
git -C ../llama.cpp status --short    # MUST be 8 modified files + ?? ggml/src/tokenscope/
```

The **contaminated tree also has exactly 9 entries**, because 03/04/05 modify
files 01 already modified. The check passes on a tree that is wrong. Only
HANDOFF's `nth * 4` grep catches it, and doc 07 does not include that grep.

Arithmetic worth recording, since it pins down the intended state: patch 01
touches 7 files, patch 02 adds `src/llama-graph.cpp` — 8 modified, plus the
untracked `ggml/src/tokenscope/` = the documented 9. **01+02 is the instrumented
tree; 03-06 are experiment arms.**

### The tree used for everything below

Rebuilt deliberately, and verified against HANDOFF's checklist:

```
git reset --hard                       # back to 4d9176092
cp src/{tokenscope.h,tokenscope-ggml.h,tokenscope.cpp} ggml/src/tokenscope/
git apply patches/01-instrument.patch
git apply patches/02-name-attn-output.patch
```

```
status --short           -> 9 entries          OK
nchunk0 * nchunk1 <      -> two `nth * 4`      OK (stock)
diff src/ vs tokenscope/ -> silent             OK
GGML_USE_OPENMP in build.ninja -> present in both arms   OK (F26)
```

---

## 3. Step 1 — the profiler's own tests under GCC

Ran both by hand and through the documented cmake/ctest path.

```
build-on      (TOKENSCOPE_ENABLED=ON)                ctest 1/1 passed
build-off     (TOKENSCOPE_ENABLED=OFF)               builds, prints the control banner
build-shared  (ENABLED=ON, BUILD_SHARED_LIBS=ON)     ctest 2/2 passed
```

`selftest` — all groups green, 0 dropped, 205,143 events. **Per-scope cost on
this CPU: 62.8 ns** (2 clock reads + 1 store).

`dlltest` — **all green.** This is the F22 guard, and it had never been built
with GCC before: CI covers ubuntu/windows/macos, but `windows-latest` is MSVC.
Two modules each with their own `ts_tls` cache resolving to one buffer per
thread, plus a module `dlopen`ed after threads already exist, all correct on
**GCC + PE/COFF**. That is a fourth toolchain for F22, though it is *not* the
ELF shared-build G9 asks for.

### W2 — a new GCC warning, not fixed

```
src/tokenscope.cpp:27:11: warning: 'NOMINMAX' redefined
```

MinGW's `bits/os_defines.h` already defines `NOMINMAX`; the guard at
`tokenscope.cpp:27` does not test for it. Harmless, MSVC never sees it. **Left
in place deliberately** — this session's remit was to measure, not to edit.

---

## 4. Step 2 — instrumented llama.cpp under GCC

Both static arms built from one tree in one session, as the docs require.

```
build-ts-on   (GGML_TOKENSCOPE=ON)   llama-bench, llama-cli    OK
build-ts-off  (GGML_TOKENSCOPE=OFF)  llama-bench               OK
```

**llama.cpp builds clean with GCC/MinGW-UCRT** — 340 targets, no errors, 8m25s
for both arms on this 15 W part.

### W3 — the documented zero-overhead check is MSVC-specific and gives a false negative

Doc 07 step 2 says:

```
nm -C build-ts-off/bin/llama-bench | grep -i tokenscope   # expect NOTHING
nm -C build-ts-on/bin/llama-bench  | grep -i tokenscope   # expect symbols
```

On GCC **both arms return nothing**, so the check cannot distinguish them — a
reader following doc 07 would conclude the ON build had failed. `nm` reads 37,920
symbols in the ON arm; none contain the string `tokenscope`, because the symbols
are named `ts_*`.

**The claim itself holds — it just needs the right predicate:**

| | `grep '\bts_'` | `strings \| grep -i tokenscope` |
|---|---|---|
| `build-ts-on`  | **84** symbols (`ts_g_level`, `ts_g_token`, `ts_flush::s_decode`, `llama_tokenize::ts_id_4428`, …) | **11** |
| `build-ts-off` | **0** | **0** |

So: **zero-overhead-when-compiled-out is confirmed on GCC**, and doc 07's
verification command needs `ts_` rather than `tokenscope` to say so.

---

## 5. Steps 3-4 — model and first trace

```
python tools/make_tiny_model.py --llama-cpp ../llama.cpp -o ../models/mid.gguf \
       --layers 24 --embd 768 --heads 12 --heads-kv 4 --vocab 8192
-> 840.3 MB, 220.24 M params        (matches the standard test model)
```

```
TOKENSCOPE_LEVEL=1 llama-bench -m ../models/mid.gguf -p 0 -n 64 -t 4
-> 321 tokens, 468,993 bytes, 0 dropped
```

### W4 — `threading=openmp` on GCC, as F26 predicted

```
clock=steady_clock  level=1  threads=1  dropped=0
threading=openmp  compute_linkage=static
```

Doc 07 asked for this explicitly, with a stop condition if it read
`ggml-threadpool`. **It does not.** GCC/libgomp builds take the same OpenMP
barrier path as MSVC/`vcomp` builds by default, so F26's conclusion carries to
this toolchain and the barrier numbers remain comparable in kind.

Trace shape is healthy: **100.0% of decode wall time attributed to a host
scope**, `graph-compute` 99.9%, 0 dropped.

### W5 — this machine is much slower, and much noisier, than machine 1

At `-t 4`: **15.35 ± 2.38 tok/s**, and within the trace `p50 54.16 ms /
p95 106.57 ms / p99 123.15 ms / max 207.67 ms` per token against a **66.70 ms
mean**.

Machine 1 sat at 38.6-46.0 tok/s. Absolute cross-machine throughput is
meaningless by the project's own rule, but two things are not:

- the **±2.38 on 15.35 is ~15% relative spread**, against machine 1's ~12% level
  switching — so this machine is at least as noisy, and the mean sitting well
  above p50 is the signature of a **long upper tail**, consistent with the
  thermal throttling doc 07 section 2b predicted for a 15 W U-series part
- `MSAcpi_ThermalZoneTemperature` returns **Not supported on this chassis too**
  (see W6), so that tail cannot be attributed on the Windows side

### W6 — M10's thermal blindness is not one bad chassis

Machine 1 could not read temperature because
`MSAcpi_ThermalZoneTemperature` returns *Not supported*. **The same query
returns *Not supported* on this completely different Dell chassis.** So M10's
blocker is a **Windows/WMI pattern, not a machine defect** — which removes the
hope that another Windows box would have fixed it, and makes the Arch side of
this machine the only route to M10.

---

## 6. Measurements

Order deviates from doc 07 section 4 deliberately: thread scaling ran **first**,
because `--min-baseline` and the overhead thread count both need this machine's
own numbers, and the project's calibration constants (40.9 / 43.5 / 46.0 tok/s)
are machine-1 specific and meaningless here.

### W7 — doc 07 section 4(a)'s overhead command does not run

```
python tools/bench_overhead.py -m ../models/mid.gguf -n 20 -t 8 --levels 3 \
    --blocks 6 --reps 10 --pair static=...
```

Two errors:

- **`-m` does not exist** on `bench_overhead.py` — the option is `--model` only
  (`-m` is `ab_throughput.py`'s spelling). The command exits with
  `error: the following arguments are required: --model` having measured
  nothing.
- **`-n 20` and `--reps 10` are the same argparse option** (`-n, --reps`). The
  `-n 20` is silently overridden. Harmless, since 10 is what F49 wants, but the
  line reads as if it sets two different things.

### 6a. Thread scaling — F10 / F14, and doc 07's written prediction

`build-ts-off`, `-n 64 -r 5`, run **three times**: ascending, descending, and
again paired against the no-OpenMP arm. The descending pass is a thermal
control.

| threads | ascending | descending | paired run | mean | vs 1 thread |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.09 ± 1.04 | 18.90 ± 0.36 | 18.99 ± 1.08 | 19.33 | 1.00× |
| **2** | **23.78 ± 0.35** | **23.07 ± 0.66** | **23.05 ± 0.51** | **23.30** | **1.21× ← peak** |
| 4 | 16.03 ± 0.26 | 15.79 ± 0.87 | 15.57 ± 0.63 | 15.80 | 0.82× |
| 8 | 10.67 ± 1.46 | 10.01 ± 0.41 | 10.28 ± 0.75 | 10.32 | 0.53× |
| 12 | 8.67 ± 0.36 | 8.83 ± 0.37 | 8.97 ± 0.50 | 8.82 | 0.46× |

Three independent replications agree closely. **Scaling peaks at 2 threads at
1.21× and then goes negative — 12 threads is 2.2× slower than one thread.**

Doc 07 section 2b predicted: *"the scaling ceiling should arrive sooner and
lower. That is a real prediction, not a repeat."* Against machine 1's 2.2×
around four threads, **1.21× at two threads is sooner and lower. The prediction
holds.**

**The thermal control matters.** If throttling drove the collapse, `t=12`
measured first (cold) should beat `t=12` measured last (hot). It does not —
**8.83 cold against 8.67 hot, inside noise.** The only order effect is at `t=1`
(20.09 cold, 18.90 hot, ~6%). So the collapse is not thermal, which is worth
having given W6 means Windows cannot show us temperature.

### 6b. F27 — and it reverses

This is the biggest result of the session, and it is a **failure to reproduce,
in the strong sense: the sign flips.**

```
ldd build-ts-off/bin/llama-bench    -> libgomp-1.dll
ldd build-noomp-off/bin/llama-bench -> no libgomp        (arms verified distinct)
```

| | machine 1 (MSVC/`vcomp`, 8P+12E) | **machine 2 (GCC/`libgomp`, 2P+8E)** |
|---|---|---|
| 8 threads | **-2.15%** | **+204.87%  [+200.16, +209.58]** |
| all threads | **-54.17%** (at 28) | **+216.71%  [+175.17, +258.26]** (at 12) |

Both machine-2 figures are `--blocks 3`, resolved, between-block interval
excludes zero. The 8-thread run is exceptionally tight — **3.78 pp** spread
across blocks.

On machine 1, turning OpenMP off **cost** more than half of decode. On this
machine, turning OpenMP off **more than triples it**.

### 6c. The control that makes 6b interpretable

`ab_throughput.py` warns that a control workload which also moves means the
comparison is wrong. Prefill did also move (+113.67% at 8 threads), so the
warning fires — but for `GGML_OPENMP` that is *expected*, since the flag changes
the barrier for all compute, not one path. The real control is **thread count**:

| threads | OpenMP (libgomp) | no-OpenMP (ggml threadpool) | ratio |
|---:|---:|---:|---:|
| 1 | 18.99 ± 1.08 | 18.37 ± 0.53 | **1.00× — agree** |
| 2 | 23.05 ± 0.51 | 29.95 ± 0.19 | 1.30× |
| 4 | 15.57 ± 0.63 | 26.09 ± 0.40 | 1.68× |
| 8 | 10.28 ± 0.75 | **32.44 ± 0.46** | **3.16×** |
| 12 | 8.97 ± 0.50 | 21.29 ± 7.61 | 2.37× |

**At one thread the two binaries are the same speed**, as they must be with no
barrier to take. The divergence is therefore threading, not codegen, and it
grows monotonically with thread count to 8. That is the shape a barrier-cost
story predicts and a compiler-difference story does not.

### W8 — the scaling collapse is libgomp's, not this CPU's

**This corrects the reading of 6a.** With ggml's own threadpool the same
silicon scales *positively* to eight threads:

| | peak | at peak | at 12 threads |
|---|---|---|---|
| libgomp | **t=2** | 1.21× | 0.46× |
| ggml threadpool | **t=8** | **1.77×** | 1.16× |

So F14's core-heterogeneity mechanism is **confounded with the OpenMP runtime on
this machine, and the runtime is the larger term.** 6a's ceiling — and the
tempting reading that it tracks the P-core count — is a property of the
*libgomp build*, not of the CPU. Machine 1's 2.2× was measured on an
OpenMP build too, so the comparable machine-2 number is **1.21×**, and this
machine's actual best is **1.77×** on a path machine 1 never used for scaling.

One oddity, recorded rather than explained: the threadpool curve is
**non-monotonic** — 29.95 at t=2, *down* to 26.09 at t=4, then up to 32.44 at
t=8. Repeatable within this session. A P-core/E-core placement effect is the
obvious guess; Windows gives no way to check, and `t=12` also carries a huge
±7.61, consistent with oversubscribing 12 logical CPUs.

### 6d. Overhead — G1's headline, and it does not resolve here

**No overhead number from this machine is quotable.** The harness's baseline
gate failed at both thread counts and said so:

> `baseline IQR [static] is 7.50% of median` ... `that is wider than the 2%
> budget being tested` ... **`DO NOT QUOTE ANY OF THE ABOVE`**

A prediction was written before the second run: *t=8 fails because it sits deep
in 6a's negative-scaling regime, where time is barrier wait and variance
explodes; at t=2, the scaling peak, the gate should pass.* **Partially
confirmed** — everything tightened by 3-5×, but the gate still failed:

| | t=8 | t=2 |
|---|---|---|
| A-arm IQR, worst block | 7.50% | **4.02%** |
| A-arm IQR, pooled | 5.34% | **1.88%** ← under the 2% budget |
| decode, level 3 | +1.50% [-0.03, +3.14] | **+0.28% [-0.16, +0.60]** |
| decode, per-block mean | +1.56 [-0.34, +3.46] | **-0.02 [-0.76, +0.73]** |
| between-block spread | 4.97 pp | **2.00 pp** |
| prefill, level 3 | +0.58% [-0.75, +1.18] | -0.35% [-1.22, +0.48] |
| elapsed | 2325 s | 1313 s |

**Pooled IQR passes the 2% budget at t=2; worst-block IQR does not**, and the
gate keys off the worst block. One bad block per run is the whole problem — the
signature of an unpredictable throttle excursion on a 15 W chassis that
Windows refuses to report (W6).

Recorded as *not resolved*, not as a result. For calibration only, and not to be
quoted: the t=2 point estimates straddle zero and are comfortably inside
machine 1's `+0.56% [-0.05, +1.16]`, which is weakly encouraging for G1's
headline without being evidence for it.

---

## 7. What this session did and did not establish

**Established:**

1. **`bootstrap.py` is broken for any fresh clone** (W1), and the documented
   check cannot detect it (W1b). Anyone bringing up machine 3 hits this first.
2. **tokenscope builds and passes its own tests under GCC** — a fourth toolchain
   for F22, and the first GCC build of the shared-library arm anywhere (§3).
3. **Zero-overhead-when-compiled-out holds on GCC** — 84 `ts_` symbols against
   0 (W3) — though the documented command cannot show it.
4. **`threading=openmp` on GCC** (W4), so F26's conclusion carries and the
   barrier numbers stay comparable in kind.
5. **F10/F14's prediction held** as written: sooner and lower (6a) — but see 8.
6. **F27 reverses sign**, hugely and tightly (6b), with a t=1 control that rules
   out codegen (6c).
7. **On this machine the OpenMP runtime dominates core heterogeneity** (W8),
   which reframes 6a and is the most interesting thing here.
8. **M10's thermal blindness is a Windows pattern, not one chassis** (W6).

**Not established:**

- **G1 is not closed.** This is GCC, not Linux. The overhead headline that the
  upstream conversation needs did not resolve here (6d).
- **Which variable drives 6b.** Machine 1 is MSVC/`vcomp` on 8P+12E; this is
  GCC/`libgomp` on 2P+8E. Runtime *and* CPU differ. The Arch side of this same
  laptop holds CPU constant and is the experiment that separates them.
- **G6 is untouched**, as doc 07 said it would be — both machines are hybrid.

**What the Arch side should do first**, in light of the above:

1. **Re-run 6b on the same silicon.** Same CPU, same libgomp, different OS. If
   the +205% survives, it is libgomp; if it moves, it is Windows' scheduler.
   This is now the highest-value single measurement in the project.
2. **Pin the governor and re-run 6d.** `cpupower frequency-set -g performance`
   plus visible `sensors` is the difference between a failed gate and G1's
   headline. 6d says the gate fails on *one bad block*, which is exactly what a
   pinned governor should remove.
3. **Re-run 6a on both threading paths.** W8 means any scaling claim must say
   which barrier it was measured on.
4. **Fix W1 before anything else**, or measure a contaminated baseline.

