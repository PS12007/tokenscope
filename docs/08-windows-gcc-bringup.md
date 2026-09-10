# 08 — Second machine, Windows/GCC: bring-up log and findings

**Machine 2, Windows side.** Everything in this project before this document was
MSVC on one Windows laptop (an 8P+12E, 28-thread part). This is a **different
CPU** — the dual-boot Dell Inspiron of [`07-linux-bringup.md`](07-linux-bringup.md)
section 2b — measured on its **Windows** side with **GCC/MinGW-UCRT**, not on the
Arch side that document was written for.

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

*Measurements — overhead, F27 and thread scaling — follow in section 6, appended
after this bring-up commit.*
