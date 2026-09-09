# 07 — Bringing tokenscope up on a second machine (Linux/GCC)

**This closes the project's biggest gap.** Every number in this repo is MSVC on
one Windows laptop with a heterogeneous CPU. A Linux machine with GCC turns
several one-machine curiosities into results, and a *dual-boot* machine does
something better still.

---

## 1. Why this machine is worth the effort

| gap | what a Linux machine gives |
|---|---|
| **G1 — no Linux, no GCC** | The blocker for the whole upstream conversation. F26 narrowed it: the barrier path measured on Windows *is* Linux's default (OpenMP), so what is untested is **libgomp and GCC**, not a different algorithm |
| **F27 on a second machine** | The biggest unexploited result here. `GGML_OPENMP=OFF` costs **-54% of decode** at 28 threads on the Windows box. Nobody knows if that is libgomp-vs-ggml or that specific CPU |
| **G6 — SMT vs core heterogeneity** | The Windows machine **cannot** separate these: it has 8 P-cores and 12 E-cores, so any no-SMT arm drags in E-cores. On a homogeneous CPU the confound simply does not exist |
| **M10 — the throughput levels** | On Windows this died because the chassis will not report temperature. **Linux will.** See section 5 |
| **G9 — shared build on Linux/ELF** | F22's fix is verified on three toolchains but only via a two-module test, never a real llama.cpp shared build outside Windows |

### The dual-boot machine is the interesting part

Same silicon, two operating systems, two compilers, two OpenMP runtimes. That is
a **controlled comparison the single-OS machines cannot do**: anything that
differs between Arch and Windows on the same hardware is OS, toolchain or
threading-runtime, not the CPU. Nothing in this project has been able to
separate those.

**Integrated graphics is irrelevant and slightly helpful** — everything here is
CPU inference, and a machine with no discrete GPU has no driver background load
to worry about.

---

## 2. The one thing that decides what is worth running

**What CPU is it?** Run this on the Arch side and keep the output:

```bash
lscpu | head -20
lscpu -e            # per-core listing: look at the MAXMHZ column
nproc
free -g
```

Then check for **P-cores and E-cores**:

```bash
# if every core has the same max frequency, the CPU is homogeneous
lscpu -e=CPU,CORE,MAXMHZ | sort -u -k3
```

- **Homogeneous CPU** (all cores identical - most Inspirons before 12th-gen
  Intel, and all Ryzen mobile parts): **this is the good case.** F10's 2.2x
  scaling ceiling and F14's core-heterogeneity mechanism predict *different*
  behaviour here, so the experiment is a real test rather than a repeat.
- **Hybrid CPU** (12th-gen Intel or later, some cores at lower MAXMHZ): still
  useful for G1 and F27, but it will not settle G6.

RAM matters too: the standard test model is ~840 MB and the harness wants
**1.5x that free**. 8 GB is plenty; 4 GB is workable with nothing else running.

---

## 3. Bring-up, with a verification ladder

Each step is checkable, and a failure at step *n* means do not bother with *n+1*.

### Prerequisites (Arch)

```bash
sudo pacman -S --needed base-devel cmake ninja git python python-numpy
gcc --version && cmake --version && ninja --version
```

### Step 1 - the profiler's own tests, no llama.cpp needed

This is the cheapest possible check that tokenscope compiles and works under
GCC. **CI already runs exactly this on ubuntu-latest**, so it should pass; if it
does not, something is wrong with the toolchain rather than the project.

```bash
git clone https://github.com/PS12007/tokenscope && cd tokenscope
cmake -S . -B build-on     -DCMAKE_BUILD_TYPE=Release -DTOKENSCOPE_ENABLED=ON
cmake -S . -B build-off    -DCMAKE_BUILD_TYPE=Release -DTOKENSCOPE_ENABLED=OFF
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release -DTOKENSCOPE_ENABLED=ON \
      -DBUILD_SHARED_LIBS=ON
cmake --build build-on && cmake --build build-off && cmake --build build-shared
ctest --test-dir build-on --output-on-failure
ctest --test-dir build-shared --output-on-failure
```

**Expected:** all green. The shared test guards F22 and runs in 0.04 s.

### Step 2 - instrumented llama.cpp, which has never been built on Linux

```bash
python scripts/bootstrap.py --dest ../llama.cpp
git -C ../llama.cpp status --short     # MUST be 8 modified files + ?? ggml/src/tokenscope/
```

Then the two static arms, **both from one tree in one session** - this is the
trap that has caught this project twice:

```bash
cmake -S ../llama.cpp -B ../llama.cpp/build-ts-on  -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DGGML_TOKENSCOPE=ON  \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake -S ../llama.cpp -B ../llama.cpp/build-ts-off -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DGGML_TOKENSCOPE=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build-ts-on  --target llama-bench llama-cli
cmake --build ../llama.cpp/build-ts-off --target llama-bench
```

**Then verify the zero-overhead claim on this toolchain** - it has only ever
been checked against MSVC's symbol table:

```bash
nm -C ../llama.cpp/build-ts-off/bin/llama-bench | grep -i tokenscope   # expect NOTHING
nm -C ../llama.cpp/build-ts-on/bin/llama-bench  | grep -i tokenscope   # expect symbols
```

### Step 3 - a model, generated not downloaded

No network, no multi-GB download. The standard test model:

```bash
mkdir -p ../models
python tools/make_tiny_model.py --llama-cpp ../llama.cpp -o ../models/mid.gguf \
       --layers 24 --embd 768 --heads 12 --heads-kv 4 --vocab 8192
```

### Step 4 - a trace, and the first real result

```bash
TOKENSCOPE_LEVEL=1 TOKENSCOPE_OUT=t.json \
  ../llama.cpp/build-ts-on/bin/llama-bench -m ../models/mid.gguf -p 0 -n 64 -t 4
python tools/trace_analyze.py t.json
```

**Read the summary line.** It prints `threading=` and `compute_linkage=`. On
Linux with default flags it should say **`threading=openmp`** - and if it says
`ggml-threadpool`, that is itself a finding, because it would mean GCC builds
take a different barrier path than MSVC ones by default. F26 exists because five
documents got this wrong for four sessions; check it, do not assume it.

---

## 4. What to measure, in priority order

### (a) Overhead on GCC - the G1 headline

```bash
python tools/bench_overhead.py -m ../models/mid.gguf -n 20 -t 8 --levels 3 \
    --blocks 6 --reps 10 \
    --pair static=../llama.cpp/build-ts-off/bin,../llama.cpp/build-ts-on/bin
```

**Use 6 blocks of 10, not 3 of 20** (session 7's F49). The gate measures
within-block baseline spread, and a shorter block contains less drift.

Windows/MSVC answer to beat: **+0.56% [-0.05, +1.16]** at 8 threads, level 3.

### (b) F27 - the barrier comparison, and the biggest prize

```bash
cmake -S ../llama.cpp -B ../llama.cpp/build-noomp-off -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DGGML_TOKENSCOPE=OFF -DGGML_OPENMP=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build-noomp-off --target llama-bench
# verify the arms really differ:
ldd ../llama.cpp/build-ts-off/bin/llama-bench   | grep -i gomp   # expect libgomp
ldd ../llama.cpp/build-noomp-off/bin/llama-bench | grep -i gomp  # expect nothing

python tools/ab_throughput.py \
    --a ../llama.cpp/build-ts-off/bin/llama-bench \
    --b ../llama.cpp/build-noomp-off/bin/llama-bench \
    -m ../models/mid.gguf -t $(nproc) --blocks 6 -n 10 --json-out f27-linux.json
```

Windows numbers to compare against: **-2.15%** at 8 threads, **-54.17%** at 28.
If libgomp behaves differently from MSVC's `vcomp`, this is where it shows.

### (c) Thread scaling - F10's 2.2x ceiling

```bash
for t in 1 2 4 8 $(nproc); do
  ../llama.cpp/build-ts-off/bin/llama-bench -m ../models/mid.gguf -p 0 -n 64 -t $t -r 5 -o csv
done
```

On the Windows box nothing beats **2.2x** and every thread past four becomes
barrier wait, *because* of P/E heterogeneity. **On a homogeneous CPU the
prediction is that scaling continues further.** If it does not, core
heterogeneity was not the whole story and F14's mechanism needs revisiting -
which would be a more interesting result than confirming it.

---

## 5. What Linux gives that Windows could not

M10 - the throughput levels - stalled on Windows because
`MSAcpi_ThermalZoneTemperature` returns *Not supported* on that chassis. **Linux
will tell you all of this:**

```bash
sudo pacman -S --needed lm_sensors cpupower linux-tools
sudo sensors-detect --auto && sensors        # actual core temperatures
cpupower frequency-info                       # governor and limits
watch -n1 'grep MHz /proc/cpuinfo'           # live per-core frequency
cat /sys/devices/system/cpu/intel_pstate/no_turbo
cat /sys/class/thermal/thermal_zone*/temp
```

**Before measuring anything, pin the governor** - this removes a variable the
Windows machine could never control:

```bash
sudo cpupower frequency-set -g performance
```

If throughput on Linux is stable where Windows wandered 39-46 tok/s, that is
strong evidence M10 is an OS or firmware policy rather than the silicon - and
**on a dual-boot machine you can test that directly**, which nobody has been
able to do.

---

## 6. Traps that will bite on the new machine

- **Both arms must be built from one tree in one session.** A stale arm gave
  +2.71% where the truth was +1.95%, with cleanly separated ranges, which made
  it *more* convincing.
- **Never `git checkout <file>` or `git apply -R` inside `../llama.cpp`.** The
  clone is patched in place, so git's baseline is upstream and either command
  silently strips the instrumentation while everything still builds. Check
  `git -C ../llama.cpp status --short` shows **9** entries.
- **Do not run anything else while measuring**, including editing files.
- **Two runs whose baseline medians differ are not comparable.** Use
  `--min-baseline` once you know what this machine's normal reading is.
- **Cross-machine absolute throughput comparison is meaningless.** Compare
  *ratios and shapes* between the two machines, never tok/s.

---

## 7. What a good outcome looks like

You do not need all of it. In descending value:

1. **Overhead under 1% on GCC/Linux** - G1's headline, and the sentence the
   upstream conversation needs.
2. **F27 reproduced, or not, on libgomp** - either answer is publishable. "It
   is the same on both" makes it a real result about barriers; "it is not"
   makes it a result about MSVC's `vcomp`.
3. **Thread scaling past 2.2x on a homogeneous CPU** - confirms F14's
   mechanism, or breaks it.
4. **A stable baseline under a pinned governor** - would give this project the
   quiet machine it has never had, and make every future measurement cheaper.
