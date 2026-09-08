# HANDOFF — state of the project, and what to do next

Updated **during session 6 (2026-09-08)**. Everything here is either a
fact about the current tree or an explicit next step. Read this first when
picking the project back up.

**Cold-start checklist, in order.** Each takes seconds and each has caught
something real:

```bash
cd "C:/-CS/TLI profiler/tokenscope"
git log --oneline origin/main..main     # MUST be empty
git status --short                      # MUST be empty
git -C ../llama.cpp status --short      # MUST match section 3's list exactly
grep -n 'nchunk0 \* nchunk1 <' ../llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c
                                        # two `nth * 4` lines = stock (F24 not applied)
for f in tokenscope.h tokenscope-ggml.h tokenscope.cpp; do
  diff -q src/$f ../llama.cpp/ggml/src/tokenscope/$f; done   # MUST be silent
grep -o 'GGML_USE_OPENMP' ../llama.cpp/build-ts-on/build.ninja | head -1
                                        # MUST print it -- see F26
```

Then, before trusting any measurement, rebuild rather than assuming the binaries
match the tree — see the A/B trap in section 3.

**New in session 5: the trace tells you how it was built.** Every trace's
provenance record now carries `threading` and `compute_linkage`, and
`trace_analyze.py` prints them on the summary line. If a barrier number ever
looks strange, read that line before theorising — the field exists because four
sessions of documents described every barrier figure here as coming from a code
path none of them came from (**F26**).

**Session 6 so far, in one paragraph.** Closed section 5 item 7, which F25 had
called "the single cheapest unfinished thing in this list", and it was.
`bench_overhead.py` now takes N build pairs and round-robins every arm of every
pair in one invocation, which is what F25's post-mortem said the question
needed and what no amount of extra reps could supply. The answer: **the shared
build's level-3 overhead is +0.87% [+0.55, +1.19], certified** — the first
measurement behind F22's inspection-only claim that per-module `ts_tls` caching
keeps a cross-DLL call off the hot path. P25.1, the difference between the two
builds, comes out **bounded but unresolved at +0.36pp [-0.32, +1.17]**, which
is a weaker claim than a value and a much stronger one than F25's "nothing was
resolved in either direction". Three other things fell out. F25's tentative
"the shared build is 0.7% slower" is **wrong in sign** once interleaved
(-0.22% [-0.79, +0.26]) — the hedge on it was justified. The harness's own
protocol had a flaw nobody was looking for: **arm order within a round was
fixed**, so a transient shorter than a round hit the same arms every time, and
in F30's run one extra warm-up round cost the static pair its certification;
order now rotates. And the compiled-out static arm read **42.94 tok/s in
session 5 and 45.92 in session 6** — same code, same machine, +6.9%, about six
times the effect being resolved, which is the sharpest statement this project
has of why cross-session comparisons of absolute throughput are worthless here.

**Session 5 in one paragraph.** Five findings, **F25 through F29**, and the
theme is that four of the five are the project auditing itself. It began by
closing section 5 item 7 — the `GGML_TOKENSCOPE=OFF` shared build now exists, so
the shared/static overhead comparison could finally run — and while configuring
it, noticed `-DGGML_USE_OPENMP` on the compile line for `ggml-cpu.c`. **Every
barrier number this project has ever taken came from `#pragma omp barrier`**,
and five documents said the opposite (**F26**). That turned into
`-DGGML_OPENMP=OFF` builds and **F27**, the largest result of the session:
ggml's own spin-wait barrier costs **-54% of decode at 28 threads** on this
machine, and -2.06% at 8, with the cause measured in the traces rather than
inferred. Then, while extending a tool for F27, found that one barrier out of
824 held 76-83% of all after-arrival time in every level-3 trace — and that it
was **tokenscope's own lazy buffer allocation**, which `--barriers` had been
blaming on ggml's thread pool since session 1 (**F28**). Fixed; the
imbalance/release split moves from 55/45 to **84.6/15.4**. **F29** is a parser
bug: `TOKENSCOPE_TOKENS=10:11` captured one token rather than two, silently, and
`imbalance_repeat.py` had defaulted to it since session 4. **F25** is the
overhead run that started everything: the static number certifies for the first
time since session 1 at **+1.16% [+0.67, +1.87]**, and the shared build's answer
was eaten by its own noise floor for a reason worth reading.

Also: the README finally has a picture. `tools/trace_svg.py` renders one token
per-thread as a theme-aware SVG from a committed trace, which closes most of
what the Perfetto item wanted, without a browser.

**Session 4 in one paragraph.** Closed section 5 item 6 — the shared-library
build works (**F22**) — then **re-scoped item 5 out from under itself (F23)**
and **acted on the re-scoped version (F24)**, which produced the first
throughput improvement this project has ever certified. Findings went F21 ->
**F24**. 18 commits, all pushed, CI green on all three platforms. The F20
upstream issue is *not* filed: llama.cpp's `AGENTS.md` forbids an agent writing
issue or PR text, so `docs/03` now holds an evidence pack to write from instead
of a draft to paste.

On F23: item 5 assumed ggml gives every thread an equal share of rows, and for
matmul that is only half true. Above `nchunk0 * nchunk1 >= nth * 4` threads
steal chunks from an atomic counter, and a matmul in that mode carries roughly a
third to a half the arrival imbalance per unit work. So ggml already solves core
heterogeneity for big matmuls, by a method that needs no model of core speed —
but the threshold contains `nth`, so **adding threads can turn it off**. Twelve
runs at each of six points, two models, with a cross-model control at a fixed
thread count; two of the comparisons have non-overlapping ranges.

Then **F24 acted on it**: one line, `nth * 4` -> `nth * 2` in `mul_mat`,
measured at **+1.95% [+1.59, +2.35] on decode, certified** by
`bench_overhead.py`'s own bootstrap over 20 interleaved rounds. That harness
has declined to certify six times across four sessions; this is the first
thing it has ever passed. Prefill is the control and stays uncertified, which
is what makes the decode number believable. `patches/03-mulmat-chunk-threshold.patch`,
**not applied to the tree** and deliberately kept out of `01-instrument.patch`.

**Read F23's reproducibility section even if you skip the rest.** Getting there
took three wrong turns, each caused by trusting too few runs: a single trace
inverted the conclusion, a six-run median invented a "28-thread anomaly" that
does not exist, and another six-run median put a prediction outside its band
that twelve runs put inside it. Per-node imbalance spreads up to **8.5×** across
identical runs. `bench_overhead.py` has interleaved arms and bootstrapped CIs
since session 1 — that discipline was never applied to numbers read *out of
traces*, which got treated as exact because the tracing is exact. The tracing is
exact. The machine is not.

On F22 specifically: F18 had called it a genuine incompatibility between the hot path's
raw thread-local and the registry's need for one instance across DLLs — both
true, but `ts_tls` is a cache and the buffer is the state, so the cache never
needed to be single-instance. Each module now keeps its own and they all resolve
to one registry-owned buffer; the hot path is unchanged. A two-module test now
guards it on all three CI platforms. The shared build's overhead is still
unmeasured.

**Session 3 in one paragraph.** Ran the 8B that section 5 item 4 was waiting on
— it was already on the machine, pulled by Ollama, no download needed. Wrote six
predictions down and committed them *before* measuring; five held and one failed
on both its specifics. Findings went F18 -> **F21**. The best result is a
controlled experiment the earlier models could not support: llama.cpp's Q4_K_M
recipe stores `ffn_down` at two different precisions in different layers, so the
same node in adjacent layers differs only in dtype, and the byte law predicts
the ratio to 2.7% with two control tensors flat at 1.00. Also found and fixed a
defect in **upstream llama.cpp** (F20) and a dead entry in **tokenscope's own**
category table (found because the first would have exposed the second). All 24
commits from sessions 2 and 3 are pushed.

---

## 1. Where things stand

**Working, measured, committed:**

- Core scope-timing mechanism (24-byte record, thread-local chunked arena,
  RAII scopes, Chrome Trace Event output). Self-test passes under 8 threads.
- Tier 1: 11 host scopes across `llama_context::decode` and `process_ubatch`.
- Tier 2: per-node work and barrier wait, per thread, in
  `ggml_graph_compute_thread`.
- Per-phase and per-layer attribution across all 24 layers.
- Barrier decomposition: arrival imbalance vs release latency, per node, with
  the matching refused rather than guessed if node/barrier alternation breaks
  (FINDINGS F9).
- Sampling, tokenizer and cell-search scopes (sites 13, 16-19), so the whole
  per-token loop is covered and not just the parts `llama-bench` reaches.
- Thread-count sweep, 1 to 28, throughput from the uninstrumented build
  (FINDINGS F10).
- Python analysis: summary, per-token, outliers-with-cause, per-layer, diff.
- Overhead: level 3 is **+1.16% [+0.67, +1.87]** at 8 threads, **static** build,
  certified in session 5 (F25), n=20; and **+0.87% [+0.55, +1.19]** for the
  **shared** build, certified in session 6 (F30). It was +0.67% [+0.12,
  +1.67] in session 1; session 2 tried at 8 and 28 threads and the harness
  **refused to certify either**, because the machine's noise floor had moved
  (baseline IQR 2-4%, wider than the effect). Quote a number with its thread
  count and its build, and expect to have to re-measure on a quiet machine.
  **The two cannot be compared across sessions** — F30 found the compiled-out
  static arm reading 42.94 tok/s in session 5 and 45.92 in session 6, the same
  code on the same machine, which is ~6x the effect being resolved. Within F30's
  single invocation the difference of overheads is **+0.36pp [-0.32, +1.17]**:
  bounded, not resolved.
- Zero-overhead-when-off verified against the symbol table.
- **Shared-library builds work (F22).** Each module keeps its own `ts_tls`
  cache over one registry-owned buffer. `ts_dlltest` builds two binaries plus a
  third opened at runtime and asserts they share one buffer per thread.
- CI for **three platforms x three configurations** (instrumentation on, off,
  and shared), including the over-attribution regression check and the F22
  two-module test.

**The upstream patch is 174 changed lines across 7 files.** F9 and F10 needed
no new instrumentation at all -- only analysis of traces the existing scopes
already produced. The growth from 104/4 is the sampling and tokenizer scopes,
which `llama-bench` never reaches.

**Not done** — the honest list is in [`FINDINGS.md`](FINDINGS.md) under
"Not yet measured" and [`02`](02-overhead-methodology.md) under "Remaining gaps".
The short version:

| Gap | Why it matters |
|---|---|
| Linux / GCC never built or measured | Everything so far is MSVC on Windows. **The barrier-path half of this gap was wrong for four sessions (F26):** F9/F10 measured the *OpenMP* path, which is also Linux's default, so what is untested is GCC and the `libgomp` runtime, not a different barrier algorithm. **F27 then measured the other barrier on this machine** and found it 3-5x more expensive, so every barrier figure here is from the cheaper implementation, not the pessimistic one |
| ~~No real quantized model~~ | **Done (F12).** Qwen2.5-0.5B Q4_K_M is in `models/`, gitignored. Largest real model measured is 630 M params |
| ~~Shared-library build is BROKEN~~ | **Fixed (F22).** F18's option 2, implemented and verified: each module caches its own `ts_tls`, all resolving to one registry-owned buffer. `BUILD_SHARED_LIBS=ON` links and traces correctly. Two-module regression test passes on all three toolchains; **llama.cpp shared on Linux still untested**, and the shared build's overhead has never been measured |
| ~~Sampling / tokenizer scopes not written~~ | **Done (F11).** `llama-cli` is now built in `build-ts-on`. Sampling + detokenization are 0.13% of a token |
| ~~Context-shift behaviour~~ | **Done (F16).** 4 spikes in 699 tokens at `-c 256`, 1.13-1.34x median |
| Concurrent sequences / server workload | The last untested prediction in F2, and F16 says it is still plausible: the head-pointer trick that makes `find_slot` O(1) is much weaker with many streams |
| ~~Larger real model~~ | **Done (F19, F21).** Three more real models measured, to 8.19 B. Biggest is now 8.19 B; **no MoE model at all**, which is the clearest remaining gap. Session 4 asked and was told **not to download one** — it needs several GB and free RAM is ~7 GB against the 8B's 4.86 GiB. Ask again rather than assuming |
| ~~No Perfetto screenshot~~ | **Sidestepped in session 5.** `tools/trace_svg.py` renders one token from a committed trace as a theme-aware SVG, and the README opens with it. That is better than a screenshot for a repo -- it is text, it diffs, and anyone who clones can regenerate it -- but it is **not** the Perfetto UI, and a post that wants to show the UI still wants a screenshot |
| ~~No thread pinning~~ | **Done (F14).** Mechanism confirmed: homogeneous cores drop spread 13%->2% and halve barrier wait. Pinning is not the fix |
| Upstream issue not filed | Two issues now, and [`03`](03-upstream-issue-draft.md) says which goes first. **The F20 naming defect is not blocked on Linux** and should be filed on its own; the instrumentation proposal still is. **An agent must not write or file it** — see the box at the top of `03` |
| ~~Shared build's overhead still unmeasured~~ | **Done (F30).** +0.87% [+0.55, +1.19] at level 3, certified. F25 was right that the fix was a harness change and not more reps: `bench_overhead.py` now takes N build pairs with `--pair` and round-robins every arm of every pair together. What is *still* open is the difference between the builds, which came out bounded but unresolved at +0.36pp [-0.32, +1.17] |
| F24 not raised upstream, and `mul_mat_id` untested | The +1.95% is one machine, one thread count, one model, and no NUMA hardware — and NUMA is what the constant was tuned for |

---

## 2. Push first. Check before anything else.

```bash
git log --oneline origin/main..main    # MUST be empty
git push origin main
```

**Session 3 ended with everything pushed**, including the 15 commits session 2
left stranded. That is the first time this has been true, so do not assume it
stays that way — run the check anyway.

Session 3's outage was total, not GitHub-specific: `ping 1.1.1.1` lost 100% of
packets and DNS to 8.8.8.8 timed out, for roughly an hour. Diagnosing that took
one command and was worth it, because "GitHub is blocked" and "this machine has
no network" call for different responses. Then it came back with no warning and
the first retry succeeded.

**What worked: a retry loop in the background** (`git push` every 45s, up to 40
times) started early and left alone while the real work continued. It landed on
its first attempt after the network returned, with no further attention. Do that
at the *start* of a session rather than pushing by hand between commits.

Networking here is **intermittent, not blocked**. Session 1 concluded GitHub was
specifically unreachable and stopped retrying, which is why 14 commits sat local
for a whole session. Session 2 saw a `git push` succeed while a `curl
https://github.com` seconds later still failed, a `git fetch` die mid-protocol
with `expected flush after ref listing`, a Hugging Face download of 469 MB
complete without a hiccup, and then hours where nothing connected at all.
Session 3 lost all networking for about an hour and got it back without doing
anything.

**The rule: a failed connection says nothing about the next one. Retry, and
retry again later.**

## 3. Environment — what had to be set up, and the traps

### Toolchain

There was no `cmake`, no `ninja`, and nothing on `PATH` at the start. What
worked:

- **MSVC 19.44** exists at
  `C:\Program Files\Microsoft Visual Studio\2022\Community` but the VS-bundled
  CMake is **not** installed (the "C++ CMake tools" component is missing).
- `pip install cmake ninja` supplies both (cmake 4.4.3, ninja 1.13.2). They land
  in `C:\Users\priya\AppData\Local\Programs\Python\Python312\Scripts`, already
  on `PATH`.
- Every build must run under `vcvars64.bat`. There is a helper in the scratchpad:

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
%*
```

  Recreate it and invoke as `cmd /c "vc.bat cmake --build ..."`.

### Paths

```
C:\-CS\TLI profiler\
├── tokenscope\            the repo (git; origin github.com/PS12007/tokenscope)
│   ├── build-on\          ninja, TOKENSCOPE_ENABLED=ON   -> ts_selftest.exe
│   ├── build-off\         ninja, TOKENSCOPE_ENABLED=OFF  -> the 0-symbol proof
│   └── build-shared\      ninja, ENABLED=ON + BUILD_SHARED_LIBS=ON
│                          -> ts_selftest + ts_dlltest (F22's two-module test)
├── llama.cpp\             upstream clone, pinned at 4d91760, PATCHED IN PLACE
│   ├── build-ts-on\       ninja, static, GGML_TOKENSCOPE=ON
│   │                      bin\llama-bench.exe, llama-cli.exe, llama-batched.exe
│   ├── build-ts-off\      ninja, static, GGML_TOKENSCOPE=OFF
│   │                      bin\llama-bench.exe   <- ALL throughput numbers
│   ├── build-ts-shared\   VS 17 2022, BUILD_SHARED_LIBS=ON, TOKENSCOPE=ON
│   │                      bin\Release\llama-bench.exe  (needs --config Release)
│   ├── build-ts-shared-off\  NEW in session 5. Same as build-ts-shared with
│   │                      GGML_TOKENSCOPE=OFF. Item 7 could not run without it
│   ├── build-ts-nshared-on\   NEW in session 6, F31. ninja, BUILD_SHARED_LIBS=ON
│   │                      + GGML_TOKENSCOPE=ON. Exists to hold the generator
│   │                      constant against build-ts-shared, which is MSBuild
│   ├── build-ts-nshared-off\  NEW in session 6, F31. Same, TOKENSCOPE=OFF
│   ├── build-ts-noomp-on\    NEW in session 5, F27. ninja, static,
│   │                      GGML_OPENMP=OFF + GGML_TOKENSCOPE=ON
│   └── build-ts-noomp-off\   NEW in session 5, F27. GGML_OPENMP=OFF + OFF, the
│                          F27 throughput arm. Verify a noomp build by grepping
│                          the exe for VCOMP -- it must find nothing
└── models\
    ├── tiny.gguf          8L,   34 MB  synthetic F32
    ├── mid.gguf           24L, 840 MB  synthetic F32   <- the workhorse
    └── qwen-q4km.gguf     Qwen2.5-0.5B-Instruct Q4_K_M, 469 MB, REAL
```

### Exactly what is non-stock in `llama.cpp`, as of the end of session 4

`models/` and `llama.cpp/` are gitignored. The clone is **modified in place** and
`git -C ../llama.cpp status --short` should show precisely this:

```
 M ggml/CMakeLists.txt          | patch 01, the instrumentation
 M ggml/src/CMakeLists.txt      |   (regenerate with bootstrap.py --make-patch)
 M ggml/src/ggml-cpu/ggml-cpu.c |
 M src/llama-context.cpp        |
 M src/llama-kv-cache.cpp       |
 M src/llama-sampler.cpp        |
 M src/llama-vocab.cpp          |
 M src/llama-graph.cpp            patch 02, the F20 naming fix -- NOT in patch 01
?? ggml/src/tokenscope/          copies of src/tokenscope.*, build inputs
```

Anything else in that list is something a previous session left behind and did
not write down.

**`patches/03-mulmat-chunk-threshold.patch` (F24) is NOT applied.** The tree is
at stock `nth * 4` at both `ggml-cpu.c:1422` and `:1698`. Check with:

```bash
grep -n 'nchunk0 \* nchunk1 <' ../llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c
```

Two `nth * 4` lines means stock. To reproduce F24, `git -C ../llama.cpp apply`
the patch, rebuild **both** `build-ts-on` and `build-ts-off`, measure, then
reverse-apply it. It is deliberately kept out of `01-instrument.patch` even
though `ggml-cpu.c` is in `bootstrap.py`'s `TOUCHED` list, for the same reason
`llama-graph.cpp` is: it is a behaviour change and has nothing to do with the
instrumentation. **`--make-patch` while it is applied would silently fold a
scheduler change into the instrumentation patch.**

Three trees, one source of truth: `tokenscope/src/tokenscope.*` is authoritative
and `llama.cpp/ggml/src/tokenscope/` are copies. Session 4 ended with them in
sync; verify before believing a build:

```bash
for f in tokenscope.h tokenscope-ggml.h tokenscope.cpp; do
  diff -q src/$f ../llama.cpp/ggml/src/tokenscope/$f
done
```

That drifted once during session 4 — a comment added to `src/tokenscope.h` after
the copy — which is harmless only because it was a comment.

### The models Ollama already has

**A real 8B is already on disk**, pulled by Ollama before this project started,
so nothing about it needs a download. Ollama stores GGUF blobs unmodified and
content-addressed; `llama-bench -m` opens one directly:

```
~/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f
```

That is **Qwen3 8B Q4_K_M**, 4.86 GiB, 8.19 B params, 36 layers, `n_embd` 4096,
`n_ff` 12288, GQA 32/8 — 13x the parameters of the Qwen2.5-0.5B used in F12.
Read the metadata with `gguf-py` rather than trusting the tag; the manifest at
`~/.ollama/models/manifests/registry.ollama.ai/library/qwen3/8b` maps tags to
blobs. `dolphin-llama3` (8B) and `dolphin-mistral` (7B) are there too, and **both
were measured in F21** — blobs `sha256-ea025c10...` and `sha256-11a57a9b...`
respectively. **No MoE model is present**, and session 4 asked and was told not
to download one.

**The constraint is RAM, not disk.** This machine has 15.7 GB total and about
7 GB free, against a 4.86 GiB model. It fits and it does not thrash — measured
38.78 pp32 / 7.39 tg16 tok/s at 8 threads — but the margin is thin enough that
anything else running can page the weights out and quietly corrupt a decode
number, because decode is bandwidth-bound (F14). Check free memory before
trusting a run at this size, and again after.

**One consequence of the F20 naming patch, for reading old traces.** Because
`llama-graph.cpp` carries it, traces from `build-ts-on` are not comparable with
pre-session-3 traces in one respect: `attn.out` appears and `~attn` nearly
vanishes. That is the fix working, not a regression. `git -C ../llama.cpp
checkout src/llama-graph.cpp` reverts it if a comparison ever needs the old
behaviour.

**And a trap that cost real time in session 4, generalised.** An A/B where one
arm is a binary built earlier is not an A/B. `build-ts-off`'s binary was three
days older than the tree, so it silently lacked the F20 patch, and a throughput
comparison came out **+2.71% when the true figure was +1.95%** — with clean
non-overlapping ranges, which made it *more* convincing rather than less. Before
trusting any A/B, check the binary's timestamp against `git -C ../llama.cpp
status`, and build both arms in the same session. (F24.)

**`llama-cli` is required** for anything involving sampling, tokenization or
context shift (F11, F16) -- `llama-bench` calls none of them. It is built in
`build-ts-on`; add it to the `--target` list when rebuilding.

**The machine matters for F10/F14.** i7-14700HX: 8 P-cores + 12 E-cores, 28
logical. P-cores are 2.88x faster than E-cores on compute-bound work (and
indistinguishable on decode, which is bandwidth-bound). Affinity via
`llama-bench -C <hex> --cpu-strict 1` works; masks used were `0x5555` (one
thread per P-core, but see F14's unexplained anomaly) and `0x0fff0000` (the 12
E-cores, which behaves cleanly).

**Env vars found in session 2:** `GGML_CPU_DISABLE_FUSION=1` turns off ggml's
only op fusion (RMS_NORM+MUL) at runtime -- that is what made F15 measurable
without patching anything.

`models/` and `llama.cpp/` are gitignored deliberately. **Both builds are
needed** — the overhead harness compares them, and rebuilding `build-ts-off`
from scratch takes several minutes.

### Traps hit, so they are not hit again

- **Bash heredocs fail on large documents** in this environment. A ~300-line
  markdown heredoc died with `unexpected EOF while looking for matching '`.
  Use the Write tool for long files; heredocs are fine for short commit
  messages.
- **`git commit -m` with multi-line bodies** is unreliable here too. Write the
  message to a file and use `git commit -F`.
- **A stalled `curl` holds its output file open**, so `rm` fails with `Device or
  resource busy`. `Stop-Process -Name curl -Force` first.
- **Hugging Face downloads kept truncating** in session 1 -- a Qwen2.5-0.5B
  GGUF stopped at 86 KB, then 0 bytes, with `curl` still exiting 0. **In
  session 2 the same download succeeded**, complete and byte-exact, using
  `curl -L --fail --retry 5 --retry-delay 3 --retry-all-errors -C -`. Verify
  the size against the HF API tree listing and check the GGUF magic before
  trusting a downloaded model. Pass `-s`; the progress meter floods the
  transcript.
- **CMake 4.x** rejects `cmake_minimum_required(VERSION < 3.5)`. llama.cpp is
  fine; other projects may not be.

---

## 4. Rebuilding from scratch

```bash
python scripts/bootstrap.py --dest ../llama.cpp    # clone, copy sources, apply patch

cmake -S ../llama.cpp -B ../llama.cpp/build-ts-on  -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF -DGGML_TOKENSCOPE=ON -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build-ts-on --target llama-bench

# same with -DGGML_TOKENSCOPE=OFF into build-ts-off

# The shared-library arm (F22). Visual Studio generator, so --build needs
# --config Release; the default is Debug and it will silently give you one.
cmake -S ../llama.cpp -B ../llama.cpp/build-ts-shared -DBUILD_SHARED_LIBS=ON \
      -DGGML_TOKENSCOPE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build-ts-shared --config Release --target llama-bench

python tools/make_tiny_model.py --llama-cpp ../llama.cpp -o ../models/mid.gguf \
       --layers 24 --embd 768 --heads 12 --heads-kv 4 --vocab 8192
```

And the repo's own three, which need no llama.cpp, no model and no network. The
third is new in session 4 and is what CI runs to guard F22:

```bash
cmake -S . -B build-on     -DCMAKE_BUILD_TYPE=Release -DTOKENSCOPE_ENABLED=ON
cmake -S . -B build-off    -DCMAKE_BUILD_TYPE=Release -DTOKENSCOPE_ENABLED=OFF
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release -DTOKENSCOPE_ENABLED=ON \
      -DBUILD_SHARED_LIBS=ON
cmake --build build-on && cmake --build build-off && cmake --build build-shared
ctest --test-dir build-on          # selftest
ctest --test-dir build-shared      # selftest + dlltest (two modules, one buffer)
```

`build-shared` adds three targets that only exist in that configuration:
`tokenscope_shared` (the DLL with the registry), `ts_dllmod` (a second module,
linked) and `ts_dynmod` (a third, opened at runtime). If `dlltest` ever fails on
its *first* assertion — "each module has its own ts_tls cache" — the modules have
collapsed into one binary and every other assertion in it has gone vacuous;
fix that before reading the rest.

### The edit loop that is easy to get wrong

`src/tokenscope.*` in this repo is the **source of truth**. The copies in
`llama.cpp/ggml/src/tokenscope/` are build inputs. After editing:

```bash
cp src/tokenscope.h src/tokenscope-ggml.h src/tokenscope.cpp \
   ../llama.cpp/ggml/src/tokenscope/
# rebuild, then, if upstream FILES changed (not tokenscope's own):
python scripts/bootstrap.py --make-patch --dest ../llama.cpp
```

Forgetting the `cp` means rebuilding the old code and debugging a fix that is
already correct. It cost time in session 1.

**And if you instrument a file that is not already in `bootstrap.py`'s
`TOUCHED` list, `--make-patch` silently regenerates the *old* patch and reports
success.** That happened twice in session 2. The list is now:
`ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`,
`ggml/src/ggml-cpu/ggml-cpu.c`, `src/llama-context.cpp`,
`src/llama-sampler.cpp`, `src/llama-vocab.cpp`, `src/llama-kv-cache.cpp`.
Add to it the moment you touch a new file.

---

## 4b. What the tools do now

```bash
python tools/trace_analyze.py T.json                 # summary (host / node / outside-slice)
python tools/trace_analyze.py T.json --tokens 40     # per-token table
python tools/trace_analyze.py T.json --outliers 10   # slowest tokens, attributed
python tools/trace_analyze.py T.json --layers 6      # per-layer thread time
python tools/trace_analyze.py T.json --barriers 12   # NEW: imbalance vs release, per node
python tools/trace_analyze.py A.json --diff B.json   # did my change help, and where
python tools/mulmat_chunking.py M.gguf -t 8,16,28    # NEW: matmul partitioning mode
python tools/imbalance_repeat.py -m M.gguf -t 8,16 -n 12 --ratio ffn_up/ffn_out
python tools/mulmat_chunking.py M.gguf -t 16 --mult 2   # model a PATCHED build
python tools/ab_throughput.py --a old.exe --b new.exe -m M.gguf -t 16 -n 20
python tools/imbalance_repeat.py -m M.gguf --metric release   # NEW: the other
                                        # half of --barriers, repeated (P27.2)
python tools/spinup_probe.py -m M.gguf --windows 1:2,40:41    # NEW: is the big
                                        # after-arrival barrier ours? (P28)
python tools/trace_svg.py T.json -o docs/token.svg            # the README
                                        # picture, from a trace, no Perfetto
python tools/bench_overhead.py -m M.gguf -n 20 -t 8 --levels 3     --pair static=../llama.cpp/build-ts-off/bin,../llama.cpp/build-ts-on/bin     --pair shared=../llama.cpp/build-ts-shared-off/bin/Release,../llama.cpp/build-ts-shared/bin/Release
                                        # NEW: N build pairs, every arm of every
                                        # pair in ONE round-robin (F30). This is
                                        # the only way a cross-build comparison
                                        # is interleaved at all
```

`--pair` is repeatable and each pair is scored against **its own** compiled-out
arm. With more than one pair you also get the difference of overheads with its
own bootstrap CI, and the compiled-out arms against each other -- which measures
DLL cost in llama.cpp, not in tokenscope. `--bin-off`/`--bin-on` still work.
Arm order rotates each round since F30; `--no-rotate` restores the old fixed
order if you need to reproduce a pre-`9ec7c82` number.

`--barriers` needs `TOKENSCOPE_LEVEL=3`. It matches the k-th barrier across
threads, which is exact because node and barrier scopes strictly alternate and
every thread walks the same node list -- and it **refuses to report anything**
rather than guess if that ever stops holding.

`split_totals` now separates per-token work *inside* the decode slice from work
*outside* it (sampling, detokenization run after `decode` returns). Charging the
latter against the former is what pushed attribution over 100% in F11.

**Nine reference traces** in `examples/`, **1164 decode tokens**, ~21 MB, all
exercised by CI. (The table said five and 1012 until session 4 counted them; the
three 8B/batched ones were added in sessions 2-3 without the table being
updated.)

| trace | level | decode tok | what it covers |
|---|---|---|---|
| `mid-24L-tg256` | 1 | 257 | host scopes, the F1 control-plane result |
| `mid-24L-L3-tok10-11` | 3 | 25 | the F9 barrier data, and F23's baseline |
| `mid-24L-L3-pp64` | 3 | 0 | level-3 prefill, the other half of F9's test |
| `mid-24L-cli-sampling` | 1 | 31 | `llama-cli`, sampling + detokenization (F11) |
| `qwen-ctxshift-c256` | 1 | 699 | real model, context shift, `find_slot` (F16) |
| `qwen-batched-np16` | 1 | 92 | 16 concurrent sequences (F17) |
| `qwen3-8b-L3-tok8-13` | 3 | 17 | the 8B, before the F20 naming fix |
| `qwen3-8b-named-attnout` | 3 | 17 | the 8B, after it — the second `trace_svg.py` figure comes from here |
| `mid-24L-L3-tok10-11-f28` | 3 | 26 | **new in session 5.** Same model, window and thread count as `mid-24L-L3-tok10-11`, taken after the F28 fix. Median of seven candidate runs by total imbalance, not the best one. The README's barrier split and its timeline figure both come from here |

**CI mechanizes five claims**, and each fails the build if the finding stops
being true: attribution never exceeds 100% (inside-slice only), every barrier is
matched by `--barriers`, F9 (elementwise nodes serial in decode, parallel in
prefill), F16 (shifts visible, `find_slot` a small part of its scope), and
**F22 (two modules, one buffer per thread)**.

The F22 one is `ctest --test-dir build-shared`, built on all three platforms
from `BUILD_SHARED_LIBS=ON`. It needs no llama.cpp, no model and no network, and
runs in 0.04 s. It exists because F18 — the shared build not linking at all —
went undetected for a session purely because nothing here ever built a DLL.

---

## 5. Next steps, in the order I would do them

**Session 5 closed item 3 (the picture, by another route) and half of item 7
(F25), and added item 8 out of F27.** The Linux item (2) is smaller than it
looked — F26 found its barrier-path premise false — and item 1 is unchanged and
still needs a person.

**Session 4 closed item 6 (F22) and item 5 (F23/F24).** The ordering below is
unchanged otherwise, but the shape of the project has changed: it now has a
measured, certified performance result in ggml itself, which is a different kind
of thing to take upstream than a profiler. Items 1 and 5 are both "decide
whether to raise this", and **neither can be done by an agent** — see the box at
the top of [`03`](03-upstream-issue-draft.md).

The two items that need hardware this machine does not have are 2 (Linux) and
the SMT question inside 5. The two that need only a person are 1 and 3.


Session 3 closed items 3 and 4 (concurrent sequences had already gone in F17;
the 8B is F19/F21) and added a new one at the top that did not exist before.

1. **File the F20 naming issue.** First because it is the only piece of this
   work that is *not* blocked on Linux, and the smallest thing a maintainer
   could say yes to. **The evidence is assembled; the prose has to be yours** — see
   [`03`](03-upstream-issue-draft.md), "Evidence pack for the F20 issue".
   A first attempt at this wrote a finished issue body, which llama.cpp's
   `AGENTS.md` forbids in terms: an agent must never write a PR description, a
   comment, or a reviewer response, and `gh issue create` is listed among the
   things not to run on a user's behalf. The measurements, file references and
   diff are all there; write it short and in your own words, and note the house
   style is ASCII-only with no em-dashes. The deeper reason is the same one
   AGENTS.md gives: the contributor has to be able to explain the change to a
   reviewer without AI assistance, and that is a bar for a person, not a
   session. The draft
   opens with three things to re-check first, because all three go stale: that
   `build_attn` still has seven overloads and still does not name the output
   projection at current `master` (the measurement is pinned at `4d91760`), that
   `CONTRIBUTING.md` has not changed, and that nobody has filed it already. `patches/02-name-attn-output.patch` is applied and measured;
   [`03`](03-upstream-issue-draft.md) has the framing. llama.cpp's `AGENTS.md`
   asks for an issue before a PR **and** asks that the contributor own the change
   and be able to defend it unaided — which for a change this size is a fair bar
   and worth meeting deliberately before filing.
2. **Linux + GCC**, unchanged from session 2 and still the blocker for the main
   upstream conversation. The detail is in section 5 of the previous revision
   and still accurate: `ggml_graph_compute_thread` is shared, so the scopes fire,
   and `ggml_barrier` is `#pragma omp barrier` -- **which is also what was
   measured here**, contrary to what this item said until session 5 (F26).
   F9's *structural* claims should transfer; every barrier *cost* number is a
   `vcomp` 2.0 number and may not transfer to `libgomp`, which is a narrower
   worry than the one this item used to state -- but **F27 showed the
   worry is not small**: swapping `vcomp` for ggml's own threadpool on this
   machine changes release latency by 2-5x and decode throughput by up to 54%,
   so barrier implementation is worth far more than it looked.
3. ~~**Perfetto screenshot.**~~ **Largely done in session 5, by other means.**
   `tools/trace_svg.py` draws one token per-thread from a committed trace and
   the README leads with `docs/token-timeline.svg`; a second figure from the 8B
   is at `docs/token-timeline-8b.svg`. What is left is only the part that
   genuinely needs a browser: an actual Perfetto screenshot for post B5, where
   the point is partly "this opens in the tool you already use". Open
   `examples/qwen3-8b-named-attnout.trace.json` there and zoom to 2-3 tokens.
4. **An MoE model.** The clearest remaining gap in the byte law. F21 covers three
   architectures and a 13.7x range of `lm_head` share, but every model measured
   is dense, and MoE is the case where bytes-streamed-per-token stops being a
   property of the file and starts depending on the router. The law as stated
   would predict expert phases from *stored* bytes and should be **wrong** there,
   which makes it the most informative test available.
5. **The mul_mat chunking threshold — which is what item 5 turned into.**
   Item 5 used to read "proportional row assignment", on the premise that ggml
   hands every thread an equal share of rows. **F23 found that premise is only
   half true**: `ggml_compute_forward_mul_mat` steals work from a shared atomic
   counter whenever `nchunk0 * nchunk1 >= nth * 4`, and a matmul in that mode has
   roughly **a third to a half** the arrival imbalance per unit work of one that is not
   (0.34–0.56 against 0.94–1.51, two models, **twelve runs per arm**, two of the
   comparisons with non-overlapping ranges). Work stealing needs no
   model of core speed, so for large matmuls ggml already solves what
   proportional assignment was going to solve, and solves it better.

   What is left is sharper and much smaller. That threshold contains `nth`, so
   **adding threads can turn the load balancer off** for a model's biggest
   matmuls — on `mid.gguf` between 8 and 16 threads, on Qwen2.5-0.5B between 16
   and 28 — and nothing reports it. The experiment is to lower the multiplier or
   make `chunk_size` adapt to `nth`, and see whether the flip stops costing.
   `tools/mulmat_chunking.py` says which matmuls flip and where, from the GGUF
   alone.

   **Session 4 did this (F24).** `nth * 4` -> `nth * 2` in `mul_mat` alone is
   **+1.95% [+1.59, +2.35] on decode, certified**, with prefill as an
   uncertified control. `patches/03-mulmat-chunk-threshold.patch` holds it and is
   **not applied to the tree**. What is left on this item is no longer "measure
   it" but "decide whether to raise it", and the honest framing is a question
   about a constant backed by a measurement, not a patch claiming to know better
   — there is no NUMA hardware here and NUMA is what the constant was tuned for
   (PR #6915). `mul_mat_id` carries the identical threshold, is the MoE path, and
   is untested and unchanged.

   Anything further here still needs **n≥12 per arm**, because on this evidence
   six cannot tell a 3× effect from noise.

   Two things F23 leaves open. The ops that are *not* matmul still use the flat
   `dr = (nr + nth - 1)/nth` that F14's 2.88× applies to, and they are where
   proportional assignment might still have a case. And SMT is unseparated from
   core heterogeneity: telling them apart needs two arms at one thread count,
   one sharing physical cores and one not, with `ffn_up` static — which needs
   ≥16 threads, and 16 threads without SMT needs more than this machine's 8
   P-cores, so the no-SMT arm has to bring in E-cores and the arms then differ
   in core type too. **The confound is in the hardware.** It wants a machine
   with homogeneous cores.
6. ~~**Fix the shared-library build (F18).**~~ **Done in session 4 (F22).** What
   is left of it: the shared build's **overhead has never been measured** (the
   hot path is unchanged by inspection, but that is not a measurement), and the
   GCC/ELF behaviour still needs checking alongside item 2. Both fold into
   items 2 and 7 rather than standing on their own.
7. ~~**Measure the shared build's overhead.**~~ **Done in session 6 (F30).**
   `BUILD_SHARED_LIBS=ON` at level 3, 8 threads, is **+0.87% [+0.55, +1.19]**,
   certified — the measurement behind F22's inspection-only claim that
   per-module `ts_tls` caching keeps a cross-DLL call off the hot path. The
   static figure from F25 stands at **+1.16% [+0.67, +1.87]**.

   F25 was right that the fix was a harness change. `bench_overhead.py` takes
   `--pair NAME=OFF_DIR,ON_DIR` repeatably and round-robins every arm of every
   pair in one invocation, and it reports two things a single pair cannot: the
   **difference of overheads** with its own bootstrap interval, and the
   **compiled-out arms against each other**, which measures what DLL boundaries
   cost llama.cpp rather than anything about tokenscope.

   Two things are left, and both are small. The difference of overheads is
   **+0.36pp [-0.32, +1.17]** — bounded but not resolved away from zero, and
   resolving it needs either a quieter machine or many more reps, and is not
   obviously worth either. And **the two pairs still use different generators**
   (Ninja for static, Visual Studio for shared), which P30 named in advance as
   the confound this design cannot remove; a Ninja shared build would remove it
   for maybe twenty minutes of configure-and-build.

   F30 also found and fixed a flaw in the harness's own protocol: arm order
   within a round was fixed, so any transient shorter than a round landed on the
   same arms every time. It cost the static pair its certification in F30's run.
   Order now rotates each round. **Any overhead number taken before commit
   `9ec7c82` was measured with a fixed order.**

8. **F27 is the biggest unexploited result in the repo, and it needs a second
   machine before it means anything general.** Turning `GGML_OPENMP` off costs
   54% of decode at 28 threads *here*. That is one hybrid x86 CPU running MSVC
   `vcomp` against ggml's spin-wait; a homogeneous server part with `libgomp`
   could plausibly reverse the sign. Whether it is worth raising upstream is a
   judgement for a person, and the same `AGENTS.md` rules in [`03`](03-upstream-issue-draft.md)
   apply: an agent must not file it or write the text. What an agent *can* do is
   run the same protocol elsewhere — `tools/ab_throughput.py` plus
   `tools/imbalance_repeat.py --metric release` on two builds differing only in
   `GGML_OPENMP`, n>=12.

## 6. Things I would tell myself

From session 1, still true:

- **The consistency checks earned their keep.** Two real bugs were caught by
  "these percentages cannot exceed 100". Keep adding checks of that shape.
- **The benchmark harness refusing to answer is a feature.** It declined four
  more times in session 2 and was right each time.
- **Do not run builds while benchmarking.**
- **Check documented features actually work.** A feature is not done when the
  code exists, only when something calls it.

Added by session 2, in rough order of how much time they would have saved:

- **Write predictions down before you can test them, and date them.** Four were
  tested in session 2. Two were wrong. Without the written version I would have
  remembered predicting whichever turned out right. This is now the single most
  valuable habit in the project.
- **A scope's name is a claim about what it measures** (F16). `kv.slot-search`
  wrapped `init_batch`, and the search was 2.7% of the number. F2 then reasoned
  from the name rather than the code and got two things wrong for one reason.
  When a scope's number looks interesting, re-read what it actually wraps.
- **Barrier count and barrier cost are different quantities** (F15). Removing
  10.6% of the barriers changed throughput by nothing measurable. The ones you
  can cheaply remove are the ones nobody was waiting at.
- **A null result is a claim about your workload.** P-core vs E-core on decode
  reads 28.0 vs 28.2 tok/s and looks like a broken CPU mask. It is not; decode
  is bandwidth-bound. The same masks on prefill give 265.95 vs 92.39. Nearly
  threw out F14 over this.
- **Trace-derived throughput is not throughput.** A traced token carries the
  recording cost on exactly the token being measured. Every tok/s number in
  FINDINGS comes from the *uninstrumented* build for this reason; a single r=1
  traced run briefly showed a 21% degradation that was pure noise.
- **The Bash tool mangles `
` inside heredocs.** Backslash escapes in Python
  written via `<<'EOF'` came out as literal newlines and broke the file three
  times. Use the Write tool for anything with escapes, or build strings with
  `chr(92)`. Line-range edits located by content are more reliable than
  `str.replace` on text containing em-dashes or `×`.
- **`bootstrap.py --make-patch` silently regenerates the old patch** if a newly
  touched file is not in its `TOUCHED` list. It reported success while ignoring
  two edited files. Add the file to the list *when you first edit it*.
- **curl progress output floods the transcript.** Use `-s` on any large
  download.
- **A feature can look dead because your test is too small.** `TOKENSCOPE_RING`
  produced byte-identical output to non-ring mode three times running, which
  reads exactly like the `TOKENSCOPE_TOKENS` bug from session 1. It works.
  Records come in 1 MiB chunks *per thread*, so ring only engages after a thread
  fills one (~43,700 records) -- all three tests were under that. Before
  declaring a feature inert, check its precondition is actually met. (It is
  genuinely inert at small budgets though, and the README now says so.)

Added by session 3:

- **A prediction can be right for a reason worth half of what it claimed.**
  P19.3 predicted `attn_v` would cost less than its 3.56x byte ratio because
  Q4_K needs dequantizing and F16 does not. It came in at 2.911, below, as
  predicted. But `Qcur`/`Kcur` — same dtype, differing only in size — undershoots
  *its* byte ratio by 10% too, so roughly half the effect was a size artifact
  present in both. **Check whether your mechanism is the only thing producing
  the sign you predicted**, using a pair where it cannot be operating.
- **Two mechanisms can move at once and you will model one.** P19.4 reasoned
  correctly about memory traffic, got the direction right and both specifics
  wrong, because F14's core-heterogeneity penalty stopped applying at the same
  time — at 8B every thread waits on memory, so slow cores cost nothing. When a
  prediction fails, check whether a *second* known mechanism changed regime.
- **The strength of a test is not the separation between the hypotheses.** I
  predicted the 8B would test the byte law weakly, because bytes and parameters
  only disagreed by 4.2 points there against 8.3 on the 0.5B. It was the
  sharpest result in the project — 0.3 points of residual. Separation and
  precision are different axes.
- **Look for the controlled experiment inside the data you already have.** The
  best result of the session cost nothing to produce: `ffn_down` is Q6_K in 18
  layers and Q4_K in 18, which is a paired experiment with controls sitting
  inside a file that had already been traced. Ask what varies *within* a
  workload before running another one.
- **A dead branch in a lookup table has no symptom.** `{ "kqv_out", ... }` was
  unreachable behind `{ "kq", ... }` for the whole project. Nothing failed,
  because no model had ever produced that node name. Found only by asking where
  a *hypothetical* name would land. The self-test now checks the table's shape
  rather than its behaviour on the inputs that happen to exist.
- **Verify a check by breaking the thing it checks.** After adding the shadow
  test I reintroduced the bug, confirmed it failed, and restored it. A check
  that has never failed is a check nobody has tested — this is the "check
  documented features actually work" lesson applied to the checks themselves.
- **`cmd /c "vc.bat <command>"` from the Bash tool silently opens an
  interactive shell** instead of running the command, and produces no error.
  Use the **PowerShell tool** with
  `cmd.exe /c "call `"$vc`" >nul 2>&1 && cd /d ... && <command>"` instead; that
  works reliably. This cost a wasted 7-minute background build.
- **The heredoc backslash trap from session 2 is still live and still bites.**
  A `\n` inside a `<<'EOF'` Python heredoc reached the file as a real newline
  and broke `ts_selftest.cpp` mid-build. The rule stands: **use the Write or
  Edit tool for anything containing escapes.**

Added by session 4:

- **When two requirements conflict, check whether they are really about the same
  object.** F18 stated a genuine incompatibility: the hot path needs a raw
  thread-local, the registry needs one instance across DLLs, and MSVC will not
  let one variable be both. Both halves were true and the conclusion did not
  follow, because `ts_tls` is a *cache* and the registry-owned buffer is the
  *state*. Nothing ever required the cache to be single-instance. The fix took
  an hour; the framing was the whole problem, and it sat unexamined for a
  session because "genuine incompatibility" reads like a finished thought.
- **The dangerous part of a small change is what it makes load-bearing.** The
  edit is ~40 lines. It also silently turned `ts_thread_init` into something
  that must be idempotent, made every caller responsible for writing the result
  back, and — worst — would have made level 2 record *no node data at all* in
  shared builds, with a valid trace, zero drops and no warning. That last one
  was found by asking "who initializes this, and in which module?", not by any
  test. **After changing where state lives, re-derive the initialization order
  for every reader of it.**
- **Reproduce the failure before fixing it.** One command, and it converts "it
  builds" from a hope into evidence. The `LNK1120` is in F22 for the same
  reason.
- **Verify a shared-state fix by counting, not by looking.** The trace says four
  worker threads at `-t 4`. Had the DLLs each built their own per-thread state,
  it would say eight — the records would all still be present, the trace would
  still parse, and every per-thread percentage would be computed on half a
  thread. A count that *could* have come out wrong is worth more than a table
  that looks right.
- **`Select-String` on a here-string of Python is not worth it.** Inline Python
  through the PowerShell tool mangled quotes twice. The session-2 rule
  generalizes: **anything with escapes or quoting goes in a file via the Write
  tool**, then gets run. This applies to PowerShell here-strings as much as bash
  heredocs — and to *backticks*, which bash command-substitutes inside double
  quotes: `python -c "...markdown with \`code\` spans..."` silently deleted two
  spans from HANDOFF and reported success, exactly the shape of the session-2
  backslash trap. Markdown is full of backticks. Use the Edit tool for it.
- **An A/B where one arm is a binary you saved earlier is not an A/B.** F24's
  first throughput run said **+2.71% with non-overlapping ranges** and was wrong.
  `build-ts-off` had last been built on 5 Sept; `llama-graph.cpp` changed on
  6 Sept for the F20 naming patch. So the "stock" binary predated a change the
  patched one contained, and the arms differed by more than the line under test.
  Rebuilding both from one tree took it to +1.49% at n=12 and +1.95% at n=20.
  Nothing looked wrong — the clean separation made it *more* convincing. This
  project already knew to interleave arms in **time**; it had not written down
  that they have to be matched in **version**. Check binary timestamps against
  `git status` before trusting any comparison.
- **Read the target project's `AGENTS.md` before writing anything aimed at it.**
  llama.cpp's forbids an agent writing PR descriptions, issue comments or
  reviewer responses, non-overridably, and lists `gh issue create` among things
  not to run for a user. Session 4 wrote a finished, paste-ready issue body for
  F20 before reading it, and had to reframe the whole section as an evidence
  pack. The measurements were the valuable part anyway; the prose was the part
  that was not wanted. `docs/03` now leads with that rule.
- **One trace is one draw, and six are not many more.** This cost three wrong
  turns in one session. A single trace inverted F23's conclusion. Six runs then
  produced a "28-thread anomaly" that does not exist — twelve runs give 1.041
  where six gave 0.558 — and **a whole prediction (P23.4) was written, committed
  and tested to explain it** before re-measuring showed there was nothing there.
  Six runs also put P23.4's own answer outside its predicted band at 0.747 where
  twelve give 0.944. Per-node imbalance spreads up to **8.5×** between identical
  runs. This project already knew not to trust one throughput number —
  `bench_overhead.py` has interleaved arms and bootstrap CIs, and has refused to
  certify six times — and then trusted trace-derived ratios at n=1 and n=6
  anyway, because the tracing is exact so the numbers *looked* exact.
  **Before explaining a surprising number, re-measure it.** That is cheaper than
  the prediction it saves you writing.
- **Predict ratios against a control, not levels.** P23.1 predicted a number
  would rise, in a regime where every comparable number also rose; it held and
  meant almost nothing. P23.2 predicted a ratio against a node that did not
  change mode, and that one carried the entire finding. When drafting a
  prediction, ask what else moves at the same time and divide by it.
- **Check the instrument can return the quantity you are predicting.** P23.3
  named `lm_head` as its control. `--barriers` matches a node to the barrier
  *after* it, and `lm_head` is the last node of the graph, so the quantity does
  not exist for it. That is a cheaper check than the measurement it wasted.

Added by session 5, in rough order of how much they would have saved:

- **A claim about how your code was BUILT is checkable in one command, and a
  caveat is a claim.** Five documents said every barrier number here came from
  ggml's spin-wait threadpool. `grep GGML_USE_OPENMP build.ninja` would have
  cost four seconds in session 1 and the answer was the opposite (F26). It
  survived four sessions because it lived only in *caveats* — the part of a
  document that exists to say what a result does not cover, which is exactly the
  part nobody re-derives. Every `#ifdef` a finding's scope depends on is worth
  one grep.
- **An anomaly detector that also explains the anomaly has two outputs, and
  usually only one of them was measured.** `--barriers` correctly flagged a
  barrier holding three quarters of all release latency, correctly excluded it
  from the corrected split, and then explained it as "thread-pool spin-up". It
  was tokenscope's own allocator (F28). The detection was real; the attribution
  was prose in an authoritative voice. F16's lesson — a scope's *name* is a
  claim — applies to diagnostic *messages* too.
- **Predict the scaling of the quantity you will actually read.** P28.2
  predicted 1.5-2.5x going 8 -> 16 threads, arguing that a mutex serialises the
  work so the total cannot double. Measured 4.0-4.5x. The mutex does serialise,
  so the *wall* duration is linear — but `--barriers` reports **thread time**,
  and n threads each sit through the whole linear stall. Linear duration summed
  over n threads is quadratic. Right physics, wrong denominator.
- **"Interleave the arms" means the arms of the comparison you are making.**
  `bench_overhead.py` has interleaved within one invocation since session 1.
  F25's question was shared-versus-static, which is a comparison *between* two
  invocations, and the machine's noise floor moved by 2x between them. The
  prediction came back neither confirmed nor falsified, which is worse than
  wrong. Same trap as F24 in a different disguise: there the arms differed in
  version, here in time.
- **Real asymmetries are not automatically the relevant ones.** P27.4 found two
  genuine differences between the threading paths, wrote them down in advance,
  and both were microseconds — and both were paid by the arm that *won*. Writing
  a mechanism down early is still right; it just does not make the mechanism
  load-bearing.
- **Prediction and control are different jobs, and a control has to be able to
  not move.** `ab_throughput.py` warns that a certified control means the
  comparison is broken. In F27 the control certified because `GGML_OPENMP`
  changes every barrier in every graph including prefill's. The tool was right to
  complain and the complaint did not apply — which means the honest report is
  "this comparison has no control", not "the warning is spurious".
- **sscanf does not care what it leaves behind.** `"%u-%u"` then `"%u"` made
  `10:11` mean token 10 alone, silently, for two sessions (F29). The cases worth
  asserting in a parser are the ones it must *refuse*; the accepted ones are the
  ones somebody already tried by hand.
- **Pick a committed reference artifact by the median, not by eye.** The new
  reference trace is the median of seven candidate runs by total imbalance. F23
  paid to learn that one trace is one draw; a repo that quotes its luckiest run
  is the same mistake with a longer half-life.
- **The blocked item may be blocked on the wrong thing.** "Perfetto screenshot"
  sat at the top of the list for four sessions needing a browser and a person.
  What the README actually needed was a picture, and a picture generated from a
  committed trace is *better* for a repo than a screenshot: it is text, it
  diffs, and anyone who clones can regenerate it. Ask what the item is for
  before assuming its stated form.


---

## 7. Numbers to quote (all reproducible from the committed code)

| Claim | Value | Source |
|---|---|---|
| Upstream patch size | 174 lines, 7 files | `patches/01-instrument.patch` |
| F20 naming fix size | 8 added, 13 removed, 1 file | `patches/02-name-attn-output.patch` |
| Per-scope cost | 52.8 ns (2 clock reads + 1 store) | `ts_selftest` |
| Level 3 overhead, static, 8 threads | **+1.16% [+0.67, +1.87]**, certified, n=20 — **as of commit `7149794`**; a re-run on the post-F28 tree was refused at a 3.78% noise floor | [`FINDINGS`](FINDINGS.md) F25 |
| ...same, session 1 | +0.67% [+0.12, +1.67] | [`02`](02-overhead-methodology.md) |
| Level 3 overhead, shared build | **unmeasured** — every interval spanned zero at a 2.22% noise floor | [`FINDINGS`](FINDINGS.md) F25 |
| Zero-overhead-when-off | 0 symbols, 864-byte archive | [`02`](02-overhead-methodology.md) §2 |
| Shared-library build | links and traces correctly on MSVC; overhead unmeasured | [`FINDINGS`](FINDINGS.md) F22 |
| Matmul arrival imbalance, work-stealing vs equal-slice | **0.34-0.56x** against 0.94-1.51; two comparisons with non-overlapping ranges, n=12 per arm | [`FINDINGS`](FINDINGS.md) F23 |
| Per-node imbalance, spread over 12 identical runs | up to **8.5x** at 8 threads, 1.3-1.5x at 16 | [`FINDINGS`](FINDINGS.md) F23 |
| Decode speedup from `nth*4` -> `nth*2` in mul_mat | **+1.95% [+1.59, +2.35]**, certified, n=20 interleaved | [`FINDINGS`](FINDINGS.md) F24 |
| ...same patch on prefill (control) | -0.81% [-2.44, +0.35], **not** certified | [`FINDINGS`](FINDINGS.md) F24 |
| Barrier wait | 11.2% of worker thread time (post-F28 trace reads 11.5%) | [`FINDINGS`](FINDINGS.md) F6 |
| ...of which arrival imbalance | **84.6%**, on a post-F28 trace with no artifact to exclude | [`FINDINGS`](FINDINGS.md) F28 |
| Decode cost of `GGML_OPENMP=OFF`, 28 threads | **-54.17% [-55.05, -53.31]** certified; -2.06% at 8 threads | [`FINDINGS`](FINDINGS.md) F27 |
| ...same, `tiny.gguf` | **-78.64%** at 28 threads, -13.07% at 8 | [`FINDINGS`](FINDINGS.md) F27 |
| Release latency per unit work, ggml pool vs OpenMP | 1.95x at 8 threads, 4.9x at 16, both non-overlapping, n=12 | [`FINDINGS`](FINDINGS.md) F27 |
| Total barrier wait per unit work at 28 threads | 0.752 (ggml pool) vs 0.249 (OpenMP) | [`FINDINGS`](FINDINGS.md) F27 |
| tokenscope's own first-touch artifact, before F28 | 76-83% of all after-arrival barrier time, scaling as thread count SQUARED | [`FINDINGS`](FINDINGS.md) F28 |
| Barriers behind single-threaded nodes | 120 of 412 per token | [`FINDINGS`](FINDINGS.md) F9 |
| Upper bound on fixing that | 1.34% of graph wall time, and F15 found no reachable part | F9, F15 |
| Barriers removed by ggml's one fusion | 49 of 461 per token, for no measurable throughput | [`FINDINGS`](FINDINGS.md) F15 |
| Best speedup at any thread count | 2.21x F32 / 3.28x Q4_K_M, both at 6 threads | F10, F12 |
| `lm_head` share, Qwen2.5-0.5B Q4_K_M | 34% of decode thread time | [`FINDINGS`](FINDINGS.md) F12 |
| ...same tensor, Qwen3-8B, identical vocabulary | 10.5% | [`FINDINGS`](FINDINGS.md) F19 |
| ...dolphin-mistral-7B, 32k vocabulary | 2.8% | [`FINDINGS`](FINDINGS.md) F21 |
| Byte law, max error over 4 models | 0.3-0.5 points (0.5B: 2.9) | F19, F21 |
| ...predicting from parameter counts instead | wrong by 1.1-7.2 points | F12, F19, F21 |
| `ffn_down` Q6_K vs Q4_K layers, same shape | **1.419x measured, 1.458x predicted**, controls at 1.007/1.009 | [`FINDINGS`](FINDINGS.md) F19 |
| `attn_v` (F16) vs `attn_k` (Q4_K), same shape | 2.911x, byte ratio 3.556x | [`FINDINGS`](FINDINGS.md) F19 |
| Anonymous attention output projection | 7.1% of decode thread time, in all 7 `build_attn` overloads | [`FINDINGS`](FINDINGS.md) F20 |
| Peak decode speedup, Qwen3-8B | 3.01x at 8 threads (0.5B: 3.28x at 6) | [`FINDINGS`](FINDINGS.md) F19 |
| Prefill speedup, same model, 28 threads | **11.07x**, against 2.95x on decode | [`FINDINGS`](FINDINGS.md) F19 |
| Barrier wait, Qwen3-8B at 6 threads | 5.6% (F6 measured 11.2% on the F32 model) | [`FINDINGS`](FINDINGS.md) F19 |
| Phase time predicted from weight BYTES | within 2.9 points on a real quantized model | [`FINDINGS`](FINDINGS.md) F12 |
| ...predicted from parameter counts | wrong by 7.2 points on the same model | [`FINDINGS`](FINDINGS.md) F12 |
| Sampling + detokenization | 0.13% of a token | [`FINDINGS`](FINDINGS.md) F11 |
| Context shift, `-c 256` | 4 spikes in 699 tokens, 1.13-1.34x median | [`FINDINGS`](FINDINGS.md) F16 |
| KV cell search (`find_slot`) | 1.17 us/token, *falls* as the cache fills, 1.55 us at 16 sequences | F16, F17 |
| Batching 16 sequences | 3.06x aggregate throughput | [`FINDINGS`](FINDINGS.md) F17 |
| Parallel efficiency at 28 threads | 7% | [`FINDINGS`](FINDINGS.md) F10 |
| P-core vs E-core, compute-bound | 2.88x (265.95 vs 92.39 tok/s) | [`FINDINGS`](FINDINGS.md) F14 |
| Barrier wait, mixed vs homogeneous cores | 21.2% -> 11.1% at 12 threads | [`FINDINGS`](FINDINGS.md) F14 |
| Host control-plane work | 0.4% of decode | [`FINDINGS`](FINDINGS.md) F1 |
| Attention math (`attn.score`) | 0.7% of thread time | [`FINDINGS`](FINDINGS.md) F7 |
| Phase time vs parameter count | within 1.5–8.6% over a 27× range — **F32 only**, see the F12 rows | [`FINDINGS`](FINDINGS.md) F7 |

Everything above is measured on **synthetic F32 weights unless the row names a
real model**, MSVC Release, Windows 11, on an i7-14700HX (8 P-cores + 12
E-cores). Say so whenever quoting them.

**Four real models have now been measured** (Qwen2.5-0.5B, Qwen3-8B,
dolphin-llama3-8B, dolphin-mistral-7B) and the F19/F21 rows come from those.
The three 7-8B ones live in Ollama's blob store; paths are in section 3.

All of it is 8 threads except the F10 and F12 rows, which are the sweeps
themselves. The level-3 overhead figure is an 8-thread number too, which
matters because docs/01 predicts it should move with thread count: at 28
threads it re-measured as +0.66% [-1.85, +4.18], which the harness declined to
certify.
