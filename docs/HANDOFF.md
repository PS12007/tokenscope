# HANDOFF — state of the project, and what to do next

Written at the end of session 1 (2026-09-04/05). Everything here is either a
fact about the current tree or an explicit next step. Read this first when
picking the project back up.

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
- Thread-count sweep, 1 to 28, throughput from the uninstrumented build
  (FINDINGS F10).
- Python analysis: summary, per-token, outliers-with-cause, per-layer, diff.
- Overhead measured and inside budget at every level. Level 3 is
  **+0.67% [+0.12, +1.67]**.
- Zero-overhead-when-off verified against the symbol table.
- CI for three platforms, including the over-attribution regression check.

**The upstream patch is 123 changed lines across 6 files.** F9 and F10 needed
no new instrumentation at all -- only analysis of traces the existing scopes
already produced. The growth from 104/4 is the sampling and tokenizer scopes,
which `llama-bench` never reaches.

**Not done** — the honest list is in [`FINDINGS.md`](FINDINGS.md) under
"Not yet measured" and [`02`](02-overhead-methodology.md) under "Remaining gaps".
The short version:

| Gap | Why it matters |
|---|---|
| Linux / GCC never built or measured | Everything so far is MSVC on Windows, and F9/F10 both measured the non-OpenMP barrier path |
| No real quantized model | All numbers are synthetic F32 weights, and F10's conclusion depends on that |
| Shared-library build untested | All measurements are `BUILD_SHARED_LIBS=OFF` |
| Sampling / tokenizer scopes not written | `llama-bench` never exercises them; needs `llama-cli`, not currently built |
| No Perfetto screenshot | Blocks the README and several posts |
| No thread pinning | Would settle the mechanism F10 records as unconfirmed |
| Upstream issue not filed | Draft ready at [`03`](03-upstream-issue-draft.md); F9 is the strongest material for it |

---

## 2. The push landed

**Session 2 pushed all 20 commits. `origin/main` is at `9d189c4`.** The repo is
no longer one machine's local state, which is what section 2 used to be about.

Networking here is **intermittent, not simply blocked**. During session 2
`github.com`, `huggingface.co` and `pypi.org` were all unreachable for over an
hour, then a `git push` succeeded while a `curl https://github.com` issued
seconds later still failed, and a `git fetch` right after that died with
`expected flush after ref listing`. So: retry rather than conclude. A failed
connection says nothing about the next one.

The Hugging Face download needed for a real quantized model is still
outstanding for this reason.

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
├── tokenscope\            the repo
├── llama.cpp\             upstream clone, pinned at 4d91760, patched in place
│   ├── build-ts-on\       GGML_TOKENSCOPE=ON
│   └── build-ts-off\      GGML_TOKENSCOPE=OFF   (baseline arm; keep it)
└── models\
    ├── tiny.gguf          8L,  34 MB   synthetic
    └── mid.gguf           24L, 840 MB  synthetic  <- all published numbers
```

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
- **Hugging Face downloads kept truncating** — a Qwen2.5-0.5B GGUF stopped at
  86 KB, then 0 bytes, with `curl` still exiting 0. This is why the project uses
  synthetic models. Retry the real-model download before quoting any number
  about a real checkpoint.
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

python tools/make_tiny_model.py --llama-cpp ../llama.cpp -o ../models/mid.gguf \
       --layers 24 --embd 768 --heads 12 --heads-kv 4 --vocab 8192
```

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
already correct. It cost time this session.

---

## 5. Next steps, in the order I would do them

Items 1, 5 and 6 of the session-1 list are done: the push landed, the
thread sweep is FINDINGS F10, and the barrier arrival spread is F9.

1. **A real quantized model.** Every number in this repo is synthetic F32, and
   F10 in particular *turns on* that fact: it concludes decode is
   bandwidth-bound, so a Q4_K_M model reading a quarter of the bytes per
   parameter should scale to more threads before hitting the same wall. That is
   a sharp, falsifiable prediction and it is the single most valuable thing left
   to test. Retry the Hugging Face download; the network here is intermittent
   rather than blocked (section 2).
2. **Perfetto screenshot.** Open `examples/mid-24L-L3-tok10-11.trace.json` at
   [ui.perfetto.dev](https://ui.perfetto.dev), zoom to 2-3 tokens so both the
   per-node structure and the barrier gaps are visible, and put it at the top of
   the README. Still the best impact-to-effort item that needs no new code.
3. **Linux + GCC.** CI covers the standalone core on ubuntu, but nobody has
   built instrumented llama.cpp there. Most likely breakages: `__thread` vs
   `__declspec(thread)` (already handled), and the OpenMP path in
   `ggml_graph_compute`, which is the default on Linux and takes a *different*
   branch from the one measured here. Note this also matters for F9 and F10:
   both were measured on the non-OpenMP barrier path.
4. **Sampling and tokenizer scopes** (sites 16-19 in
   [`00`](00-architecture-map.md) section 8). Needs `llama-cli`, which is not
   currently built -- add it to the build-ts-on target list.
5. **Three-model decode table** for the README.
6. **Thread pinning**, which would settle the mechanism F10 records as
   unconfirmed: whether the per-thread spread above 8 threads really is the
   even-row-split-across-uneven-cores story. `-C`/`--cpu-mask` on llama-bench,
   or `SetThreadAffinityMask`. A cheap experiment with a clear yes/no.
7. **Fusion experiment for F9.** F9 says the promising fix for the near-serial
   elementwise nodes is fusing them, not parallelizing them, and llama.cpp
   already has `ggml_cpu_try_fuse_ops`. Checking what it currently fuses on this
   graph would turn F9's closing recommendation from reasoning into a
   measurement.
8. **File the upstream issue** - draft and prerequisites in
   [`03`](03-upstream-issue-draft.md). Do not file before items 1 and 3. F9 is
   now much the strongest material for it, and it comes with a mechanism from
   the source and a confirmed prediction, which is what a good issue needs.

---

## 6. Things I would tell myself

- **The consistency checks earned their keep.** Two real bugs were caught by
  "these percentages cannot exceed 100" — scope misparenting (F5) and summing
  wall time with thread time (the 894% report). Keep adding checks of that
  shape; they find things that inspection does not.
- **The benchmark harness refusing to answer is a feature, not a nuisance.** It
  declined three times this session and was right each time. The machine's noise
  floor moved 5× between sessions on identical binaries.
- **Do not run builds while benchmarking.** Two overhead runs were wasted
  because a compile was competing. The measurement takes ~10 minutes; do nothing
  heavy during it.
- **Check documented features actually work.** `TOKENSCOPE_TOKENS` was in the
  README and did nothing for most of the session — `ts_token_selected()` existed
  and was never called. It works now, but the class of error is worth watching:
  a feature is not done when the code exists, only when something calls it.

---

## 7. Numbers to quote (all reproducible from the committed code)

| Claim | Value | Source |
|---|---|---|
| Upstream patch size | 123 lines, 6 files | `patches/01-instrument.patch` |
| Per-scope cost | 52.8 ns (2 clock reads + 1 store) | `ts_selftest` |
| Level 3 overhead | +0.67% [+0.12, +1.67] | [`02`](02-overhead-methodology.md) |
| Zero-overhead-when-off | 0 symbols, 864-byte archive | [`02`](02-overhead-methodology.md) §2 |
| Barrier wait | 11.2% of worker thread time | [`FINDINGS`](FINDINGS.md) F6 |
| ...of which arrival imbalance | 83.7% (spin-up excluded) | [`FINDINGS`](FINDINGS.md) F9 |
| Barriers behind single-threaded nodes | 120 of 412 per token | [`FINDINGS`](FINDINGS.md) F9 |
| Upper bound on fixing that | 1.34% of graph wall time | [`FINDINGS`](FINDINGS.md) F9 |
| Best speedup at any thread count | 2.21x, at 6 threads | [`FINDINGS`](FINDINGS.md) F10 |
| Parallel efficiency at 28 threads | 7% | [`FINDINGS`](FINDINGS.md) F10 |
| Host control-plane work | 0.4% of decode | [`FINDINGS`](FINDINGS.md) F1 |
| Attention math (`attn.score`) | 0.7% of thread time | [`FINDINGS`](FINDINGS.md) F7 |
| Phase time vs parameter count | within 1.5–8.6% over a 27× range | [`FINDINGS`](FINDINGS.md) F7 |

Everything above is measured on **synthetic F32 weights**, MSVC Release,
Windows 11, on an i7-14700HX (8 P-cores + 12 E-cores). Say so whenever quoting
them. All of it is 8 threads except the F10 rows, which are the sweep itself --
and the level-3 overhead figure is an 8-thread number too, which matters
because docs/01 predicts it should move with thread count. At 28 threads it
re-measured as +0.66% [-1.85, +4.18], which the harness declined to certify.
