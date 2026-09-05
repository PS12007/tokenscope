# HANDOFF — state of the project, and what to do next

Updated at the end of **session 2 (2026-09-05)**. Everything here is either a
fact about the current tree or an explicit next step. Read this first when
picking the project back up.

**Session 2 in one paragraph.** Pushed the 14 commits session 1 could not.
Added the `--barriers` analysis, sampling/tokenizer/cell-search scopes, and a
real quantized model. Findings went from F8 to **F16**. Four of those tested
predictions the project had written down in advance: F10's scaling prediction
and F1's context-shift prediction were confirmed; **F2's `find_slot` prediction
and F9's own fusion recommendation were falsified** and now carry corrections
inline. The upstream draft was rewritten to lead with findings.

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
- Overhead: level 3 was **+0.67% [+0.12, +1.67]** at 8 threads in session 1.
  Re-measured in session 2 at 8 and 28 threads, the harness **refused to
  certify either** because the machine's noise floor had moved (baseline IQR
  2-4%, wider than the effect). Quote the session-1 number only with its
  thread count, and expect to have to re-measure on a quiet machine.
- Zero-overhead-when-off verified against the symbol table.
- CI for three platforms, including the over-attribution regression check.

**The upstream patch is 174 changed lines across 7 files.** F9 and F10 needed
no new instrumentation at all -- only analysis of traces the existing scopes
already produced. The growth from 104/4 is the sampling and tokenizer scopes,
which `llama-bench` never reaches.

**Not done** — the honest list is in [`FINDINGS.md`](FINDINGS.md) under
"Not yet measured" and [`02`](02-overhead-methodology.md) under "Remaining gaps".
The short version:

| Gap | Why it matters |
|---|---|
| Linux / GCC never built or measured | Everything so far is MSVC on Windows, and F9/F10 both measured the non-OpenMP barrier path |
| ~~No real quantized model~~ | **Done (F12).** Qwen2.5-0.5B Q4_K_M is in `models/`, gitignored. Largest real model measured is 630 M params |
| **Shared-library build is BROKEN** | **Not untested any more (F18): `ggml-cpu.dll` fails to link on `ts_tls`, and MSVC forbids `dllexport` on a `__declspec(thread)` variable (C2492). Blocks in-tree adoption; two candidate fixes in F18** |
| ~~Sampling / tokenizer scopes not written~~ | **Done (F11).** `llama-cli` is now built in `build-ts-on`. Sampling + detokenization are 0.13% of a token |
| ~~Context-shift behaviour~~ | **Done (F16).** 4 spikes in 699 tokens at `-c 256`, 1.13-1.34x median |
| Concurrent sequences / server workload | The last untested prediction in F2, and F16 says it is still plausible: the head-pointer trick that makes `find_slot` O(1) is much weaker with many streams |
| Larger real model | Biggest measured is 630 M. F12's `lm_head` result shrinks with size and its bandwidth result should move |
| No Perfetto screenshot | Blocks the README and several posts |
| ~~No thread pinning~~ | **Done (F14).** Mechanism confirmed: homogeneous cores drop spread 13%->2% and halve barrier wait. Pinning is not the fix |
| Upstream issue not filed | Draft ready at [`03`](03-upstream-issue-draft.md); F9 is the strongest material for it |

---

## 2. The push landed

**Session 2 pushed everything.** The repo is no longer one machine's local
state, which is what section 2 used to be about. Check `git log --oneline
origin/main..main` is empty before doing anything else; session 2's last push
hit `Empty reply from server` and had to be retried.

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
    ├── tiny.gguf          8L,  34 MB   synthetic F32
    ├── mid.gguf           24L, 840 MB  synthetic F32
    └── qwen-q4km.gguf     Qwen2.5-0.5B-Instruct Q4_K_M, 469 MB, REAL
```

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
```

`--barriers` needs `TOKENSCOPE_LEVEL=3`. It matches the k-th barrier across
threads, which is exact because node and barrier scopes strictly alternate and
every thread walks the same node list -- and it **refuses to report anything**
rather than guess if that ever stops holding.

`split_totals` now separates per-token work *inside* the decode slice from work
*outside* it (sampling, detokenization run after `decode` returns). Charging the
latter against the former is what pushed attribution over 100% in F11.

**Five reference traces** in `examples/`, 1012 decode tokens, all exercised by
CI:

| trace | what it covers |
|---|---|
| `mid-24L-tg256` | level 1, host scopes |
| `mid-24L-L3-tok10-11` | level 3, the F9 barrier data |
| `mid-24L-L3-pp64` | level 3 prefill, the other half of F9's test |
| `mid-24L-cli-sampling` | `llama-cli`, sampling + detokenization (F11) |
| `qwen-ctxshift-c256` | real model, context shift, `find_slot` (F16) |

**CI mechanizes four claims**, and each fails the build if the finding stops
being true: attribution never exceeds 100% (inside-slice only), every barrier is
matched by `--barriers`, F9 (elementwise nodes serial in decode, parallel in
prefill), and F16 (shifts visible, `find_slot` a small part of its scope).

---

## 5. Next steps, in the order I would do them

Session 2 closed items 1, 5, 6 and 7 of the session-1 list, plus the real
model. What is left, in the order I would do it:

1. **Linux + GCC.** Now the single most valuable thing, and the only remaining
   blocker on filing upstream. CI covers the standalone core on ubuntu, but
   nobody has built *instrumented llama.cpp* there. The reason it matters more
   than it looks: `GGML_USE_OPENMP` is the **default on Linux** and takes a
   different branch in `ggml_graph_compute` from the one every barrier finding
   (F9, F10, F14, F15) was measured on. Those findings are currently claims
   about the non-OpenMP path only. Expected breakages: `__thread` vs
   `__declspec(thread)` (already handled), and the OpenMP branch not having the
   `TS_NODE_*` macros wired at all -- check that first.
2. **Perfetto screenshot.** Still the best impact-to-effort item that needs no
   new code. Open `examples/mid-24L-L3-tok10-11.trace.json` at
   [ui.perfetto.dev](https://ui.perfetto.dev), zoom to 2-3 tokens, put it at the
   top of the README. Blocks post B5 and improves several others.
3. **Concurrent sequences.** The last of F2's predictions still standing, and
   F16 explains why it is the one most likely to be *right*: `find_slot` is
   O(1) because of a per-stream head pointer, and that argument weakens with
   many streams competing for cells. Needs `llama-server` or `llama-bench -np`.
   This is the most likely source of the next real finding.
4. **A 7-8B Q4_K_M model.** F12's two headline results move in opposite
   directions with size -- `lm_head` share shrinks, the bandwidth wall moves --
   so a bigger model tests both at once. The download works; see section 2.
5. **Proportional row assignment**, which is what F14 ends up arguing for and
   the largest change this project has pointed at. ggml's
   `dr = (nr + nth - 1)/nth` gives every thread the same row count, optimal only
   when every core is equally fast; F14 measured P-cores at **2.88x** E-cores.
   A prototype weighting the split by measured per-thread throughput would test
   whether that waste is recoverable. **Do this before proposing it upstream** --
   F15 is what happens when a plausible fix goes out unmeasured.
6. **File the upstream issue.** Draft at [`03`](03-upstream-issue-draft.md),
   rewritten in session 2 to lead with findings rather than architecture. Do not
   file before item 1.
7. **Fix the shared-library build (F18).** It does not link, and this is a
   blocker for upstreaming rather than a nice-to-have, since llama.cpp ships
   shared libraries. MSVC forbids `dllexport` on `__declspec(thread)` (C2492),
   so the raw-TLS design and the exported-registry design are incompatible as
   written. F18 lays out two fixes; the per-DLL-TLS-with-shared-registry one
   keeps the hot path intact and is the one I would try. **Re-measure level 3
   overhead after, whichever is chosen** -- both touch the node loop's hot path.
   Check the GCC/Linux behaviour at the same time (item 1); ELF may not have
   this restriction at all.

---

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

---

## 7. Numbers to quote (all reproducible from the committed code)

| Claim | Value | Source |
|---|---|---|
| Upstream patch size | 174 lines, 7 files | `patches/01-instrument.patch` |
| Per-scope cost | 52.8 ns (2 clock reads + 1 store) | `ts_selftest` |
| Level 3 overhead | +0.67% [+0.12, +1.67] | [`02`](02-overhead-methodology.md) |
| Zero-overhead-when-off | 0 symbols, 864-byte archive | [`02`](02-overhead-methodology.md) §2 |
| Barrier wait | 11.2% of worker thread time | [`FINDINGS`](FINDINGS.md) F6 |
| ...of which arrival imbalance | 83.7% (spin-up excluded) | [`FINDINGS`](FINDINGS.md) F9 |
| Barriers behind single-threaded nodes | 120 of 412 per token | [`FINDINGS`](FINDINGS.md) F9 |
| Upper bound on fixing that | 1.34% of graph wall time, and F15 found no reachable part | F9, F15 |
| Barriers removed by ggml's one fusion | 49 of 461 per token, for no measurable throughput | [`FINDINGS`](FINDINGS.md) F15 |
| Best speedup at any thread count | 2.21x F32 / 3.28x Q4_K_M, both at 6 threads | F10, F12 |
| `lm_head` share, Qwen2.5-0.5B Q4_K_M | 34% of decode thread time | [`FINDINGS`](FINDINGS.md) F12 |
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

All of it is 8 threads except the F10 and F12 rows, which are the sweeps
themselves. The level-3 overhead figure is an 8-thread number too, which
matters because docs/01 predicts it should move with thread count: at 28
threads it re-measured as +0.66% [-1.85, +4.18], which the harness declined to
certify.
