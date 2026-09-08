# HANDOFF — state of the project, and what to do next

Updated during **session 4 (2026-09-07)**. Everything here is either a fact
about the current tree or an explicit next step. Read this first when picking
the project back up.

**Session 4 so far.** Closed section 5 item 6: the shared-library build works
(**F22**). F18 had called it a genuine incompatibility between the hot path's
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
| ~~Shared-library build is BROKEN~~ | **Fixed (F22).** F18's option 2, implemented and verified: each module caches its own `ts_tls`, all resolving to one registry-owned buffer. `BUILD_SHARED_LIBS=ON` links and traces correctly. Two-module regression test passes on all three toolchains; **llama.cpp shared on Linux still untested**, and the shared build's overhead has never been measured |
| ~~Sampling / tokenizer scopes not written~~ | **Done (F11).** `llama-cli` is now built in `build-ts-on`. Sampling + detokenization are 0.13% of a token |
| ~~Context-shift behaviour~~ | **Done (F16).** 4 spikes in 699 tokens at `-c 256`, 1.13-1.34x median |
| Concurrent sequences / server workload | The last untested prediction in F2, and F16 says it is still plausible: the head-pointer trick that makes `find_slot` O(1) is much weaker with many streams |
| ~~Larger real model~~ | **Done (F19, F21).** Three more real models measured, to 8.19 B. Biggest is now 8.19 B; **no MoE model at all**, which is the clearest remaining gap |
| No Perfetto screenshot | Blocks the README and several posts. **Still the best impact-to-effort item that needs no new code**, and the reference trace for it is now `examples/qwen3-8b-named-attnout.trace.json` |
| ~~No thread pinning~~ | **Done (F14).** Mechanism confirmed: homogeneous cores drop spread 13%->2% and halve barrier wait. Pinning is not the fix |
| Upstream issue not filed | Two issues now, and [`03`](03-upstream-issue-draft.md) says which goes first. **The F20 naming defect is not blocked on Linux** and should be filed on its own; the instrumentation proposal still is |

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
├── tokenscope\            the repo
├── llama.cpp\             upstream clone, pinned at 4d91760, patched in place
│   ├── build-ts-on\       GGML_TOKENSCOPE=ON,  static  (ninja, bin\)
│   ├── build-ts-off\      GGML_TOKENSCOPE=OFF, static  (baseline arm; keep it)
│   └── build-ts-shared\   BUILD_SHARED_LIBS=ON (VS generator, bin\Release\)
└── models\
    ├── tiny.gguf          8L,  34 MB   synthetic F32
    ├── mid.gguf           24L, 840 MB  synthetic F32
    └── qwen-q4km.gguf     Qwen2.5-0.5B-Instruct Q4_K_M, 469 MB, REAL

**A real 8B model is already on disk**, pulled by Ollama before this project
started, so item 4 of section 5 needs no download at all. Ollama stores GGUF
blobs unmodified and content-addressed; `llama-bench -m` opens one directly:

```
~/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f
```

That is **Qwen3 8B Q4_K_M**, 4.86 GiB, 8.19 B params, 36 layers, `n_embd` 4096,
`n_ff` 12288, GQA 32/8 — 13x the parameters of the Qwen2.5-0.5B used in F12.
Read the metadata with `gguf-py` rather than trusting the tag; the manifest at
`~/.ollama/models/manifests/registry.ollama.ai/library/qwen3/8b` maps tags to
blobs. `dolphin-llama3` (8B) and `dolphin-mistral` (7B) are there too, and **both were
measured in F21** — blobs `sha256-ea025c10...` and `sha256-11a57a9b...`
respectively.

**The constraint is RAM, not disk.** This machine has 15.7 GB total and about
7.4 GB free, against a 4.86 GiB model. It fits and it does not thrash — measured
38.78 pp32 / 7.39 tg16 tok/s at 8 threads — but the margin is thin enough that
anything else running can page the weights out and quietly corrupt a decode
number, because decode is bandwidth-bound (F14). Check free memory before
trusting a run at this size.
```

**`build-ts-on` now contains a change that is NOT upstream.** Session 3 applied
`patches/02-name-attn-output.patch` to `llama.cpp/src/llama-graph.cpp` in place
and rebuilt, so `llama-bench` from that directory names the attention output
projection where stock llama.cpp does not. Traces taken from it are not
byte-comparable with pre-session-3 traces in that one respect — `attn.out`
appears and `~attn` nearly vanishes (F20). `git -C ../llama.cpp diff
src/llama-graph.cpp` shows it; `git -C ../llama.cpp checkout src/llama-graph.cpp`
reverts it. `bootstrap.py --make-patch` does **not** touch it, deliberately:
`llama-graph.cpp` is kept out of `TOUCHED` so the naming fix stays a separate
patch from the instrumentation.

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

Session 3 closed items 3 and 4 (concurrent sequences had already gone in F17;
the 8B is F19/F21) and added a new one at the top that did not exist before.

1. **File the F20 naming issue.** New, and first because it is the only piece of
   this work that is *not* blocked on Linux, and the smallest thing a maintainer
   could say yes to. `patches/02-name-attn-output.patch` is applied and measured;
   [`03`](03-upstream-issue-draft.md) has the framing. llama.cpp's `AGENTS.md`
   asks for an issue before a PR **and** asks that the contributor own the change
   and be able to defend it unaided — which for a change this size is a fair bar
   and worth meeting deliberately before filing.
2. **Linux + GCC**, unchanged from session 2 and still the blocker for the main
   upstream conversation. The detail is in section 5 of the previous revision
   and still accurate: `ggml_graph_compute_thread` is shared, so the scopes fire,
   but `ggml_barrier` is `#pragma omp barrier` instead of the atomic spin-wait
   measured here. F9's *structural* claims should transfer; every barrier *cost*
   number may not transfer at all.
3. **Perfetto screenshot.** Still the best impact-to-effort item that needs no
   new code, and now with a better trace to use — open
   `examples/qwen3-8b-named-attnout.trace.json`, zoom to 2-3 tokens, put it at
   the top of the README. Blocks post B5.
4. **An MoE model.** The clearest remaining gap in the byte law. F21 covers three
   architectures and a 13.7x range of `lm_head` share, but every model measured
   is dense, and MoE is the case where bytes-streamed-per-token stops being a
   property of the file and starts depending on the router. The law as stated
   would predict expert phases from *stored* bytes and should be **wrong** there,
   which makes it the most informative test available.
5. **Proportional row assignment**, unchanged and still the largest change this
   project has pointed at. ggml's `dr = (nr + nth - 1)/nth` gives every thread
   the same row count; F14 measured P-cores at 2.88x E-cores. **Do this before
   proposing it upstream** — F15 is what happens when a plausible fix goes out
   unmeasured. Note F19 narrows where it matters: at 8B the heterogeneity
   penalty vanishes into the bandwidth wall, so this is a prefill and
   small-model fix, not a universal one.
6. ~~**Fix the shared-library build (F18).**~~ **Done in session 4 (F22).** What
   is left of it: the shared build's **overhead has never been measured** (the
   hot path is unchanged by inspection, but that is not a measurement), and the
   GCC/ELF behaviour still needs checking alongside item 2. Both fold into
   items 2 and 7 rather than standing on their own.
7. **Re-measure overhead on a quiet machine.** The harness refused to certify
   twice in session 2 and nothing has changed. Every quoted overhead number is
   still the session-1 8-thread one.

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
  heredocs.

---

## 7. Numbers to quote (all reproducible from the committed code)

| Claim | Value | Source |
|---|---|---|
| Upstream patch size | 174 lines, 7 files | `patches/01-instrument.patch` |
| F20 naming fix size | 8 added, 13 removed, 1 file | `patches/02-name-attn-output.patch` |
| Per-scope cost | 52.8 ns (2 clock reads + 1 store) | `ts_selftest` |
| Level 3 overhead | +0.67% [+0.12, +1.67] | [`02`](02-overhead-methodology.md) |
| Zero-overhead-when-off | 0 symbols, 864-byte archive | [`02`](02-overhead-methodology.md) §2 |
| Shared-library build | links and traces correctly on MSVC; overhead unmeasured | [`FINDINGS`](FINDINGS.md) F22 |
| Barrier wait | 11.2% of worker thread time | [`FINDINGS`](FINDINGS.md) F6 |
| ...of which arrival imbalance | 83.7% (spin-up excluded) | [`FINDINGS`](FINDINGS.md) F9 |
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
