# 03 — Draft: the upstream issue

Not filed yet. Filed **before** proposing a PR, because writing code a
maintainer did not ask for and then asking them to review it is a way of
transferring your work onto someone else.

The draft is kept in the repo so the reasoning is visible, and so it can be
revised as the data improves.

**Prerequisites before filing:**

- [x] Tier 2 working, so the issue can show a per-layer breakdown rather than
      "graph-compute: 99.6%"
- [x] Overhead measured at levels 2 and 3, not just level 1
- [x] Tested against a real quantized model, not only synthetic weights
      (Qwen2.5-0.5B Q4_K_M, [`FINDINGS`](FINDINGS.md) F12)
- [x] Tested at a size a maintainer would care about (Qwen3-8B Q4_K_M,
      [`FINDINGS`](FINDINGS.md) F19) — and the 0.5B results did *not* all
      survive, which is worth saying in the issue rather than hiding
- [ ] At least one Perfetto screenshot
- [ ] Tested on Linux/GCC as well as Windows/MSVC
- [x] `BUILD_SHARED_LIBS=ON` linking — F18 found it broken, F22 fixed it. Each
      module caches its own thread-local and they all resolve to the one
      registry-owned buffer, so the hot path keeps its single unguarded load.
      The two-module regression test runs on MSVC, GCC and Clang; llama.cpp
      with shared libraries on Linux is still untested and folds into the
      Linux gap below

**The Linux gap is the one that should block filing.** Every number below comes
from the non-OpenMP barrier path, and `GGML_USE_OPENMP` is the default on Linux
and takes a *different* branch in `ggml_graph_compute`. Filing an issue whose
central measurements a maintainer cannot reproduce on their own machine is
worse than not filing.

---

## File this one first: the naming defect (F20)

**This is a different issue from the one below, and it should go first.**

[`F20`](FINDINGS.md) found that the attention output projection has no name in
any of `build_attn`'s seven overloads, so it appears in every ggml-name-based
profile as `node_1182`. On Qwen3-8B it is 7.1% of decode thread time — the
largest single node in the attention block.

It is worth filing separately because it is **not about tokenscope at all**:

- it is a defect, not a feature request, so it does not need the "do you want
  this in-tree" conversation
- it is reviewable in a minute: one line inside the `if (wo)` that does the
  matmul, in each overload, plus three dead `if (wo_b) { }` blocks removed
- `cb()` assigns a name at graph-build time, which graph reuse makes about once
  per run (F1), so there is no measurable cost and no behaviour change
- the evidence is a before/after that needs no profiler to believe: the node
  goes from anonymous to named, and every existing tool that groups by name
  picks it up

`patches/02-name-attn-output.patch` is applied and measured locally. Per
llama.cpp's `AGENTS.md`, open an **issue** describing the defect first rather
than a PR — and note that AGENTS.md also asks that a contributor own and be
able to defend the change without assistance, which for a change this size is a
reasonable bar to meet before filing.

Unlike the instrumentation proposal, **this one is not blocked on Linux.** It is
a graph-construction change with no threading or platform dependency, and the
evidence for it does not rest on any timing number that the OpenMP path could
invalidate.

---

## Target

`ggml-org/llama.cpp` → Issues → "Feature request" template.

Read `CONTRIBUTING.md` again immediately before filing; it changes.

---

## Draft

**Title:** Optional per-token profiling instrumentation emitting Chrome Trace
Event JSON

---

### What

I've built an optional instrumentation layer that attributes each generated
token's wall-clock time to phases of the forward pass, and writes Chrome Trace
Event Format JSON that opens directly in Perfetto.

It's behind a CMake option (`GGML_TOKENSCOPE`, default OFF) that compiles to
nothing when off. Before proposing a PR I'd like to know whether this is
something you'd want in-tree at all.

Repo, with the full design docs and measured overhead:
https://github.com/PS12007/tokenscope

### Why — four things it found

Rather than argue that per-token profiling is useful in the abstract, here is
what it actually turned up on CPU decode. All are reproducible from the repo;
all are on one machine (i7-14700HX, Windows, MSVC) and say so.

1. **~29% of decode barriers are paid for nodes only one thread worked on.**
   Elementwise ops partition over rows (`dr = (nr + nth - 1)/nth`), and at batch
   size 1 a hidden state is a single row — so `ir0 = ith >= 1 = ir1` and seven
   of eight threads get an empty range, then hit `ggml_barrier` anyway. Confirmed
   by the same nodes going to 7.7/8 busy threads during prefill.

2. **Barrier wait halves when the cores are the same speed.** Twelve threads
   unpinned across P- and E-cores: 13% per-thread compute spread, 21.2% of
   worker time in `ggml_barrier`. Twelve threads pinned to twelve identical
   E-cores: **2% spread, 11.1% barrier** — same model, same graph, same thread
   count. Equal row counts to unequal cores is the mechanism. (Pinning is not
   the fix; it costs 8-10% throughput. Proportional row assignment would be.)

3. **Per-phase time tracks weight *bytes*, not parameter counts.** The clean
   test is inside your own quantization recipe: Q4_K_M stores `ffn_down` at
   Q6_K in 18 of Qwen3-8B's 36 layers and Q4_K in the other 18. Same tensor,
   same shape, same op, adjacent layers, dtype the only difference.

   ```
                   Q6_K layers    Q4_K layers    ratio
     ffn_out        38516.6 us     27138.5 us    1.419    <- the test
     ffn_gate       26735.9 us     26485.5 us    1.009    <- control
     ffn_up         26681.6 us     26506.4 us    1.007    <- control

     predicted from bytes alone: 6.5625 / 4.5 = 1.458
   ```

   Across the whole model, phase time predicted from byte share is accurate to
   **0.3 points**, and from parameter counts wrong by 4.5. `lm_head` is 34% of
   decode on Qwen2.5-0.5B and 10.5% on Qwen3-8B against an identical 151,936
   vocabulary, so per-tensor quantization choices are directly visible in the
   profile and small-model results do not transfer.

4. **A negative result, included because it corrected me.** I expected op fusion
   to help: `GGML_CPU_DISABLE_FUSION=1` removes 49 of 461 barriers per token,
   10.6% of them. Throughput change: none measurable, in three regimes including
   one where barrier wait is 54% of thread time. The barriers a fusion removes
   are the ones threads arrive at together.

None of these are visible to the existing timers, which give two aggregate
scalars:

```
llama_perf_context_print: prompt eval time = ...
llama_perf_context_print:        eval time = ...
```

In particular, without a work/wait split, barrier time is indistinguishable from
compute, so "ffn: 69% across 8 threads" silently includes seven of them
spinning.

I also note the existing split is heuristic — `synchronize()` classifies by
`n_queued_tokens == 1`, with the `FIXME` at `llama-context.cpp:714` noting it
misattributes when a caller doesn't synchronize per token.

### How it works

Two tiers, because llama.cpp builds a graph and then executes it, so the
functions named like they do work mostly emit nodes:

- **host scopes** — RAII, in `llama-context.cpp`, around genuinely serial work
  (`batch-init`, `kv.slot-search`, `graph-build`, `set-inputs`,
  `logits-readback`, …), plus `sample` and `tok.encode`/`tok.decode`
- **node scopes** — two scopes in `ggml_graph_compute_thread`, around
  `ggml_compute_forward` and `ggml_barrier`, giving per-node work and wait per
  worker thread

Per-layer attribution is free: `graph_get_cb` already names every tensor
`<role>-<layer>`, so node names are parsed at flush rather than instrumented
per phase.

Design constraints, all enforced rather than intended:

- **No new dependencies.** C++17 standard library only.
- **Nothing when disabled.** Macros expand to no tokens. Verified against the
  symbol table, not a benchmark: the OFF build's archive contains zero
  tokenscope symbols.
- **No locks or allocation in the hot path.** Thread-local buffers merged at
  flush; node events store a graph node index and never touch a string.
- **Never silently drops data.** Bounded budget with a drop counter in every
  trace's metadata.

### Cost

Measured with interleaved arms and bootstrap CIs (`tools/bench_overhead.py`),
24-layer model, MSVC Release. Level 3 is the expensive mode — every node event
on every thread.

```
  8 threads
  arm                       median tok/s     IQR   overhead vs A
  --------------------------------------------------------------------
  A: compiled out                  43.16    2.0%                  -
  B: in, level 0                   43.31    2.5%    -0.36%  [-2.27, +1.77]
  C3: active level 3               43.11    1.2%    +0.12%  [-1.59, +1.43]

  28 threads
  C3: active level 3               36.50    3.0%    +0.66%  [-1.85, +4.18]
```

Every interval contains zero. **The harness refused to certify either
measurement**, because this machine's baseline IQR is wider than the 2% effect
being tested — so the honest claim is "not resolvable at ±2-4% here", not a
number. The same machine on a quieter day resolved level 3 as +0.67% [+0.12,
+1.67] at 8 threads; its noise floor moved 5x between sessions on identical
binaries, which is itself worth knowing before trusting anyone's sub-1%
profiler-overhead claim.

I flag this because a barrier makes the graph pay the `max` of per-thread
overhead rather than the mean, so I'd expect overhead to grow with thread count.
I could not detect that growth up to 28 threads, which is weaker than saying it
does not happen.

### Size of the change

Currently 123 changed lines across 6 existing files (`ggml/CMakeLists.txt`,
`ggml/src/CMakeLists.txt`, `ggml/src/ggml-cpu/ggml-cpu.c`,
`src/llama-context.cpp`, `src/llama-sampler.cpp`, `src/llama-vocab.cpp`), plus
one self-contained translation unit under `ggml/src/tokenscope/`.

The two lines in `ggml_graph_compute_thread` are the ones that need the most
scrutiny; everything else is in cold code.

### Questions

1. Is optional profiling instrumentation something you'd want in-tree, or is
   this better as an out-of-tree patch set that people apply themselves?
2. **`BUILD_SHARED_LIBS=ON` works, and I'd value a sanity check on how.** It did
   not link at first: `ggml-cpu.dll` needs `ts_tls`, the thread-local buffer
   pointer read on the hot path, and MSVC refuses to `dllexport` a
   `__declspec(thread)` variable at all (C2492). Rather than an exported
   accessor — which would put a cross-DLL call on the hottest loop in the
   project — each module now compiles its own copy of that pointer and fills it
   from the exported `ts_thread_init()`, which hands back the one registry-owned
   buffer for the calling thread. The pointer is only a cache; the buffer is the
   state, and `ggml-base` owns it. The hot path keeps its single unguarded load.
   **Measured on MSVC only.** Is `ggml-base` the right home for the shared
   registry, or is there a convention here I should follow? And is there
   anything about `GGML_BACKEND_DL`, where backends are loaded at runtime, that
   this would break?
3. Would you want the node-level scopes in `ggml_graph_compute_thread` at all?
   That's the hottest loop in the project and I understand the reluctance —
   though when compiled out there is nothing there, and I have the numbers for
   when it is compiled in.
4. Is there an existing effort here I've missed? I found `eval-callback` and
   the `GGML_PERF` history, and concluded neither gives per-token attribution
   without changing the thing being measured, but I may well have missed
   something.

Happy to split this into smaller PRs, drop tiers, or change the naming to fit
existing conventions.

---

## Notes to self

- Do not open this until Tier 2 works. "Here's a profiler that tells you 99.6%
  of your time is in `graph_compute`" is not a compelling pitch. **(Done.)**
- Lead with the question, not the patch. The first ask is "do you want this",
  not "please review this".
- **Lead the body with findings, not architecture.** A maintainer's first
  question is "what did this tell you that we didn't already know", and the
  four-item list answers it before any design discussion starts. This is the
  biggest change from the session-1 draft, which described the tool for four
  paragraphs before showing anything it found.
- Findings 1 and 2 are the ones that might interest a maintainer independently
  of whether they want the tool. Keep them first, and keep the mechanism
  (`dr = (nr + nth - 1)/nth`) visible, because it is checkable in ten seconds
  by someone who knows the file.
- **Keep the negative result (finding 4).** It is the cheapest possible signal
  that the numbers are not being curated, and it costs three lines.
- Question 3 is the one that decides everything. If the answer is no, the
  project stays out-of-tree and that is a perfectly good outcome — worth saying
  so in the thread rather than arguing.
- Do not file without Linux. See the prerequisites note: the OpenMP path is the
  default there and is a different branch from everything measured here.
- Keep it short enough to read on a phone. The findings list is four bullets on
  purpose; the design docs are one link away for anyone who wants them.
