# 03 — Draft: the upstream issue

Not filed yet. Filed **before** proposing a PR, because writing code a
maintainer did not ask for and then asking them to review it is a way of
transferring your work onto someone else.

The draft is kept in the repo so the reasoning is visible, and so it can be
revised as the data improves.

**Prerequisites before filing:**

- [ ] Tier 2 working, so the issue can show a per-layer breakdown rather than
      "graph-compute: 99.6%"
- [ ] Overhead measured at levels 2 and 3, not just level 1
- [ ] At least one Perfetto screenshot
- [ ] Tested on Linux/GCC as well as Windows/MSVC
- [ ] Tested against a real quantized model, not only synthetic weights

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

### Why

The current timers give two aggregate scalars:

```
llama_perf_context_print: prompt eval time = ...
llama_perf_context_print:        eval time = ...
```

They can't answer "why did token 340 take 3× the median", "how much of decode
is barrier wait rather than compute", or "did my change help, and where". For
optimization work those are the questions, and today the answer usually comes
from `perf`/VTune plus guesswork about which sample belongs to which token.

I also note the existing split is heuristic — `synchronize()` classifies by
`n_queued_tokens == 1`, with the `FIXME` at `llama-context.cpp:714` noting it
misattributes when a caller doesn't synchronize per token.

### How it works

Two tiers, because llama.cpp builds a graph and then executes it, so the
functions named like they do work mostly emit nodes:

- **host scopes** — RAII, in `llama-context.cpp`, around genuinely serial work
  (`batch-init`, `kv.slot-search`, `graph-build`, `set-inputs`,
  `logits-readback`, …)
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

<!-- REPLACE with levels 1/2/3 once measured; do not file with only level 1 -->

Measured with interleaved arms and bootstrap CIs (`tools/bench_overhead.py`),
24-layer model, 8 threads, MSVC Release:

```
  arm                       median tok/s     IQR   overhead vs A
  --------------------------------------------------------------------
  A: compiled out                  42.23    0.4%                  -
  B: in, level 0                   42.19    0.8%    +0.11%  [-0.25, +0.76]
  C1: active level 1               42.25    0.7%    -0.04%  [-0.48, +0.45]
```

Every interval contains zero, so the claim is "not distinguishable from zero at
±0.5% on this system", not "0.11%".

### Size of the change

Currently ~97 changed lines across 3 existing files
(`ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`, `src/llama-context.cpp`),
plus one self-contained translation unit under `ggml/src/tokenscope/`.

Tier 2 adds a handful of lines to `ggml-cpu.c`.

### Questions

1. Is optional profiling instrumentation something you'd want in-tree, or is
   this better as an out-of-tree patch set that people apply themselves?
2. If in-tree: is `ggml-base` the right home for the shared registry? It needs
   to be visible from both `ggml-cpu` and `llama`, and I'd rather not force
   `BUILD_SHARED_LIBS=OFF` on anyone.
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
  of your time is in `graph_compute`" is not a compelling pitch.
- Lead with the question, not the patch. The first ask is "do you want this",
  not "please review this".
- Question 3 is the one that decides everything. If the answer is no, the
  project stays out-of-tree and that is a perfectly good outcome — worth saying
  so in the thread rather than arguing.
- Keep it short enough to read on a phone. The current draft is already near
  the limit; the design docs are one link away for anyone who wants them.
