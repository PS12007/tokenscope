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
- [ ] At least one Perfetto screenshot
- [ ] Tested on Linux/GCC as well as Windows/MSVC

**The Linux gap is the one that should block filing.** Every number below comes
from the non-OpenMP barrier path, and `GGML_USE_OPENMP` is the default on Linux
and takes a *different* branch in `ggml_graph_compute`. Filing an issue whose
central measurements a maintainer cannot reproduce on their own machine is
worse than not filing.

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

3. **Per-phase time tracks weight *bytes*, not parameter counts, and on a real
   quantized model the difference matters.** On Qwen2.5-0.5B Q4_K_M — where
   `output.weight` is Q8_0 while the rest is nearer 5.5 bits — predicting phase
   time from bytes is accurate to 2.9 points and from parameters is wrong by
   7.2. `lm_head` alone is **34% of decode** on that model.

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
