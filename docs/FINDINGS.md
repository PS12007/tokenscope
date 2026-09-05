# Findings

Things the traces showed that were not obvious before taking them. Each entry
states the workload, because a finding without a workload is an opinion.

Numbers here come from real traces produced by the committed code. Where a
finding is provisional, it says so.

---

## F1 — Host-side control-plane work is not where decode time goes. At all.

**Workload:** 24-layer / `n_embd=768` / 12 heads (4 KV) / `n_ff=3072` F32 model,
220 M params, 8 threads, `tg256`, MSVC Release, `TOKENSCOPE_LEVEL=1`.

```
decode: 257 tokens, 6182.7 ms total, 24.06 ms/tok (41.6 tok/s)

  category                 total       %      per-tok
  --------------------------------------------------
  graph-compute       6157.54 ms   99.6%     23.959 ms
  kv.slot-search        14.35 ms    0.2%      0.056 ms
  ubatch                 2.64 ms    0.0%      0.010 ms
  batch-init             2.39 ms    0.0%      0.009 ms
  set-inputs             1.32 ms    0.0%      0.005 ms
  logits-readback        1.08 ms    0.0%      0.004 ms
  output-reserve        365.3 us    0.0%      0.001 ms
  kv.update             275.4 us    0.0%      0.001 ms
  graph-build           206.4 us    0.0%      0.001 ms
  graph-alloc           137.7 us    0.0%      0.001 ms
  sched-reserve          55.3 us    0.0%      0.000 ms

  100.0% of decode wall time is attributed to a scope.
```

**What I expected.** Going in, the project pitch listed KV cache reallocation,
logits buffer reservation and graph rebuilds as prime suspects for per-token
stalls — the "token 340 stalled for 90 ms" story. Every one of those is
instrumented here.

**What the trace says.** They are not suspects on this workload. Summed across
all 257 tokens:

- `graph-build` — **206 µs total**, i.e. 0.8 µs per token. llama.cpp's graph
  reuse (`llama-context.cpp:1348`) is doing its job essentially perfectly. A
  rebuild costs on the order of a millisecond, so this says the reuse check
  misses roughly never after the first token.
- `kv.update` — **275 µs total.** No context shift was triggered in a 256-token
  run inside a 2048-token window, which is the expected and boring outcome.
- `output-reserve` — **365 µs total.** The logits buffer is reserved once and
  then hits the already-large-enough path.

All host-side work together is **0.4%** of decode.

**Why this is worth recording rather than embarrassing.** This is the profiler
doing its actual job. The stall hypotheses were plausible enough to be worth
instrumenting, and the honest result is that on a steady-state generation loop
inside a comfortable context window, llama.cpp's control plane is not costing
you anything. That is a real answer to a real question, and it is one nobody
could have given from the built-in timers.

The corollary sets up the next milestone: **every remaining question lives
inside `graph_compute`.** Which is exactly what docs/00 predicted, and exactly
why Tier 2 is the part that matters.

**Caveats, stated plainly:**

- A 2048-token context with 256 generated tokens never triggers a context
  shift. A run that *does* — long generation in a tight window — is the case
  where `kv.update` should light up, and it has not been tested yet.
- Single sequence, batch size 1. Server-style concurrent decoding exercises
  `kv.slot-search` far harder, and it is already the largest host cost here.
- Synthetic F32 weights. See [`02`](02-overhead-methodology.md) section 1 for
  why that is fine for overhead and not fine for absolute performance claims.

---

## F2 — `kv.slot-search` is the only host-side cost with a real shape to it

Same run. At 56 µs/token it is 0.23% of decode — but it is **26× the next
largest host cost** and 260× `graph-build`.

`llama_kv_cache::find_slot` (`llama-kv-cache.cpp:898`) is a linear scan for free
cells, so its cost grows with cache occupancy. In a 256-token run in a
2048-cell cache it stays flat and cheap. The prediction this trace makes, and
that has not yet been tested: on a long run in a nearly-full cache, or with many
concurrent sequences, this is the host-side cost that stops being a rounding
error.

Recorded now precisely so that the prediction is on the record *before* the
measurement that tests it.

---

## F3 — The outliers are all inside the graph, which is a finding about the tool

```
 token        ms   x median  dominant excess
------------------------------------------------------------------
    56     33.05      1.38x  graph-compute (+9.09 ms)
   197     29.61      1.24x  graph-compute (+5.66 ms)
   195     29.18      1.22x  graph-compute (+5.25 ms)
```

p50 23.92 ms, p99 28.75 ms, max 33.05 ms — a 1.38× spread between the median
token and the worst.

Every outlier attributes to `graph-compute`, which at level 1 is one opaque
scope. So the honest reading is: **the tool has correctly localized the
variance and cannot yet explain it.** That is a better state than a plausible
guess, and it is the concrete argument for Tier 2 — per-node work and barrier
wait, split per thread.

The two candidate explanations Tier 2 will distinguish between:

1. genuine compute variance (memory pressure, frequency scaling, OS scheduling);
2. barrier imbalance — one thread arriving late and 7 spinning, which shows up
   as compute time in any tool that does not separate work from wait.

If it is (2), the fix is a partitioning change and the finding is actionable. If
it is (1), the fix is elsewhere and the finding is "stop looking here". Either
way the current data cannot tell them apart, and saying so is the point.

---

## F4 — Where you draw the boundary changes the number by 3%

The first trace, on an 8-layer model, read **1405 tok/s** for decode while
`llama-bench` read **1452 tok/s** for the same run.

Neither is wrong. `llama-bench` times its own generation loop; tokenscope's
slice runs from `llama_context::decode` entry to exit, and
`llama_synchronize()` is called *after* `decode()` returns
(`tools/llama-bench/llama-bench.cpp:2215`). The 3% is real work outside our
boundary.

On the 24-layer model the same comparison is 41.6 vs 42.0 tok/s — under 1%,
because the synchronize is a roughly fixed cost and the token got 33× longer.

Worth recording because it is a compact demonstration of the thing docs/00
complains about: a timing number without a stated boundary is not a measurement.
Ours is stated. llama.cpp's built-in split
([`llama-context.cpp:714`](00-architecture-map.md#1-the-prefill--decode-boundary))
carries a maintainer `FIXME` saying its boundary is sometimes wrong.

---

## F5 — A profiler can misparent its own scopes with correct timestamps

Not a finding about llama.cpp, but the most useful bug of the project so far.

The first per-category breakdown reported **100.4%** of decode time attributed —
which is impossible, and therefore informative.

Cause: two adjacent scopes share an instant. `set-inputs` ended at
`t0 + dur = 4990158.100000001` µs; `graph-compute` began at `4990158.100`. The
analyzer decided the second was *inside* the first, reparented a sibling as a
child, and double-counted 25 ms on one token out of 257.

An epsilon would have papered over it. The actual fix was to stop inferring
structure that the profiler already knows: `tokenscope::scope` now records
nesting **depth** explicitly (one thread-local increment, in a field that was
already in the 24-byte record), and the analyzer uses it, falling back to
timestamp containment only for traces from other producers.

Two things worth keeping from this:

- The bug was found by a **consistency check**, not by inspection — "these
  percentages must sum to ≤100" caught a 0.4% error in a 300 KB trace. That
  check is now permanent output.
- Floating-point microseconds cannot distinguish "ends exactly where the next
  begins" from "contains the next". Any tool that reconstructs a call tree from
  timestamps alone has this bug latent in it.

---

## Not yet measured

Listed so the gaps are explicit rather than implied:

- per-layer and per-phase breakdown (needs Tier 2)
- work vs barrier-wait split (needs Tier 2)
- real quantized models — everything above is synthetic F32 weights
- context-shift behaviour, i.e. the case where `kv.update` should be expensive
- concurrent sequences / server workload
- sampling and tokenization, which `llama-bench` never exercises
- levels 2 and 3 overhead
