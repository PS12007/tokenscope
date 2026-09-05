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

## F6 — 11.2% of worker thread time is barrier wait, not compute

**Workload:** same 24-layer model, 8 threads, `tg32`, `TOKENSCOPE_LEVEL=3`.

```
  graph nodes -- thread time across 8 workers (6160.1 ms busy of 6200.9 ms available)

  category                 total       %      per-tok
  --------------------------------------------------
  ffn                 4249.84 ms   69.0%    128.783 ms
  barrier              692.53 ms   11.2%     20.986 ms
  attn.qkv             640.73 ms   10.4%     19.416 ms
  attn.out             359.43 ms    5.8%     10.892 ms
  lm_head              160.83 ms    2.6%      4.874 ms
  attn.score            42.35 ms    0.7%      1.283 ms
  norm                   6.54 ms    0.1%      0.198 ms
  attn.kv_rw             4.10 ms    0.1%      0.124 ms
  residual               3.05 ms    0.0%      0.092 ms

  11.2% of worker thread time is barrier wait, not compute.
  99.3% of available thread time is inside a node scope.
```

`ggml_barrier` runs after **every node** — roughly 700 barriers per token here,
on each of 8 threads. One eighth of the total CPU budget is spent spinning in
`ggml_thread_cpu_relax()` rather than doing arithmetic.

This is the number the project was built to produce, and it is invisible to
every aggregate timer: without a work/wait split, those 692 ms are
indistinguishable from compute, and "attention took X across 8 threads" silently
includes seven of them waiting.

**What it does not yet say.** 11.2% is not automatically 11.2% of recoverable
time. Some of it is unavoidable: the graph has real serial dependencies, and a
barrier after a node whose work genuinely cannot be split is not waste. Telling
"structurally required wait" from "wait caused by bad partitioning" needs the
per-node, per-thread arrival spread, which is in the level-3 trace and is not
yet reduced into the report. That is the next analysis feature, not a claim to
make now.

**Answered in [F9](#f9--most-barrier-wait-is-arrival-imbalance-and-a-third-of-the-barriers-are-paid-for-nodes-that-only-one-thread-worked-on).**
Once the spread was reduced: 83.7% of the wait is arrival imbalance rather than
release latency, and 29% of the barriers follow a node that only one thread
worked on. The recoverable share is real but small -- an upper bound of 1.34% of
graph wall time -- and the promising fix is fusion, not partitioning.

---

## F7 — Phase times track parameter counts to within a few percent, which
## validates both the workload model and the tool

This one is a check I expected to fail, and it did not.

If decode is bandwidth-bound on streaming weights — the standard mental model
for single-token CPU inference — then time per phase should be proportional to
**bytes of weights read**, and for a uniform dtype that is just parameter count.
The model's dimensions are known exactly, so this is a falsifiable prediction
rather than a story.

Predicting every phase from the FFN measurement alone:

```
phase             params   pred ms   meas ms    delta
ffn          169,869,312    4249.8   4249.84    +0.0%   (reference)
attn.qkv      23,592,960     590.3    640.73    +8.6%
attn.out      14,155,776     354.2    359.43    +1.5%
lm_head        6,291,456     157.4    160.83    +2.2%
```

Three independent phases, spanning a **27× range** in weight volume, predicted
from parameter counts to within 1.5–8.6%.

Two conclusions, and the second matters more:

1. **Decode on this model is almost purely weight-streaming.** The arithmetic is
   free; the time is the memory traffic. `attn.score` — the actual attention
   computation, scores and softmax — is **0.7%**. At batch size 1 with a short
   context there is essentially nothing there. Anyone optimizing attention math
   for CPU decode is optimizing 0.7% of the workload.

2. **The tool is measuring what it says it is measuring.** A profiler that
   misattributed nodes to phases, or double-counted across threads, or drifted
   its clock, would not reproduce a 27× spread to within a few percent by
   accident. This is the strongest evidence so far that the per-phase numbers
   can be trusted.

The +8.6% on `attn.qkv` is the residual worth noting rather than smoothing over:
that bucket contains `Qcur`/`Kcur`/`Vcur`, which carry RoPE and the KV cache
write on top of the projection matmul. Extra work beyond the weight read is
exactly what should make it the one phase that overshoots.

---

## F8 — About 15% of ggml graph nodes have no meaningful name, and on this model
## they are the attention core

`graph_get_cb` names tensors `<role>-<layer>`, which docs/00 identified as the
mechanism that makes per-layer attribution nearly free. That is true for the
tensors it names. It does not name all of them.

The first Tier 2 trace, by raw node name:

```
ffn_out        2434.4 ms   n=6720
ffn_gate       2395.7 ms   n=6720
ffn_up         2392.6 ms   n=6720
Qcur            652.4 ms   n=13440
attn_out        629.2 ms   n=6720
...
node_21           8.2 ms   n=280
node_579          7.7 ms   n=280
node_300          7.7 ms   n=280       <- ~200 of these
```

`node_<index>` is ggml's automatic fallback name. There is no `kq` or `kqv` node
at all: the attention core is entirely unnamed, which is the phase a per-layer
profiler most needs to report.

Resolved two ways, in order of confidence:

1. **Op type.** `ts_graph_set_node` was already registering `ggml_op_name(node->op)`
   and the emitter was ignoring it. `SOFT_MAX` and `FLASH_ATTN_EXT` → `attn.score`,
   `ROPE` → `rope`, `SET_ROWS`/`CPY`/`CONT` → `attn.kv_rw`. This is a fact about
   the node, not a guess.
2. **Graph position.** Anything still unresolved inherits the phase of the last
   named marker before it in graph order. This is an *inference*, so those
   categories are prefixed `~` in the output and the report says so explicitly.

After the op mapping, the inferred bucket is **0.008 ms/token** — essentially
nothing needs guessing. But the `~` prefix stays, because the moment a reader
cannot tell a measurement from an inference, neither is worth much.

---

---

## F9 — Most barrier wait is arrival imbalance, and a third of the barriers are
## paid for nodes that only one thread worked on

**Workload:** same 24-layer synthetic F32 model, 8 threads, MSVC Release,
`TOKENSCOPE_LEVEL=3`, two steady-state decode tokens
([`examples/mid-24L-L3-tok10-11.trace.json`](../examples/mid-24L-L3-tok10-11.trace.json)).
Reproduce with `trace_analyze.py <trace> --barriers`.

[F6](#f6--112-of-worker-thread-time-is-barrier-wait-not-compute) measured 11.2%
of worker thread time as barrier wait and explicitly refused to call it
recoverable, because telling structurally-required wait from bad partitioning
needs the per-node arrival spread. That measurement now exists.

### The split

```
  total barrier wait      62.31 ms   thread-time
    arrival imbalance     34.59 ms    55.5%   threads idle, waiting for the last
    after last arrival    27.72 ms    44.5%   release latency and spin-up
```

44.5% looks alarming until you look at where it is. **One barrier** — the first
one on the first traced token — accounts for 20.98 ms of it. That is the thread
pool spinning up, not a property of the graph. Excluding it the split is

```
  83.7% arrival imbalance / 16.3% release latency
```

So the answer to F6 is: **the great majority of barrier wait is threads that
finished early standing around.** That is the shape of a partitioning problem,
not of an unavoidable serial dependency.

### Where the imbalance comes from

Two populations, and they want opposite fixes.

```
  node                   work   imbalance     n  threads busy
  -------------------------------------------------------------
  ffn_out            88.93 ms    10.19 ms    48          8.0
  ffn_up             88.39 ms     5.61 ms    48          8.0
  ffn_gate           88.61 ms     4.05 ms    48          8.0
  Qcur               23.19 ms     3.14 ms    96          7.9
  attn_out           22.55 ms     2.90 ms    48          8.0
  Kcur                8.35 ms     2.13 ms    96          6.2
  ffn_swiglu         267.0 us     1.75 ms    48          1.8
  l_out              204.3 us     1.30 ms    48          2.0
  Vcur                9.51 ms     1.27 ms    48          8.0
  attn_norm          211.9 us    657.9 us    48          1.8
  ffn_norm           184.4 us    386.1 us    48          1.4
```

The big matmuls use all 8 threads and leak 4–9% of their own work to skew —
ordinary, and roughly what a row-partitioned matmul on a noisy machine should
do.

The second population is the interesting one:

```
  10 node types cost more in other threads' waiting than in their own work:
  ffn_swiglu, l_out, attn_norm, ffn_norm, embd, ffn_inp.
  Together     1.08 ms of compute causes     4.78 ms of waiting -- 14% of all
  imbalance from 0.32% of the work.
```

These run on **1.4–2.0 threads out of 8**.

### Why, from the source rather than from the trace

Every elementwise op in ggml partitions over rows
([`ggml/src/ggml-cpu/ops.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ops.cpp)):

```c
const int nr = ggml_nrows(src0);
const int dr = (nr + nth - 1)/nth;   // rows per thread
const int ir0 = dr*ith;
const int ir1 = MIN(ir0 + dr, nr);   // this thread's range
```

At batch size 1 a hidden-state tensor is a **single row**. With `nr = 1` and
`nth = 8`: `dr = 1`, so thread 0 gets `[0,1)` and every other thread gets
`ir0 = ith >= 1 = ir1` — an empty range. **One thread does the work and seven
fall through to the barrier.**

`ggml_get_n_tasks` does not save them. `GGML_OP_SOFT_MAX` clamps itself with
`n_tasks = MIN(n_threads, ggml_nrows(node->src[0]))`, but `ADD`, `GLU` and the
norms take `n_tasks = n_threads` unconditionally — and in any case `n_tasks` is
consulted only by `ggml_graph_plan` for work-buffer sizing
([`ggml-cpu.c:2841`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c)).
It never gates dispatch. `ggml_graph_compute_thread` runs every node on every
thread and calls `ggml_barrier` after each one unconditionally.

### The prediction this makes, and the test of it

If the mechanism is `nrows == 1`, then the *same nodes* in a workload with many
rows should parallelize normally. Prefill is exactly that workload: `nr` becomes
`n_tokens`.

Measured, same model and thread count, `pp64` against the `tg` trace:

```
  node           decode busy  pp64 busy   dec work us   pp work us
  ---------------------------------------------------------------
  ffn_swiglu            1.79       7.67           5.6        208.5
  l_out                 2.04       7.88           4.3         36.1
  attn_norm             1.75       8.00           4.4        116.6
  ffn_norm              1.44       7.71           3.8         87.3
  ffn_inp               1.42       7.71           3.2         35.2
  ffn_out               8.00       8.00        1852.7      18945.8
  Qcur                  7.86       8.00         241.6       3178.3
```

Every near-serial node goes to 7.7–8.0 busy threads at `pp64`, while the
matmuls, which were already parallel, do not move. The mechanism is confirmed,
and it is **specific to batch-size-1 decode** — which is the most common CPU
inference workload there is.

The same collapse shows up in the whole-graph report, which is why the prefill
trace is committed alongside the decode one:

```
$ trace_analyze.py examples/mid-24L-L3-pp64.trace.json --barriers

  category             work   imbalance  wait/work  nodes  serial
  ---------------------------------------------------------------
  ffn            1398.36 ms     5.09 ms         0%    120       2
  norm              4.90 ms     1.02 ms        21%     49       2
  attn.kv_rw       397.7 us    152.1 us        38%     48       0
```

Against the decode run the `serial` column for those three phases reads
84, 86 and 92 out of the same node counts. The set of nodes whose barrier
costs more than their work shrinks from ten types carrying 14% of all
imbalance to two carrying 0%.

### What it is worth, stated as an upper bound

This is where the finding has to be careful, because the tempting number is
wrong. 120 of the 412 barriers per token (29%) follow a node only one thread
worked on. But the *thread* time those threads waste is not *wall* time saved:

```
  graph wall time                      25.207 ms/tok
  near-serial node wall time            0.385 ms/tok   1.53% of graph wall
  barrier release after those nodes     0.157 ms/tok
  ------------------------------------------------------------------------
  total attributable                    0.542 ms/tok   2.15% of graph wall

  perfect 8-way parallelization of those nodes would save 0.337 ms/tok
                                                        1.34%  (upper bound)
```

**1.34% is an upper bound that will not be reached.** These tensors are 1×768
and 1×3072 — a few microseconds of work each. Splitting three microseconds
across eight threads costs a dispatch and leaves each thread with less work than
the barrier that follows. The realistic version of this fix is not "parallelize
them", it is **fuse them**, so that a chain of single-row elementwise ops pays
one barrier instead of five.

Recorded as a bounded, checkable claim rather than a headline: the honest
statement is *"29% of decode barriers are paid for single-threaded nodes, worth
at most 1.3% of graph wall time, and the promising fix is fusion, not
partitioning."*

**Caveats:**

- Synthetic F32 weights. Q4_K_M shifts the mix toward compute in the matmuls,
  which would make the serial nodes a *smaller* share, not a larger one.
- 8 threads on one machine. The waste from a serial node scales with thread
  count, so this is a floor for wider machines, not a ceiling.
- Two decode tokens. The per-node structure is identical token to token, but
  the absolute wall figures carry that token count's noise.
- llama.cpp is under active development and already fuses some op chains
  (`ggml_cpu_try_fuse_ops`); this measurement is against the pinned commit
  `4d91760` and says nothing about what a newer tree does.

---

## Not yet measured

Listed so the gaps are explicit rather than implied:

- real quantized models — everything above is synthetic F32 weights
- context-shift behaviour, i.e. the case where `kv.update` should be expensive
- concurrent sequences / server workload
- sampling and tokenization, which `llama-bench` never exercises
