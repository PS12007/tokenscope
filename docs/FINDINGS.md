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

**Answered in [F9](#f9--most-barrier-wait-is-arrival-imbalance-and-a-third-of-the-barriers-are-paid-for-nodes-only-one-thread-worked-on).**
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

**Update (F12).** This law was later tested on a real Q4_K_M model, where
bytes and parameter counts stop agreeing because tensors are quantized
differently by role. The **byte** form held to within 2.9 points; the
**parameter-count** form was wrong by 7.2. The shortcut used throughout this
finding is safe only for a uniform dtype, and F12 says so in more detail.

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

## F9 — Most barrier wait is arrival imbalance, and a third of the barriers are paid for nodes only one thread worked on

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

- Synthetic F32 weights. **Retested on a real Q4_K_M model in F12, where the
  effect is larger, not smaller: 0.74% of the work causing 19% of the
  imbalance, against 0.32% and 14% here.** The prediction in this bullet was
  backwards -- cheaper matmuls make the fixed-cost serial nodes a bigger
  share, not a smaller one.
- 8 threads on one machine. The waste from a serial node scales with thread
  count, so this is a floor for wider machines, not a ceiling.
- Two decode tokens. The per-node structure is identical token to token, but
  the absolute wall figures carry that token count's noise.
- llama.cpp is under active development and already fuses some op chains
  (`ggml_cpu_try_fuse_ops`); this measurement is against the pinned commit
  `4d91760` and says nothing about what a newer tree does.

---

---

## F10 — Nothing beats 2.2×. Every thread past four turns into barrier wait.

**Workload:** same 24-layer synthetic F32 model, `tg32`, Windows 11, MSVC
Release, on an i7-14700HX (8 P-cores + 12 E-cores, 28 logical). Throughput
from the **uninstrumented** build, `-r 5`. Structural columns from level-3
traces of 6 decode tokens at each thread count.

```
 thr   tok/s  speedup  par.eff  barrier%  imbal%  thr-spread  serial imb%
--------------------------------------------------------------------------
   1   20.25    1.00x     100%      0.1%    0.0%          0%        0.0%
   2   35.12    1.73x      87%      2.4%   64.2%          1%       12.5%
   4   42.34    2.09x      52%      5.7%   66.6%          1%       15.5%
   6   44.82    2.21x      37%      8.4%   65.3%          1%       16.3%
   8   44.59    2.20x      28%     12.2%   71.5%          2%       10.1%
  12   40.59    2.00x      17%     21.2%   77.3%         13%        8.5%
  16   41.50    2.05x      13%     22.0%   73.8%         27%        7.4%
  20   41.83    2.07x      10%     22.9%   69.7%         23%       15.3%
  28   39.33    1.94x       7%     22.9%   56.1%         29%       15.2%
```

`barrier%` is the share of worker thread time spent in `ggml_barrier`.
`imbal%` is the share of that wait caused by uneven arrival rather than
release latency (F9). `thr-spread` is the gap between the busiest and
idlest worker's compute time, as a percentage of the busiest.

### The headline

**No thread count on this machine beats 2.21×.** Four threads already reach
2.09×. Everything after that buys at most 6% more throughput, and past six
threads it buys *less than nothing* — 28 threads is 12% slower than 6 while
occupying seven times the cores.

Parallel efficiency falls from 100% to **7%**.

### Where the thread time goes, which is the part a normal profiler hides

Barrier wait rises monotonically with thread count — 0.1% → 12.2% at eight
threads → 22.9% at twenty and beyond. By 12 threads, **more than a fifth of
all worker CPU time is spinning in `ggml_thread_cpu_relax()`**.

The causal direction matters, and it is the opposite of the tempting reading.
The barrier is not what makes the extra threads useless. F7 established
that decode on this model is almost purely weight-streaming: the time is memory
traffic, not arithmetic. Once memory bandwidth is saturated — which happens
somewhere around four threads — an additional thread cannot go faster, so it
finishes its slice late or early relative to the others and the difference is
paid at the next barrier.

**The barrier is where wasted parallelism becomes visible, not the cause of
it.** And that is exactly the point of the work/wait split: without it those
threads are indistinguishable from threads doing arithmetic. A tool that
reports "ffn: 69% across 8 threads" cannot tell you that a fifth of that number
is spinning, and it is spinning *because you asked for too many threads.*

### The thread-spread step at eight

`thr-spread` is 0–2% for every thread count up to and including eight, then
jumps to 13% at twelve and 23–29% above that.

Eight is the P-core count on this CPU. Up to eight threads the scheduler has
equivalent cores to hand out and ggml's even row split is a fair split. Past
eight, threads land on cores of a different speed while ggml keeps handing out
**equal numbers of rows**, so the slowest core sets the pace at every one of
the ~412 barriers per token.

**A prediction I made and did not confirm.** (**Now settled in F14** by
pinning: twelve threads on twelve identical E-cores drop the spread from 13% to
2% and halve barrier wait. The mechanism is confirmed; the paragraph below
stands as written because the *evidence available at the time* did not support
it.) I expected the per-thread compute times to fall into two tight clusters,
one per core type. They do not — at 16
threads the distribution runs 30.0, 30.2, 30.5, 33.3, 33.7, 35.1, 37.2 … 44.9
ms, continuously. So "even split across uneven cores" is consistent with the
step at eight, but the clean bimodal signature that would prove it is absent.
Thread pinning and per-core identification would settle it; neither is done
here, and Windows is free to migrate threads mid-run. **Recorded as an
unconfirmed mechanism with a confirmed symptom.**

### Testing docs/01's overhead prediction — not confirmed

[`docs/01`](01-design-scope-timing.md) section 8 predicts that profiler
overhead should *compound* with thread count, because a barrier makes the graph
pay the `max` of per-thread overhead rather than the mean, and lists
pre-touching chunks and never allocating mid-graph as the mitigations.

Measured with the interleaved A/B/C harness, `-n 14`, level 3:

| threads | overhead vs compiled-out | baseline IQR |
|---|---|---|
| 8 | +0.12% [-1.59, +1.43] | 2.0% |
| 28 | +0.66% [-1.85, +4.18] | 3.9% |

Both confidence intervals span zero, and the harness **refused to certify
either**, because this machine's baseline spread is wider than the effect being
tested. So the prediction is neither confirmed nor refuted. What can honestly
be said: if overhead compounds with thread count, it does not compound enough
to resolve at 28 threads on this machine — which is a weaker claim than the
mitigations working, and is the claim the data supports.

### Caveats

- **Synthetic F32 weights, and this finding is more sensitive to that than any
  other in this file.** The whole result turns on the workload being
  bandwidth-bound. A Q4_K_M model reads a quarter of the bytes per parameter
  and should therefore scale to more threads before hitting the same wall. This
  table is a statement about *this* workload's arithmetic intensity, not about
  llama.cpp's threading in general. **F12 tested that prediction on a real
  quantized model and confirmed it: peak speedup rose from 2.21x to 3.28x. The
  peak stayed at six threads.**
- One machine, one OS, hybrid core layout, no thread pinning.
- Structural columns come from single traces of 6 tokens; the throughput column
  is `-r 5` on the uninstrumented build. They are not the same runs, and the
  throughput column is the one to quote.
- `thr-spread` is measured per thread across a whole token, so it mixes core
  speed with scheduling noise and with the near-serial nodes of F9.

### Reproduce

```bash
# throughput column
for T in 1 2 4 6 8 12 16 20 28; do
  llama-bench -m mid.gguf -p 0 -n 32 -t $T -r 5
done

# structural columns
TOKENSCOPE_LEVEL=3 TOKENSCOPE_TOKENS=8-13 TOKENSCOPE_OUT=t$T.json \
  llama-bench -m mid.gguf -p 0 -n 20 -t $T -r 1 --no-warmup
trace_analyze.py t$T.json --barriers
```

---

---

## F11 — Sampling and detokenization are 0.13% of a token, and finding that out broke the attribution check

**Workload:** same 24-layer synthetic F32 model, `llama-cli` (not `llama-bench`),
6 threads, 31 decode tokens, `TOKENSCOPE_LEVEL=1`
([`examples/mid-24L-cli-sampling.trace.json`](../examples/mid-24L-cli-sampling.trace.json)).

Every number in F1 through F10 came from `llama-bench`, which never samples and
never tokenizes. Those two are the last pieces of the per-token loop that had
never been measured, so this closes the gap the "Not yet measured" list has
carried since F1.

```
  per-token work OUTSIDE the decode slice (sampling, detokenization)

  category                 total       %      per-tok
  --------------------------------------------------
  sample                886.9 us    0.1%      0.029 ms
  tok.decode             19.2 us    0.0%      0.001 ms

  906.1 us on top of decode, i.e. 0.1% more wall time per token.
```

- **`sample` — 28.6 µs/token, 0.129% of decode.** That is the whole CPU sampler
  chain: `set_logits`, the reasoning-budget and grammar samplers, and the chain
  apply, over an 8192-entry vocabulary.
- **`tok.decode` — 0.62 µs/token.** Detokenization is not a cost.
- **`tok.encode` — 206 µs, once,** for the whole prompt. It is a prefill cost
  and does not recur.

So sampling joins KV-slot search, graph building and logits readback on the
list of things that sound expensive and are not. On this model, **99.87% of a
generated token is `llama_context::decode`, and 99.5% of that is
`graph_compute`.** F1 said every remaining question lives inside the graph;
with sampling now measured rather than assumed, that statement is complete
rather than provisional.

### The part worth more than the number

Adding these scopes made the summary report **100.1% of decode wall time
attributed to a host scope** — which is impossible, and is the same consistency
check that caught the scope-misparenting bug in F5.

The cause was not a timing bug. It was a **boundary** bug, and a conceptual one:
`common_sampler_sample` runs *after* `llama_context::decode` returns. The scopes
were correctly timed and correctly associated with their token — but they lie
outside that token's slice, so charging them against it adds time the slice
never contained. 341 of 341 sampling scopes were outside; 16 of 31 tokens went
over.

The fix is a distinction the report now makes explicitly: per-token work that
happens *inside* the decode slice is charged against it, and per-token work that
happens *outside* it is reported separately, with its own note saying so. Both
are real costs of producing a token. Only one is part of `decode`.

This is F4's lesson arriving from the other direction. F4 was about two tools
drawing the boundary differently; this is about one tool drawing it in one place
and then quietly billing work from outside it. **A per-token profiler has to
answer "per token" and "inside what" as separate questions,** and the 0.1%
overshoot was the only thing that made the difference visible.

The CI check now applies the same inside-the-slice rule, so the two cannot drift
apart, and the `llama-cli` trace is committed as a reference so the sampling
path stays covered.

### Caveats

- One sampler chain (`llama-cli` defaults) and an 8192-token synthetic
  vocabulary. A real 128k vocabulary makes every softmax and sort roughly 16×
  larger, and a grammar-constrained sampler is a different workload entirely.
  **0.13% is not a general claim about sampling.**
- `llama_sampler_apply` is scoped rather than `common_sampler_sample`, so
  `llama_synchronize` and `set_logits` inside the latter are *not* in the
  number. The scope covers the samplers, not the wait for the graph.
- 6 threads, single sequence, greedy-ish default chain.

---

---

## F12 — The first real quantized model, and three standing predictions tested against it

**Workload:** Qwen2.5-0.5B-Instruct **Q4_K_M** (the published GGUF: 24 layers,
`n_embd` 896, `n_ff` 4864, 14 heads / 2 KV heads, **vocab 151,936**, 630 M
params, 469 MiB on disk). Throughput from the uninstrumented build, `-r 10` for
the headline points. Structural columns from level-3 traces of 6 decode tokens.

Every finding before this one was measured on synthetic F32 weights, and each
carried that as its first caveat. This is the model that tests whether they
survive contact with a real one. **Two predictions held, one held only in its
correct form, and the correct form is not the one the earlier finding leaned
on.**

### Prediction 1 (F10): a quantized model should scale to more threads. Confirmed.

F10 concluded decode was bandwidth-bound and predicted that a model reading
fewer bytes per parameter would scale further before hitting the same wall.

```
              synthetic F32 (220M)      Qwen2.5-0.5B Q4_K_M (630M)
 thr   tok/s  speedup  par.eff      tok/s  speedup  par.eff
   1   20.25    1.00x     100%      27.06    1.00x     100%
   2   35.12    1.73x      87%      52.85    1.95x      98%
   4   42.34    2.09x      52%      77.49    2.86x      72%
   6   44.82    2.21x      37%      88.88    3.28x      55%
   8   44.59    2.20x      28%      88.20    3.26x      41%
  12   40.59    2.00x      17%      70.80    2.62x      22%
  28   39.33    1.94x       7%      66.07    2.44x       9%
```

**Peak speedup rises from 2.21× to 3.28×**, and parallel efficiency is higher at
every thread count up to the peak — 98% vs 87% at two threads, 72% vs 52% at
four. The prediction was right.

What the prediction did *not* say, and is worth recording: **the peak is still at
six threads.** The wall moved up, not out. And the fall past eight threads is
*steeper* on the quantized model (−18% by ten threads) than on the F32 one
(−9% by twelve). That is consistent with the same core-heterogeneity story F10
told: a more compute-bound workload is hurt more by slow cores, not less. It is
consistent with, not evidence for — see F10's unconfirmed-mechanism note.

### Prediction 2 (F7): phase time tracks weight bytes. Confirmed — and the parameter-count shortcut is now refuted.

F7 predicted time per phase should be proportional to **bytes of weights read**,
and added: "for a uniform dtype that is just parameter count." Every F7 number
used the parameter-count form, because on an F32 model the two are the same
thing.

A real Q4_K_M file is *not* uniform. llama.cpp quantizes tensors differently by
role, and in this file the output projection is **Q8_0 at 8.50 bits/weight while
everything else sits near 5.5**. So the two forms of the prediction finally
disagree, and can be told apart.

```
phase       bits/w  param share  byte share  time share  err(param)  err(byte)
------------------------------------------------------------------------------
ffn           5.51        63.5%       55.2%       56.4%       -7.2       +1.2
lm_head       8.50        27.6%       36.9%       34.0%       +6.4       -2.9
attn.qkv      5.70         5.0%        4.5%        6.0%       +1.0       +1.5
attn.out      5.50         3.9%        3.4%        3.6%       -0.3       +0.2

max error predicting time from parameters: 7.2 points
max error predicting time from bytes:      2.9 points
```

**Bytes win, and they win exactly where the two disagree.** The parameter-count
form is wrong by 7.2 points on `ffn` and 6.4 on `lm_head` — in opposite
directions, which is the signature of a share being moved from one to the other
by nothing but dtype. The byte form holds to within 2.9 points across a 16×
range.

So F7's law survives, and F7's *convenience* does not. Anyone reusing that
result on a quantized model must read the tensor types, not the config.

### The finding that only a real model could produce: `lm_head` is a third of decode

On the synthetic model `lm_head` was 2.6% of thread time and easy to ignore.
Here it is **34%**, second only to the entire FFN stack across all 24 layers.

Two things compound:

- **Vocabulary 151,936 against `n_embd` 896.** The output projection is
  136 M parameters — 28% of everything streamed at decode — in a model whose
  every other matrix is sized for a 0.5B model.
- **It is the least compressed tensor in the file.** Q8_0, while the FFN it
  competes with is Q5_0/Q4_K/Q6_K. It is 27.6% of the parameters and 36.9% of
  the bytes.

The practical reading, stated as an arithmetic consequence rather than a
recommendation: requantizing that one tensor from Q8_0 to ~5.5 bits would remove
roughly 35% of its bytes and, if the byte law holds, about **12% of decode
time** — from a single tensor. Whether that is a good trade is a quality
question this project has not measured and is not qualified to answer; output
layers are quantized conservatively for a reason.

The general point stands on its own: **on small models with large vocabularies,
the output projection is a first-class cost, and per-tensor quantization choices
are visible in the profile.**

### Prediction 3 (F9): the near-serial elementwise nodes. Confirmed, and worse.

```
node          work    imbalance    n   threads busy (of 6)
ffn_swiglu   531.2 us    1.92 ms  144       1.0
l_out        253.5 us    1.03 ms  144       1.0
attn_norm    328.6 us    1.01 ms  144       1.1
ffn_norm     299.5 us   545.2 us  144       1.0
ffn_inp      243.5 us   420.1 us  144       1.0
```

Eleven node types cost more in waiting than in their own work: **0.74% of the
work causing 19% of all imbalance**, against 0.32% and 14% on the synthetic
model. The mechanism is dtype-independent, as it should be — a single row is a
single row whatever it is quantized to — and its relative cost grows as the
matmuls around it get cheaper.

**A new one, specific to GQA.** `Kcur` uses 2.5 of 6 threads and `Vcur` 3.0,
where `Qcur` uses 4.0 and the FFN matmuls use all 6. Qwen2.5-0.5B has 14 query
heads and **2** KV heads, so the K and V projections produce 128-wide outputs
against 896 for Q. They are too narrow to fill the pool. Grouped-query attention
shrinks the KV cache, and the same narrowing shows up here as a partitioning
problem — a cost of GQA that a wall-clock timer cannot see.

### Caveats

- One model, one quantization, one machine. "Q4_K_M" is a recipe, not a dtype;
  this file's actual mix is 133 Q5_0 tensors, 121 F32, 13 Q8_0, 12 Q6_K, 12
  Q4_K. Another Q4_K_M export may differ, which is itself the point of this
  finding.
- 630 M parameters is still small. The bandwidth/compute balance shifts again at
  7B, and the `lm_head` share in particular shrinks fast as models grow, since
  vocabulary is fixed while everything else scales.
- Byte counts come from the GGUF tensor table via `gguf-py` and are what is
  *stored*, not what is *fetched* — they ignore cache reuse, which at batch
  size 1 with a single sequence is close to nil for weights but not exactly nil.
- Structural columns are single traces of 6 tokens; throughput is `-r 10`.

---

---

## F13 — Three models, three different answers to "where does decode time go"

**Workload:** 6 threads, `TOKENSCOPE_LEVEL=3`, 6 decode tokens each, same
machine and binary. Thread-time share per phase.

```
model                 params      ffn  attn.qkv  attn.out  lm_head  attn.score  barrier
---------------------------------------------------------------------------------------
tiny   8L F32 synth     8.9 M   27.6%     10.7%      3.2%     0.9%       2.7%     54.0%
mid   24L F32 synth     220 M   70.1%     10.4%      6.0%     2.6%       0.7%     10.1%
Qwen2.5-0.5B Q4_K_M     630 M   49.2%      5.3%      3.2%    29.6%       1.1%     11.3%
```

- `tiny`: 8 layers, `n_embd` 256, `n_ff` 1024, MHA. 761 tok/s.
- `mid`: 24 layers, `n_embd` 768, `n_ff` 3072, 12 heads / 4 KV. 32.5 tok/s.
- `Qwen2.5-0.5B`: 24 layers, `n_embd` 896, `n_ff` 4864, 14 heads / 2 KV,
  vocab 151,936, Q4_K_M. 88.9 tok/s.

Read the columns down rather than across. **The largest single consumer of
decode is a different thing in each row:**

- On `tiny` it is the **barrier** — 54%. The model is small enough that the
  work between two rendezvous is smaller than the rendezvous. ~200 nodes per
  token, six threads, and each node's arithmetic finishes before the
  synchronization protecting it does. This is not llama.cpp being slow; it is a
  graph executor being used far below the size it is designed for.
- On `mid` it is the **FFN**, at 70%, with the barrier down to 10%. This is the
  textbook weight-streaming picture that F7 modelled and F10 explained.
- On `Qwen2.5-0.5B` it is the FFN *and* **`lm_head` at 29.6%**, because a
  151,936-token vocabulary against `n_embd` 896 makes the output projection a
  first-class cost (F12).

So the honest general statement is that **there isn't one.** "Where does CPU
decode time go" has no model-independent answer, and any advice of the form
"optimize X for CPU inference" is implicitly quantified over a model shape that
usually goes unstated.

Two things are stable across all three:

- **`attn.score` is 0.7-2.7%.** The actual attention computation — scores and
  softmax — is nearly free at batch size 1 with a short context, on every model
  measured. F7 said this and it now has three data points instead of one.
- **`norm` never exceeds 0.4%** of thread time while being, per F9, one of the
  larger sources of *barrier* time. The cheapest nodes stay the most expensive
  rendezvous.

### Caveats

- Two of the three models are synthetic F32 with random weights. Weight values
  do not affect timing on these kernels, but the *dtype mix* does, which is
  exactly what makes the Qwen row different (F12).
- 6 threads throughout. F10 shows the barrier column in particular is strongly
  thread-count-dependent, so these are three points on one slice of a larger
  surface, not three model summaries.
- Single traces of 6 tokens each, so the small columns carry real noise. The
  ordering of the large columns is far outside it.

---

---

## F14 — Core heterogeneity is the mechanism. Pinning proves it and does not fix it.

**Workload:** 24-layer F32 synthetic model, i7-14700HX (8 P-cores + 12 E-cores,
28 logical), Windows 11. Throughput from the uninstrumented build, `-r 5`.
Structure from level-3 traces of 6 decode tokens. Affinity via llama-bench
`-C <hex> --cpu-strict 1`.

F10 measured that per-thread compute spread is 0-2% up to eight threads and
13-29% above, noted that eight is this CPU's P-core count, and then
**deliberately declined to claim the mechanism** because the expected bimodal
signature was absent. This is the experiment that settles it.

### First: the cores really are that different

Four threads, `pp128` (compute-bound, unlike decode):

```
  4 threads on P-cores  (mask 0x55)       265.95 ± 2.40 tok/s
  4 threads on E-cores  (mask 0xf0000)     92.39 ± 0.89 tok/s
  4 threads unpinned                      266.79 ± 4.98 tok/s
```

**P-cores are 2.88× faster than E-cores here**, and the unpinned scheduler
picks P-cores, as it should. (This gap is invisible at one thread on *decode* —
28.0 vs 28.2 tok/s — because single-thread decode is bandwidth-bound and both
core types wait on the same memory. That near-miss is worth recording: the
first version of this experiment used decode, found no difference, and would
have concluded the mask was broken.)

### The test: make the cores homogeneous and see if the spread goes away

```
case                       thr   spread   min ms   max ms   barrier%
--------------------------------------------------------------------
12 thr unpinned (mixed)     12      13%    112.3    129.6      21.2%
12 thr E-cores only         12       2%    134.8    137.0      11.1%
```

Twelve threads on twelve **identical** E-cores: per-thread compute times run
134.8, 134.9, 134.9, 135.0, 135.2, 135.3, 135.3, 135.4, 136.6, 136.9, 136.9,
137.0 ms. The spread collapses from 13% to 2%, and **barrier wait halves, from
21.2% to 11.1%**, with no change to the model, the graph, the thread count, or
the code.

**F10's mechanism is confirmed.** ggml gives every thread the same number of
rows; when the cores behind those threads are not the same speed, the slowest
one sets the pace at every one of the ~412 barriers per token. Equal work to
unequal workers is the whole story.

### And yet pinning makes it slower

```
  6 thr  unpinned                45.15 ± 0.53
  8 thr  unpinned                43.60 ± 0.63
 12 thr  unpinned                40.89 ± 0.09
  8 thr  P-cores only            39.21 ± 0.44      -10% vs unpinned
 12 thr  E-cores only            37.51 ± 3.15       -8% vs unpinned
 20 thr  P8 + E12                35.76 ± 1.09
```

Every pinned configuration is **worse** than letting the scheduler choose. The
homogeneous E-core run halves its barrier waste and still loses, because twelve
slow equal cores beat neither eight fast ones nor the scheduler's mix.

That is the useful conclusion, and it is not the obvious one:

**The waste is real, the mechanism is confirmed, and pinning is the wrong fix.**
Removing heterogeneity by refusing to use the fast cores costs more than the
heterogeneity did. What the measurement actually argues for is **proportional
work assignment** — giving a P-core more rows than an E-core — rather than
ggml's current

```c
const int dr = (nr + nth - 1)/nth;   // every thread gets the same count
```

which is optimal exactly when every worker is equally fast, and is a growing
tax as consumer CPUs get less uniform. That would keep the fast cores *and*
close the arrival gap. It is a much larger change than anything else this
project has suggested, and it is the one the data points at.

### An anomaly, unexplained

Eight threads pinned to `0x5555` — intended as one thread per P-core — do
**not** behave homogeneously. Across three runs, seven threads cluster tightly
and exactly one is 12-20% slower:

```
  127 128 128 128 128 129 129 146
  122 123 123 123 123 124 125 140
  124 124 124 124 125 125 126 156
```

The slow thread is `tid 0` twice and `tid 7` once, so it is not simply the main
thread doing host work between graphs. Candidate explanations not tested:
hyperthread sibling collision from a wrong assumption about this CPU's logical
numbering, or `--cpu-strict` assigning two threads to one bit. **Recorded rather
than explained**, and it is why the confirmation above rests on the twelve-core
E-only run, which is clean, rather than on this one.

### Caveats

- One machine, one OS, one hybrid layout. AMD CCD topologies and Apple's
  P/E split will differ in ways this does not predict.
- The mask-to-core mapping is assumed from the usual Windows enumeration
  (P-core threads first, E-cores from logical 16), and the anomaly above is a
  reason to hold that assumption loosely.
- Throughput and structure come from different runs; the throughput column is
  the one to quote.
- Synthetic F32 weights. F12 showed the quantized model falls off *more*
  steeply past the P-core count, which is consistent with this being the cause,
  but that combination was not measured under pinning.

---

## Not yet measured

Listed so the gaps are explicit rather than implied:

- larger real models — the biggest measured is 630 M parameters (F12)
- context-shift behaviour, i.e. the case where `kv.update` should be expensive
- concurrent sequences / server workload
