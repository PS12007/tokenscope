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

**That measurement is F16, and this prediction was wrong.** `find_slot` keeps a
rotating head pointer rather than scanning from zero, so its cost is amortized
O(1) and *fell* 40% as the cache filled. Worse, the 56 us/token quoted above is
97% batch splitting: the scope wraps `init_batch`, which calls `find_slot` but
is not `find_slot`. The actual cell search is 1.17 us/token.

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

**That recommendation was tested in F15 and did not hold.** ggml already fuses
exactly this kind of chain (`RMS_NORM` + `MUL`), removing 49 of 461 barriers per
token; turning it off changes throughput by no measurable amount, in three
regimes including one where barrier wait is 54% of thread time. The barriers a
fusion removes are the ones threads arrive at together, so removing them saves
nothing. The measurement above stands; the fix proposed here does not.

Recorded as a bounded, checkable claim rather than a headline: the honest
statement is *"29% of decode barriers are paid for single-threaded nodes, worth
at most 1.3% of graph wall time, and the promising fix is fusion, not
partitioning."*

**Caveats:**

- Synthetic F32 weights. **Retested on a real Q4_K_M model in F12, where the
  effect is larger, not smaller: 0.74% of the work causing 19% of the
  imbalance, against 0.32% and 14% here.** And **F17 shows it disappears
  entirely under batched decode** -- 1.0 of 6 busy threads becomes 6.0 at
  `-np 16` -- so this is a batch-size-1 finding, which is to say a
  single-user-local-inference finding. The prediction in this bullet was
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

---

## F15 — Fusion removes 10.6% of the barriers and buys nothing. F9's recommendation was wrong.

**Workload:** as labelled below; uninstrumented build for throughput, level-3
traces for structure. ggml exposes `GGML_CPU_DISABLE_FUSION=1`, which turns off
its op fusion at runtime — so the value of fusion can be measured directly
rather than argued about.

F9 closed by recommending fusion: the near-serial elementwise nodes cost more in
other threads' waiting than in their own work, so *"the realistic version of
this fix is not 'parallelize them', it is fuse them, so that a chain of
single-row elementwise ops pays one barrier instead of five."*

That was reasoning, not measurement. Here is the measurement.

### What ggml already fuses, and what it removes

`ggml_cpu_try_fuse_ops` implements exactly **one** pattern: `RMS_NORM` + `MUL`.
That is the norm chain — precisely the kind of single-row elementwise work F9
was talking about.

Turning it off, on the 24-layer F32 model at 6 threads:

```
                     barriers/token   barrier share of thread time
  fusion enabled          412                    6.8%
  fusion disabled         461                    8.5%
```

**49 barriers per token** — 24 layers × 2 norms, plus the final one — or 10.6%
of all barriers in the graph. Barrier thread-time share falls by a fifth
relative. By F9's argument this should show up as throughput.

### It does not

Interleaved arms, `-r 5` per round:

```
  24L F32, 6 threads   (barrier ~7% of thread time)
    fusion enabled    44.75   44.78   45.06     mean 44.86
    fusion disabled   44.81   44.86   45.00     mean 44.89

  24L F32, 20 threads  (barrier ~23%)
    fusion enabled    42.06   41.71               mean 41.89
    fusion disabled   41.62   41.86               mean 41.74

  tiny 8L, 6 threads   (barrier 54%)
    fusion enabled  1789.92 1933.83               mean 1861.9
    fusion disabled 1854.92 1875.68               mean 1865.3
```

**No measurable difference in any regime**, including the one where barrier wait
is more than half of all worker thread time. The signs are not even consistent:
−0.06%, +0.35%, −0.18%, all inside the run-to-run spread.

### Why, and what it means for F9

The barriers fusion removes are **cheap** barriers. A barrier's wall-clock cost
is set by how far apart the threads arrive at it, and threads arrive at a norm
almost together — the node before it is tiny and everyone finishes it at
roughly the same moment. Removing a rendezvous that nobody was waiting long at
saves nothing, no matter how many of them you remove.

F9's own table said this and I misread it. `ffn_swiglu` carries 1.75 ms of
imbalance; `ffn_out` carries **10.19 ms**. The waiting that matters is at the
barriers after the *big matmuls*, where threads genuinely arrive at different
times — and F14 then showed why they do: unequal cores given equal row counts.

So the three findings resolve into one story, with the middle step corrected:

- **F9** — the tiny elementwise nodes are single-threaded and cost more in
  waiting than in work. **True, and still true on a real quantized model.**
- **F9's proposed fix** — fuse them. **Tested here. No measurable benefit.**
- **F14** — the imbalance with real wall-clock cost comes from core
  heterogeneity in the large matmuls, and wants proportional work assignment,
  not fewer rendezvous.

F9's *measurement* stands. F9's *recommendation* does not, and F9 now says so
inline. The 1.34% figure there was already labelled an upper bound that would
not be reached; this is how far short it falls — the reachable part is not
distinguishable from zero by this experiment.

### The general point

Barrier count and barrier cost are different quantities, and a profiler that
reports the first invites you to optimize the second by proxy. Ten percent of
the rendezvous carried approximately none of the waiting. **"Reduce
synchronization" is not a strategy; "reduce the arrival spread at the
synchronizations that have one" is.**

### Caveats

- This tests one fusion pattern, the only one ggml has. It is a norm chain, and
  the residual-add and SwiGLU chains F9 also named are not fused by anything, so
  they are not directly tested. But they are the same *kind* of node — cheap,
  single-row, entered by threads that arrive together — so the prior that
  fusing them would pay should now be much weaker, not merely unproven.
- Interleaved arms but not the full bootstrap treatment of
  [`02`](02-overhead-methodology.md); this is a null result on a machine whose
  noise floor is 1-3%, so it bounds the effect rather than excluding it. A real
  effect smaller than about 1% would not be visible here.
- The tiny-model arm has ±50-110 tok/s of spread on a ~1860 tok/s median, which
  is why it is reported as three regimes agreeing rather than as one precise
  number.

---

---

## F16 — Context shift is real and cheap. F2's prediction was wrong, and the scope name is why.

**Workload:** Qwen2.5-0.5B Q4_K_M, `llama-cli`, **`-c 256`** with `-n 700` so the
context fills and shifts repeatedly, 6 threads, `TOKENSCOPE_LEVEL=1`. 699 decode
tokens, median 11.60 ms.

F1 and F2 each recorded a prediction about the KV cache *before* the experiment
that could test it, which is the only reason either is worth anything now. This
is that experiment.

### F1's prediction: confirmed

F1 measured `kv.update` at 275 µs across 257 tokens and said the case where it
should light up — long generation in a tight window — "has not been tested yet."

```
context-shift events (kv.update > 1 ms): 4 of 699 tokens

  token  218   13.34 ms  (1.15x median)  kv.update 1.72 ms
  token  345   15.54 ms  (1.34x median)  kv.update 1.80 ms
  token  472   13.48 ms  (1.16x median)  kv.update 1.99 ms
  token  599   13.13 ms  (1.13x median)  kv.update 1.62 ms
```

Four shifts, **evenly spaced 127 tokens apart** in a 256-cell cache, each
costing 1.6-2.0 ms on an 11.6 ms token. `kv.update` in total went from 275 µs
(no shift) to 7.83 ms — **28×** — exactly as predicted.

And it is the first thing in this project that produces a *periodic* per-token
spike rather than a flat cost, which is precisely the "token 340 stalled" shape
the tool was built to catch. It is also, honestly, small: a shift token is 13-34%
slower than median, not 3×.

### F2's prediction: falsified

F2 said of `kv.slot-search`:

> `llama_kv_cache::find_slot` is a linear scan for free cells, so its cost grows
> with cache occupancy. […] on a long run in a nearly-full cache […] this is the
> host-side cost that stops being a rounding error.

This run is exactly that: a 256-cell cache held at capacity for 500 tokens.

```
  quarter 1: find-slot  1.59 us/tok    slot-search  43.31 us/tok
  quarter 2: find-slot  1.20 us/tok    slot-search  42.81 us/tok
  quarter 3: find-slot  0.95 us/tok    slot-search  42.56 us/tok
  quarter 4: find-slot  0.94 us/tok    slot-search  41.65 us/tok
```

It does not grow. It **shrinks**, by 40%, as the cache fills.

**Why.** `find_slot` is not a scan from zero. It keeps a per-stream head pointer
(`v_heads[]`) and starts from there, so in steady-state single-sequence decode
the cell it wants is the one immediately after the last one it took, and the
loop exits almost immediately. There is even an explicit reset —
`if (head_cur > cells.get_used() + 2*n_tokens) head_cur = 0;` — for the case
where enough space has opened up behind it. The cost is amortized O(1) per
token, not O(occupancy). F2 described an algorithm llama.cpp does not use.

### The part that is a lesson rather than a correction

F2 quoted 56 µs/token and attributed all of it to the cell search. Adding the
scope for the cell search itself — site 13 in [`00`](00-architecture-map.md),
listed since the first day and never implemented — shows what that number was:

```
  kv.slot-search self    42.59 us/tok      (init_batch, minus find_slot)
  kv.find-slot            1.17 us/tok      (the actual cell search)

  find_slot is 2.7% of what F2 called "kv.slot-search".
```

**97% of it was batch splitting, not searching.** The scope wraps
`memory->init_batch(...)` at `llama-context.cpp:1785`, which splits the batch
into ubatches *and then* calls `find_slot`. It was named for the interesting
half and measured both.

So F2 was wrong twice over, and the profiler carried the error: a name that
over-claimed what a scope covered, and a prediction reasoned from the name
rather than from the code. **A scope's name is a claim about what it measures,
and it is exactly as checkable as any other claim in this file.**

The fix is the measurement above — the nested scope now separates them
permanently, so the split is visible rather than assumed. The enclosing scope
keeps its name for compatibility with the committed reference traces, which is a
compromise, and the architecture map now says what it actually covers.

### Caveats

- Single sequence. F2's *other* prediction — that concurrent sequences stress
  `find_slot` far harder — remains untested, and the head-pointer mechanism is
  much weaker with many streams competing, so it is still plausible.
- A 256-cell cache is small. The head-pointer argument says occupancy should not
  matter at any size, but only 256 was tested.
- `kv.update` at 1.6-2.0 ms per shift is for a 24-layer 0.5B model; the shift
  copies KV data, so it should scale with layers × heads × context.

---

---

## F17 — Concurrent sequences: F2's last prediction also fails, and the workload broke tokenscope's own prefill/decode boundary

**Workload:** Qwen2.5-0.5B Q4_K_M, `llama-batched -np {1,4,16} -kvu`, 96 tokens
per sequence, 6 threads, `TOKENSCOPE_LEVEL=1`, 92 decode steps each.

F2 made two predictions about `find_slot`. F16 killed the first (occupancy).
This is the second: *"with many concurrent sequences […] this is the host-side
cost that stops being a rounding error."* F16 explicitly left it open, and
argued it was the more likely of the two to hold, because the per-stream head
pointer that makes `find_slot` O(1) should weaken when many streams compete.

```
  np  ms/step    tok/s  vs np=1  find-slot  slot-search  graph-compute
   1    11.58     86.4    1.00x     0.89 us     37.75 us      11.42 ms
   4    20.18    198.2    2.29x     0.93 us     44.86 us      19.86 ms
  16    60.57    264.2    3.06x     1.55 us     61.07 us      59.65 ms
```

`find_slot` does grow — **1.7× from one sequence to sixteen** — so the
directional half of the prediction is right. But it grows from 0.89 µs to 1.55
µs on a step that costs 60,570 µs. It is **0.0026% of a decode step at 16
concurrent sequences.** It does not stop being a rounding error; it is not
within three orders of magnitude of stopping.

Both of F2's predictions are now tested and both fail. What survives is the
observation that started it — `kv.slot-search` is the largest *host-side* cost —
and F16 already showed that number is 97% batch splitting.

**The incidental result:** batching 16 sequences gives **3.06× aggregate
throughput**, which is the expected shape for a weight-streaming workload (F7,
F10) — the weights are read once per step regardless of how many sequences ride
along. It is 3.06× and not 16× because a 16-sequence step still costs 5.2× a
one-sequence step; the per-sequence attention and the wider matmuls are real.

### Batching dissolves the F9 problem entirely

F9 found that the single-row elementwise nodes run on one thread at batch size
1, and confirmed the mechanism by showing they parallelize during prefill,
where there are many rows. Concurrent decode is the third case: `n_seqs` rows
instead of one, on the *decode* path.

```
  level 3, 6 threads, 4 decode steps each

  np=1    barrier 13.6%   busy threads: ffn_swiglu 1.0  l_out 1.0  attn_norm 1.0
                                        ffn_norm 1.0    ffn_inp 1.0
  np=16   barrier  7.2%   busy threads: ffn_swiglu 6.0  l_out 5.8  attn_norm 6.0
                                        ffn_norm 6.0    ffn_inp 5.9
```

Every one of them goes from **1.0 of 6 threads to essentially all 6**, and
barrier wait **halves**, 13.6% to 7.2%, on the same model and thread count.

So F9 is precisely a batch-size-1 problem, and anyone already serving
concurrent requests has it fixed for free. That also narrows who the finding is
for: single-stream local inference -- `llama-cli` on a laptop, one user, one
sequence -- which is a large share of how llama.cpp is actually run, and the
exact case where nothing amortizes it.

### What batching amortizes, and what it does not

The same two level-3 traces give the whole phase mix, and the multipliers are
more informative than the shares. Sixteen sequences means 16x the tokens:

```
  phase          np=1 %   np=16 %   np=1 ms   np=16 ms   work x
  ffn             47.8%     58.4%      3.71      22.78     6.1x
  lm_head         29.0%     21.5%      2.25       8.40     3.7x
  barrier         13.6%      7.2%      1.05       2.79     2.7x
  attn.qkv         5.1%      6.1%      0.40       2.40     6.0x
  attn.score       1.1%      2.2%      0.09       0.87     9.8x
  norm             0.2%      0.3%      0.01       0.11     9.1x
```

Nothing costs 16x. The weight-bound phases amortize hard, because a weight
matrix is read **once per step** however many sequences ride along: `lm_head`
does 16x the arithmetic for **3.7x** the time, and the FFN stack for 6.1x. The
sequence-bound work does not amortize -- `attn.score` is 9.8x, because every
sequence has its own KV and its own scores to compute.

So batching does not just make decode faster, it **changes what decode is**.
`lm_head`, the phase F12 called a first-class cost at 29%, falls to 21.5%
purely because its weight read is now shared sixteen ways. Attention, 1.1% and
ignorable at batch size 1, doubles its share and is the only thing on the list
heading towards dominance as concurrency rises.

This is F7's byte model holding in a third regime, and it is the cleanest
statement of why the answer to "what should I optimize" depends on how the
model is being served, not only on the model.

### The bug this workload found, which is worth more than the finding

The first run at `-np 4` reported **zero decode tokens.** Everything was
classified as prefill.

`TS_TOKEN_SCOPE` drew the boundary as `batch_inp.n_tokens > 1`. A batched
generation step submits **one token per sequence**, so `n_tokens` is 4 or 16 and
every generation step looked like a prompt.

This is precisely the failure [F4](#f4--where-you-draw-the-boundary-changes-the-number-by-3) describes, and the same class as the
maintainer `FIXME` at `llama-context.cpp:714` that this project quoted
approvingly about llama.cpp's own heuristic. **tokenscope had shipped the same
bug it was built to criticize**, and no single-sequence workload could reveal
it — every trace in this repo before today was `-np 1`.

The first fix was also wrong. `n_tokens > n_seqs` handles batched decode, then
gets the *shared prompt* backwards: with `-kvu`, four prompt tokens each belong
to sixteen sequences, so `4 > 16` is false and the prompt was classified as
decode. Fixing one direction broke the other.

The correct quantity is neither. It is the **maximum number of tokens the batch
submits for any one sequence**:

- generation step, any `-np` → 1 per sequence → decode ✓
- prompt, single sequence → N → prefill ✓
- prompt shared across sequences → N per sequence → prefill ✓

Verified on all four: `-np 1/4/16` now each report 92 decode + 1 prefill, and
single-sequence `llama-bench pp64 tg16` still reports 16 decode + 1 prefill.

**The lesson is about coverage, not arithmetic.** The boundary was correct for
every workload that had ever been run, and wrong for the first new one. A
heuristic that has only met one case has not been tested; it has been agreed
with.

### Caveats

- One model, one machine, up to 16 sequences with a shared prompt and `-kvu`
  (unified KV). Separate per-sequence prompts, or non-unified KV, exercise
  `find_slot` differently and are untested.
- 96 tokens per sequence is short; a long multi-sequence run in a nearly-full
  shared cache combines both of F2's conditions and is still untested.
- `llama-batched` is an example program, not a server. Real serving adds arrival
  and eviction patterns that nothing here models.

---

---

## F18 — The shared-library build does not link, and MSVC says the obvious fix is illegal

**Configuration:** `-DBUILD_SHARED_LIBS=ON -DGGML_TOKENSCOPE=ON`, MSVC 19.44,
Visual Studio generator. Every other measurement in this repo is
`BUILD_SHARED_LIBS=OFF`, which the gap list has flagged since session 1.

It does not build:

```
ggml-cpu.c.obj : error LNK2001: unresolved external symbol ts_tls
ggml-cpu.dll   : fatal error LNK1120: 1 unresolved externals
```

With shared libraries, `ggml-cpu` is its own DLL. `ts_tls` — the thread-local
pointer to the current thread's record buffer, read on the hot path by every
node scope — is declared `extern __declspec(thread)` **without** `TS_API`, while
every plain global next to it (`ts_g_level`, `ts_g_token`, `ts_g_capture`) has
it. So `ggml-cpu` references a symbol nothing exports.

### The obvious fix is not available

Adding `TS_API` to the two thread-locals is a three-line change. MSVC rejects it
outright:

```
tokenscope.h(162): error C2492: 'ts_depth': data with thread storage
                   duration may not have dll interface
tokenscope.cpp(47): error C2492: 'ts_tls': ...
```

**A `__declspec(thread)` variable cannot be `dllexport`ed in MSVC.** Not
"should not" — the compiler refuses. So this is not an oversight that a missing
annotation explains; it is a genuine incompatibility between two decisions the
design made independently:

- [`docs/01`](01-design-scope-timing.md) chose a raw `__declspec(thread)`
  pointer with a constant initializer *specifically* to avoid MSVC's
  `__dyn_tls_on_demand_init` guard on every access ("The Windows TLS trap").
  That decision is correct and measured.
- The registry is exported across DLL boundaries so that `ggml-cpu` and `llama`
  share one instance.

The first requires the TLS variable to be raw. The second requires it to cross a
DLL boundary. On MSVC those cannot both hold for the same variable.

### The two ways out, neither implemented

1. **An exported accessor.** `TS_API ts_buffer * ts_get_tls(void)`, with the TLS
   variable private to the translation unit. Correct and simple, and puts a
   non-inlinable cross-DLL call on the hottest path in the project — the one
   `docs/01` went out of its way to keep to a single load. It would need
   re-measuring at level 3, where it is executed twice per node per thread.
2. **Per-DLL TLS, shared registry.** `ts_tls` is only a cache; the buffer it
   points at is owned by the registry. So each consumer could compile its own
   copy of the TLS variable and call the *exported* `ts_thread_init()` to obtain
   a registry-owned buffer. Flush walks the registry, so events recorded through
   either copy are found. This keeps the hot path exactly as it is and moves the
   cost to thread setup, which is the right place for it.

Option 2 looks right and is more invasive than anything that should be attempted
without a benchmark to check it against.

### Why this matters more than a build-flag footnote

The upstream draft ([`03`](03-upstream-issue-draft.md)) asks, as its second
question:

> is `ggml-base` the right home for the shared registry? It needs to be visible
> from both `ggml-cpu` and `llama`, and **I'd rather not force
> `BUILD_SHARED_LIBS=OFF` on anyone.**

That question was asked speculatively. It now has an answer, and the answer is
that the current design *does* force `BUILD_SHARED_LIBS=OFF`. llama.cpp ships
shared libraries, so this is a blocker for in-tree adoption rather than a
nice-to-have — and it should be in the issue as a known limitation with the two
options above, not discovered by a maintainer.

**No measurement in this repo is invalidated**; they were all static builds and
all say so. What changes is the scope of the claim: tokenscope currently works
in static builds, and the gap list's "shared-library build untested" was
understating it. It is not untested any more. It is broken, for a reason worth
writing down.

### Caveats

- MSVC only. The `__thread` path on GCC/Clang with
  `__attribute__((visibility("default")))` may well work, since ELF handles
  thread-local symbols across shared objects differently from PE/COFF. **Untested
  — this is another thing the Linux gap is hiding.**
- The Visual Studio generator defaults `--build` to Debug; the link failure
  reproduces under both Debug and `--config Release`.

---

## A caveat that applies to every barrier number here

All of them were measured on the **non-OpenMP** threading path (Windows/MSVC,
ggml's own threadpool), where `ggml_barrier` is an atomic spin-wait using
`ggml_thread_cpu_relax()`. On Linux `GGML_USE_OPENMP` is the default and
`ggml_barrier` becomes `#pragma omp barrier` (`ggml-cpu.c:577`) -- a different
implementation, and OpenMP runtimes generally spin then park rather than
spinning throughout.

The instrumentation itself is fine there: the OpenMP branch calls the same
`ggml_graph_compute_thread`, so every node and barrier scope is present.

So, splitting the claims by how far they should travel:

- **Should transfer** -- F9's structural results (which nodes run on one
  thread, and why), F12's byte model, F13's phase mix, F17's batching results.
  These are about ggml's row partitioning and about memory traffic, not about
  how threads wait.
- **May not transfer** -- every barrier *cost* figure: F6's 11.2%, F9's
  imbalance/release split, F10's rise to 22.9%, F14's halving under homogeneous
  cores, F15's null result on fusion. These measure one barrier implementation.

Untested either way. Stated here rather than repeated in eight caveat sections.

---

## P19 — Six predictions for an 8B model, recorded before the run

**Dated 2026-09-06, session 3.** No 8B measurement has been taken at the time
this is written; the only thing run so far is a 12-second smoke test
(`38.78 pp32 / 7.39 tg16` at 8 threads) to confirm the file loads. This section
exists so the predictions are in the commit history *before* the data, which is
the habit the last two sessions found most valuable — four predictions were
tested in session 2 and two of them were wrong, and without the written version
I would have remembered predicting whichever turned out right.

**The model.** Qwen3 8B Q4_K_M, the GGUF Ollama had already pulled onto this
machine. 8.19 B parameters, 36 layers, `n_embd` 4096, `n_ff` 12288, 32 query
heads against 8 KV heads, vocab 151,936, 4.86 GiB on disk. Against F12's
Qwen2.5-0.5B that is **13x the parameters** and **12.4x the bytes streamed per
token**, at a *lower* 5.15 bits/weight against 6.35.

Structural inputs, from `tools/model_bytes.py` (tensor table only, nothing run):

```
phase            n  bits/w  param share  byte share
---------------------------------------------------
ffn            108    4.84        71.8%       67.6%
attn.qkv       108    6.42        12.0%       14.9%
lm_head          1    6.56         8.2%       10.5%
attn.out        36    4.50         8.0%        7.0%
norm           145   32.00         0.0%        0.0%

streamed per token: 7.568 G params, 4.535 GiB, 5.15 bits/weight
token_embd (gathered, not streamed): 0.622 G params, 0.326 GiB
```

Untied embeddings: `output.weight` is a separate Q6_K tensor, so `lm_head`
really is streamed at decode rather than aliasing `token_embd`.

### P19.1 — `lm_head` collapses from 34% to about 10%

F12 measured `lm_head` at **34% of decode thread time** on the 0.5B and its
caveat said the share "shrinks fast as models grow, since vocabulary is fixed
while everything else scales". Vocabulary is in fact *identical* here — 151,936
both times — while everything else grew 13x, so this is as clean a test of that
sentence as the two models allow.

**Predict: 7.5%–13.5% of decode thread time**, centred on the 10.5% byte share,
a fall of roughly 3.2x. Falsified if it lands above 15% or below 6%.

### P19.2 — the byte law holds again, but this model tests it more weakly

**Predict byte-share error under 5 points, and bytes beating parameters.**

Stated with the caveat up front: on F12's model the two predictions disagreed by
up to 8.3 points, because one tensor sat at 8.50 bits/weight while the rest were
near 5.5. Here the spread is narrower and the largest disagreement is 4.2 points
on `ffn`. **A weaker test, and it should be reported as one** — if bytes win by
a point and a half, that is consistent with the law but is not strong evidence
for it, and P19.3 is the test that carries the weight.

### P19.3 — the controlled experiment: `attn_v` against `attn_k`

The sharpest test available, and one F12's model could not run. In all 36
layers, `attn_k` and `attn_v` have **identical shape** `[4096, 1024]`, 4.2 M
parameters each, the same matmul, the same output width, in the same phase and
the same layer. The only difference is dtype: **`attn_k` is Q4_K at 4.50
bits/weight, `attn_v` is F16 at 16.00** — a 3.56x byte ratio at identical
arithmetic. (An odd choice for a file labelled Q4_K_M, and itself worth noting:
"Q4_K_M" is a recipe, not a dtype, which was already F12's point.)

If per-phase time tracks bytes, `attn_v` should cost about **3.56x** `attn_k`.

**Predict the ratio lands in 2.0–4.5, and specifically below 3.56.** Below,
because the two effects pull apart here in a way the aggregate table hides:
bytes say V is 3.56x worse, but Q4_K must be *dequantized* and F16 need not be,
so compute works in V's favour while bandwidth works against it. That is the
first place in this project where the byte law and a compute effect make
opposite-signed predictions about the same pair of tensors.

Falsified as a byte law if the ratio comes out near 1.0 — that would say the
cost is the arithmetic, not the traffic, and would put F12's headline result in
question rather than confirming it.

### P19.4 — thread scaling gets worse, not better

F10 predicted and F12 confirmed that a quantized model scales further than an
F32 one: peak speedup rose 2.21x -> 3.28x, both peaking at six threads.

The naive extension says 8B is more compressed still (5.15 vs 6.35 bits/weight,
so more compute per byte) and should scale better again. **I predict the
opposite.** The thing that sets the wall is absolute bandwidth demand, and this
model streams 4.535 GiB per token against 0.365 — **12.4x more traffic** —
against a memory system that has not changed. The bits/weight effect is real but
second-order to that.

**Predict peak speedup below 3.28x, in the range 2.0x–3.0x, peaking at six
threads or fewer.** Falsified if it beats 3.28x, which would mean bits/weight
governs and absolute traffic does not.

This is the prediction I am least confident in, and it is the one where the two
effects are closest in size.

### P19.5 — GQA narrowing should mostly disappear

F12 found `Kcur` filling 2.5 of 6 threads and `Vcur` 3.0, against 4.0 for `Qcur`
and 6.0 for the FFN matmuls, and attributed it to width: Qwen2.5-0.5B has 2 KV
heads, so K and V produce **128-wide** outputs against 896 for Q. Too narrow to
partition across the pool.

Qwen3 8B has 8 KV heads at head_dim 128, so K and V are **1024 wide** — eight
times wider in absolute terms, even though the Q:KV head ratio only moves from
7:1 to 4:1. Width is what ggml partitions on, not the ratio.

**Predict K and V reach at least 5 of 6 threads busy**, close to Q. Falsified if
they stay near 2.5–3.0, which would mean the ratio governs and the absolute
width does not — and would make GQA narrowing a permanent cost rather than a
small-model artifact.

### P19.6 — the near-serial elementwise nodes shrink in relative cost

F9 found elementwise nodes (`ffn_swiglu`, `l_out`, `attn_norm`, ...) running on
one thread and causing imbalance out of all proportion to their work: 0.32% of
work causing 14% of imbalance on the synthetic F32 model, **0.74% causing 19%**
on the 0.5B. F12 read that as a trend and named the mechanism: "its relative
cost grows as the matmuls around it get cheaper."

That mechanism run backwards predicts a fall here, because the matmuls got much
more expensive — the elementwise nodes scale with `n_embd` (4.6x) while the
matmuls scale with `n_embd`-squared-ish (13x in parameters).

**Predict the elementwise share of total imbalance falls below 14%**, i.e. below
even the synthetic model. Falsified if it holds near 19% or rises.

### What would make this whole section uninteresting

If the machine pages. 4.86 GiB of weights against ~7.4 GB free is a thin margin,
and decode is bandwidth-bound (F14), so eviction would corrupt exactly the
numbers P19.1–P19.4 depend on while still producing a plausible-looking table.
Free memory gets checked before and after each run, and any run that pages is
thrown out rather than reported.
---

## F19 — The 8B run: five of six predictions hold, and the byte law gets a controlled experiment

**Workload:** Qwen3 8B Q4_K_M (8.19 B params, 36 layers, `n_embd` 4096, `n_ff`
12288, 32 query heads / 8 KV heads, vocab 151,936, 4.86 GiB). Throughput from
the **uninstrumented** build, `-r 3` for decode and `-r 2` for prefill.
Structural columns from a level-3 trace of 6 decode tokens at 6 threads,
6,742,824 bytes, **0 records dropped**. Free memory 8.30 GB before and 9.29 GB
after, against 4.86 GiB of weights: **the run did not page**, which
[P19](#p19--six-predictions-for-an-8b-model-recorded-before-the-run) named as
the thing that would invalidate it.

The six predictions in P19 were committed before any of this was measured
(`b4a140d`, and this finding is a later commit). **Five hold. P19.4 fails on
both of its specific claims while holding on its general one**, and the failure
is more interesting than the successes.

### Scorecard

| | prediction | measured | |
|---|---|---|---|
| P19.1 | `lm_head` 7.5–13.5% of decode | **10.5%** | ✅ dead on the byte share |
| P19.2 | byte error < 5 points, bytes beat params | **0.3 vs 4.5 points** | ✅ far stronger than predicted |
| P19.3 | `attn_v`/`attn_k` in 2.0–4.5, below 3.56 | **2.91** | ✅ both parts |
| P19.4 | peak speedup 2.0–3.0x, at ≤6 threads | **3.01x at 8 threads** | ❌ both specifics |
| P19.5 | K and V reach ≥5 of 6 threads | **5.1 and 6.0** | ✅ |
| P19.6 | elementwise imbalance share < 14% | **12%** | ✅ |

### P19.1 — `lm_head` fell from 34% to 10.5%, and vocabulary is why

F12 measured `lm_head` at 34% of decode thread time on Qwen2.5-0.5B and called
it "a first-class cost". Here the same tensor, against a **numerically identical
vocabulary of 151,936**, is 10.5%.

```
                        Qwen2.5-0.5B      Qwen3 8B
  vocab                    151,936        151,936     unchanged
  n_embd                       896          4096       4.6x
  lm_head params              136 M         622 M      4.6x
  everything else             494 M       7,568 M     15.3x
  lm_head time share          34.0%         10.5%     0.31x
```

The output projection did not get cheaper — it grew 4.6x. It shrank *as a share*
because the rest of the model grew 15.3x around it. F12's phrasing, that the
share "shrinks fast as models grow, since vocabulary is fixed while everything
else scales", is exactly right and the mechanism is exactly the stated one.

The practical reading of F12 needs the qualifier attached, though. Its estimate
that requantizing this one tensor could remove "about 12% of decode time" was a
0.5B result; the same arithmetic here gives **about 3%**, and on a 70B it would
be under 1%. **`lm_head` is a small-model problem.**

### P19.2 — the byte law, and a test I called weak that turned out decisive

```
phase          bits/w  param share  byte share  time share  err(param)  err(byte)
---------------------------------------------------------------------------------
ffn              4.84        71.8%       67.6%       67.3%        -4.5       -0.3
attn.qkv         6.42        12.0%       14.9%       15.1%        +3.1       +0.1
lm_head          6.56         8.2%       10.5%       10.5%        +2.3       +0.0
attn.out         4.50         8.0%        7.0%        7.1%        -0.8       +0.2

max error predicting time from parameters: 4.5 points
max error predicting time from bytes:      0.3 points
```

**0.3 points.** F12's byte prediction was good to 2.9 points and that was already
the headline; this is an order of magnitude tighter, on a model 13x larger.

P19.2 predicted this would be a *weaker* test than F12's, because the two rival
predictions disagree by at most 4.2 points here against 8.3 there. That
reasoning was sound and the conclusion was still too pessimistic: a narrower gap
between the hypotheses does not make the winner's residual larger, and the
residual is what carries the information. Worth remembering — the strength of a
test is not only the separation between hypotheses.

Time shares are of the four weight-bearing phases, renormalised from 99.62%;
the missing 0.38% is `attn.score`, `norm`, `residual` and `attn.kv_rw`, which
stream no weights and so appear in no byte column.

### The controlled experiment: same node, same shape, adjacent layers, different dtype

Everything above is still a correlation across phases that differ in many ways
at once. This file allows something better.

llama.cpp's Q4_K_M recipe stores `ffn_down` at **Q6_K in 18 layers and Q4_K in
the other 18** — layers 0-3, then every third, then 30-35. Same tensor, same
shape `[12288, 4096]`, same op, same graph position, same token, same thread
pool. **The dtype is the only difference**, and `ffn_gate` and `ffn_up` are Q4_K
in every layer and serve as controls.

```
                Q6_K layers    Q4_K layers    ratio
  ffn_out        38516.6 us      27138.5 us   1.419     <- the test
  ffn_gate       26735.9 us      26485.5 us   1.009     <- control
  ffn_up         26681.6 us      26506.4 us   1.007     <- control

  predicted from bytes alone: 6.5625 / 4.5 = 1.458
```

**1.419 measured against 1.458 predicted, a 2.7% error, with both controls flat
at 1.00.** The controls are what make this an experiment rather than an
observation: if the Q6_K layers were slower for any reason other than the dtype
of that one tensor — placement, scheduling, cache, position in the graph — the
gate and up projections in those same layers would show it too. They do not.

This is the strongest form of F7's law the project has produced, and it upgrades
it from a cross-phase correlation to a paired within-model result.

### P19.3 — `attn_v` against `attn_k`, where bytes and compute disagree

Same shape `[4096, 1024]`, 4.2 M parameters each, all 36 layers, `MUL_MAT` only
(Qwen3's QK-norm shares the `Kcur` node name, so the op filter matters — without
it `Kcur` carries 1,296 extra RMS-norm events).

```
  Kcur MUL_MAT    91.1 ms      Q4_K, 4.50 bits/weight
  Vcur MUL_MAT   265.1 ms      F16,  16.00 bits/weight
  Qcur MUL_MAT   328.7 ms      Q4_K, 4.50 bits/weight, 4x the rows

  V/K = 2.911   against a byte ratio of 3.556
  Q/K = 3.608   against a byte ratio of 4.000
```

P19.3 predicted 2.0–4.5 **and specifically below 3.56**, on the argument that
Q4_K must be dequantized where F16 need not be, so compute pulls the other way
from bandwidth. 2.911 is below, and the direction of the miss is the predicted
one.

The `Q/K` row is the check that keeps this honest: Q and K are the *same dtype*
and differ only in size, and they come in at 3.608 against a byte ratio of 4.000
— a 10% shortfall from fixed per-node overhead that has nothing to do with
dtype. So of V/K's 18% shortfall from its byte ratio, roughly half is the same
size effect visible in Q/K, and only the remainder is attributable to
dequantization. **The dequantization effect is real but smaller than the raw
number suggests**, and P19.3's reasoning was right for a reason that accounts
for about half of what it predicted.

### P19.4 — wrong twice, and the interesting part is where

```
              Qwen2.5-0.5B Q4_K_M          Qwen3 8B Q4_K_M
 thr    tok/s  speedup  par.eff       tok/s  speedup  par.eff
   1    27.06    1.00x     100%        2.56    1.00x     100%
   2    52.85    1.95x      98%        4.85    1.89x      95%
   4    77.49    2.86x      72%        6.82    2.66x      67%
   6    88.88    3.28x      55%        7.50    2.93x      49%
   8    88.20    3.26x      41%        7.70    3.01x      38%
  12    70.80    2.62x      22%        7.49    2.93x      24%
  28    66.07    2.44x       9%        7.55    2.95x      11%
```

P19.4 got the general claim right — peak speedup fell from 3.28x to 3.01x, so
the 12.4x rise in bytes streamed per token beats the drop from 6.35 to 5.15
bits/weight, and **absolute traffic sets the wall, not compression ratio**.

Both specifics failed. The peak is 3.01x, marginally outside the stated 2.0–3.0x
range; that is a near miss but the range was stated and 3.01 is outside it. And
the peak moved **out** to 8 threads, not in to six or fewer, which was the
confident half of the prediction.

The reason is visible in the last two rows and it is not a detail. **The 0.5B
collapses past its peak and the 8B does not.** The small model falls 3.26x ->
2.44x from 8 to 28 threads, a 25% loss; the 8B goes 3.01x -> 2.95x, a 2% loss.
F14 established the mechanism for that collapse — heterogeneous cores, P-cores
2.88x faster than E-cores, every barrier waiting on the slowest. That mechanism
needs the workload to be compute-bound enough for core speed to matter. At 8B
decode every thread is waiting on memory, the E-cores are no slower at waiting,
and **the heterogeneity penalty disappears into the bandwidth wall.**

So a prediction reasoned from one mechanism (traffic) got the direction right
and the shape wrong, because a second mechanism (heterogeneity) stopped applying
at the same time. Both moved together and I only modelled one.

### Prefill is the control, and it scales

The same sweep on `pp256`, which is compute-bound rather than bandwidth-bound:

```
 thr    tok/s   speedup   par.eff
   1     7.08     1.00x      100%
   2    14.52     2.05x      103%
   4    28.56     4.03x      101%
   6    42.28     5.97x       99%
   8    52.23     7.38x       92%
  12    53.51     7.56x       63%
  28    78.40    11.07x       40%
```

**Near-perfect scaling to six threads and still climbing at 28**, against decode
on the identical model, weights and machine plateauing at 3.01x. This is the
cleanest statement of "decode is bandwidth-bound" the project has: not an
inference from a null result as in F14, but the same model doing the same
arithmetic on the same cores, differing only in how many tokens share each
weight read.

One anomaly, recorded and not explained: the 8 -> 12 thread step is nearly flat
(52.23 -> 53.51) before jumping to 78.40 at 28. That does not fit a smooth
curve. 12 threads on an 8 P-core / 12 E-core machine is where the scheduler must
start mixing core types, so F14's mechanism is the obvious suspect, but this run
does not test it and the affinity sweep that would has not been done at 8B.

### P19.5 — GQA narrowing was a width problem, and the width grew

F12 found `Kcur` filling 2.5 of 6 threads and `Vcur` 3.0 on the 0.5B, where K
and V outputs are 128 wide. Qwen3 8B's are 1024 wide.

```
  node        threads busy (of 6)     0.5B, for comparison
  Qcur                6.0                     4.0
  Vcur                6.0                     3.0
  Kcur                5.1                     2.5
  ffn_gate/up/out     6.0                     6.0
```

Confirmed. The Q:KV head ratio only improved from 7:1 to 4:1, but the absolute
width went up 8x, and **ggml partitions rows, so absolute width is what
matters**. GQA narrowing is a small-model artifact, not a permanent cost of
grouped-query attention — which is the opposite of what a reader of F12 alone
might reasonably conclude.

`Kcur` at 5.1 rather than 6.0 uses `--barriers`' stricter definition (a thread
counts as busy if it takes more than 25% of the busiest thread's time). By the
looser "did this thread touch the node at all" count it is 6.0 like the rest.

### P19.6 — elementwise imbalance fell to 12%

```
                          work share   of all imbalance
  synthetic F32 220M          0.32%          14%
  Qwen2.5-0.5B Q4_K_M         0.74%          19%
  Qwen3 8B Q4_K_M             0.18%          12%
```

54 node types cost more in other threads' waiting than in their own work — 7.58
ms of compute causing 29.19 ms of waiting. But the share fell below even the
synthetic model's, as predicted, and F12's stated mechanism run backwards is
why: the elementwise nodes scale with `n_embd` while the matmuls around them
scale far faster, so the same serial nodes are a smaller fraction of a much
larger total.

`ffn_swiglu`, `attn_norm`, `l_out`, `ffn_norm` and `ffn_inp` are still the worst
offenders and still run on 1.0–1.7 threads, exactly as F9 described. The
structure did not change; its weight did.

**Barrier wait is also down**: 5.6% of worker thread time here, against 11.2% in
F6 and 22.9% at high thread counts in F10. Bigger matmuls amortise the same
barriers over more work. This is consistent with F15's conclusion that barrier
*count* and barrier *cost* are different quantities.

### Caveats

- One model, one quantization, one machine, MSVC Release on Windows 11,
  i7-14700HX. The barrier and thread numbers are the non-OpenMP path; see the
  Linux section.
- Structural columns are a single trace of 6 decode tokens at 6 threads.
  Throughput is `-r 3` decode / `-r 2` prefill from the uninstrumented build.
- The `ffn_down` experiment is the one result here that does not depend on
  cross-phase attribution at all, and is the one to quote if only one survives.
- `attn.out` in the byte table is measured as tokenscope's `~attn` bucket. That
  it *is* the output projection is established in F20, not assumed here.
- "Q4_K_M" remains a recipe rather than a dtype. This file's `attn_v` is F16 in
  all 36 layers, which is not what most Q4_K_M exports do, and is the reason
  P19.3 was testable at all.

---

## F20 — The attention output projection is anonymous in every llama.cpp graph, and it is 7% of decode

Found while checking which trace bucket the 8B's output projection landed in.
This one is a defect in upstream llama.cpp rather than a property of a workload.

### What the trace showed

Tokenscope reported a category `~attn` at **7.1% of decode thread time** — the
tilde meaning the phase was inferred from graph position because the node had no
usable name. The nodes were `node_27`, `node_62`, `node_97`, ... spaced exactly
35 apart, one per layer, all `MUL_MAT`, all on 6 threads.

The identification is arithmetic rather than a guess:

```
  ~attn total (36 nodes, one per layer)   328.63 ms
  Qcur MUL_MAT                            328.67 ms
```

`attn_output` is `[4096, 4096]` Q4_K and `attn_q` is `[4096, 4096]` Q4_K — same
shape, same dtype, same op. **They agree to 0.01%.** Combined with one instance
per layer at a fixed graph offset, the anonymous nodes are the output
projections.

### Why they have no name

`build_attn` in `llama-graph.cpp` has **seven overloads**, one per attention
input type, and every attention architecture routes through one of them. Not one
of them names the output projection. There are two distinct reasons, and the
first draft of this finding got them the wrong way round.

**Three overloads** — the no-cache, the ISWA-K and the cross-attention ones —
carry this:

```c
    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        //cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }
```

The naming call is commented out, **and** it is in a block guarded by the
attention output *bias* rather than by the weight `wo` whose matmul it would
name. Uncommenting it would still name nothing on any model without that bias.
The leftover `if (wo_b) { }` is dead either way.

**The other four**, including `build_attn(llm_graph_input_attn_kv *)` — the one
nearly every decoder-only model uses, Qwen3 included — have **no naming call at
all**, not even a commented one.

So the node this finding measured is anonymous for the simpler of the two
reasons. The mis-guarded block is real and worth fixing, but it is not the cause
on the path that was profiled, and saying so was wrong.

### The fix, applied and measured

`patches/02-name-attn-output.patch`: move the call inside the `if (wo)` block
that performs the matmul, in all seven overloads, and delete the three dead
blocks. **8 lines added, 13 removed.**

```c
    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
        cb(cur, "kqv_wo", il);
    }
```

Rebuilt and re-traced the same workload. `cb` only assigns a name, so nothing
about the computation changes:

```
                    before          after
  attn.out        (absent)       321.06 ms   6.7%
  ~attn           328.63 ms       40.1 us    0.0%
```

**The whole bucket moved.** That converts the identification above from an
inference about shapes into a demonstration: naming the node relocated exactly
the time in question out of the graph-position fallback and into `attn.out`, and
left 40 microseconds of genuinely unnamed nodes behind.

The residual 2.3% between 328.63 and 321.06 ms is run-to-run variation — these
are single 6-token traces, and the two runs measured 6.98 and 7.48 tok/s.

**This also needed a fix on tokenscope's side**, which is how the dead entry in
its own category table came to light: `"kqv_wo"` would have been classified as
`attn.score`, because `{ "kq", ... }` sat above `{ "kqv_out", ... }` in a
first-match-wins table and shadowed it. Both are fixed, and the self-test now
fails if any entry is ever shadowed again.

### Why it is worth reporting

This is the largest single node in the attention block — bigger than `Qcur`,
6.6x `Kcur` — and it is invisible to anything that groups by node name. That
includes tokenscope without its graph-position fallback, and it includes
`GGML_SCHED_DEBUG` and any profiling built on ggml's own names.

F8 recorded that about 15% of graph nodes carry no meaningful name and treated
it as a general property of ggml. It is partly that, but the single most
expensive anonymous node is anonymous because of **an omission in seven
overloads of one function**, it is worth 7% of decode time, and it affects every
architecture that has an attention output projection. F8's framing was too
resigned.

This is now the best-evidenced item for the upstream conversation
([`03`](03-upstream-issue-draft.md)): a one-line-per-overload change that
deletes more than it adds, a named node worth 7% of decode, no behaviour change
and no measurable cost — `cb` assigns a name at graph-build time, which happens
about once per run thanks to graph reuse (F1).

Per llama.cpp's `AGENTS.md`, this goes to an **issue first**, not a PR.

### Caveats

- ~~The identification is inference from shape, dtype, count and spacing.~~
  **No longer inference**: the patch was applied and the bucket moved from
  `~attn` to `attn.out` wholesale. The shape argument is retained above because
  it is what pointed at the answer before the patch existed.
- Measured on Qwen3 8B. The 7.1% share is model-dependent; the naming defect is
  not, since `build_attn` is shared by every attention architecture.
- Not yet tested against a model that *does* have `wo_b` (an attention output
  bias), which is the case where the existing mis-guarded block would have
  fired. Qwen3 has none.
- The patch is applied locally and measured; it has **not** been sent anywhere.
  It touches seven overloads and the three dead blocks, and a maintainer may
  reasonably want only the one-line-in-`if (wo)` part.

---

## F21 — The byte law across three architectures, and the vocabulary story it explains

**Workload:** two more real models that were already on the machine, both
`arch=llama`, both traced identically to F19 — level 3, 6 decode tokens, 6
threads, 0 dropped. `dolphin-2.9-llama3-8b` (8.03 B params, 4.06 GiB, vocab
128,256) and `dolphin-2.9.3-mistral-7b` (7.24 B, 3.76 GiB, vocab **32,000**).

F19 established the byte law on one model, with a controlled experiment inside
it. The open question was whether it is a property of memory traffic in general
or of Qwen3 in particular.

```
dolphin-llama3-8B            bits/w  param%   byte%   time%   err(param)  err(byte)
  ffn                          4.50   75.1    72.8    72.3       -2.8       -0.5
  attn.qkv                     4.50   10.7    10.4    10.6       -0.1       +0.2
  lm_head                      6.56    7.0     9.9    10.2       +3.2       +0.3
  attn.out                     4.50    7.2     6.9     6.8       -0.3       -0.1
                                              max:              3.2        0.5

dolphin-mistral-7B           bits/w  param%   byte%   time%   err(param)  err(byte)
  ffn                          4.50   79.3    78.6    78.2       -1.1       -0.4
  attn.qkv                     4.50   11.3    11.2    11.6       +0.3       +0.4
  attn.out                     4.50    7.6     7.5     7.5       -0.1       -0.0
  lm_head                      6.56    1.8     2.7     2.8       +0.9       +0.1
                                              max:              1.1        0.4
```

**Byte error 0.5 and 0.4 points**, against F19's 0.3 on Qwen3. Three
architectures, three quantization mixes, same answer.

**The mistral row is a weak test and should be read as one.** Its file is
uniform Q4_K apart from `lm_head`, so the two rival predictions only disagree by
0.9 points anywhere. It confirms nothing that F19 did not already establish; it
is included because excluding a weak-but-consistent result and keeping the
strong ones is how a law stops being falsifiable.

The llama3 row is the real test of the pair: `lm_head` at Q6_K against a
uniformly Q4_K body puts 2.9 points between the hypotheses, and the byte form
takes it by 3.2 to 0.3.

### What four models say about `lm_head`, which is the practical result

F12 called the output projection "a first-class cost" at 34%. F19 found 10.5%
and blamed vocabulary. These two models vary vocabulary independently of size,
which neither Qwen model could:

```
model                    vocab    n_embd   lm_head byte%   lm_head time%
Qwen2.5-0.5B Q4_K_M    151,936       896          36.9            34.0
Qwen3-8B     Q4_K_M    151,936      4096          10.5            10.5
llama3-8B    Q4_K      128,256      4096           9.9            10.2
mistral-7B   Q4_K       32,000      4096           2.7             2.8
```

**A 13.7x range in `lm_head`'s share, tracked to within 0.3 points on three of
the four**, and 2.9 on the fourth. Two 8B models at the same `n_embd` differ
only by vocabulary and land 10.5 against 10.2; drop vocabulary to 32,000 at the
same size and it collapses to 2.8.

So the rule is `vocab x n_embd x bits`, against the rest of the model, and
nothing else needs to be known:

- **Qwen2.5-0.5B's 34% was a small-model artifact**, as F19 said, but the
  vocabulary half of the explanation is now tested directly rather than inferred
  from two models that shared a vocabulary.
- **On a 32k-vocabulary model the output projection is not worth instrumenting.**
  2.8% is inside the noise of most of what this project measures.
- The practical consequence for anyone reading F12: whether `lm_head` matters is
  decided before you run anything, by two integers in the config and one dtype
  in the tensor table.

### Caveats

- Both models are `arch=llama`, so this is two quantization recipes and two
  vocabularies more than F19, but only one additional *architecture family*. A
  MoE model would be the interesting next test and none is available here.
- Single 6-token level-3 traces at 6 threads, as in F19. No throughput sweeps
  were run on either model, so nothing here speaks to F19's scaling result.
- Traces not committed. The two files are ~5 MB each and the reference set is
  already 13 MB; they add nothing CI does not already check on the Qwen3 pair.
  Reproduce with the command in F19 against the blob paths in HANDOFF section 3.
- The `~post-attn` and `other` buckets are together under 0.05% on both models,
  so the F20 naming fix is doing its job on `arch=llama` too — the output
  projection is named there as well, not just on Qwen3.
---

## Not yet measured

Listed so the gaps are explicit rather than implied:

- larger real models — the biggest measured is now 8.19 B (F19, F21);
  nothing above that, and **no MoE model at all**, which is the most obvious
  gap in the byte law's coverage
- server workloads with real arrival and eviction patterns (F17 covers
  `llama-batched` only, up to 16 sequences)
