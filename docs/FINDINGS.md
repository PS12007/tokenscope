# Findings

Things the traces showed that were not obvious before taking them. Each entry
states the workload, because a finding without a workload is an opinion.

Numbers here come from real traces produced by the committed code. Where a
finding is provisional, it says so.

> ### On the word "certified" — read this before quoting a percentage
>
> It appears throughout F10–F31 and it **overpromised**. It meant only that a
> run's bootstrap interval excluded zero while its baseline IQR cleared 2% —
> that is **resolved within one invocation**, not reproducible. F31 put two
> "certified" intervals for *one quantity on the same unrebuilt binaries* side
> by side and they **did not overlap**; F36's block interval later contained
> both, so the runs never disagreed, only their intervals did.
>
> The uses below are left in place deliberately. They are what the sessions
> believed at the time, and editing them out would hide the thing worth
> learning. **Treat every "certified" here as "resolved within that run", and
> check [`04-project-audit.md`](04-project-audit.md)'s trust ladder before
> quoting the number.** The tools no longer print the word.
>
> Superseded by measurement, not just by wording: **F24 → F33**, and
> **F25 / F30 / F31's percentages → F36**.

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
- ~~The mask-to-core mapping is assumed from the usual Windows enumeration~~
  **Tested in session 6 ([F37](#f37--three-hypotheses-for-m10-all-refuted-and-a-correction-to-a-claim-made-an-hour-earlier)).**
  Prefill at two threads gives `0x0101` (logical 0,8) **1.87x** the throughput of
  `0x0003` (logical 0,1), so logical 0 and 1 share a physical core and the
  numbering is interleaved: core N = logical 2N, 2N+1. **`0x5555` is one thread
  per P-core, as assumed here.** So the first candidate above — wrong logical
  numbering — is **eliminated**, and `--cpu-strict` bit assignment is the
  surviving hypothesis for the anomaly.
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

**Rewritten in session 5. The first version of this section had it exactly
backwards, and [`F26`](#f26--every-measurement-in-this-project-was-taken-on-the-openmp-path-and-five-documents-said-the-opposite)
is why.**

All of them were measured on the **OpenMP** threading path.
`GGML_OPENMP` defaults to ON with no platform condition
(`ggml/CMakeLists.txt:247`), CMake finds MSVC's `vcomp` here, and every build
this project has measured compiles `ggml-cpu.c` with `-DGGML_USE_OPENMP` --
so `ggml_barrier` is `#pragma omp barrier` (`ggml-cpu.c:583`) and
`llama-bench.exe` imports `_vcomp_barrier` and `_vcomp_fork` to prove it.
The atomic spin-wait built from `n_barrier` / `n_barrier_passed` and
`ggml_thread_cpu_relax()` -- which this section used to claim was the thing
being measured -- **has never been measured by this project at all**.

`GGML_USE_OPENMP` is also the default on Linux, so the barrier *mechanism* is
the same there. What differs is the OpenMP *runtime*: `vcomp` 2.0 here against
`libgomp` or `libomp` there, whose spin-then-park policies are tunable
(`GOMP_SPINCOUNT`, `KMP_BLOCKTIME`) in a way `vcomp` does not document.

The instrumentation is not affected either way: both branches call the same
`ggml_graph_compute_thread`, so every node and barrier scope is present.

So, splitting the claims by how far they should travel:

- **Should transfer** -- F9's structural results (which nodes run on one
  thread, and why), F12's byte model, F13's phase mix, F17's batching results.
  These are about ggml's row partitioning and about memory traffic, not about
  how threads wait.
- **May not transfer** -- every barrier *cost* figure: F6's 11.2%, F9's
  imbalance/release split, F10's rise to 22.9%, F14's halving under homogeneous
  cores, F15's null result on fusion. These measure one barrier implementation.
  That is still true, but it is now a **narrower** claim than it was: the
  implementation is an OpenMP one on both platforms, so what is untested is
  `libgomp`/`libomp` against `vcomp`, not a spin-wait against a barrier
  pragma.

**And session 5 put a size on "may not transfer", by building the other one.**
[`F27`](#f27--ggmls-own-barrier-is-not-a-cheaper-spin-wait-than-openmps-on-this-machine-at-28-threads-it-costs-54-of-decode)
measured `GGML_OPENMP=OFF` on this machine: release latency per unit work is
**1.95x higher at 8 threads and 4.9x higher at 16**, both with non-overlapping
ranges, and total barrier wait per unit work is 0.752 against 0.249 at 28
threads. Decode throughput falls 2.06% at 8 threads and **54.17% at 28**.

So the numbers above are from the **cheaper** of the two barriers this machine
can build, which is the opposite of what a reader would assume from a caveat.
How much of that carries to `libgomp` against ggml's threadpool on Linux is
still untested -- but "one barrier implementation" is no longer a shrug, it is
a factor of three to five.

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
  i7-14700HX. The barrier and thread numbers are the OpenMP path (`vcomp`);
  see the Linux section, and F26 for why this line used to say the opposite.
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

## F22 — The shared-library build works, because the thread-local was never the thing that had to be shared

**Session 4, 2026-09-07.** [F18](#f18--the-shared-library-build-does-not-link-and-msvc-says-the-obvious-fix-is-illegal)
left `BUILD_SHARED_LIBS=ON` broken with two candidate fixes and neither
implemented. Option 2 is now implemented and measured. The configuration builds,
links, runs, and produces a trace containing records from two different DLLs.

Reproduced first, so that "it links now" means something:

```
ggml-cpu.c.obj : error LNK2019: unresolved external symbol ts_tls
                 referenced in function ggml_graph_compute_thread
ggml-cpu.dll   : fatal error LNK1120: 1 unresolved externals
```

### The reframing that makes it easy

F18 stated the conflict as a genuine incompatibility, and it is one *for a
single variable*: `docs/01` needs `ts_tls` to be a raw `__declspec(thread)`
pointer with a constant initializer, so that reading it does not go through
MSVC's `__dyn_tls_on_demand_init` guard; the registry needs one instance visible
to `ggml-cpu` and to `llama`. MSVC will not export thread-storage data, so those
cannot both hold.

They do not have to. **`ts_tls` is a cache, not state.** The state is the
`ts_buffer`, which the registry owns and which `ts_flush` reaches by walking the
registry rather than by walking any thread-local. The only two values `ts_tls`
can ever hold are null and the address of this thread's one buffer. So the
number of copies of `ts_tls` in the process does not matter at all — what
matters is that they all resolve to the same buffer.

So each module compiles its own `ts_tls` (a `static` in the header) and fills it
by calling the exported `ts_thread_init()`. Two modules, two pointers, one
buffer. The hot path is byte-for-byte the design `docs/01` argued for: one load
of a plain thread-local, no guard, no cross-DLL call. Option 1's exported
accessor would have put a non-inlinable cross-DLL call on the hottest path in
the project, executed twice per node per thread at level 3, which is why F18
preferred option 2 and why that preference was right.

The cost moves to thread setup, and amounts to one extra cold call per
translation unit per thread.

### Three consequences that had to be handled, two of them silent

The change is small. What it makes load-bearing is not.

**1. `ts_thread_init()` has to become idempotent per thread.** Every module
calls it once, and every module has to be handed the *same* buffer. Without
that, N modules produce N `thread_state`s per thread — N sets of records under N
trace thread ids, and N times the memory budget consumed. A thread-local in
`tokenscope.cpp` holds the owning state; that TU is compiled into exactly one
module, so unlike the header's cache it genuinely is one slot per thread per
process.

**2. Every caller must write the result back.** `ts_thread_init()` can only
assign to its own module's copy of the cache. A caller that ignored the return
value would call it again on every event — correct, but a cross-DLL call on the
hot path, i.e. option 1 by accident.

**3. `ts_acc` would have silently recorded nothing at level 2.** This is the one
worth writing down. It used to bail on a null `ts_tls`:

```c
struct ts_buffer * b = ts_tls;
if (b == 0 || node_n >= b->acc_n) return;
```

That was correct when `ts_tls` was one symbol, because `ts_thread_prepare()` set
it before the node loop began. With a per-module cache, `ts_thread_prepare()`
runs in `ggml-base` and fills *ggml-base's* copy; `ts_acc` runs in `ggml-cpu`
and reads a copy that is still null on the first node. Level 2 would have
produced a trace, with host scopes, zero drops, no warning, and no node data —
failing in exactly the shape this project has twice decided is the worst
available failure. Caught by reasoning about the initialization order rather
than by a test, and then tested.

**And `ts_depth` moved into `ts_buffer`.** It hit the same C2492 and had to
leave the header either way. Per-module would have made it report each module's
nesting rather than the thread's — host scopes already nest across *translation
units* today (`kv.find-slot` inside `kv.slot-search`, `graph-compute` inside
`decode`) and would nest across modules the moment ggml grows a host scope. The
buffer is the thing that is genuinely per-thread and genuinely shared, so the
counter belongs in it.

### What was verified

| Check | Result |
|---|---|
| `ggml-cpu.dll` links, `BUILD_SHARED_LIBS=ON` | yes, was `LNK1120` |
| Full `llama-bench.exe` links against 5 DLLs | yes |
| Level-3 trace from the shared build | 442 KB, 17 tokens, **0 dropped** |
| Host scopes from `llama.dll` present | `kv.slot-search`, `batch-init`, `ubatch`, … |
| Node scopes from `ggml-cpu.dll` present | `ffn` 71.1%, `barrier` 9.0%, `attn.qkv` 10.4% |
| Worker threads in the trace | **4, not 8** — see below |
| Level 2 in the shared build | host scopes, 100.0% attributed, 0 dropped |
| Recorded scope depths vs pre-change static build | **identical**, all 13 scopes |
| Static build event count / thread count | identical (205,143 events, 161 threads) |
| Self-test | passes, 52.6 ns/scope against a documented 52.8 |
| Disabled build | still 0 symbols, still an 864-byte archive |
| Barrier decomposition on a fresh level-3 trace | every barrier matched |
| Two-module test, standalone shared build | passes, and fails when the fix is reverted — see below |
| A module opened at runtime, mid-run | joins the calling thread's existing buffer (the GGML_BACKEND_DL shape) |

**The thread count is the real evidence.** `llama-bench -t 4` with the fix
produces a trace with four worker threads. Had the two DLLs each built their own
`thread_state` per thread — the failure mode consequence 1 exists to prevent —
it would say eight, with the work split across pairs of ids and every per-thread
percentage in the analyzer computed on half a thread. It says four.

The phase mix from the shared build (`ffn` 71.1%, `attn.qkv` 10.4%,
`attn.out` 6.0%) matches the static build on the same model, which is the
second check: the records are not merely present, they are attributed to the
same places.

### The regression test, and breaking it on purpose

F18 existed because nothing ever built a shared library. Fixing that without
adding a build of one would leave the next regression to be found the same way —
by hand, a session late. So `ts_dllmod.cpp` and `ts_dlltest.cpp` build two
separate binaries that each include `tokenscope.h`, and CI now runs them on all
three platforms. No llama.cpp, no model, no network; the whole thing takes 0.04
seconds.

The test asserts the invariant directly rather than through its symptoms:

```
[ ok ] each module has its own ts_tls cache (the control: without this the rest is vacuous)
[ ok ] both modules resolve to the same buffer on each thread
[ ok ] both modules share one host-scope depth counter
[ ok ] different threads still get different buffers
```

The first line is a control, and it is the reason the other three mean anything.
If the two "modules" ever collapsed into one binary — a build-system change, a
static link, an inlining decision — every remaining assertion would still pass
while checking that a variable equals itself. The fourth is the opposite guard:
sharing *everything* would satisfy lines two and three and reintroduce the
cross-thread contention the whole design exists to avoid.

A second phase covers the case the design is most likely to get wrong: a module
that arrives **after** threads already exist and have already recorded. That is
the `GGML_BACKEND_DL` shape, where backends are `dlopen`ed rather than linked,
and the newcomer's cache starts null at a point where the registry already has
an answer — so it must join the existing buffer rather than allocate a second.

```
[ ok ] a module loaded mid-run joins this thread's existing buffer
[ ok ] a thread created after the load also sees one buffer
[ ok ] ...and it is that thread's own, not the main thread's
```

It works, and it works for a reason worth stating: `ts_thread_init()` consults
the registry, not the caller, so "which module asked" and "when it asked" are
both irrelevant to the answer. The third line is there because the first two
would both pass if every thread shared one buffer, which would be a different
and worse bug.

**Then the check was tested by breaking what it checks**, as F20's shadow test
was. Disabling `ts_thread_init`'s idempotency — one `if` — and rebuilding:

```
[ ok ] each module has its own ts_tls cache
[FAIL] both modules resolve to the same buffer on each thread
[FAIL] both modules share one host-scope depth counter
[FAIL] a module loaded mid-run joins this thread's existing buffer
[FAIL] a thread created after the load also sees one buffer
[ ok ] no records were dropped
       12 distinct trace thread ids, expected 6
[FAIL] one trace thread id per thread, not one per thread per module
```

Note what did **not** fail. The trace was written. It was valid JSON. Zero
records were dropped. Every scope name from both modules was present, correctly
nested. `llama-bench` would have printed a plausible table. The only visible
symptom was the thread count — exactly doubled, 12 where 6 were used — and it
had to be taken deliberately to be seen.

The count is asserted as **equality against a number the test knows exactly**,
not as an upper bound. The first version of it used a bound, which passed a
run it should have failed as soon as the test grew a sixth thread; a bound loose
enough to survive edits to the test is loose enough to miss a doubling.

That is the third time this project has found a defect whose entire symptom is a
number being quietly wrong (F5's over-100% attribution, F20's dead table entry,
this). The pattern is consistent enough to state as a rule: **for a profiler,
the default failure mode is not a crash, it is a plausible answer.** Checks have
to assert quantities, not the absence of errors.

### Caveats

- **The mechanism is verified on GCC and Clang; llama.cpp's shared build on
  Linux is not.** These are different claims and the difference matters. The
  two-module test passes on ubuntu-latest (GCC 13.3.0) and macos-latest, so
  "each shared object gets its own `ts_tls`, and they all resolve to one
  registry-owned buffer" is measured on all three toolchains rather than
  reasoned about — which is more than the first draft of this section claimed,
  and it was written before CI had run. What remains untested is llama.cpp
  itself with `BUILD_SHARED_LIBS=ON` on Linux, which is a bigger configuration
  than the test: real `.so` boundaries between `ggml-base`, `ggml-cpu` and
  `llama`, and the OpenMP threading path rather than ggml's own pool. This
  narrows the Linux gap. It does not close it.
- **No overhead re-measurement.** The hot path is unchanged by inspection — same
  instruction sequence, one load of a thread-local — but the shared build has
  never had its overhead measured at all, and the harness has declined to
  certify the static number twice already (docs/02). Unmeasured, and quoting the
  static figure for a shared build would not be justified.
- One extra cold `ts_thread_init()` call per translation unit per thread, and a
  pointer-sized TLS slot per TU. Neither is on a hot path.
- The scheme is robust to COMDAT folding, which is the obvious objection: an
  inline function touching an internal-linkage variable may leave several TUs
  sharing one slot. That is fine here for the same reason the whole approach
  works — the slot is a cache whose only possible values are null or this
  thread's buffer, so folding changes the number of cold init calls and nothing
  observable.
- `GGML_BACKEND_DL` — llama.cpp loading backends with `dlopen`/`LoadLibrary`
  rather than linking them — is **covered as a shape but not as a build.** The
  test opens a third module at runtime and checks it joins correctly (below);
  llama.cpp configured that way has not been built. The mechanism is the part
  that could have been wrong, and it is the part that is now tested.

### What this changes about the upstream story

[`docs/03`](03-upstream-issue-draft.md) asks whether `ggml-base` is the right
home for the registry, and says "I'd rather not force `BUILD_SHARED_LIBS=OFF` on
anyone". F18 answered that the design did force it. It does not any more, on
MSVC, which removes a known blocker from the instrumentation proposal rather
than a hypothetical one. The answer to the question is now: yes, `ggml-base` is
the right home, and the thing that has to live there is the registry — not the
thread-local that points into it.

---

## P23 — ggml's matmul has two partitioning modes, and adding threads can lose you the good one

**Dated 2026-09-07, session 4. Written and committed before the measurement**,
which is the habit sessions 2 and 3 found most valuable. Four predictions were
tested in session 2 and two were wrong; six in session 3 and one was wrong.

### The mechanism, from the source

`ggml_compute_forward_mul_mat` picks its partitioning at runtime:

```c
int chunk_size = 16;
if (nr0 == 1 || nr1 == 1) chunk_size = 64;      // decode: nr1 == 1

int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
    nchunk0 = nr0 > nr1 ? nth : 1;              // one chunk per thread
    nchunk1 = nr0 > nr1 ? 1 : nth;
}
...
int current_chunk = ith;
while (current_chunk < nchunk0 * nchunk1) {
    ...
    current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
}
```

— `ggml/src/ggml-cpu/ggml-cpu.c`, at the pin `4d91760`.

Above the threshold there are more chunks than threads and they are claimed from
a shared atomic counter: a slow core takes fewer chunks and the node still ends
when the *work* ends. **That is work stealing, and it makes core heterogeneity
free.** Below the threshold each thread gets exactly one equal slice and the
node ends when the slowest thread ends.

The condition is `nchunk0 * nchunk1 < nth * 4`. `nchunk0` is fixed by the
model's shape; `nth` is not. **So raising the thread count can move a matmul
from the self-balancing mode into the equal-slice mode.** `tools/mulmat_chunking.py`
computes which mode each matmul takes, from the GGUF tensor table alone:

```
mid.gguf (24L, n_embd 768, n_ff 3072)     t=8         t=16        t=28
  attn_q            nr0    768        static/8    static/16   static/28
  attn_output       nr0    768        static/8    static/16   static/28
  ffn_down          nr0    768        static/8    static/16   static/28
  ffn_gate          nr0   3072         dynamic    static/16   static/28   <- flips
  ffn_up            nr0   3072         dynamic    static/16   static/28   <- flips
  output            nr0   8192         dynamic     dynamic     dynamic
```

`ffn_gate` and `ffn_up` are 48 natural chunks. At 8 threads the threshold is 32
and they self-balance; at 16 it is 64 and they do not.

### The baseline is already in a committed trace

`examples/mid-24L-L3-tok10-11.trace.json`, 8 threads, `--barriers`. Imbalance
divided by work, for the matmuls, with the mode this tool assigns:

| node | mode at t=8 | work | imbalance | imbalance/work |
|---|---|---|---|---|
| `ffn_out` (`ffn_down`) | static/8 | 88.93 ms | 10.19 ms | **0.115** |
| `attn_out` | static/8 | 22.55 ms | 2.90 ms | **0.129** |
| `Qcur` | static/8 | 23.19 ms | 3.14 ms | **0.135** |
| `ffn_up` | dynamic | 88.39 ms | 5.61 ms | **0.063** |
| `ffn_gate` | dynamic | 88.61 ms | 4.05 ms | **0.046** |

Three static matmuls cluster at 0.115–0.135. Two dynamic ones sit at 0.046–0.063,
roughly half. And `ffn_down` against `ffn_up`/`ffn_gate` is close to a controlled
comparison: same phase, same layer, adjacent in the graph, the same 768×3072
parameter count, the same dtype, and **work within 0.6% of each other** (88.93,
88.39, 88.61 ms). The mode is the salient difference.

That is suggestive and it is not yet a test, because it is one trace at one
thread count and the modes are confounded with `nr0`. The prediction below is
the test, because it moves the mode while holding the tensor fixed.

### P23.1 — `ffn_up` and `ffn_gate` lose self-balancing between 8 and 16 threads

**Predict their imbalance/work roughly doubles**, from 0.046–0.063 at 8 threads
to **above 0.10** at 16 and 28, joining the static cluster.

Falsified if they stay below 0.08 at 16 threads, which would say the mode does
not govern and the 8-thread gap was about `nr0` or about where the node sits in
the graph.

### P23.2 — the controls do not move the same way

`ffn_down`, `attn_out` and `Qcur` are static at every thread count tested, so
their imbalance/work should **not** show the same jump. It need not be flat —
more threads means more arrivals to wait for, and F10 already measured total
barrier wait rising from 11.2% to 22.9% between 8 and 28 threads — but the
*ratio* of `ffn_up` to `ffn_down` should collapse toward 1.

**Predict `ffn_up`/`ffn_down` imbalance-per-work goes from ~0.55 at 8 threads to
above 0.8 at 16 threads.** This is the sharper form of P23.1 and the one I would
defend, because it divides out any effect that raises every node's imbalance
together.

### P23.3 — `output` (`lm_head`) stays self-balancing throughout

8192 rows is 128 chunks, above the threshold even at 28 threads (112). It is the
control that stays in the good mode the whole way.

**Predict its imbalance/work stays below the static cluster at every thread
count tested.** Falsified if it rises with the others, which would mean the
whole effect is thread count rather than mode.

### What this would mean for section 5 item 5

Item 5 is proportional row assignment — give faster cores more rows — and the
HANDOFF calls it the largest change this project has pointed at. If P23 holds it
reshapes the proposal in two ways, and both are worth having before writing a
patch rather than after:

1. **ggml already solves this for large matmuls, by work stealing rather than by
   prediction.** Proposing static proportional assignment for nodes that are
   already dynamically chunked would be proposing a worse version of a mechanism
   that is already there. F15 is what happens when a plausible fix goes out
   unmeasured.
2. **The interesting change may be the threshold, not the assignment.** If the
   equal-slice fallback is what costs, then `nchunk0 * nchunk1 < nth * 4` is the
   line to argue about — and lowering the multiplier, or chunking more finely
   when threads are heterogeneous, is a much smaller change than reworking row
   assignment across every op.

Neither follows unless P23 holds. Recorded before measuring so that it cannot be
remembered as having been obvious.

### What would make this uninteresting

If total barrier wait at 16 and 28 threads is dominated by something else
entirely — spin-up, or the near-serial elementwise nodes F9 found — then the
matmul mode could flip exactly as predicted and be worth nothing. F9 measured
the elementwise nodes at 14% of imbalance on this model at 8 threads; if that
share grows sharply with thread count, the matmul story is a footnote. The
measurement should report both.

---

## F23 — ggml already solves core heterogeneity for big matmuls, and a thread count can take that away

**Session 4, 2026-09-07.** Testing [`P23`](#p23--ggmls-matmul-has-two-partitioning-modes-and-adding-threads-can-lose-you-the-good-one),
written and committed before any of this was measured.

**P23.1 holds but is nearly uninformative. P23.2, the sharp one, holds, with
non-overlapping ranges on two independent comparisons. P23.3 turned out to be
unmeasurable as written. P23.4 was a prediction about an anomaly that did not
survive being measured properly, and is withdrawn along with it.**

> **This section was rewritten once.** Its first version reported six runs per
> arm and built an argument on a 28-thread "anomaly". Twelve runs per arm made
> the anomaly disappear — it was sampling noise — and moved two other numbers
> materially. The first version's *conclusion* survived; several of its numbers
> did not. What that cost, and why the table below is uniformly n=12, is in
> "The reproducibility problem" below, which is the more useful half of this
> finding.

### The result

`ffn_up` and `ffn_down` are the pair. Same phase, same layer, adjacent in the
graph, the same parameter count, the same dtype, work within 0.6% of each other.
What differs is `nr0` — 3072 against 768 on `mid.gguf` — and `nr0` is what
decides which side of `nchunk0 * nchunk1 < nth * 4` a matmul falls on. So for a
**fixed model** the pair's shapes are fixed and only the *mode* changes with
thread count, which is the comparison P23.2 asked for.

Arrival imbalance per unit work, `ffn_up` ÷ `ffn_down`. **Median of 12 identical
runs at every point**, with the full range, because the ranges are the story:

| model | threads | `ffn_up` mode | ratio | range over 12 runs |
|---|---|---|---|---|
| `mid.gguf` F32 | 8 | **dynamic** | **0.557** | 0.253 – 1.350 |
| `mid.gguf` F32 | 16 | static | 0.967 | 0.685 – 1.358 |
| `mid.gguf` F32 | 20, pinned 1/core | static | 0.944 | 0.677 – 1.823 |
| `mid.gguf` F32 | 28 | static | 1.041 | 0.439 – 1.217 |
| Qwen2.5-0.5B Q4_K_M | 16 | **dynamic** | **0.338** | 0.265 – 0.406 |
| Qwen2.5-0.5B Q4_K_M | 28 | static | 1.505 | 0.983 – 1.879 |

Every arm in which `ffn_up` self-balances sits below 0.56. Every arm in which it
does not sits above 0.94. **Two comparisons have non-overlapping ranges**, which
is what carries this:

- **Within Qwen**, 16 threads against 28: 0.265–0.406 against 0.983–1.879. Same
  model, same tensors, same machine. The only thing that changed is that
  `ffn_up`'s 76 natural chunks stopped clearing the threshold when it rose from
  64 to 112.
- **Across models at a fixed 16 threads**, Qwen against `mid`: 0.265–0.406
  against 0.685–1.358. Same thread count, same machine, same analysis. `mid`'s
  `ffn_up` is 48 chunks and falls below the 64 threshold; Qwen's is 76 and does
  not.

The second is the control the original prediction did not think to ask for, and
it is the one that separates *mode* from *thread count* — the first comparison
alone could be explained by 28 threads simply being worse than 16.

`mid.gguf` at 8 threads is the weakest row: right direction, but a range of
0.253–1.350 that overlaps everything. It is reported rather than dropped.

Underlying medians, same 12 runs:

```
mid.gguf     t=8   ffn_up 0.056 (dyn)  ffn_gate 0.034 (dyn)  ffn_down 0.095 (sta)  attn_out 0.090 (sta)
             t=16  ffn_up 0.210 (sta)  ffn_gate 0.226 (sta)  ffn_down 0.214 (sta)  attn_out 0.222 (sta)
             t=28  ffn_up 0.137 (sta)  ffn_gate 0.156 (sta)  ffn_down 0.145 (sta)  attn_out 0.157 (sta)
Qwen 0.5B    t=16  ffn_up 0.072 (dyn)  ffn_gate 0.065 (dyn)  ffn_down 0.214 (sta)
             t=28  ffn_up 0.270 (sta)  ffn_gate 0.300 (sta)  ffn_down 0.195 (sta)
```

At every all-static row the values converge: `mid` t=16 spans 0.210–0.226 across
four nodes, t=28 spans 0.137–0.157. They do not merely rise, they **land on the
static value**, which is what "the mode is the variable" predicts and what a
general thread-count effect does not.

**A matmul that steals work carries roughly a third to a half the arrival
imbalance per unit of work of one handed an equal slice.**

### P23.1: technically right, and it is the control that says so

Predicted `ffn_up` and `ffn_gate` imbalance/work above 0.10 at 16 threads.
Measured 0.210 and 0.226. Held.

But `ffn_down` and `attn_out` were static at *both* thread counts and rose just
as far — 0.095 → 0.214 and 0.090 → 0.222. **A prediction that a number goes up,
in a regime where every comparable number also goes up, is worth very little.**
P23.1 was written as the headline and P23.2 as its refinement; the refinement is
the entire result. Noted for the next prediction: state it as a ratio against a
control wherever a control exists.

### P23.3: not falsified, not confirmed, unmeasurable as written

`lm_head` was to be the control that never leaves the dynamic mode. It cannot
be: `result_output` is the last node of the graph, and `--barriers` matches a
node to *the barrier that follows it*. There is no following barrier, so the
node has no arrival-imbalance figure at all and is absent from the analysis
rather than reported as zero.

The prediction was written from the tensor table without checking that the
quantity it names exists for that node. **A prediction has to be about something
the instrument can return**, which is a cheaper check than the one it replaces.

### P23.4 and the anomaly that was not there

The first version of this finding reported `mid.gguf` at 28 threads as an
anomaly: both nodes static, and a ratio of **0.558** where ~1.0 was expected.
P23.4 was then written and committed, predicting that pinning 20 threads one per
physical core would restore it to 0.8–1.2 — on the theory that 28 logical
threads over 20 physical cores introduces SMT contention as a second mechanism.

The 20-thread pinned run gives **0.944**, inside the predicted band. But by then
the premise was gone: re-running the 28-thread arm at twelve repetitions instead
of six gives **1.041**, not 0.558. **There was no anomaly.** It was a six-run
median that happened to land low, in an arm whose range turned out to be
0.439–1.217.

So P23.4 is not scored. It predicted the resolution of something that did not
need resolving, and its test — while consistent with everything else here — was
answered by re-measuring the original number rather than by the intervention it
proposed.

The SMT question it was invented for remains genuinely open and this machine
cannot settle it. Separating SMT contention from core heterogeneity needs two
arms at the same thread count, one sharing physical cores and one not, in a
regime where `ffn_up` is static. Static needs ≥16 threads; 16 threads without
SMT needs more than the 8 P-cores available, so the no-SMT arm must include
E-cores and the arms then differ in core *type* as well. **The confound is in the
hardware, not in the experiment design**, and it should be tested somewhere with
more homogeneous cores.

### The reproducibility problem, which is the transferable half

P23's baseline came from a single committed trace,
`examples/mid-24L-L3-tok10-11.trace.json`, reading `ffn_up`/`ffn_down` = 0.554
at 8 threads. The first fresh trace at the same thread count, same model, same
binary, gave **1.086** — which reads exactly like the mode not mattering, and
was written down as "P23.2 falsified" before any repeats were run.

That is three separate occasions in one session where a median moved enough to
change the conclusion:

| quantity | n=1 | n=6 | n=12 |
|---|---|---|---|
| `mid` t=8 ratio | 1.086 (and 0.554 from the committed trace) | 0.330 | **0.557** |
| `mid` t=28 ratio | 0.996 | 0.558 → *called an anomaly* | **1.041** |
| `mid` t=20 pinned ratio | — | 0.747 → *outside predicted band* | **0.944** |

Every one of those n=6 columns was used, at the time, to say something the n=12
column does not support. The spread of the underlying per-node quantity over
identical runs explains why:

| | 8 threads | 16 threads | 28 threads |
|---|---|---|---|
| `ffn_up` | 3.7× | 1.5× | 4.0× |
| `ffn_down` | **8.5×** | 1.4× | 2.5× |
| `attn_out` | 6.4× | 1.3× | 2.9× |

Up to **8.5×** between the smallest and largest of twelve identical runs. The
16-thread arm is the well-behaved one at 1.3–1.5×, and the 8-thread arm is the
worst — the opposite of the intuition that more threads means more noise.

So: **every per-node imbalance figure in this repo that came from a single trace
should be read as one draw, not as a measurement.** That includes F9's table. It
does not affect F9's *structural* claims — which threads get work, and why —
since those are about row counts rather than timing, and it does not affect any
throughput number, which all come from the uninstrumented build and were always
repeated.

This project already had the discipline. `bench_overhead.py` interleaves arms and
bootstraps confidence intervals, and the harness has refused to certify a result
six times across four sessions because the noise exceeded the effect. That
machinery was built for throughput and never applied to numbers read *out of
traces*, which were treated as exact because the tracing is exact. The tracing
is exact. The machine underneath it is not.

### What this does to section 5 item 5

Item 5 is **proportional row assignment** — give faster cores more rows — and the
HANDOFF has called it the largest change this project has pointed at, since F14
measured P-cores at 2.88× E-cores against ggml's equal `dr = (nr + nth - 1)/nth`.

F23 does not kill it, but it moves the target:

1. **For large matmuls, ggml already solves this, and by a better method.**
   Work stealing needs no model of how fast each core is, no calibration, and no
   assumption that speeds are stable — a slow core simply takes fewer chunks.
   Proposing static proportional assignment for nodes that are already
   dynamically chunked would be proposing a worse mechanism than the one in the
   tree. F15 is what happens when a plausible fix is proposed unmeasured.

2. **The interesting line is the threshold, not the assignment.**
   `nchunk0 * nchunk1 < nth * 4` decides which nodes get the good mode, and the
   `nth` in it means the answer changes as you add threads. On `mid.gguf` the two
   largest nodes in the graph lose work stealing between 8 and 16 threads. On
   Qwen2.5-0.5B they lose it between 16 and 28. **A user adding threads to go
   faster silently turns off the load balancer for their biggest matmuls**, and
   nothing reports that.

3. **The cheap experiment is now obvious and was not before.** Lower the
   multiplier, or make `chunk_size` adapt to `nth` instead of holding at 64, and
   the flip moves or stops happening. That is a one-line change to a heuristic
   whose own comment admits it was tuned empirically on NUMA
   ("*In theory, chunking should be just as useful on NUMA and non NUMA systems,
   but testing disagreed with that*"), which is a far easier thing to propose
   than reworking row assignment across every op — and it is testable here.

4. **Whatever is proposed, it needs n≥12 per arm.** On this evidence a six-run
   comparison of this quantity cannot tell a 3× effect from noise reliably.

Not attempted in this session. Written down so item 5 is re-scoped by evidence
rather than by memory.

### Caveats

- **One machine**, i7-14700HX, 8 P + 12 E, Windows/MSVC, on the OpenMP
  threading path (F26 -- this line originally said the opposite). The mode
  logic is platform-independent source and sits in `mul_mat`, not in the
  barrier, so it is the caveat least affected by which of the two applies;
  every imbalance number here is still this machine's.
- **Two models**, both small and dense, F32 and Q4_K_M. The 8B was not run: its
  `ffn_up` is 192 chunks and stays dynamic through 28 threads, so it offers no
  flip to observe on this machine — which is itself the point that the flip
  depends on shape.
- Medians of 12 with ranges given. These are **not** confidence intervals, and
  the runs were not interleaved between arms the way `bench_overhead.py`
  interleaves; a determined version would randomize arm order and bootstrap.
  Two comparisons have non-overlapping ranges, which is the strongest statement
  the data supports without that.
- `imbalance / work` is a ratio of two quantities from the same trace, so it is
  insensitive to the machine being globally fast or slow that minute. That is
  why it is used instead of raw imbalance — and the 8.5× spread shows it is not
  immune.
- The pairing argument needs `ffn_up` and `ffn_down` comparable in everything but
  mode. They are equal in parameters, dtype, phase and layer, and measured within
  0.6% on work — but `ffn_down` reads the SwiGLU output while `ffn_up` reads the
  layer norm, so their inputs differ in provenance if not in size. The
  cross-model control at fixed thread count is what makes the argument, not the
  pairing alone.

---

## P24 — Lowering ggml's chunking threshold: predictions, and the protocol, before the build

**Dated 2026-09-07, session 4. Written and committed before the patched binary
exists.** [`F23`](#f23--ggml-already-solves-core-heterogeneity-for-big-matmuls-and-a-thread-count-can-take-that-away)
ended by naming this as the experiment to run before proposing anything
upstream, and [`F15`](#f15--removing-106-of-the-barriers-changes-throughput-by-nothing-measurable)
is what happens when a plausible fix goes out unmeasured.

### The change

One line, in `ggml_compute_forward_mul_mat` only:

```c
-    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
+    if (nchunk0 * nchunk1 < nth * 2 || ggml_is_numa()) {
```

**`ggml_compute_forward_mul_mat_id` is deliberately left alone.** It carries the
identical threshold at `ggml-cpu.c:1698` and it is the MoE path — for which this
machine has no model, so it cannot be tested here. Changing an untested code path
is the F15 mistake with extra steps.

Modelled with `tools/mulmat_chunking.py --mult 2`, on `mid.gguf` at 16 threads:

```
                stock (nth*4)   patched (nth*2)
  ffn_gate        static/16        dynamic       <- flips
  ffn_up          static/16        dynamic       <- flips
  ffn_down        static/16        static/16     <- control, 12 chunks < 32
  attn_output     static/16        static/16     <- control
  output           dynamic         dynamic       <- was never static
```

`ffn_up` is 48 natural chunks: below 64 (`16*4`), at or above 32 (`16*2`). So it
returns to work stealing with about 3 chunks per thread, while `ffn_down` at 12
chunks stays static in both arms. **That is F23's exact pair, with the mode moved
by a source change instead of by a thread count** — which is the one way of
varying it that F23 could not use.

### Protocol, fixed now so it cannot be chosen after seeing the numbers

- **Imbalance: n=12 per arm**, `mid.gguf`, 16 threads, level 3,
  `TOKENSCOPE_TOKENS=10:11`, via `tools/imbalance_repeat.py`. F23 measured six to
  be too few by a wide margin, three separate times.
  *(Session 5 note: at the time this was written `10:11` parsed as token 10
  alone — [`F29`](#f29--tokenscope_tokens1011-captured-one-token-and-had-done-so-since-session-4).
  Both arms used it, so the comparison holds; the capture was half the size it
  says here.)*
- **Throughput comes from the uninstrumented build**, `build-ts-off`, never from
  a trace. A traced token carries the recording cost on the token being measured;
  this project nearly reported a 21% degradation that was pure noise for exactly
  that reason.
- **The two throughput arms are interleaved.** `build-ts-off` is a static build,
  so `llama-bench.exe` is self-contained: the stock binary is copied aside, the
  patched one built, and the two are then run alternately in the same session
  rather than in two blocks. Consecutive-block A/B on this machine is what F23's
  reproducibility section is about.
- Free memory checked before and after; any run that pages is discarded.

### P24.1 — the mode flips back, and the imbalance follows

Stock at 16 threads, n=12 medians: `ffn_up` 0.210, `ffn_down` 0.214, ratio 0.967.
The dynamic-mode arms elsewhere in F23 sit at 0.056–0.072 with ratios of
0.338–0.557.

**Predict `ffn_up` imbalance/work falls below 0.12, and the `ffn_up`/`ffn_down`
ratio falls below 0.6.** Falsified if the ratio stays above 0.8, which would say
the mode is not what F23 thinks it is — and would put F23's own conclusion in
question, since this is the same claim tested a different way.

Weaker than F23's arms in one respect, stated up front: 3 chunks per thread is
thin. Work stealing with 48 chunks over 16 threads can only ever redistribute in
units of a third of a thread's share, so **the effect should be smaller than the
0.338 seen where chunks were plentiful.** A ratio landing at 0.6–0.8 would be
consistent with the mechanism and weak evidence for it; below 0.6 is the call.

### P24.2 — and the throughput does not move measurably

**Predict the change is under 2%, and that the honest report is "not
measurable".** Falsified by an improvement above 3%.

This is a prediction of a null, and it is the one I hold most confidently, for
reasons this project has already established:

- [`F9`](#f9) put the **upper bound on removing all elementwise barrier
  imbalance at 1.34% of graph wall time.** The imbalance addressed here is a
  subset of a different node class, but the order of magnitude is the point.
- [`F15`](#f15) removed 10.6% of all barriers per token and moved throughput by
  nothing measurable, in three regimes.
- Arrival imbalance is thread-time idled, not wall time added. Threads that
  arrive early wait at a barrier they were going to wait at anyway; the node ends
  when the *slowest* thread ends, and work stealing only helps if it moves work
  off that thread specifically.
- `mid.gguf` is F32 and 220 M parameters, so decode at 16 threads is closer to
  bandwidth-bound than compute-bound, and [`F14`](#f14) found core heterogeneity
  stops mattering in that regime.

**If P24.1 holds and P24.2 also holds, that is the interesting outcome, not a
disappointing one.** It would say the threshold governs a real and measurable
property of the schedule that does not reach the user — which is precisely what
should be established *before* a maintainer is asked to look at a patch, and
precisely what F15 wishes someone had established for op fusion.

### P24.3 — the controls do not move

`ffn_down` and `attn_output` are static in both arms. **Predict their
imbalance/work medians stay inside the stock n=12 ranges** (`ffn_down`
0.205–0.291, `attn_out` 0.186–0.240). Falsified if either shifts outside, which
would mean the patch changed something other than the two nodes it was aimed at
— most likely by changing how much barrier time is available to be attributed
anywhere, and would make P24.1 hard to read.

### What would make this uninteresting

If the patched build changes `output`/`lm_head` or any node the model says
should be untouched, the one-line change is not doing the one thing it looks
like it does, and the whole comparison is confounded. `mulmat_chunking.py --mult`
says which nodes should move; anything else moving is a finding about the model
being wrong, and gets reported as one.

---

## F24 — One line, +1.95% decode, and the first certified speedup in this project

> **Superseded by [`F33`](#f33--f24-survives-at-162-the-interval-that-can-see-drift-works-and-the-control-stopped-being-clean), session 6.
> The effect is real; `+1.95% [+1.59, +2.35]` is not the right number for it.**
> Re-measured over six blocks and two independent runs with an interval that
> can see between-run drift: **+1.62% [+1.10, +2.15]**. The old point estimate
> sits inside the new interval; the new one sits at the edge of the old.
>
> Two things below no longer stand as written. The word **"certified"** meant
> only "resolved within this run" — see F31. And **the prefill control is no
> longer a clean null**: across six blocks it reads -1.30% [-2.67, +0.06], and
> every run has come back negative. The argument quoted below — "the arm that
> should not move does not" — is the weakest part of this finding now, and it
> was its strongest part when written.
>
> The two arms also differ by 1,137,994 bytes, so **code layout is not held
> constant** and some unknown fraction of the effect may be layout rather than
> chunking. Read F33 before quoting anything here.


**Session 4, 2026-09-07.** Testing [`P24`](#p24--lowering-ggmls-chunking-threshold-predictions-and-the-protocol-before-the-build),
committed before the patched binary existed.

**P24.1 holds, with non-overlapping ranges. P24.3 holds. P24.2 splits: it got the
magnitude right and the certifiability wrong — I predicted "not measurable" and
the benchmark harness certified it.** That harness has declined to certify six
times across four sessions. This is the first thing it has ever certified.

```
mul_mat: -    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
         +    if (nchunk0 * nchunk1 < nth * 2 || ggml_is_numa()) {
```

`patches/03-mulmat-chunk-threshold.patch`. `mul_mat_id` carries the identical
threshold and is deliberately untouched — it is the MoE path and no MoE model was
available, and shipping an untested behaviour change is what F15 exists to
record.

### P24.1 — the mode flips, and the imbalance follows it

`mid.gguf`, 16 threads, n=12 per arm, both arms level-3 instrumented:

| | `ffn_up` imb/work | `ffn_up`/`ffn_down` ratio | range |
|---|---|---|---|
| stock, `nth*4` (`ffn_up` static) | 0.210 | **0.967** | 0.685 – 1.358 |
| patched, `nth*2` (`ffn_up` dynamic) | **0.087** | **0.446** | 0.185 – 0.631 |

Predicted below 0.12 and below 0.6. Measured 0.087 and 0.446, and **the ranges
do not overlap**.

This is the third independent way F23's claim has now been tested, and the only
one that moves the mode by a source change rather than by a thread count. F23 had
to vary `nth`, which varies other things too; here `nth` is fixed at 16, the
model is fixed, the machine is fixed, and one constant in one line differs.

It landed stronger than P24 expected. The prediction warned that 48 chunks over
16 threads is only 3 per thread, so stealing could only redistribute in thirds of
a share, and that 0.6–0.8 would be the honest expectation. It reached 0.446 —
close to the 0.338 seen where chunks were plentiful. **Three chunks per thread
recovers most of the benefit**, which is itself the useful engineering fact,
because it says the threshold does not need to be large to work.

### P24.3 — the controls held

`ffn_down` and `attn_output` are static in both arms and should not move.
`attn_out` went 0.222 → 0.198, inside its stock range. `ffn_down` went 0.214 →
0.204, and the prediction named the stock *range* 0.205–0.291, so the patched
median lands **0.001 below the stated floor**. That is reported rather than
rounded away, and it is not meaningful: the two ranges (0.205–0.291 and
0.182–0.293) overlap almost entirely.

### P24.2 — right about the size, wrong about whether it would show

Both arms **uninstrumented** (`build-ts-off`), interleaved within one session
with the leading arm alternating each round, 20 rounds per arm, three
`llama-bench` repetitions each. CI from `bench_overhead.py`'s own
`bootstrap_ratio_ci`, so this is the same standard every overhead number in this
repo is held to:

| workload | stock | patched | change | 95% CI | verdict |
|---|---|---|---|---|---|
| **tg64** (decode) | 40.17 tok/s | 40.96 tok/s | **+1.95%** | **[+1.59, +2.35]** | **certified** |
| pp64 (prefill, control) | 739.94 tok/s | 733.96 tok/s | −0.81% | [−2.44, +0.35] | not certified |

Predicted "under 2%, and the honest report is *not measurable*". The first half
is right — 1.95% is under 2%. The second half is wrong: the interval excludes
zero.

**The prefill row is why the decode row is believable.** Prefill has `nr1 = 64`,
so `chunk_size` is 16 and there are 768 chunks — far above the threshold in
*both* arms, so the patch cannot reach it. Same harness, same session, same
machine, same binaries: the arm that should move moves and certifies, the arm
that should not move does not and does not certify. A build artifact or a
thermal drift would not respect that distinction.

Why I expected a null and did not get one, since the reasoning was not stupid:
F9 bounded the elementwise-barrier prize at 1.34% of graph wall time, and F15
removed 10.6% of barriers for nothing. But both of those are about *removing
barriers*. This does not remove any barrier — it changes **who does the work
before each one**, so it moves the arrival time of the slowest thread rather
than the count of things to wait at. Those are different quantities and I applied
the wrong prior. F15's lesson was "the barriers you can cheaply remove are the
ones nobody was waiting at"; the corollary, which this is, is that the barriers
people *are* waiting at are worth attacking from the work side.

### The confound that would have inflated this by 1.2 points

The first throughput run reported **+2.71%, with non-overlapping ranges**, and
was wrong.

`build-ts-off` had last been built on 5 September. `src/llama-graph.cpp` was
modified on 6 September by session 3's F20 naming patch. So the binary saved as
"stock" predated a change the patched binary contained, and the two arms differed
by more than the line under test. Rebuilding both from the identical tree took
the effect from +2.71% to +1.49% at n=12, and +1.95% at n=20.

Caught by checking file timestamps against the binary's, not by anything going
wrong. Nothing about the run looked suspicious — the ranges separated cleanly,
which if anything made it *more* convincing.

**An A/B where one arm is a binary you saved earlier is not an A/B.** Build both
arms from the same tree in the same session, or the thing you are measuring
includes every commit in between. This project already knew to interleave arms in
time; it had not written down that the arms have to be interleaved in *version*
too.

### What this is and is not

It **is**: evidence that ggml's chunking threshold is load-bearing at ordinary
thread counts, that it is reachable by a user simply adding threads, and that
moving it is worth about 2% of decode on one machine and one model — with the
mechanism visible in the trace rather than inferred from the throughput.

It is **not** an argument that `2` is the right constant. The comment being
edited says the `nth * 4` form was tuned empirically and that NUMA testing
disagreed with theory (PR #6915), and there is no NUMA hardware here to check
that against. The honest upstream framing is a question about a constant, backed
by a measurement, not a patch that claims to know better.

### Caveats

- **One machine** (i7-14700HX, 8 P + 12 E), **one thread count** (16), **one
  model** for the throughput arm (`mid.gguf`, F32, 220 M). The imbalance result
  has two models behind it via F23; the throughput result has one.
- 16 threads is where `mid.gguf` flips. A user at 8 threads on this model sees
  none of this, and a user at 28 sees `ffn_up` static in both arms. **The size of
  the effect depends on where your model's shapes sit relative to your thread
  count** — which is the finding, but it also means +1.95% is not a number to
  quote as ggml's headroom.
- No NUMA hardware, which is the case the original constant was tuned for and
  the one most likely to regress.
- `mul_mat_id` untested and unchanged.
- Prefill is a control here, not a result. It says the patch does not *hurt*
  prefill on this model; at n=20 its interval still spans −2.44 to +0.35.
- The decode workload is `-n 64` at batch 1. Longer generations, other batch
  sizes and other architectures are untested.

---


## P25 — What the shared build's overhead should cost, written while the harness runs

**Dated 2026-09-08, session 5. Written and committed after the two shared
binaries exist and the harness was started, and before any of its output was
read.** Section 5 item 7 has said since session 4 that
[`F22`](#f22--the-shared-library-build-works-because-the-thread-local-was-never-the-thing-that-had-to-be-shared)
left the hot path unchanged *by inspection*, which is not a measurement. This is
the prediction that goes with the measurement.

### The arms

A `GGML_TOKENSCOPE=OFF` shared build did not exist; there was only the ON one,
which is why item 7 had never been runnable. `build-ts-shared-off` is that
build, configured identically to `build-ts-shared` — same Visual Studio 17 2022
generator, `BUILD_SHARED_LIBS=ON`, same options — and both were rebuilt in the
same session, because [`F24`](#f24--one-line-195-decode-and-the-first-certified-speedup-in-this-project)
is what happens when one arm of an A/B is a binary somebody built earlier. The
static pair was checked with `ninja` and reported `no work to do`, so it matches
the tree too.

`ggml-cpu.dll` in the OFF build contains zero occurrences of the string
`tokenscope`; the ON build's contains one. That is the same check as the
zero-overhead-when-off proof in [`02`](02-overhead-methodology.md), applied to a
DLL instead of an archive.

Both pairs run the same workload as session 1: `mid.gguf`, 8 threads, pp512 /
tg256, `-n 20` interleaved with the first rep discarded, levels 1, 2 and 3.

### What actually changes in the hot path, at the instruction level

F22's "unchanged by inspection" is true of the *source* and not quite true of
the *instructions*, which is the whole reason to measure. Three differences, all
of them from `TS_API` becoming `__declspec(dllimport)` instead of nothing:

1. `ts_g_capture`, `ts_g_level`, `ts_g_token` and `ts_g_graph` are exported
   **data**. A dllimport data read on MSVC is a load of the import address table
   slot followed by a load through it, where the static build gets one
   RIP-relative load. Every `ts_reserve` reads one, every `ts_emit` reads two.
2. `ts_now()` is **not** `__rdtsc` here — `TOKENSCOPE_TSC` is OFF in every build
   this project has measured, so `ts_now()` is `ts_now_slow()`, which is
   `TS_API`. In the shared build each of the two clock reads per scope becomes
   an indirect call through the IAT instead of a direct call.
3. `ts_tls` is unchanged: `static __declspec(thread)`, one per module by
   design, so its cost is the same TEB-relative access in both.

Against those: one scope costs **52.8 ns** (`ts_selftest`), and two
`steady_clock::now()` calls on Windows are two `QueryPerformanceCounter`s, which
is essentially all of that 52.8 ns. An IAT indirection is a load that hits L1 and
a correctly-predicted indirect branch — low single-digit nanoseconds against
that.

### P25.1 — shared costs more than static at the same level, and by less than 50%

The comparison that matters is a **ratio against a control**, not a level
(P23.2's lesson): each build is compared to *its own* compiled-out arm, so
everything that makes a DLL build slower than a static one — no cross-module
inlining, indirect calls everywhere, no whole-program optimisation — cancels
out of both sides.

Direction: shared >= static, because instructions were added to the hot path
and none were removed. Size: **less than 1.5x the static figure at the same
level**, because the added work is a handful of nanoseconds on a path whose cost
is two kernel-ish clock reads.

Falsified if shared comes out *below* static by more than the intervals allow,
or above it by more than 1.5x.

### P25.2 — level 0 stays unresolvable in the shared build

B vs A is one load of `ts_g_level` and one predictable branch per site, plus an
IAT indirection in the shared arm. That was already below this machine's noise
floor when it was one load. **Prediction: no certified difference between A and
B in either build**, at any level.

### P25.3 — at least one of the eight comparisons is refused

Four levels' worth of comparison in each of two builds. The harness has declined
to certify **six times across four sessions**, the baseline IQR on this machine
was 2-4% in session 2, and every effect here is expected under 2%. Predicting a
clean sweep would be predicting against the instrument's own history.

### P25.4 — the static level-3 point estimate lands under 2% on decode

Session 1 measured +0.67% [+0.12, +1.67] at 8 threads, and session 2 could not
certify it twice. This is the same workload, the same thread count and the same
binary pair, so the *point estimate* should land under 2% whether or not it is
certified. A point estimate above 2% would mean something changed in the
instrumentation between session 1 and now, and 174 lines of patch say it did
not.

### One contaminated observation, disclosed

Before starting the harness both shared binaries were smoke-tested with a single
`-r 1` run each to prove the DLLs resolve — ON first, then OFF, which read the
840 MB model cold and then warm. It showed the ON build 12% slower on decode.
**That number is an artifact of the ordering and is not evidence for anything**;
it is written down because it was seen before these predictions, and because
"trace-derived throughput is not throughput" and "one run is one draw" are two
lessons this project has already paid for. The harness interleaves for exactly
this reason.

### What would make this uninteresting

If both builds land under 1% and neither certifies, the finding is "still below
the noise floor, in both configurations" — which closes item 7 as an honest
negative and is worth the run, but is not a result anybody would quote.

---

## F25 — The static overhead certifies for the first time since session 1, and the shared build's noise floor ate its own answer

> **Superseded by [`F36`](#f36--the-overhead-numbers-settled-every-build-under-1-linkage-and-generator-both-null-and-two-f31-claims-retracted).**
> `+1.16% [+0.67, +1.87]` was a single-run bootstrap and is too narrow; the
> current figure for this quantity is **+0.56% [-0.05, +1.16]** over three
> blocks. F36's interval contains this one's point estimate. **What stands is
> this finding's own post-mortem** — that a comparison spanning two invocations
> has not been interleaved — which is what led to `--pair`, `--blocks`, and
> everything after it.


**Workload:** `mid.gguf` (24L, F32, 220 M), 8 threads, pp512 / tg256, MSVC
Release, `-n 20` interleaved per arm with the first rep discarded, five arms
(compiled out / level 0 / levels 1, 2, 3), `tools/bench_overhead.py`. Two pairs,
run back to back: shared first (792 s), static second (785 s).

Section 5 item 7 has been unrunnable since session 4 for one reason — no
`GGML_TOKENSCOPE=OFF` **shared** build existed, so there was nothing for the ON
one to be compared against. `build-ts-shared-off` is that build. Both shared
arms were rebuilt in the same session as it; both static arms reported
`ninja: no work to do` against an unchanged tree.
[`P25`](#p25--what-the-shared-builds-overhead-should-cost-written-while-the-harness-runs)
holds the predictions, committed before any output was read.

### The result

**Static, decode.** Baseline IQR 1.06%, so the harness answered.

```
  arm                       median tok/s     IQR   overhead vs A
  A: compiled out                  42.94    1.1%                  -
  B: in, level 0                   42.84    0.8%    +0.23%  [-0.20, +0.92]
  C1: active level 1               42.58    1.0%    +0.86%  [+0.34, +1.58]
  C2: active level 2               42.70    1.2%    +0.56%  [+0.08, +1.54]
  C3: active level 3               42.45    0.7%    +1.16%  [+0.67, +1.87]
```

**Level 3 is +1.16% [+0.67, +1.87], certified** — the first overhead number this
project has been able to stand behind since session 1, after two refusals in
session 2 and a whole gap-table row saying every quoted figure was the old one.
It supersedes session 1's +0.67% [+0.12, +1.67]: same workload, same thread
count, same machine, wider and higher, and this time with the level-0 arm and
prefill both behaving as controls should.

Prefill is the control and stays uncertified at every level (C3 +1.28% [-0.21,
+1.93]), which is the pattern that makes the decode number believable rather
than a machine having a good day.

**Shared, decode.** Baseline IQR 2.22%, and the harness said so:

```
  A: compiled out                  42.65    2.2%                  -
  B: in, level 0                   42.68    2.0%    -0.08%  [-1.29, +1.47]
  C1: active level 1               42.58    3.1%    +0.16%  [-1.28, +2.02]
  C2: active level 2               42.64    1.6%    +0.02%  [-0.95, +1.38]
  C3: active level 3               42.54    0.8%    +0.26%  [-0.68, +1.50]

  baseline IQR is 2.22% of median.
  NOTE: that is wider than the 2% budget being tested.
```

Every shared interval spans zero. The shared build's overhead is **still
unmeasured**, and the row in the gap table stays.

### Scoring the predictions

| | claim | outcome |
|---|---|---|
| **P25.1** | shared >= static at the same level, by < 1.5x | **not tested.** The point estimates went the other way (+0.26% shared against +1.16% static at level 3) but the shared interval, [-0.68, +1.50], contains the static point estimate outright. Nothing was resolved in either direction |
| **P25.2** | level 0 unresolvable in both builds | **held.** +0.23% [-0.20, +0.92] static, -0.08% [-1.29, +1.47] shared. Both span zero |
| **P25.3** | at least one of the eight comparisons refused | **held**, and by more than predicted — the whole shared decode block was refused, and shared *prefill* produced C2 at -0.66% [-1.88, -0.12], an interval excluding zero in which the instrumented build is **faster** than the compiled-out one. That is not a result, it is the shape of structured noise, and it is worth more than the refusal notice as a warning |
| **P25.4** | static level-3 point estimate under 2% on decode | **held**, +1.16% |

P25.1 is the one worth dwelling on. It was not falsified and it was not
confirmed; it was **unanswerable with the design that was run**, and the design
was chosen before the run. That is a worse outcome than a wrong prediction,
because a wrong prediction teaches something.

### Why the design could not answer it, which is the transferable part

`bench_overhead.py` interleaves arms **inside** one invocation, so thermal
drift and background load hit A, B, C1, C2 and C3 equally. It has done that
since session 1 and it is why the static block above is trustworthy.

It does not interleave across invocations. The shared pair and the static pair
were two separate runs, thirteen minutes apart, and the machine's baseline IQR
between them moved by more than a factor of two — 2.22% then 1.06%. **The
comparison P25.1 wanted is between two runs, and between-run noise is exactly
what the harness's whole design avoids by never comparing between runs.**

So the question "does the shared build cost more?" needs the four builds'
arms interleaved **together in one invocation**: A_static, C3_static,
A_shared, C3_shared, round-robin. That is a change to the harness, not a
longer version of the run that was done. Session 1's lesson was
"interleave the arms"; this one is **the arms are whatever you are comparing,
and if your comparison spans two invocations you have not interleaved it**.

The same trap caught F24 in a different disguise — there the two arms differed
in *version*, here they differ in *time*. Both are the arms not being matched
on something the design assumed was constant.

### One observation the design does support, weakly

Both A arms are uninstrumented builds of the same tree at the same commit,
differing only in `BUILD_SHARED_LIBS`. Shared read 42.65 tg / 682.76 pp against
static's 42.94 / 688.62 — the DLL build about **0.7% slower on decode and 0.9%
on prefill**. That is the cost of DLL boundaries in llama.cpp, not of
tokenscope, and it is *small*, which is mildly interesting given that a shared
build gives up cross-module inlining entirely.

It is also a between-run comparison, so it inherits everything in the section
above. Take it as an order of magnitude and nothing more.

### Caveats

- **The binaries predate the same session's own instrumentation change.** All
  four were built from the tree at commit `7149794`, before `ts_note_build`
  (F26's provenance field) added one call per thread per graph and before F28
  moved the buffer allocation. The numbers above describe the code at that
  commit. The addition is off the node path and guarded by `ts_g_level`, but "by
  inspection" is the phrase F22 got caught by, so it is stated rather than
  dismissed.

  **It was re-run at the end of session 5 on the post-F28 binaries, and the
  harness refused the whole table.** Baseline IQR 3.78% on decode and 2.97% on
  prefill, against 1.06% and 0.99% earlier the same day on the same binaries'
  predecessors; every interval spans zero and every point estimate is negative,
  which is what noise looks like when it is wider than the effect. That is the
  seventh refusal in five sessions, it says nothing about whether the new code
  costs more or less, and it is a third independent demonstration of the thing
  this finding is really about: **on this machine the noise floor moves by more
  than the quantity being measured, within a single day.** `+1.16% [+0.67,
  +1.87]` stands as the best measurement of the instrumentation as of `7149794`,
  and the current tree's overhead is unmeasured.
- One machine, one model, one thread count, one workload. F10 predicts overhead
  grows with thread count and only 8 has ever been measured.
- The shared arms are Visual Studio / MSBuild builds and the static arms are
  Ninja builds, because that is how the two configurations have existed since
  session 4. Within each pair both arms share a generator, so each pair's
  internal comparison is clean; the cross-pair comparison has that difference
  in it as well as the timing one.
- Level 2 reading lower than level 1 in both pairs (+0.56% against +0.86%
  static) is not a real ordering — the intervals overlap almost completely.

---

## F26 — Every measurement in this project was taken on the OpenMP path, and five documents said the opposite

**Workload:** none. This is a fact about the build, found by reading
`build-ts-on/build.ninja` while configuring an unrelated one, and confirmed four
ways.

Since session 1 this project has told itself, in
[`FINDINGS`](#a-caveat-that-applies-to-every-barrier-number-here), in
[`HANDOFF`](HANDOFF.md), and in [`03`](03-upstream-issue-draft.md), that the
barrier numbers here come from ggml's **own threadpool** — the atomic spin-wait
built out of `n_barrier` / `n_barrier_passed` and `ggml_thread_cpu_relax()` —
and that `#pragma omp barrier` is the *Linux* path that the results might not
transfer to.

That is backwards. `GGML_OPENMP` defaults to **ON**
(`ggml/CMakeLists.txt:247`), with no platform condition, and CMake finds OpenMP
here. Every build this project has ever measured compiled `ggml-cpu.c` with
`-DGGML_USE_OPENMP`, so `ggml_barrier` has always been the two lines at
`ggml-cpu.c:583`:

```c
#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    ... the atomic spin-wait this project believed it was measuring ...
#endif
```

### Four independent confirmations

1. **The option.** `option(GGML_OPENMP "ggml: use OpenMP" ON)`, unconditional.
   `CMakeCache.txt` in `build-ts-on`, `build-ts-off` and `build-ts-shared` all
   carry `GGML_OPENMP:BOOL=ON` and `GGML_OPENMP_ENABLED:INTERNAL=ON`.
2. **The configure log.** `-- Found OpenMP: TRUE (found version "2.0")` — MSVC's
   `vcomp`, which implements OpenMP **2.0**, a 2002 specification.
3. **The compile line for the exact translation unit.** In
   `build-ts-on/build.ninja`, the rule producing
   `ggml-cpu.dir/ggml-cpu/ggml-cpu.c.obj` has
   `DEFINES = ... -DGGML_USE_OPENMP -DTOKENSCOPE_ENABLED -DTOKENSCOPE_STATIC ...`
   and passes `-openmp`. Not an inherited property on some other target: the
   file that contains `ggml_barrier`.
4. **The linked binary.** `llama-bench.exe` imports `VCOMP140.DLL`, and the
   named imports it uses are `_vcomp_fork`, `_vcomp_barrier`,
   `_vcomp_single_begin`, `_vcomp_single_end`, `_vcomp_set_num_threads`,
   `_vcomp_for_dynamic_init`, `_vcomp_for_dynamic_next` and
   `_vcomp_reduction_i4`. `_vcomp_barrier` is `#pragma omp barrier`;
   `_vcomp_fork` is the `#pragma omp parallel num_threads(n_threads)` at
   `ggml-cpu.c:3427` that dispatches every graph.

The fourth is the one that settles it, because it is evidence from the artifact
rather than from the build system's intentions.

### What this changes

**It does not change a single measured number.** Everything recorded in F1-F24
was produced by these binaries; nothing about them has moved. What changes is
the *label* on those numbers, and labels are what determines how far a result is
allowed to travel.

- **The barrier caveat inverts.** "These are spin-wait numbers, and Linux uses
  OpenMP" becomes "these are OpenMP numbers, and Linux uses OpenMP too". The
  barrier *mechanism* is the same on both platforms. What differs is the
  *runtime*: MSVC `vcomp` 2.0 here against GNU `libgomp` or LLVM `libomp`
  there, and their spin-then-park policies are genuinely different — `libgomp`
  has `GOMP_SPINCOUNT`, `libomp` has `KMP_BLOCKTIME`, and `vcomp` documents
  neither. So the caveat survives, at a much smaller size: it is now a claim
  about one OpenMP implementation versus another, not about two different
  synchronisation algorithms.
- **The Linux gap gets smaller and better defined.** [`HANDOFF`](HANDOFF.md)
  section 5 item 2 justified itself with "`ggml_barrier` is `#pragma omp
  barrier` instead of the atomic spin-wait measured here". The premise was
  false, so the item's *stated* risk was overstated. Linux/GCC is still
  unmeasured and still worth doing — different compiler, different vectoriser,
  different allocator, ELF instead of PE — but not because the barrier is a
  different thing there.
- **[`03`](03-upstream-issue-draft.md) is affected in the direction that
  matters.** It says "Do not file without Linux ... the OpenMP path is the
  default on Linux", implying the measurements are from a path most users do
  not run. They are from the path most users *do* run. That is an argument for
  the evidence being more transferable than the document claims, not less.
- **The spin-wait path has never been measured by this project at all.** That
  is the opposite of what `FINDINGS` said, and it is now a thing that can be
  tested on this machine in an afternoon: `-DGGML_OPENMP=OFF` builds it, so
  both barrier implementations are available on one CPU, at one thread count,
  with one model. See [`P27`](#p27--two-barrier-implementations-on-one-machine-predictions).

### The specific sentences that were wrong

| Where | Said | Actually |
|---|---|---|
| `FINDINGS` "A caveat that applies to every barrier number here" | "measured on the **non-OpenMP** threading path (Windows/MSVC, ggml's own threadpool)" | measured on the OpenMP path, `_vcomp_barrier` |
| `FINDINGS` F19 caveats | "The barrier and thread numbers are the non-OpenMP path" | OpenMP path |
| `FINDINGS` F23 caveats | "ggml's own threadpool and not OpenMP" | OpenMP |
| `HANDOFF` gap table | "F9/F10 both measured the non-OpenMP barrier path" | OpenMP barrier path |
| `HANDOFF` section 5 item 2 | "`ggml_barrier` is `#pragma omp barrier` instead of the atomic spin-wait measured here" | it is `#pragma omp barrier` in both places |

F22's caveat in F22 was the one that had it right by accident — "the OpenMP
threading path rather than ggml's own pool" is listed there as something Linux
would add, which is true of the *shared* configuration question it was about and
happened not to repeat the error.

### Why it survived four sessions

Nothing depended on it. No test asserted it, no number moved with it, and it
appears only inside caveats — the part of a document that exists to say what the
result does *not* cover, which is exactly the part nobody re-derives. It read
like a fact about Windows ("MSVC has poor OpenMP support, so llama.cpp uses its
own pool") that is plausible, was never written down as a measurement, and was
copied forward five times because each copy was quoting the last one.

The general lesson is narrower than "check your assumptions", which is useless
advice. It is: **a claim about how your code was built is checkable in one
command, and a caveat is a claim.** `grep GGML_USE_OPENMP build.ninja` would
have cost four seconds in session 1. The same command answers it for any
`#ifdef` that a finding's scope depends on, and this project has several — the
first thing F26 did after finding this was check `TOKENSCOPE_TSC`, which is
**OFF**, meaning `ts_now()` is `steady_clock::now()` and not `__rdtsc()` in
every number quoted here too. That one the documents had right.

---

## P27 — Two barrier implementations on one machine: predictions

**Dated 2026-09-08, session 5. Written and committed before the `GGML_OPENMP=OFF`
builds exist.** [`F26`](#f26--every-measurement-in-this-project-was-taken-on-the-openmp-path-and-five-documents-said-the-opposite)
found that the spin-wait path — the one this project spent four sessions
believing it was measuring — has never been run here at all. It is one CMake
flag away, which makes "how much does a barrier *cost* figure depend on the
barrier implementation?" answerable on this machine instead of blocked behind
Linux.

That question has been open since session 1. The standing caveat says F6's
11.2%, F9's imbalance/release split, F10's rise to 22.9% and F14's halving
"measure one barrier implementation" and might not transfer. Nobody has ever
put a number on *might*.

### The arms

Two more static Ninja builds, `-DGGML_OPENMP=OFF`, otherwise identical to the
existing pair and built in the same session as their comparisons:

```
build-ts-noomp-on    GGML_TOKENSCOPE=ON  GGML_OPENMP=OFF   traces
build-ts-noomp-off   GGML_TOKENSCOPE=OFF GGML_OPENMP=OFF   throughput
```

Verification that the flag took, before any measurement: `llama-bench.exe` must
**not** import `VCOMP140.DLL`, and the new provenance field must read
`threading=ggml-threadpool`. Both are checks that could come out wrong, which is
the only kind worth running.

### Three things change at once, and only one of them is the barrier

This is the trap from session 3 — "two mechanisms can move at once and you will
model one" — so all three go on the record before the measurement:

1. **The barrier.** `#pragma omp barrier` (`_vcomp_barrier`) against the atomic
   spin-wait on `n_barrier_passed` with `ggml_thread_cpu_relax()`.
2. **Fork/join per graph.** The OpenMP path enters `#pragma omp parallel
   num_threads(n)` once per graph — one `_vcomp_fork` per token in decode. The
   non-OpenMP path wakes persistent workers through
   `ggml_graph_compute_kickoff`. Both cost something; they are not the same
   something.
3. **A syscall per thread per graph, on the OpenMP path only.**
   `ggml_graph_compute` calls `ggml_thread_apply_priority(threadpool->prio)`
   *inside* the parallel region (`ggml-cpu.c:3440`), so every thread runs it at
   the start of every graph. On Windows at the default `GGML_SCHED_PRIO_NORMAL`
   that function still calls `SetThreadInformation(..., ThreadPowerThrottling,
   ...)` before its early return — a kernel transition, `n_threads` of them per
   token. The non-OpenMP path calls it once per thread at threadpool creation
   (`:3384`) and on resume (`:3253`, `:3305`).

Point 3 was found while writing this and is the reason P27.4 exists. It is a
**fixed per-graph cost proportional to thread count**, which is a different
shape from anything the barrier does, and it is the one that could be
interesting upstream on its own.

### P27.1 — arrival imbalance is unchanged. This is the control.

`--barriers` splits barrier time into arrival imbalance and release latency.
Imbalance is set by how work is divided among threads, and the division is
identical source in both builds — `mul_mat`'s chunking and the flat `dr` for
everything else are untouched by `GGML_USE_OPENMP`.

**Prediction: the per-node arrival-imbalance figures agree within the run-to-run
spread F23 measured for them, which is wide — up to 8.5x at 8 threads, 1.3-1.5x
at 16.** So this control is only informative at 16 threads or above, and n>=12
per arm, both of which F23 paid to learn.

If imbalance *does* move, something is wrong with the comparison rather than
interesting about barriers.

### P27.2 — release latency is lower on the spin-wait path

A thread in the spin-wait polls one relaxed atomic in a `ggml_thread_cpu_relax()`
loop and leaves within tens of nanoseconds of the last arrival. `_vcomp_barrier`
is a library call into a runtime whose release policy is undocumented and which
implements a 2002 specification.

**Prediction: median release latency in the non-OpenMP build is lower, by at
least 20%, at 8 threads on `mid.gguf` at level 3.** Direction plus a size, so it
can fail two ways.

The reasoning is not one-sided, which is why the size is modest: at 8 threads on
28 logical CPUs nothing is oversubscribed, so a spin-then-park runtime never
reaches the park, and both are then spinning on something.

### P27.3 — decode throughput at 8 threads moves by less than 3%

F15 removed 10.6% of the barriers and changed throughput by nothing measurable.
F14 established decode is bandwidth-bound. Whatever the barrier does differently,
decode at 8 threads is the workload least able to show it.

**Prediction: |difference| < 3% on tg256, and there is a real chance
`ab_throughput.py` declines to certify it at all** — which would be the sixth
time this project's harness has said no, and consistent with every other attempt
to move decode by changing how threads wait.

### P27.4 — the gap grows with thread count, and grows more on a small model

The interesting prediction, and the one that follows from mechanism 3 rather
than from the barrier.

Mechanisms 2 and 3 are **fixed costs per graph**; mechanism 3 scales with
`n_threads` on top of that. A graph is one decode token. So the OpenMP path's
extra cost per token is roughly `n_threads * (one SetThreadInformation) +
(one fork/join)`, independent of how much arithmetic the token needs.

Two consequences, both testable here:

- **28 threads shows a larger gap than 8**, in favour of the non-OpenMP build.
- **`tiny.gguf` shows a larger relative gap than `mid.gguf`**, because the same
  fixed microseconds sit on top of a much shorter token. `tiny.gguf` is 34 MB
  and 8 layers; if a token there costs ~1 ms and 28 syscalls cost ~2 us each,
  that is ~5% — an effect a whole order of magnitude above anything the barrier
  is expected to do.

**Prediction: measured as (noomp tok/s / omp tok/s), the ratio is ordered
`tiny@28 > tiny@8 >= mid@28 > mid@8`, and at least the first and last differ by
more than their intervals.** If that ordering holds it is a statement about
per-graph fixed cost, not about barriers, and the barrier question is then
answered by P27.1 and P27.2 alone.

### What would make this uninteresting

Everything within noise everywhere. That is a real possibility at 8 threads on
`mid.gguf` and it is why the design includes 28 threads and `tiny.gguf` — a
null result at one operating point is a claim about that point, which is the
lesson F14 cost.

### What it would change if P27.2 holds and P27.4 does not

The standing caveat gets a number. "Barrier cost figures may not transfer" would
become "barrier cost figures shift by roughly X% between two implementations on
one machine, so treat F9's split as accurate to that", which is a caveat a
reader can act on instead of one they can only worry about.

---

## F27 — ggml's own barrier is not a cheaper spin-wait than OpenMP's. On this machine at 28 threads it costs 54% of decode

**Workload:** `mid.gguf` (24L F32, 220 M) and `tiny.gguf` (8L, 34 MB), 8 / 16 /
28 threads, MSVC Release, Windows 11, i7-14700HX. Throughput from the two
**uninstrumented** builds via `tools/ab_throughput.py`, 20 interleaved rounds
per arm with the lead flipping each round. Trace-derived barrier quantities from
the two instrumented builds via `tools/imbalance_repeat.py`, **n=12** per arm at
8 and 16 threads and n=8 at 28. Predictions in
[`P27`](#p27--two-barrier-implementations-on-one-machine-predictions), committed
before the builds existed.

[`F26`](#f26--every-measurement-in-this-project-was-taken-on-the-openmp-path-and-five-documents-said-the-opposite)
made this measurable: the spin-wait path had never been run here, and
`-DGGML_OPENMP=OFF` is one flag.

### The arms are what they claim to be

Checked before measuring, because both checks could have come out wrong:

- `llama-bench.exe` in `build-ts-noomp-on` and `build-ts-noomp-off` imports **no
  `VCOMP` symbols**; the OpenMP control still imports `VCOMP140.DLL`.
- The instrumented no-OpenMP build writes `"threading":"ggml-threadpool"` in its
  provenance record, against `"threading":"openmp"` from the other. That field
  exists because of F26 and this is the first thing it has distinguished.
- The two throughput binaries are byte-different (SHA-256) despite being the
  same size, and neither prints llama.cpp's "cplan requested more threads than
  available" warning at `-t 28`.

### Throughput: `GGML_OPENMP=OFF`, relative to the default ON

```
  model      threads   decode (tg64)                     prefill (pp64)
  mid.gguf         8    -2.06%  [-2.42, -1.56]  cert.     -1.15%  [-1.62, +0.12]
  mid.gguf        28   -54.17%  [-55.05, -53.31] cert.   -45.02%  [-47.94, -43.03] cert.
  tiny.gguf        8   -13.07%  [-15.35, -10.07] cert.     -0.77%  [-9.55,  +8.60]
  tiny.gguf       28   -78.64%  [-80.40, -77.60] cert.   -38.60%  [-50.26, -34.81] cert.
```

At 28 threads on `mid.gguf`, decode goes from **39.09 tok/s to 17.91 tok/s**.
On `tiny.gguf` it goes from 921 to 197.

### Why: the barrier, measured directly

Total barrier wait per unit node work, from level-3 traces:

```
                       OpenMP                   ggml threadpool
   8 threads     0.019 [0.018, 0.026]      0.037 [0.035, 0.042]     release only
  16 threads     0.028 [0.025, 0.030]      0.136 [0.116, 0.152]     release only
  28 threads     0.249 [0.184, 0.380]      0.752 [0.508, 1.145]     wait total
```

The 8- and 16-thread rows are **release latency** — the part of the wait after
the last thread has arrived, which is exactly the part a barrier implementation
controls. Both comparisons have **non-overlapping ranges**, and so does the
28-thread total: 0.752 against 0.249, with the ranges clearing each other
(0.508 against 0.380).

At 28 threads the ggml barrier costs **0.75 units of wait per unit of work** —
three quarters as much time waiting as computing. That is the throughput
collapse, in the quantity that causes it.

### Scoring the predictions

**P27.2 — falsified, with the sign inverted.** I predicted median release
latency would be *lower* on the spin-wait path by at least 20%, reasoning that a
thread polling a relaxed atomic in a `ggml_thread_cpu_relax()` loop leaves within
tens of nanoseconds of the last arrival, while `_vcomp_barrier` is a library
call into a 2002-vintage runtime. It is **higher**: 1.95x at 8 threads, 4.9x at
16, both with non-overlapping ranges. The prediction was wrong in direction, in
size, and in mechanism.

What the reasoning missed, stated after the fact and not before: the release of
a spin-wait barrier is a **cache-coherence broadcast**. The last thread's
`fetch_add` on `n_barrier_passed` has to reach *n-1* cores that are all hammering
that one line with `pause` loops, on a CPU whose P-cores and E-cores sit in
different cache clusters. "Leaves within tens of nanoseconds" describes one
waiter. It does not describe twenty-seven of them contending for the same line,
and the contention slows the arriving thread as well as the waiting ones.

**P27.3 — held.** Predicted `|difference| < 3%` on decode at 8 threads on
`mid.gguf`; measured **-2.06%**. I also predicted a real chance the harness
would decline to certify it. It certified — the interval is
[-2.42, -1.56], comfortably clear of zero.

**P27.1 — held at 8 threads, and its premise is wrong.** Arrival imbalance per
unit work, whole-trace: 0.106 [0.050, 0.151] against 0.115 [0.052, 0.167] at 8
threads, thoroughly overlapping as predicted. At 16 threads: 0.218 [0.164,
0.281] against 0.281 [0.241, 0.320] — still overlapping, so not falsified, but
the medians differ by **29%** in a consistent direction.

The prediction rested on "imbalance is set by how work is divided, and the
division is identical source in both builds". That is not sufficient.
**Imbalance is measured between arrival timestamps, and when a thread arrives at
barrier *k+1* depends on when it was released from barrier *k*.** A barrier with
a slower, more skewed release feeds that skew forward into the next node's
arrival spread. The control was not as clean as its argument claimed, and the
28-thread numbers say so more loudly than the 16-thread ones.

**P27.4 — direction right twice, ordering wrong, mechanism wrong.** Predicted
the gap grows with thread count and grows more on a small model, ordered
`tiny@28 > tiny@8 >= mid@28 > mid@8`. Both growth claims hold enormously. The
ordering does not: measured `tiny@28 (78.6%) > mid@28 (54.2%) > tiny@8 (13.1%)
> mid@8 (2.1%)`. **Thread count dominates model size**, and my ordering had them
interleaved.

More importantly the *mechanism* was wrong. P27.4 attributed the growth to two
fixed per-graph costs — the `_vcomp_fork` per graph, and
`ggml_thread_apply_priority` calling `SetThreadInformation` once per thread per
graph inside the parallel region. Those are microseconds. They cannot produce
-78%, and they point the wrong way anyway: **both of those costs are paid by
the OpenMP arm**, which is the *faster* one. I found two real asymmetries, wrote
them down in advance, and they were noise against the thing I had reasoned
myself out of.

### The design flaw, which the tool caught

`ab_throughput.py` prints "A certified control workload means the comparison is
wrong, not that the change is good" — and at 28 threads the prefill control
certified in both models.

The tool is right to complain and the complaint does not apply. That warning
exists for F24, where the change was a threshold inside `mul_mat` that prefill
sat above in both arms, so prefill genuinely could not move. `GGML_OPENMP` is
not a targeted change: it swaps the barrier every node in every graph, prefill
included. **This comparison has no control**, and no arrangement of these two
binaries provides one. That is a real weakness and it is why the barrier
measurement above matters more than the throughput table — the throughput says
something changed by a lot, and only the trace says it was the barrier.

### What this changes

- **F26's caveat now has a size, and it points the other way.** This project's
  barrier-cost figures — F6's 11.2%, F9's split, F10's rise to 22.9% — are all
  from the **cheaper** of the two implementations available on this machine. A
  reader who assumed they were the pessimistic case had it backwards.
- **`GGML_OPENMP` is a much larger option than its one-line description
  suggests.** It reads as a build convenience. On this machine, turning it off
  costs 2% at 8 threads and **more than half of decode at 28**, and nothing in
  the build output says so.
- **It does not affect anyone using defaults.** `GGML_OPENMP` is ON by default
  on every platform and CMake found OpenMP here without help. This is a finding
  about a configuration people can select, not about what llama.cpp ships.
- **It does not transfer to Linux without testing**, and that cuts both ways:
  `libgomp`'s barrier is not `vcomp`'s, and glibc's threading is not Windows'.
  The one thing that does transfer is the *question* — anyone comparing the two
  paths now has a protocol and a reason to run it.

### Caveats

- **One machine, one OS, one compiler.** i7-14700HX, 8 P + 12 E, Windows 11,
  MSVC 19.44. A hybrid CPU is close to the worst case for a spin barrier and
  close to the best case for a runtime that parks threads; a homogeneous server
  part could easily reverse the sign.
- **No control workload exists for this comparison**, as above.
- Two dense models, both small. The 28-thread rows put 28 spinning threads on 28
  logical CPUs, which is total subscription — the regime where a spin barrier is
  most exposed, and the regime `llama-bench -t $(nproc)` puts people in.
- The trace-derived rows at 8 and 16 threads measure *release latency*; the
  28-thread row measures *total* barrier wait, because at 28 threads the
  release/imbalance split is itself affected by the release skew feeding forward
  (see P27.1 above), so the total is the more honest number.
- `ab_throughput.py` reports medians and bootstrap CIs over interleaved rounds.
  The trace rows report medians and **ranges**, which are not confidence
  intervals; the claim rests on non-overlap, as in F23.

---

## P28 — The "thread-pool spin-up" barrier is probably tokenscope's own allocator

**Dated 2026-09-08, session 5. Written and committed before the test.** Found
while extending `imbalance_repeat.py` to report release latency for
[`P27`](#p27--two-barrier-implementations-on-one-machine-predictions), which
meant looking at what release latency actually contains.

### The observation

`--barriers` on the committed reference trace `mid-24L-L3-tok10-11` says:

```
  total barrier wait      62.31 ms   thread-time
    arrival imbalance     34.59 ms    55.5%
    after last arrival    27.72 ms    44.5%   release latency and spin-up

  A single barrier accounts for    20.98 ms of the after-arrival time: token 10,
  before "embd". On the first traced token that is thread-pool spin-up,
  not a property of the graph.
```

One barrier out of 824 is **76% of all after-arrival time in the trace**. The
tool already flags it and excludes it from the headline split, which is good
practice — and then explains it with a sentence nobody has ever tested.

### Why the stated explanation does not survive reading

"Thread-pool spin-up" would be a cost paid when the pool is cold. This trace
captures **tokens 10 and 11**, which is roughly the twelfth graph of the
process: two prefill batches and ten decode steps have already run. The pool
was warm long before the barrier in question, and there is no mechanism that
makes graph number twelve special.

What *is* special about token 10 is that it is the first token inside
`TOKENSCOPE_TOKENS=10:11` — the first token where `ts_g_capture` is true.

### The mechanism this predicts instead

At level 3 a worker thread's buffer is allocated **lazily, on its first
record**. `ts_reserve` returns early while `ts_g_capture` is false, so no
worker touches `ts_thread_init()` until capture opens. On the first captured
token, all eight workers then call it at once: each takes the registry mutex,
allocates a 1 MiB chunk, and first-touches 256 pages.

And that cost lands **in the barrier**, not in the node, because of how the
two-clock-read optimisation is arranged:

```
TS_NODE_WORK_END:  ts_t1 = ts_now();      <- clock read happens FIRST
                   ts_emit(...)           <- ts_thread_init() happens HERE
                   ts_t_mark = ts_t1;
ggml_barrier(...);
TS_NODE_WAIT_END:  ts_t1' = ts_now();
                   wait = ts_t1' - ts_t_mark
```

The allocation happens after `ts_t1` is read and before `ts_t1'`, so the node's
recorded work *excludes* it and the following barrier's recorded wait
*includes* it. Every thread pays it once, on the same node, in the same
barrier. That is exactly the shape observed: one barrier, first captured token,
before the first node of the graph.

If this is right it is a **measurement artifact produced by the instrument**,
sitting in the quantity the instrument exists to measure, wearing a label that
blames the thing being measured. That is the worst category of profiler bug,
and this project has an entry for its own version of it already: F16, where a
scope's name was a claim about what it wrapped and the claim was wrong.

### P28.1 — the spike does not decay with pool age

Level 3, `mid.gguf`, 8 threads, two capture windows: `1:2` (pool almost cold)
and `40:41` (pool warm through forty graphs). n=8 per arm, because F23.

- **Thread-pool spin-up predicts** the `40:41` spike is much smaller than the
  `1:2` spike — a warm pool has nothing to spin up.
- **First-touch allocation predicts** the two are the same size within their
  ranges, because the allocation is paid once wherever the window opens.

**Prediction: they are the same size, and the medians differ by less than 2x
where the spin-up story needs an order of magnitude.**

### P28.2 — the spike scales with thread count, not with graph size

Each thread allocates one 1 MiB chunk. Total artifact time should be roughly
linear in the number of threads and independent of the model.

**Prediction: at 16 threads the artifact's total thread-time is 1.5x to 2.5x
its value at 8, on the same model and window.** Not exactly 2x, because the
mutex serialises the allocations and page-faulting is not perfectly parallel
anyway.

### P28.3 — pre-touching the buffer removes it

The fix, if the diagnosis holds, is one line in the right place:
`TS_THREAD_PREPARE` already runs once per thread per graph, **outside** the
node loop, and already exists to keep level 2 from allocating mid-graph. It is
the same problem and the same answer. Calling `ts_buffer_get()` there — at every
level, not only level 2, and regardless of whether the current token is inside
the capture window — moves the allocation to the first graph of the run, where
nothing is being measured.

**Prediction: with that change, the largest single after-arrival barrier in a
`10:11` trace drops by more than 80%, and the trace-wide imbalance/release split
moves from roughly 55/45 to something near the 84/16 the tool currently reports
only after manually excluding the spike.**

Falsified if the spike survives, which would mean the allocation is not what is
being timed and the tool's original sentence deserves more credit than this
prediction gives it.

### What it changes if all three hold

Every level-3 trace this project has taken has a corrupted first captured
token, and `--barriers` has been printing an explanation that points at ggml
for something tokenscope did. The headline number F6 quotes — 11.2% of worker
thread time is barrier wait — is computed over a whole run, so the artifact is
diluted there; the ones at risk are the narrow-window level-3 traces, which is
most of what sessions 3 and 4 used. **The imbalance figures are unaffected**:
imbalance is measured between arrival timestamps, and the artifact is entirely
in the after-arrival term. F23 and F24 rest on imbalance, so they should
survive intact. That prediction is part of this one.

---

## F28 — The barrier this profiler blamed on ggml was the profiler's own allocator, and it was 76-83% of all release latency

**Workload:** `mid.gguf`, level 3, 8 and 16 threads, two capture windows (`1:2`
and `40:41`), **n=8 runs per cell**, `tools/spinup_probe.py`. Then the same
sixteen cells again after a four-line change. Predictions in
[`P28`](#p28--the-thread-pool-spin-up-barrier-is-probably-tokenscopes-own-allocator),
committed before either run.

### Before

```
                     biggest single barrier      trace after-arrival   imbalance
  t=8   window 1:2         17.899 ms  (75.1%)          23.682 ms       28.527 ms
  t=8   window 40:41       21.372 ms  (78.0%)          27.533 ms       48.191 ms
  t=16  window 1:2         80.133 ms  (81.8%)          97.615 ms      139.163 ms
  t=16  window 40:41       84.572 ms  (83.4%)         101.810 ms      135.316 ms
```

In **32 of 32 runs** that barrier was the one before `embd` — the first node of
the graph — on the **first token of the capture window**, wherever the window
was put. Medians of 8; the spike's own run-to-run spread is 1.0-1.2x, which for
this project is remarkably tight and is itself a clue: a mechanism this
repeatable is not contention with the rest of the machine.

### After

Four lines: `ts_thread_prepare()` now calls `ts_buffer_get()` on entry, at
every level from 2 up and regardless of the capture window. It already ran
once per thread per graph, outside the node loop, and already existed to stop
level 2 allocating mid-graph. It was the right place for this too.

```
                     biggest single barrier      trace after-arrival   imbalance
  t=8   window 1:2          0.085 ms  ( 1.5%)           5.567 ms       30.051 ms
  t=8   window 40:41        0.134 ms  ( 2.3%)           5.982 ms       32.346 ms
  t=16  window 1:2          0.296 ms  ( 1.6%)          17.408 ms      144.126 ms
  t=16  window 40:41        0.322 ms  ( 1.9%)          16.949 ms      139.597 ms
```

The spike is gone — **160x to 271x smaller**, and no longer attached to any
particular node or token: across the eight post-fix cells the largest barrier
lands on `ffn_up-1`, `ffn_out-13`, `l_out-22`, `Qcur-10`, `node_579` and others,
one run each. That is what noise looks like, and it is the strongest evidence
in the whole finding: the *pattern* dissolved, not just the magnitude.

Total after-arrival time fell **76% to 83%**. Arrival imbalance did not move:
three of the four cells changed by 3-5%, against a run-to-run spread of
2.0-2.8x for that quantity.

### The mechanism, confirmed

At level 3 a worker's buffer was allocated lazily on its first record.
`ts_reserve` returns early while `ts_g_capture` is false, so no worker reached
`ts_thread_init()` until the capture window opened — and then all of them did
at once, each taking the registry mutex, allocating 1 MiB and pre-touching 256
pages (`chunk::chunk` uses `resize`, not `reserve`, deliberately).

It landed in the barrier rather than the node because of the two-clock-read
optimisation:

```
TS_NODE_WORK_END:  ts_t1 = ts_now();      <- clock read happens FIRST
                   ts_emit(...)           <- ts_thread_init() happens HERE
                   ts_t_mark = ts_t1;
ggml_barrier(...);
TS_NODE_WAIT_END:  wait = ts_now() - ts_t_mark   <- so the allocation is in here
```

The node's recorded work excludes it; the following barrier's recorded wait
includes it. Every thread, once, on the same node, in the same barrier.

### Scoring the predictions

**P28.1 — held, decisively.** Thread-pool spin-up needed the `40:41` spike to
be much smaller than the `1:2` one. It was **larger**: 21.372 against 17.899 at
8 threads (1.19x), 84.572 against 80.133 at 16 (1.06x). The prediction allowed
2x in either direction and the effect came in at 1.1x — a warm pool of forty
graphs makes no difference, because the pool was never what was being timed.

**P28.2 — failed, and the failure is the more useful half.** Predicted 1.5x to
2.5x going from 8 to 16 threads, reasoning that the mutex serialises the
allocations so the total could not double. Measured **4.48x** and **3.96x** —
which is 2², not 2, and not 2^0.7.

The error is a category error, and a clean one. The mutex *does* serialise, so
the stall's **wall-clock** duration grows about linearly with thread count. But
the quantity being measured is **thread time**: every one of the *n* threads
sits in the barrier for that whole stall. Linear duration, summed over a linear
number of threads, is **quadratic**. I predicted the scaling of a wall-clock
quantity for a measurement denominated in thread-time, and the sub-linear
argument I was so pleased with was answering a different question.

*Predict the scaling of the quantity you are actually going to read.* Every
number in `--barriers` is thread time; the project has known that since F1 and
still made this mistake.

**P28.3 — held, including the number.** Predicted "more than 80%" off the
biggest barrier (measured 99.5-99.6%) and that the trace-wide split would move
"from roughly 55/45 to something near the 84/16 the tool currently reports only
after manually excluding the spike". The new reference trace measures
**84.6% imbalance / 15.4% release latency**. The old trace with its spike
manually excluded said 83.7/16.3. Two routes to the same split is the
cross-check that makes the diagnosis a mechanism rather than a story.

### What this invalidates, precisely

- **The 55.5% / 44.5% split quoted in the README and F9 is wrong.** The real
  steady-state split on this workload is **84.6 / 15.4**. Barrier wait is much
  more dominated by threads waiting for each other, and much less by release
  latency, than this project has been saying.
- **F6's 11.2%** — barrier wait as a share of worker thread time — **survives.**
  It comes from a whole-run trace with no capture window, so the artifact is
  paid once over 257 tokens rather than once over two. The new reference trace
  reads 11.5% on the same model.
- **F23 and F24 survive**, as P28 predicted they would. Both rest on *arrival
  imbalance*, which is computed between arrival timestamps and cannot contain
  an interval that ends before the first arrival. The measurement above
  confirms it rather than assuming it: imbalance moved 3-5% across the fix.
- **`examples/mid-24L-L3-tok10-11.trace.json` is kept** rather than regenerated,
  because F9, F23 and CI all reference it and silently swapping the data under a
  finding is worse than carrying an old file. `mid-24L-L3-tok10-11-f28.trace.json`
  is its post-fix counterpart — same model, same window, same thread count,
  chosen as the **median of seven candidate runs by total imbalance**, not the
  prettiest.

### The tool was lying in a sentence, which is the part worth keeping

`--barriers` did flag the spike, did exclude it from the corrected split, and
then explained it: *"On the first traced token that is thread-pool spin-up, not
a property of the graph."* Everything up to the explanation was good practice.
The explanation was a guess written once, and it pointed at ggml for something
tokenscope did.

**An anomaly detector that also explains the anomaly has two outputs, and only
one of them was measured.** The detection was real; the attribution was prose.
This project already knew that a scope's *name* is a claim (F16) — a diagnostic
message is a claim too, printed in a more authoritative voice.

---

## F29 — `TOKENSCOPE_TOKENS=10:11` captured one token, and had done so since session 4

**Workload:** none. Found by reading the parser while writing
[`P28`](#p28--the-thread-pool-spin-up-barrier-is-probably-tokenscopes-own-allocator),
reproduced against the same C runtime the binaries use before anything was
changed.

The window parser was two `sscanf` calls in order:

```c
if (std::sscanf(win, "%u-%u", &lo, &hi) == 2) { r.tok_lo = lo; r.tok_hi = hi; }
else if (std::sscanf(win, "%u", &lo) == 1)    { r.tok_lo = lo; r.tok_hi = lo; }
```

`sscanf` stops at the first character the format does not match and returns how
many conversions it completed. It does not care what is left over. So for
`"10:11"` the first call fails at the literal `-`, returns 1, and the second
call succeeds with `lo = 10` — giving `tok_lo = tok_hi = 10`, a window of **one
token**, with no error anywhere.

Reproduced before fixing, through `msvcrt`'s own `sscanf`:

```
10:11    %u-%u -> n=1 lo=10 hi=0   |  %u -> n=1 lo=10
10-11    %u-%u -> n=2 lo=10 hi=11  |  %u -> n=1 lo=10
```

### Where it mattered

`tools/imbalance_repeat.py` has had `--tokens` defaulting to `"10:11"` since it
was written in session 4, and every F23 and F24 imbalance run used that default.
Each of those runs captured **token 10 only**, not tokens 10 and 11.

The committed reference trace `mid-24L-L3-tok10-11.trace.json` is *not*
affected — it carries two captured tokens, so it was produced with the hyphen
form by hand. The colon form appears in exactly two places: that tool's default
and P24's protocol paragraph, which quotes the tool.

### What it does and does not invalidate

**F23 and F24 stand.** Both rest on *ratios between nodes inside one trace* —
`ffn_up` against `ffn_out`, patched against stock — and both arms of every
comparison used the identical window, so halving the capture halves both sides.
What changes is that "twelve runs of two tokens" was twelve runs of one, which
is half the data per run and makes the wide spreads F23 documented (up to 8.5x
between identical runs) less surprising than they looked.

It also means F23's ranges were, if anything, pessimistic: the same tool now
gathers twice the barriers per run for the same wall time.

### The fix

`ts_parse_token_window()` is now a separate exported function that parses
`N`, `N-M` and `N:M` and **refuses everything else**, including trailing
characters, reversed bounds, and spaces around the separator. `ts_init_from_env`
prints a warning and captures the whole run when it refuses, because a typo
that silently narrows the capture window is worse than one that ignores it.

The self-test drives it directly with nine cases, five of which must be
refused. That is the shape this project keeps arriving at: the assertions worth
writing are the ones about inputs the code should *reject*, since the accepted
ones tend to be the ones somebody already tried by hand.

### Why nobody noticed

A trace with one captured token instead of two looks completely normal. It has
tokens, it has barriers, it has nodes, every percentage is well-formed, and the
only visible difference is that a number in a report reads `1 of 25` instead of
`2 of 25` in a line most readers skim. There is no failure to see.

This is the same shape as the dead category-table entry in F20 and the
`kv.slot-search` scope in F16: **a silent narrowing produces valid-looking
output, so it is only ever found by reading the code that produced it, never by
looking at the result.** Three instances now, which is enough to call it the
project's characteristic bug rather than three unlucky ones.

---

## P30 — What the interleaved four-arm run should say, written before it runs

**Dated 2026-09-08, session 6. Written and committed before the harness was
pointed at the real builds.** [`F25`](#f25--the-static-overhead-certifies-for-the-first-time-since-session-1-and-the-shared-builds-noise-floor-ate-its-own-answer)
ended with a design fault rather than a wrong number: P25.1 asked whether the
shared build's overhead exceeds the static build's, and the run that was
supposed to answer it put the two halves of the comparison in two invocations
thirteen minutes apart. `bench_overhead.py` now takes N build pairs and
round-robins every arm of every pair together, so the question is askable for
the first time.

These predictions are the point of writing them down: P25.1 was *unanswerable*,
which F25 called a worse outcome than being wrong, because a wrong prediction
teaches something. This one should at least be capable of being wrong.

### The mechanism, which is narrower than "DLLs are slower"

The hot path is `TS_SINLINE` in both builds and inlines identically. `ts_now()`
is `__rdtsc()`. `ts_tls` is a `__declspec(thread)` pointer that each module
caches for itself — that is exactly what F22 built, and it means the buffer
lookup costs the same either way.

What does *not* inline across a DLL boundary is the four globals. In
`tokenscope.h` they are

```c
TS_API extern int      ts_g_level;
TS_API extern int      ts_g_capture;
TS_API extern uint32_t ts_g_token;
TS_API extern uint16_t ts_g_graph;
```

and `TS_API` is `__declspec(dllimport)` in every consuming module. A read of an
imported global is **two dependent loads** — fetch the address from the import
table, then the value — where the static build has one load from a fixed
address, usually folded into the instruction that uses it.

Count them per event: `ts_g_capture` in `ts_reserve`, `ts_g_token` and
`ts_g_graph` in `ts_emit`, and `ts_g_level` on every scope entry and exit. So
roughly four extra dependent loads per record, all from one import-table cache
line that is about as hot as memory gets, and several of them hoistable out of
a loop by the optimiser.

That gives a **direction with confidence and a magnitude with none**: shared
should cost at least as much as static, by a margin small enough that this
machine will probably not resolve it.

### The predictions

**P30.1 — the difference of overheads at level 3 is positive but its interval
spans zero.** Shared minus static, on decode, in percentage points: positive
point estimate, interval containing 0. The mechanism above is real but it is
four L1 hits against a 24-byte record write and an `__rdtsc`, and F25's static
level-3 overhead was only +1.16% to begin with. A margin that is a fraction of
that is below what a 1% noise floor resolves. **Predicted point estimate under
+0.5pp.**

**P30.2 — level 0 stays unresolvable in both builds, and its difference too.**
Arm B is the residual-branch test, and under `dllimport` the residual branch is
the *most* affected thing per unit work — it is a bare `ts_g_level` read with no
record write to hide behind. If any arm shows the linkage cost, this is the one.
It will still span zero, because the branch is predicted and the whole arm was
+0.23% [-0.20, +0.92] statically.

**P30.3 — the compiled-out arms differ, and this is the number that comes out
sharpest.** A_shared against A_static contains no tokenscope code at all; it is
llama.cpp losing cross-module inlining across `ggml.dll`, `ggml-cpu.dll` and
`llama.dll`, which is a far larger surface than four imported globals. F25 saw
0.7% on decode and 0.9% on prefill and could only call it "an order of magnitude
and nothing more", because it was a between-run comparison. Interleaved, I
expect **0.5% to 1.5% on decode with an interval excluding zero** — the first
properly-measured statement about DLL cost in this repo, and note that it is a
fact about llama.cpp rather than about this profiler.

**P30.4 — prefill shows a larger DLL penalty than decode.** Decode is
bandwidth-bound (F14) and prefill is compute-bound, so lost inlining should
matter more where the CPU is the constraint. F25's aside had it the same way
round, 0.9% against 0.7%, from a comparison too weak to lean on.

**P30.5 — at least one block is refused.** This has happened in every session
since session 1 and the harness has now refused seven times. Six arms a round
makes each round longer than F25's five, so the run is longer and there is more
of the day for the machine to drift through.

### What would falsify the mechanism rather than the numbers

If the shared build comes out **faster** at level 3 with an interval excluding
zero, the `dllimport` story is wrong and something else is going on — most
likely that the two builds' compilers made different inlining decisions
somewhere off the tokenscope path, which would make the whole four-arm design
measure a confound instead of linkage. The static pair is `ninja`/`cl` and the
shared pair is the Visual Studio generator; both are MSVC 19.44 Release, but
they are not the same command line, and that is a caveat this design cannot
remove.

---

## F30 — The shared build's overhead certifies at +0.87%, the static build's does not, and F25's DLL penalty was noise

> **Corrected by [`F31`](#f31--two-certified-intervals-for-one-quantity-that-do-not-overlap-and-level-3-turns-out-to-be-a-leveller), later the same session.
> The headline `+0.87% [+0.55, +1.19]` is over-precise.** F31 re-measured the
> same quantity on the same unrebuilt binaries about two hours later and got
> `+0.31% [+0.01, +0.52]` — an interval that does not overlap this one, from
> the same harness, both "certified". The `+0.36pp` difference of overheads
> below re-measured at `-0.85pp [-1.16, -0.45]`, outside its own interval and
> excluding zero on the other side.
>
> What survives: the shared build's overhead is **measured** and is of the same
> order as the static build's, somewhere around 0.3-1.2% at level 3 on this
> machine; the retirement of F25's DLL penalty (F31 finds the compiled-out
> arms differing by well under 1% too); and every methodological point below.
> What does not survive is the precision. The bootstrap interval is a
> *within-invocation* interval and this project had not previously tested it
> across invocations. Read F31 before quoting any number from here.


**Workload:** `mid.gguf` (24L, F32, 220 M), 8 threads, pp512 / tg256, MSVC
19.44 Release, `-n 20` interleaved with the first rep discarded, **six arms
across two build pairs in one invocation**, 899 s, `tools/bench_overhead.py`.
All four binaries rebuilt in this session before the run.
[`P30`](#p30--what-the-interleaved-four-arm-run-should-say-written-before-it-runs)
holds the predictions, committed before any output was read.

This closes section 5 item 7 and the gap-table row that has said "the shared
build's overhead has never been measured" since session 4.

### The result

**Decode.** Every baseline IQR under 2%, so the harness answered everything it
was asked — the first time in six sessions that nothing was refused.

```
  arm                               median tok/s     IQR   overhead vs A
  A: compiled out [static]                 45.92    1.4%                  -
  B: in, level 0 [static]                  46.01    1.5%    -0.19%  [-0.84, +0.93]
  C3: active level 3 [static]              45.69    0.7%    +0.50%  [-0.23, +1.05]

  A: compiled out [shared]                 46.03    0.3%                  -
  B: in, level 0 [shared]                  46.06    0.8%    -0.06%  [-0.31, +0.61]
  C3: active level 3 [shared]              45.63    0.6%    +0.87%  [+0.55, +1.19]
```

**The shared build's level-3 overhead is +0.87% [+0.55, +1.19], certified.**
That is the number the gap table has been waiting for since F18 broke the shared
build and F22 fixed it, and it says the fix costs what the static one costs: the
per-module `ts_tls` cache does not put a cross-DLL call on the hot path, which
is exactly what F22 claimed by inspection and never demonstrated.

**And the static build's level-3 overhead did not certify: +0.50% [-0.23,
+1.05].** The roles are the reverse of F25's, in the same invocation. Nothing
about the static build got noisier in any absolute sense — its baseline IQR was
1.40% against the shared pair's 0.28%, and section "the confound this design
still has" below says where that asymmetry comes from.

### P25.1, answered at last — as a bound, not a value

```
  overhead difference vs pair [static], percentage points
  B:  shared - static                 +0.13pp  [-1.02, +1.07]
  C3: shared - static                 +0.36pp  [-0.32, +1.17]
```

F25 asked whether the shared build's overhead exceeds the static build's and
could not answer, because the two halves of the comparison were separate
invocations. Interleaved, the answer is **+0.36pp [-0.32, +1.17]** — still not
resolved away from zero, but for the first time it is *bounded*. The shared
build does not cost dramatically more to instrument; whatever it costs is under
about 1.2 percentage points, against a static overhead of the same order.

That is a weaker claim than "shared costs X" and a much stronger one than F25's
"nothing was resolved in either direction". A bound is a result. It is also the
right shape of answer for the mechanism P30 predicted from: four `dllimport`
globals, an extra dependent load each, all from one hot import-table line, is
not a thing that should show up at 1% resolution.

### F25's DLL penalty does not survive being measured properly

F25 offered one observation "weakly", from between-run data: the compiled-out
shared build ran 0.7% slower on decode and 0.9% on prefill than the compiled-out
static one, and it said to "take it as an order of magnitude and nothing more."

Interleaved, both arms in the same rounds:

```
  compiled-out arms only
  decode    A: shared vs static      -0.22%   [-0.79, +0.26]
  prefill   A: shared vs static      -0.23%   [-0.49, +0.14]
```

**Consistent with zero, and the point estimates have the opposite sign.** The
0.7%/0.9% was between-run noise. F25 was right to hedge it and the hedge was
not excessive caution — the number was wrong in direction, not just in
magnitude.

What replaces it is a real if unglamorous result: **on this workload,
llama.cpp's shared build costs nothing measurable against its static build**,
to a resolution of about half a percent. Given that `BUILD_SHARED_LIBS=ON`
gives up cross-module inlining across `ggml.dll`, `ggml-cpu.dll` and
`llama.dll` entirely, that is worth knowing, and it is a fact about llama.cpp
rather than about this profiler.

### Scoring the predictions

| | claim | outcome |
|---|---|---|
| **P30.1** | C3 difference positive, spans zero, point under +0.5pp | **held, on all three specifics.** +0.36pp [-0.32, +1.17] |
| **P30.2** | level 0 unresolvable in both builds, and its difference too | **held.** -0.19% [-0.84, +0.93] static, -0.06% [-0.31, +0.61] shared, difference +0.13pp [-1.02, +1.07] |
| **P30.3** | compiled-out arms differ by 0.5-1.5% on decode, excluding zero | **failed, on both specifics.** -0.22% [-0.79, +0.26]: wrong sign and spans zero. This was the prediction with the most confidence behind it and the most reasoning — lost cross-module inlining across three DLLs is a far larger surface than four imported globals — and the reasoning was sound while the conclusion was wrong. Inlining across `ggml.dll`'s boundary evidently is not on any path that matters at 512-token prefill or 256-token decode |
| **P30.4** | prefill shows a larger DLL penalty than decode | **not testable.** It presupposed P30.3. Decode -0.22% and prefill -0.23% are indistinguishable and both span zero; there is no penalty in either to compare |
| **P30.5** | at least one block refused | **failed.** Baseline IQRs 0.28%, 0.38%, 0.65%, 1.40% — all four under 2%, no refusal anywhere. Six sessions, seven previous refusals, and the quietest machine this project has seen |

Two of five held, one failed, one failed loudly, one was unanswerable for the
same structural reason F25 hit — a prediction that depends on another
prediction cannot be scored when the first one fails.

### The confound this design still has

Arm order within a round is **fixed**: `A_static, B_static, C3_static,
A_shared, B_shared, C3_shared`, every round. Any transient shorter than a round
therefore lands on the same arms every time, which is a different failure mode
from the one interleaving was built to prevent.

It is visible in the raw data. In rep 2 all three static arms read ~43.0 while
all three shared arms read ~45.7 — one extra warm-up round, absorbed entirely
by the arms that run first. That single rep is most of why the static pair's
baseline IQR is 1.40% against the shared pair's 0.28%, and it is why the static
overhead lost its certification while the shared one kept it.

It does not change any conclusion here. Discarding a second warm-up rep:

| | as run, n=19 | 2 reps discarded, n=18 |
|---|---|---|
| C3 static | +0.50% [-0.23, +1.05] | +0.56% [-0.13, +1.02] |
| C3 shared | **+0.87% [+0.55, +1.19]** | **+0.89% [+0.55, +1.24]** |
| C3 difference | +0.36pp [-0.32, +1.17] | +0.33pp [-0.28, +1.08] |
| A shared vs static | -0.22% [-0.79, +0.26] | -0.08% [-0.73, +0.25] |
| baseline IQR, static | 1.40% | 0.96% |

Every conclusion survives, which is why the headline numbers are the ones
actually produced by the committed protocol rather than the reprocessed ones.
But the fix is obvious and unimplemented: **rotate the arm order each round**,
so position within a round is not confounded with arm. Session 1's lesson was
"interleave the arms", F25's was "the arms are whatever you are comparing", and
this one is **an interleave with a fixed order is a Latin square with one row**.

Rep 13 is the contrasting case and the reassuring one: all six arms dip
together, which is drift landing on everything equally, which is what the
round-robin is for.

### The between-session drift, which is larger than everything above

The compiled-out static arm read **42.94 tok/s in F25 and 45.92 tok/s here** —
**+6.9%**, on the same model, thread count, workload and machine. That arm is
built with `GGML_TOKENSCOPE=OFF` and contains no tokenscope code at all;
neither F26 nor F28 changed anything that survives the compile-out, so this is
very nearly the same code measured twice.

**That between-session difference is roughly six times the largest effect any
of these arms is trying to resolve.** Every number in this finding is a
within-invocation comparison and is unaffected. Every cross-session comparison
of absolute throughput in this repo is worth very little, and the +1.16% of F25
and the +0.50% here cannot be compared to each other to argue that F28 made the
instrumentation cheaper — they are two different days, and the days differ by
7%.

### Caveats

- The static pair is built by Ninja and the shared pair by the Visual Studio
  generator, as they have been since session 4. Both are MSVC 19.44 Release;
  they are not the same command line. Within each pair the generator is shared,
  so each pair's own overhead figure is clean — the cross-pair comparisons carry
  this difference in addition to linkage, and P30 named it in advance as the
  thing that could make the design measure a confound instead.
- One machine, one model, one thread count, one workload. F10 predicts overhead
  grows with thread count and only 8 has ever been measured.
- `+0.87%` describes level 3, the most expensive level, on a 220 M F32 model.
  A model with more arithmetic per node dilutes it.
- The shared arms load `ggml.dll`, `ggml-cpu.dll`, `ggml-base.dll` and
  `llama.dll` from a build tree; the static arm is one executable. Process
  start-up differs, and `llama-bench` reports steady-state throughput, so this
  should not enter the numbers — but it is the kind of thing that has caught
  this project before.

---

## P31 — Separating linkage from generator, which F30 could not

**Dated 2026-09-08, session 6. Written and committed before the run.**
[`F30`](#f30--the-shared-builds-overhead-certifies-at-087-the-static-builds-does-not-and-f25s-dll-penalty-was-noise)
answered section 5 item 7 and left one caveat that
[`P30`](#p30--what-the-interleaved-four-arm-run-should-say-written-before-it-runs)
had named in advance: the static pair is built by Ninja and the shared pair by
the Visual Studio generator, so **every cross-pair number in F30 has linkage and
generator in it together**. F25 stated the same caveat and neither session could
do anything about it, because with one pair per invocation there was no way to
hold one of the two constant.

With N pairs there is. `build-ts-nshared-on` and `build-ts-nshared-off` are new:
`BUILD_SHARED_LIBS=ON` under **Ninja**, otherwise identical to the VS shared
pair. Three pairs in one round-robin then give two clean contrasts and one
confounded one:

| contrast | holds constant | isolates |
|---|---|---|
| `nshared` vs `static` | generator (Ninja) | **linkage** |
| `shared` vs `nshared` | linkage (shared) | **generator** |
| `shared` vs `static` | nothing | what F30 measured |

### What the build files already say

The flags are equivalent, which is worth checking before predicting rather than
after. Ninja compiles `ggml-cpu.c` with `/O2 /Ob2 /arch:AVX2 -MD /DNDEBUG`; the
VS project sets `Optimization=MaxSpeed`, `InlineFunctionExpansion=AnySuitable`,
`EnableEnhancedInstructionSet=AdvancedVectorExtensions2`,
`RuntimeLibrary=MultiThreadedDLL` and `NDEBUG` — the same four things under
different names, from the same CMake configuration and the same `cl.exe`
19.44. So the *expectation* is no generator effect, and the reason to measure
is that "by inspection" is the phrase F22 got caught by.

### The predictions

**P31.1 — the pure linkage contrast is consistent with zero, like the
confounded one was.** `A_nshared` against `A_static` on decode: interval
containing zero, point estimate under 0.5% in absolute value. F30 already found
the confounded version at -0.22% [-0.79, +0.26], and for that to be hiding a
real linkage cost, a real generator effect would have to be cancelling it
almost exactly — possible, but it would be a coincidence, and predicting a
coincidence is not a prediction.

**P31.2 — the pure generator contrast is consistent with zero too, and this is
the one I would least like to be wrong about.** Same flags, same compiler, same
source. If this comes back non-zero and excluding zero, then MSBuild and Ninja
are not interchangeable on this project, which would put a footnote on **every
cross-generator number this repo has ever printed**, including F30's.

**P31.3 — the three level-3 overheads all land between +0.4% and +1.3%, and the
two shared pairs agree with each other more closely than either agrees with
static.** The mechanism from P30 says the shared builds pay four `dllimport`
loads per record that the static build does not; if that is real at all, the
two shared pairs should sit together and slightly above static. Point estimates
only — I do not expect the intervals to separate.

**P31.4 — F30's +0.36pp re-measures inside its own interval.** The
`shared - static` difference of overheads was +0.36pp [-0.32, +1.17]. A
re-measurement of the same quantity in a fresh invocation should land inside
that, and if it does not, F30's number was a one-run artifact. This is the only
prediction here that can falsify a published result.

**P31.5 — the baseline IQRs across the three pairs come out more similar than
F30's did.** F30's static pair ran at 1.40% and its shared pair at 0.28%,
almost entirely because one extra warm-up round hit whichever arms held the
first slots. Arm order now rotates (`9ec7c82`). If rotation works, the spread
of baseline IQRs across pairs should be visibly tighter than 5x. **This is a
test of the fix, not of the builds**, and it is the reason to state it as a
prediction rather than just look afterwards.

### What this cannot settle

Nine arms in a round is a round of roughly seven minutes, so a transient
shorter than one round but longer than one run still lands unevenly *within* a
round — rotation spreads that evenly across arms over the whole run, which is
weaker than eliminating it. And all three pairs are MSVC on one machine, so
"generator does not matter" would mean it does not matter *here*, which is the
standing limitation on everything in this repo.

---

## F31 — Two certified intervals for one quantity that do not overlap, and level 3 turns out to be a leveller

> **Two claims here are retracted by [`F36`](#f36--the-overhead-numbers-settled-every-build-under-1-linkage-and-generator-both-null-and-two-f31-claims-retracted).**
>
> **The generator effect is gone.** "+0.40% [+0.14, +0.69] slower under MSBuild
> than Ninja, same linkage" re-measures at **+0.07% [-0.40, +0.55]**. No effect.
>
> **"Level 3 is a leveller" does not reproduce, and it was the worse error.**
> The A/C3 spread ordering *reverses* on decode in F36 (A 0.16% vs C3 0.32%,
> against F31's A 0.58% vs C3 0.27%). Six numbers within half a percent of each
> other were read as structure. This document called that the durable,
> non-Tier-D part; it was Tier D all along, disguised by being a comparison of
> spreads rather than a percentage.
>
> What **stands** is the finding this document is named for: two bootstrap
> intervals for one quantity did not overlap, and that was real. F36's
> block-based interval for that quantity, `+0.77% [+0.05, +1.48]`, **contains
> both** of them — so the runs never disagreed, only their intervals did.


**Workload:** `mid.gguf` (24L, F32, 220 M), 8 threads, pp512 / tg256, MSVC
19.44 Release, `-n 20`, **nine arms across three build pairs in one
invocation**, rotating order, 1339 s, `tools/bench_overhead.py`.
[`P31`](#p31--separating-linkage-from-generator-which-f30-could-not) holds the
predictions, committed before the run.

Three pairs: `static` (Ninja, `BUILD_SHARED_LIBS=OFF`), `nshared` (Ninja,
`ON` — new), `shared` (Visual Studio, `ON`). `nshared` vs `static` isolates
linkage; `shared` vs `nshared` isolates generator.

**This finding contradicts [`F30`](#f30--the-shared-builds-overhead-certifies-at-087-the-static-builds-does-not-and-f25s-dll-penalty-was-noise),
published earlier the same session.** That is the most important thing in it and
it is treated as the result rather than as a caveat.

### The result

```
decode (tg)
  arm                               median tok/s     IQR   overhead vs A
  A: compiled out [static]                 46.19    0.3%                  -
  B: in, level 0 [static]                  46.06    0.5%    +0.27%  [-0.07, +0.52]
  C3: active level 3 [static]              45.66    0.5%    +1.16%  [+0.86, +1.37]

  A: compiled out [nshared]                46.11    0.4%                  -
  B: in, level 0 [nshared]                 46.09    0.3%    +0.05%  [-0.17, +0.25]
  C3: active level 3 [nshared]             45.73    0.4%    +0.84%  [+0.60, +1.04]

  A: compiled out [shared]                 45.93    0.4%                  -
  B: in, level 0 [shared]                  45.95    0.7%    -0.06%  [-0.48, +0.35]
  C3: active level 3 [shared]              45.78    0.2%    +0.31%  [+0.01, +0.52]
```

All three level-3 overheads certify. **They also disagree with each other, and
the static one reproduces F25's `+1.16%` to the second decimal** — which is
either reassuring or a coincidence, and with what follows it is impossible to
say which.

### The part that invalidates a published number

The `shared` pair's binaries were **not rebuilt** between F30 and F31. Same
executables, same DLLs, same model, same machine, same protocol, about two
hours apart:

| | level-3 overhead, decode | A arm | C3 arm |
|---|---|---|---|
| F30 | **+0.87% [+0.55, +1.19]** | 46.03 | 45.63 |
| F31 | **+0.31% [+0.01, +0.52]** | 45.93 | 45.78 |

**The two intervals do not overlap.** Both were "certified" — both had baseline
IQRs well under the 2% gate, both excluded zero, both came out of the harness
that has refused seven times in five sessions precisely so that its answers
could be trusted.

The static pair does the same thing in the other direction: +0.50% [-0.23,
+1.05] in F30 against +1.16% [+0.86, +1.37] in F31, overlapping only in their
last 0.2pp.

So the honest reading of F30's headline is that **`+0.87% [+0.55, +1.19]` was
over-precise**, and the correct statement about the shared build's level-3
overhead on this machine is something like *"between roughly 0.3% and 1.2%,
and a single run's interval understates that."* A correction is recorded in F30
itself rather than only here.

### Why the bootstrap does not know this

`bootstrap_ratio_ci` resamples the measurements *within* one invocation. It
therefore estimates "if I re-drew these 19 runs from the same afternoon, how
much would the median move" — which is a real question and not the question
anyone is asking. The quantity people want is "if I ran this again tomorrow",
and nothing in the harness estimates it.

This project has now demonstrated the same gap three times, in three places:
**F23** found per-node imbalance spreading 8.5x across identical runs, after
treating trace numbers as exact because the tracing is exact; **F25** found the
noise floor moving by a factor of two within one day; and **F31** finds two
non-overlapping certified intervals for one quantity. Each time the tooling was
statistically careful *inside* its own sample and silent about the sample being
one sample.

The fix is not a wider interval, it is **repetition at the level of the
invocation** — run the whole thing three times and report the spread of the
point estimates alongside the bootstrap. That is 45 minutes for the run above
and it is the obvious next thing.

### What the three explanations are, and that they are not separated

F30 and F31 differ in three ways at once, which is the same disease F30
diagnosed in F25:

1. **Arm order** — F30 used a fixed order, F31 rotates (`9ec7c82`).
2. **Round length** — six arms against nine, so ~45 s against ~68 s per round.
3. **Time** — two hours apart, on a machine F30 itself showed drifting 6.9%
   between sessions.

The rotation hypothesis is attractive and unproven. In F30's fixed order
`A_static` ran first in every round and `C3_shared` ran last; the arms at the
front absorbed an extra warm-up round, which would depress `A_static` and so
*understate* static's overhead, and if the end of a round runs warmer that would
depress `C3_shared` and *overstate* shared's. Both F30 numbers moved in exactly
those directions. That is a coherent story that fits, which is not the same as
evidence — F23's "28-thread anomaly" was also a coherent story that fitted.

Under rotation the artifact is gone: in F31 no arm's first retained rep is
systematically depressed, where in F30 all three static arms sat 3 tok/s low.

### Scoring the predictions

| | claim | outcome |
|---|---|---|
| **P31.1** | pure linkage consistent with zero, point under 0.5% | **held**, narrowly. `A_nshared` vs `A_static` is +0.17% [-0.00, +0.40] on decode — the lower bound is zero to two decimals — and -0.12% [-0.30, +0.22] on prefill |
| **P31.2** | pure generator consistent with zero | **failed on decode.** `A_shared` vs `A_nshared` is **+0.40% [+0.14, +0.69]**, excluding zero: the MSBuild shared build is slower than the Ninja shared build with identical linkage, identical flags and the same `cl.exe`. On prefill it is -0.29% [-0.61, +0.10] — spanning zero and pointing the other way. This is the prediction P31 said it would least like to be wrong about |
| **P31.3** | three overheads in +0.4%..+1.3%, the two shared pairs clustering | **failed on both specifics.** +1.16%, +0.84%, +0.31%: the shared pair is below the floor, and the two shared pairs (0.84, 0.31) are further apart than `nshared` and `static` (0.84, 1.16). The predicted mechanism also has the sign backwards — both shared builds show *less* overhead, not more |
| **P31.4** | F30's +0.36pp re-measures inside [-0.32, +1.17] | **failed.** It is -0.85pp [-1.16, -0.45], outside that interval and excluding zero on the other side. P31 called this the only prediction that could falsify a published result, and it did |
| **P31.5** | baseline IQRs across pairs more similar than F30's 5x | **held, decisively.** 0.32% / 0.37% / 0.40% on decode — a 1.25x spread against F30's 5x — and the per-round warm-up artifact is absent. The rotation fix works |

Two held, three failed. P31.3's failure is the one that taught something, below.

### Level 3 is a leveller, which reframes the whole question

P31.3 got the sign wrong because the question was the wrong shape. Look at the
absolute medians rather than the ratios:

| decode | static | nshared | shared | spread |
|---|---|---|---|---|
| A (compiled out) | 46.19 | 46.11 | 45.93 | **0.58%** |
| C3 (level 3) | 45.66 | 45.73 | 45.78 | **0.27%** |

and prefill is starker — A spreads 0.41% and C3 spreads **0.06%**.

**The instrumented builds all run at the same speed. The differences live
entirely in the baselines.** At level 3 the instrumentation costs enough to
dominate whatever linkage and generator do, and the three builds converge.

That means "overhead", as this harness defines it — each build against its own
compiled-out arm — **is a ratio whose denominator varies more than its
numerator**. The shared build shows the lowest overhead not because
instrumenting it is cheaper but because its baseline is slower. P25.1 as
originally posed ("does the shared build's overhead exceed the static build's?")
partly asks about the baselines, and F30 answered it in those terms without
noticing.

The better-posed question is whether the *instrumented* builds differ, and the
answer is that they barely do: 45.66 / 45.73 / 45.78 tok/s, a quarter of a
percent apart, across static, Ninja-shared and MSBuild-shared. If what you want
to know is "what does it cost me to profile", that convergence is the useful
result, and it is more favourable to the shared build than F30's framing was.

### Caveats

- Three pairs, one machine, one model, one thread count. Everything above is
  MSVC 19.44 on an i7-14700HX.
- The generator effect in P31.2 shows on decode and not prefill, with opposite
  signs. A codegen difference should appear more on the compute-bound phase, and
  it does not, so the likelier mechanism is link order changing code and data
  layout — untested, and stated as a guess.
- `nshared` was built and first used in the same session, so it has no history.
  The three-way agreement of its C3 arm with the other two is the only evidence
  it behaves.
- Nine arms make a ~68 s round. A transient shorter than a round still lands
  unevenly within it; rotation spreads that across arms over the run rather than
  removing it.

---

## P32 — What F24 looks like once the interval can see between-run drift

**Dated 2026-09-08, session 6. Written and committed before the re-measurement.**
[`F31`](#f31--two-certified-intervals-for-one-quantity-that-do-not-overlap-and-level-3-turns-out-to-be-a-leveller)
established that this project's bootstrap intervals are *within-run* intervals.
[`F24`](#f24--one-line-195-decode-and-the-first-certified-speedup-in-this-project)
— **+1.95% [+1.59, +2.35]**, the only throughput improvement this project has
ever claimed and the one thing in it a maintainer might act on — was measured
that way, at one block, with a half-width of 0.38pp.

F30 and F31 disagreed about a level-3 overhead by **0.66pp**. If a comparable
drift term sits under F24, its stated interval is roughly half as wide as it
should be. This run measures that instead of assuming it: same protocol as F24
(`mid.gguf`, 16 threads, tg64, pp64 as the control, `--reps 3`, 20 rounds per
arm), now `--blocks 3`, both arms rebuilt in this session.

### Verification done before predicting

Two stock builds of the same tree differ by **4 bytes** — the PE timestamp at
`0x110` and one word at `0x3e2ed4`. So MSVC is reproducible here and byte
comparison is a usable check, which is worth knowing given F24's own history of
comparing an arm against a stale binary.

Stock against patched differ by **1,137,994 bytes**, spanning `0x130` to
`0x45bc78`. One constant changed, a quarter of the image moved. Most of that is
almost certainly downstream address shift rather than different decisions, but
it means **code layout is not held constant between the arms**, and layout alone
can move throughput by around a percent on this kind of workload. That is an
alternative explanation for F24 that no run of this design can exclude, and it
is written down here rather than discovered later.

### The predictions

**P32.1 — the block-to-block spread exceeds F24's whole stated interval width
(0.76pp).** F30/F31 gave 0.66pp on a quantity a third the size. Predicted
spread across three blocks: **0.5pp to 1.5pp**.

**P32.2 — the t interval still excludes zero, so F24's conclusion survives while
its precision does not.** The effect is ~2% against a drift term of ~0.5pp; even
at t=4.303 for three blocks that should clear zero. This is the prediction that
decides whether F24 is repairable or retracted.

**P32.3 — the mean of the block estimates lands within 0.6pp of +1.95%.** If it
comes back at, say, +0.9%, then the original number was not merely over-precise
but wrong, and the layout confound above moves from a caveat to a suspect.

**P32.4 — the pp64 control stays unresolved on the t interval.** It was -0.81%
[-2.44, +0.35] originally. A control that resolves would mean the comparison is
broken, and with a wider interval it should be even harder to resolve.

**P32.5 — the bootstrap interval on the pooled data is narrower than the t
interval.** Nearly tautological given F31, but it is the direct demonstration on
the project's flagship number, and if it comes out false the whole diagnosis is
wrong.

---

## F33 — F24 survives at +1.62%, the interval that can see drift works, and the control stopped being clean

**Workload:** `mid.gguf`, **16 threads**, tg64 with pp64 as the control, both
arms uninstrumented and rebuilt in this session, `--reps 3`, 20 rounds per arm,
**`--blocks 3`, run twice** (six blocks, 240 rounds per arm in total),
`tools/ab_throughput.py`.
[`P32`](#p32--what-f24-looks-like-once-the-interval-can-see-between-run-drift)
holds the predictions.

This is the repair of
[`F24`](#f24--one-line-195-decode-and-the-first-certified-speedup-in-this-project),
the only throughput improvement this project claims, after
[`F31`](#f31--two-certified-intervals-for-one-quantity-that-do-not-overlap-and-level-3-turns-out-to-be-a-leveller)
showed its interval was a within-run interval and roughly half as wide as the
truth.

### The result

| | tg64 (decode) | pp64 (control) |
|---|---|---|
| run A | +1.91% [+0.92, +2.91] | -0.43% [-2.26, +1.41] |
| run B | +1.33% [+0.19, +2.48] | -2.18% [-5.08, +0.72] |
| **pooled, six blocks** | **+1.62% [+1.10, +2.15]** | **-1.30% [-2.67, +0.06]** |

Block estimates, decode: `+1.59 +2.36 +1.79 | +0.80 +1.62 +1.58`.

**F24's effect is real.** Six blocks, two independent runs, `t(5)` interval
excluding zero. What changes is the number and its precision: **+1.95% [+1.59,
+2.35] becomes +1.62% [+1.10, +2.15]**. The old point estimate sits inside the
new interval; the new point estimate sits at the very edge of the old one.

### The fix demonstrably works, which is the methodological result

F31's charge against the bootstrap was that two of its intervals, for one
quantity on unrebuilt binaries, **did not overlap**. The same test on the new
interval:

```
  run A   t interval  [+0.92, +2.91]
  run B   t interval  [+0.19, +2.48]     -> OVERLAP
```

Two independent three-block runs of the same comparison produce intervals that
agree. That is the property the bootstrap failed and the reason to quote the
`t` column. It is one test on one quantity, not a proof, but it is the
difference between a fix that is argued for and one that has been checked.

### The control stopped being clean, and that is the real news

F24's strongest argument was never its interval. It was the structure:

> the arm that should move moves and certifies, the arm that should not move
> does not and does not certify. A build artifact or a thermal drift would not
> respect that distinction.

Prefill sits far above the chunking threshold in **both** arms, so the patch
cannot reach it. It should be a flat null. Across six blocks it reads
**-1.30% [-2.67, +0.06]** — still not resolved, but only just, and every run
has come back negative: -0.81% originally, -0.43% in run A, -2.18% in run B.

So the clean version of F24's argument no longer holds. The honest version is
weaker: **decode moves by more than prefill does, in the right direction, and
prefill's own movement is unstable across runs** (-0.43% against -2.18%, a
1.75pp swing that is larger than either run's internal spread). That
instability is the best evidence it is drift rather than a property of the
binaries — but it is no longer the clean null the original finding leaned on.

### Which makes the layout confound evidence rather than speculation

[`P32`](#p32--what-f24-looks-like-once-the-interval-can-see-between-run-drift)
recorded before the run that the two arms differ by **1,137,994 bytes**, so code
layout is not held constant, and layout alone can move throughput by around a
percent. That was written as a caveat nobody could exclude.

A prefill arm that drifts persistently negative is exactly what a layout effect
would look like: prefill cannot see the scheduler change, so anything moving it
is *not* the thing F24 claims to have measured. This does not overturn F24 —
decode moves ~3x more than prefill and in the opposite direction, which layout
alone has no reason to produce — but it means **some unknown fraction of the
+1.62% may be layout rather than chunking**, and no run of this design can
separate them.

Separating them needs an arm that changes layout without changing behaviour —
padding the function, or reordering something inert — which is a real experiment
and is not done.

### Scoring the predictions

| | claim | outcome |
|---|---|---|
| **P32.1** | block spread in 0.5-1.5pp, exceeding F24's whole stated width of 0.76pp | **held, by 0.01pp.** 0.77pp in run A and 0.82pp in run B. Held on both specifics and far too narrowly to be credited as skill |
| **P32.2** | the `t` interval still excludes zero | **held**, in run A, in run B and pooled. This is what makes F24 repairable rather than retracted |
| **P32.3** | mean within 0.6pp of +1.95% | **held for the run it was written for** (+1.91%, 0.04pp) and **marginally failed for run B** (+1.33%, 0.62pp). Pooled, +1.62% is 0.33pp away |
| **P32.4** | the control stays unresolved on the `t` interval | **held in letter, and the interesting part is how narrowly.** Pooled upper bound +0.06 |
| **P32.5** | the bootstrap is narrower than the `t` interval | **failed, and backwards.** Bootstrap 6.10pp wide against the `t` interval's 1.98pp in run A, and 5.61 against 2.29 in run B |

**P32.5 is worth more than the four that held.** The pooled bootstrap resamples
240 rounds spanning three blocks *and* every contaminated round in them; per-arm
minima reached 18.30 tok/s against a 39.09 median. Block medians absorb those
outliers, so the three block estimates stay tight while the pooled bootstrap
inherits the whole mess.

So the block method is **robust to the contamination that widens a pooled
bootstrap**, which is not why it was built and is a better argument for it than
the one in F31. The corollary is a warning: a bootstrap interval is not reliably
narrower than the truth, it is reliably *wrong about* the truth, and on a dirty
run it errs the other way.

### Caveats

- **Run A was contaminated by this session's own activity.** Documentation was
  written and commits made while it ran, against the harness's own instruction
  to close background work. It is reported rather than discarded because run B
  reproduces it, but it is the reason there are two runs.
- **Run B was cleaner, not clean.** 13 of 120 decode rounds and 23 of 120
  prefill rounds landed more than 5% below their own block's median. This
  machine has background activity that no amount of care removes.
- Two runs of three blocks is six numbers. `t(5)` is honest about that and the
  interval is still 1.05pp wide.
- One machine, one thread count, one model, no NUMA — unchanged from F24, and
  NUMA is the case the `nth * 4` constant was tuned for.
- Six blocks at 16 threads say nothing about other thread counts, and the whole
  point of F23 is that this threshold's behaviour depends on `nth`.

---

## P34 — The overhead numbers, with an interval that can see drift

**Dated 2026-09-08, session 6. Written and committed before the run.**
[`F33`](#f33--f24-survives-at-162-the-interval-that-can-see-drift-works-and-the-control-stopped-being-clean)
repaired F24 and demonstrated that two independent `t` intervals for one
quantity overlap where two bootstrap intervals did not. The remaining Tier D
numbers — the **overheads** in F25, F30 and F31 — have never had that treatment.

Three pairs, `--no-level0` to keep the round affordable, `--blocks 3`, n=15:
`static` (ninja/static), `nshared` (ninja/shared), `shared` (MSBuild/shared).
The level-0 arm is dropped because it has been unresolvable every time it has
been measured (P30.2, P31.2 both held it spanning zero), and it is the cheapest
thing to give up.

### The four quantities this settles

1. **static level-3 overhead** — F25 said +1.16% [+0.67, +1.87], F30 +0.50%, F31 +1.16%
2. **shared level-3 overhead** — F30 said +0.87% [+0.55, +1.19], F31 +0.31% [+0.01, +0.52], **non-overlapping**
3. **the shared−static difference** — F30 +0.36pp, F31 −0.85pp, also inconsistent
4. **the generator contrast** — F31's +0.40% [+0.14, +0.69] on the compiled-out arms

### The predictions

**P34.1 — every `t` interval here is wider than its bootstrap counterpart, and
at least one of the four spans zero that previously did not.** F33 found the
bootstrap running 2–3× *wider* than the `t` on a dirty run, so this is not
automatic; on a clean run the ordering should revert. The one to watch is the
shared overhead, which "resolved" at +0.87% and again at +0.31% without those
intervals meeting.

**P34.2 — the shared and static overheads become indistinguishable.** F31's
structural finding was that the instrumented arms converge (0.27% apart on
decode, 0.06% on prefill) while the baselines spread. Overhead-as-a-ratio is
therefore mostly reporting baseline differences, and once the interval is
honest the two should overlap heavily. **Predicted: both means in +0.4%..+1.3%,
and the difference spanning zero.**

**P34.3 — F31's generator effect does not survive.** +0.40% [+0.14, +0.69] was
a single-run bootstrap on exactly the kind of small effect this session has
twice found over-precise. Predicted to span zero once it has a `t` interval.
If it *does* survive, it is the more interesting outcome, because it footnotes
every cross-generator comparison in the repo.

**P34.4 — the block-to-block spread on the overheads is 0.4pp to 1.0pp**, in
line with F33's 0.77/0.82pp on a larger effect and with the 0.66pp gap between
F30 and F31 that started all of this.

**P34.5 — the static overhead's mean lands in +0.8%..+1.4%**, i.e. near F25's
and F31's agreeing +1.16% rather than F30's +0.50%. F30's static arm ran first
in every round under a fixed order, and that is the one number with a named
mechanism for being wrong.

---

## F34 — A void run, and the lesson that blocks do not defeat contamination

**Workload:** intended as the re-measurement of every remaining Tier D number —
`mid.gguf`, 8 threads, three build pairs, `--blocks 3`, n=15, 2640 s.
[`P34`](#p34--the-overhead-numbers-with-an-interval-that-can-see-drift) holds
the predictions. **The run is void and none of the predictions can be scored.**

It is written up anyway, because how it failed is worth more than what it was
going to measure.

### What it produced

```
  arm                               median tok/s     IQR   overhead vs A
  A: compiled out [static]                 35.05    4.4%                  -
  C3: active level 3 [static]              35.72    3.4%    -1.89%  [-3.43, -0.23]
  A: compiled out [nshared]                35.30    4.3%                  -
  C3: active level 3 [nshared]             35.87    2.8%    -1.59%  [-3.01, -0.61]
  A: compiled out [shared]                 35.43    3.5%                  -
  C3: active level 3 [shared]              35.73    3.7%    -0.85%  [-2.31, +0.39]
```

**Every instrumented arm is faster than its own compiled-out baseline.** That is
not a small effect in an unexpected direction; it is impossible. Adding 2.9
million record writes does not speed a build up.

Throughput is ~35 tok/s where F30 and F31 both saw ~46. Prefill baseline IQR
reached **37.6%, 43.5% and 32.6%**.

### Why

`javaw` — unrelated to this project — started at 17:10:46 with a **5.17 GiB
working set**, and the run launched into **870 MB of free physical memory
against an 840 MB model**. Decode is bandwidth-bound (F14). Once the weights
stop staying resident, every number is about paging.

**The launch command printed the free-memory figure and started the run in the
same breath**, which is no check at all. `preflight_ram()` now refuses to start
below 1.5× the model size, with `--force` to override.

### The part that matters: blocks agreed with each other and were wrong

The block machinery reported this for the shared pair:

```
  C3: active level 3 [shared]    -0.76   -1.29   -1.30    0.55   -1.12   [-1.89, -0.34]
```

Three block estimates, a spread of **0.55pp — the tightest in the whole run** —
and a `t` interval **excluding zero**. By the rule this session spent hours
building, that is a resolved result. It is also nonsense.

**Blocks defend against drift *between* passes. They do nothing about
contamination that spans every pass.** Three blocks taken back to back inside
one bad window agree with each other, and their agreement reads exactly like
precision. Worse, the contamination *tightened* the interval: paging dominated
the timing so completely that it swamped the machine's ordinary variability.

So [`F33`](#f33--f24-survives-at-162-the-interval-that-can-see-drift-works-and-the-control-stopped-being-clean)'s
conclusion needs qualifying. It said the block method is "robust to the
contamination that widens a pooled bootstrap", and that is true of *sporadic*
contamination — the odd bad round a median absorbs. It is false of *sustained*
contamination, where blocks do not merely fail to help, they actively
manufacture confidence.

**The defence is the baseline-IQR gate, which fired correctly**, and which the
tool was simultaneously undermining by printing "QUOTE THE t INTERVAL"
underneath it. Two sentences that cannot both be true. Fixed: when the gate
fails, the tool now refuses the whole block table, and it counts arms whose
interval excludes zero on the impossible side.

### The ordering of trust, corrected

Session 6 spent most of its length arguing that the bootstrap interval was the
weak link and the `t` interval the fix. F34 says the ordering is:

1. **The gate first.** If the baseline cannot resolve the effect, nothing below
   is worth reading, whatever its interval says.
2. **Then physical plausibility.** An overhead that is negative, or a control
   that moves as much as the treatment, voids the run regardless of statistics.
3. **Then the `t` interval**, which is only meaningful once 1 and 2 pass.

The `t` interval is not a licence to stop looking at the machine. It was
promoted, in this session's own prose, into something close to one.

### What this costs

**F25, F30 and F31's overhead percentages remain un-re-measured.** They are
still Tier D on a within-run bootstrap. The run that was going to fix them has
to happen again on a quiet machine, and this one needs `javaw` closed — which
is the user's call, not an agent's.

F24/F33 is unaffected: it was measured before `javaw` started, at 16 threads,
and its control behaved (badly, but in a documented way).

---

## F36 — The overhead numbers, settled: every build under 1%, linkage and generator both null, and two F31 claims retracted

**Workload:** `mid.gguf`, 8 threads, pp512 / tg256, three build pairs
(`static` ninja/static, `nshared` ninja/shared, `shared` MSBuild/shared),
`--no-level0`, **`--blocks 3`**, n=15, 2020 s, 7.32 GiB free at launch.
Baseline IQRs **0.83% / 0.97% / 0.93%** on decode — the gate passed for the
first time since the blocks machinery existed.
[`P34`](#p34--the-overhead-numbers-with-an-interval-that-can-see-drift) holds
the predictions, and this is its third attempt: F34 was voided by paging and a
second attempt was refused by the gate at 3.5–4.7% IQR.

### The result

| build | level-3 overhead, decode | block spread | prefill (control) |
|---|---|---|---|
| `static` (ninja, static) | **+0.56% [-0.05, +1.16]** | 0.46pp | +0.13% [-0.13, +0.40] |
| `nshared` (ninja, shared) | **+0.92% [+0.38, +1.46]** | 0.43pp | -0.01% [-0.37, +0.34] |
| `shared` (MSBuild, shared) | **+0.77% [+0.05, +1.48]** | 0.57pp | +0.10% [-0.42, +0.62] |

All three agree with each other. Every difference spans zero:

```
  C3: nshared - static     +0.36pp  [-0.18, +0.82]
  C3: shared  - static     +0.27pp  [-0.38, +0.71]
  A:  nshared vs static    -0.03%   [-0.45, +0.24]     pure linkage
  A:  shared  vs nshared   +0.07%   [-0.40, +0.55]     pure generator
```

**The honest headline for the README is one number for all builds: level 3
costs under 1% on decode, and the three builds cannot be told apart.** Prefill
is the control and is unresolvable everywhere, which is what makes the decode
column believable.

### This resolves the non-overlapping pair that started everything

[`F31`](#f31--two-certified-intervals-for-one-quantity-that-do-not-overlap-and-level-3-turns-out-to-be-a-leveller)
found the shared build's overhead at `+0.87% [+0.55, +1.19]` and then
`+0.31% [+0.01, +0.52]` — two bootstrap intervals for one quantity that did not
meet. F36's interval for that same quantity is **`+0.77% [+0.05, +1.48]`, and it
contains both of them.**

The static arm does the same. Four measurements across three sessions:

| | F25 | F30 | F31 | **F36** |
|---|---|---|---|---|
| static level-3 | +1.16% | +0.50% | +1.16% | **+0.56% [-0.05, +1.16]** |

F36's interval contains all three earlier point estimates. **There was never a
contradiction between those runs — there was a contradiction between their
intervals**, and the intervals were the thing that was wrong. That is exactly
what F31 diagnosed and what `--blocks` was built to fix, now shown working on
the quantity that motivated it.

### Two F31 claims do not survive

**The generator effect is retracted.** F31 reported the MSBuild shared build
`+0.40% [+0.14, +0.69]` slower than the Ninja shared build on decode, with
identical linkage and flags, and called it the prediction it would least like to
be wrong about — because it would footnote every cross-generator number in the
repo. Measured with an interval that can see drift: **+0.07% [-0.40, +0.55]**.
No effect. The footnote is withdrawn.

**"Level 3 is a leveller" does not reproduce, and it was the worse error.**
F31 observed that the instrumented arms converged (0.27% apart on decode) while
the compiled-out arms spread (0.58%), and concluded that overhead-as-a-ratio has
a denominator varying more than its numerator. F36:

| | A arms spread | C3 arms spread | |
|---|---|---|---|
| F31 decode | 0.58% | 0.27% | C3 tighter |
| **F36 decode** | **0.16%** | **0.32%** | **reversed** |
| F31 prefill | 0.41% | 0.06% | C3 tighter |
| **F36 prefill** | **0.17%** | **0.14%** | marginal |

The ordering flips on decode and all but vanishes on prefill. **Six numbers
within half a percent of each other were read as structure.** That claim was
promoted in `04-project-audit.md` as F31's "durable part", explicitly marked as
*not* Tier D and therefore exempt from re-measurement. It was Tier D all along;
being a comparison of spreads rather than a percentage disguised it.

The lesson is narrower than "be careful": **a claim built from differences
between Tier D numbers inherits Tier D, however structural it sounds.**

### Scoring the predictions

| | claim | outcome |
|---|---|---|
| **P34.1** | every `t` interval wider than its bootstrap, and at least one quantity loses a resolution it had | **held on both.** Widths 1.21 vs 0.84, 1.08 vs 0.56, 1.43 vs 0.67 — and static's `t` interval spans zero where its bootstrap did not |
| **P34.2** | shared and static become indistinguishable; both means in +0.4%..+1.3%; difference spans zero | **held on all three specifics.** +0.56% and +0.77%, difference +0.27pp [-0.38, +0.71] |
| **P34.3** | F31's generator effect does not survive | **held.** +0.07% [-0.40, +0.55] |
| **P34.4** | block spread 0.4pp to 1.0pp | **held.** 0.46, 0.43, 0.57 |
| **P34.5** | static mean in +0.8%..+1.4%, near F25/F31's +1.16% rather than F30's +0.50% | **failed.** +0.56%, which is F30's number, not F25's. The reasoning was that F30's fixed arm order gave it a named mechanism for being wrong — the mechanism is real (F30) but it was not what made F30 differ from F31 |

Four held, one failed. P34.5's failure closes the rotation question the other
way: F30's fixed order was a genuine defect and worth fixing, but it was never
the explanation for the F30/F31 disagreement. **The explanation was always just
that both intervals were too narrow.**

### Caveats

- Three blocks is three numbers per arm; `t(2) = 4.303` and the intervals are
  correspondingly wide. Static's spans zero, so **the static build's level-3
  overhead is still not resolved away from zero** — it is bounded under +1.16%.
- One machine, one model, 8 threads, one workload.
- The `shared` and `nshared` pairs use different generators, which F36 now shows
  does not matter here; that is one measurement, not a general result.
- Getting a run the gate accepted took **three attempts across one evening** —
  voided by paging, refused at 3.5–4.7% IQR, then accepted at 0.83–0.97%. The
  machine's ability to resolve a 1% effect varies by more than the effect, and
  nothing in this project predicts when it will be able to.

---

## F37 — Three hypotheses for M10, all refuted, and a correction to a claim made an hour earlier

**Workload:** `mid.gguf`, 8 threads unless stated, `build-ts-off`, decode
(`-p 0 -n 256`) except where prefill is named. Not a finding about llama.cpp —
a finding about **this machine**, and about how many plausible explanations it
can eat.

[`M10`](04-project-audit.md) is the observation that decode throughput on this
machine sits sometimes near **46 tok/s** and sometimes near **40**, a ~13%
swing that is roughly ten times any effect this project measures and that
decides whether a run passes the harness gate. Four candidate causes were
tested. **All four are dead**, and one of them was a claim this session had
already published.

### 1. Thermal — refuted

The first test looked decisive: a hot burst at 41.75 tok/s, seven minutes idle,
then 45.97. **+10.11%**, and it was written up in the moment as confirmed.

The follow-up killed it. After **eight** minutes of cooling the machine started
at 39.95 and stayed flat across 30 runs — total decay **-1.16%**:

```
  run  1  t=   6.9s   39.95      run 15  t= 103.0s   40.95
  run  5  t=  35.0s   34.38      run 20  t= 137.2s   39.71
  run 10  t=  69.4s   40.13      run 30  t= 204.5s   41.38
```

**More cooling produced a lower result, and the decay curve is flat.** Re-reading
the first test, its post-idle burst had median 45.97 with min 41.24 and IQR
7.13% — bimodal, not a decay. A story was fitted to one measurement.

### 2. Core placement / pinning — refuted

The harness's own gate advises pinning threads, and scheduler migration was
M3's leading candidate. Pinning made it **worse**:

```
  free (no mask)         med  40.51   IQR  0.33%
  0x00FF  logical 0-7    med  40.69   IQR  0.37%
  0xFF00  logical 8-15    med  41.87   IQR  0.57%
  0xAAAA  odd 1-15       med  39.52   IQR  2.41%
  0x5555  even 0-14      med  36.09   IQR 19.92%   range 21.58-38.91
  0x0FF0000  E-cores     med  25.20   IQR  6.14%
```

### 3. CPU frequency — refuted, and backwards

The slow state runs at **118% of base (~2487 MHz)**, and 46/41 ≈ 1.12 against a
plausible 134/118 ≈ 1.14, so frequency looked like an excellent fit. Sampling
`\Processor Information(_Total)\% Processor Performance` alongside 40 timed
runs:

```
  n=40   freq 110-132%   tg 38.62-45.03   Pearson r = -0.423
```

**Negative.** Higher measured frequency goes with *lower* throughput. The
counter is `_Total` across all 28 logical processors, so it most likely tracks
how busy the *rest* of the machine is — other cores boosting while our eight
threads get less of the memory system. Whatever it is, it is not the cause, and
the direction rules out the mechanism rather than merely failing to support it.

That run also re-characterised M10: the range **38.62–45.03 appeared inside a
single 4.5-minute window**. This is not two stable regimes over hours. It is
high run-to-run variance whose *median* moves — F36 sat tightly at 46.0 with
0.83% IQR, and an hour later the same command wanders between 38.6 and 45.0.

### 4. The logical-processor numbering — refuted, and it was mine

[`F14`](#f14--core-heterogeneity-is-the-mechanism-pinning-proves-it-and-does-not-fix-it)
records an unexplained anomaly under `0x5555` — seven threads clustered, one
12–20% slower — and lists two untested candidates, the first being
"hyperthread sibling collision from a wrong assumption about this CPU's logical
numbering". The `0x5555` row above (19.92% IQR, worst median) looked like exactly
that, and **this session published the conclusion that `0x5555` is not one
thread per P-core**, in `04-project-audit.md` and a commit message, before
testing it.

The test is clean, because the two topologies predict opposite results. On
compute-bound prefill at two threads:

```
  0x0003 (logical 0,1)  pp512 median   91.47 tok/s
  0x0101 (logical 0,8)  pp512 median  171.10 tok/s      ratio 0.53x
```

Logical 0 and 1 **share a physical core**; logical 0 and 8 do not. The numbering
is interleaved, core N = logical 2N, 2N+1. **`0x5555` really is one thread per
P-core, F14's description was right all along, and the claim published an hour
ago was wrong.** It is corrected in the audit document.

What this *does* buy: F14's first candidate explanation is now **tested and
eliminated**, leaving its second — `--cpu-strict` assigning two threads to one
bit — as the surviving hypothesis for the anomaly.

### The observation that survives, and is odd

With the topology settled, the pinning table reads strangely:

| mask | distinct P-cores | threads | decode |
|---|---|---|---|
| `0x00FF` / `0xFF00` | **4** (both siblings each) | 8 | 40.69 / 41.87 |
| `0x5555` / `0xAAAA` | **8** (one thread each) | 8 | 36.09 / 39.52 |

**Eight threads on eight distinct cores is slower than eight threads on four
cores.** For a compute-bound workload that would be absurd; for a
bandwidth-bound one it is not, and F14 established decode here *is*
bandwidth-bound. Four cores can already saturate the path to DRAM, and eight
add contention without adding bandwidth. Recorded as an observation, not a
mechanism — testing it means measuring memory bandwidth per configuration,
which nothing here does.

### What M10 is now

Still open, better bounded, four explanations poorer:

- **not** thermal (more cooling gave a lower result; the decay curve is flat)
- **not** thread placement (pinning is worse, and the best mask is no mask)
- **not** CPU frequency (correlation is negative)
- **not** the logical-processor mapping (tested directly; the mapping is as documented)
- **not** the `-p 0` vs `-p 512` workload difference (-0.91%, interleaved test)

Untested and still live: page-cache and standby-list state for an 840 MB model
read once per invocation; Windows power throttling policy per-process; and
whatever the `_Total` frequency counter is actually tracking.

**The practical rule needs no mechanism.** A run's compiled-out arm reports its
absolute median, so **that number identifies which state the machine was in**,
and two runs with different A-arm medians must not be compared. F36's A arms
read 45.98/45.99/46.05; F35's read 43.50/43.65/43.12. That is visible in the
existing output and in every `--json-out` file already written.

### The pattern worth recording

Three hypotheses formed and refuted in one sitting, one of them after being
published. Each was plausible, each had a mechanism, and each died to a test
that took under ten minutes. **The tests were cheap and the publishing was not.**
The only reason the topology claim did not survive into the documentation
permanently is that it happened to be cheap to check afterwards.

---

## F38 — F24's two arms differ in 31% of their code section, and the control built to test it does not

**Artefacts, not a workload:** `bench-stock.exe`, `bench-patched.exe` (F24's
`nth*4`->`nth*2` in `mul_mat`) and `bench-layoutctl.exe` (the same edit applied
to `mul_mat_id` instead), all built from one tree in one session, plus a second
stock build as a null reference. No measurement — this is a fact about the
binaries, established while the machine was too busy to measure on.

Audit issue **M6** says F24's arms are not layout-controlled and that layout
alone can move throughput by around a percent. That was written as a caveat
nobody could exclude. It is now quantified, and it is worse than it read.

### How much actually differs

| comparison | bytes differing |
|---|---|
| stock vs a second stock build | **4** (PE timestamp `0x110`, one word at `0x3e2ed4`) |
| stock vs **F24's patched arm** | **1,137,994** |
| stock vs the `mul_mat_id` control | **5** |

MSVC is reproducible here to 4 bytes, so the comparison is meaningful. The
differences are not scattered: they form one contiguous block from roughly
`0x260000` to `0x380000` at **~94% byte density**, and parsing the PE section
table puts that block inside **`.text`** (raw `0x400`–`0x386400`). That is
**1.1 MB of a 3.67 MB code section — about 31% of all code in the binary.**

### Moved, regenerated, or both

Thirty-six 256-byte code probes taken from stock's changed region and searched
across the patched `.text`:

```
  same address :  0
  moved        : 14
  not present  : 22
```

**Zero at the same address** is the number that matters. Fourteen probes are
provably the same code at a different offset. The twenty-two unmatched ones are
*not* proof of regeneration — a 256-byte window containing a relocated absolute
address or a relative jump will not match byte-exactly even when the function is
otherwise identical — so the honest statement is:

> Roughly a third of the code section is relocated, regenerated, or both,
> between the two binaries whose 1.62% throughput difference F33 attributes to
> a scheduler constant.

No constant shift explains it (`p[base+k] == s[base]` fails for every
`k` in ±4096 at a probe inside the block), so this is not simple relocation of
an otherwise-identical image.

> **This paragraph is wrong, and [`F42`](#f42--f24s-11-mb-of-regenerated-code-is-a-uniform-16-byte-shift-and-m6-becomes-a-question-about-one-number)
> shows why.** The linker map says it *is* simple relocation: `mul_mat` shrinks
> by 16 bytes and all 7,721 functions after it move down by exactly 16, with
> every other function unmoved. A byte-window probe cannot detect that, because
> relocated code carries absolute addresses that also change. The 22 "not
> present" probes are the signature of relocation, not evidence against it.

### The control does not control

`patches/04-layout-control.patch` applies the *identical* edit to
`ggml_compute_forward_mul_mat_id` — the MoE expert path, which a dense model
never executes — so a patched binary must behave identically on every model in
this repo while perturbing the same file. The design intent was a layout arm.

**It perturbs 5 bytes.** One of code, four of build metadata. The compiler
emitted the same instruction with a different immediate and nothing moved.

So the control is **not** a layout control, and saying otherwise would repeat
this session's habit of publishing a mechanism before testing it. What it *is*
is a **null control**: two binaries that must behave identically on this
workload and are laid out identically too. Measuring it against stock therefore
estimates the harness's false-positive rate directly, which nothing in this
project has ever done. That is worth running and is not what M6 needs.

### What M6 needs instead, and why it is hard

A real layout arm has to change `mul_mat`'s *size* without changing what it
does — padding, an uncalled-but-retained function, forced alignment — and then
survive the objection that the padding itself costs something. The asymmetry
here is the discouraging part: a one-byte immediate in `mul_mat_id` moves
nothing, while a one-byte immediate in `mul_mat` moves a third of the image. The
same edit is layout-neutral in one function and layout-catastrophic in another,
which means **layout perturbation cannot be dialled in by choosing a small
edit**. It is a property of where the edit lands.

### What this does to F24/F33

`+1.62% [+1.10, +2.15]` stands as a measurement of *those two binaries*. What it
cannot yet claim is that the scheduler constant is why. The supporting
arguments, in descending strength:

- **Decode moves ~3x further than prefill, in the opposite direction.** Layout
  has no reason to prefer one phase's sign over the other.
- The mechanism is specific and predicted in advance (F23 said which matmuls
  flip mode, before F24 measured).
- Against it: F33's prefill control has been negative in **every** run
  (-0.81%, -0.43%, -2.18%), which is what a layout effect on a phase that
  cannot see the change would look like.

**Nothing here is a reason to withdraw F24.** It is a reason that the phrase
"one line, +1.62%" should be "one line, +1.62%, on binaries that also differ in
31% of their code layout" — and a reason the upstream evidence pack should say
so, which it now does.

---

## P39 — what a null control should look like, written before the run

Session 7, `HANDOFF.md` section 5 item A1. The measurement below has **no
treatment**. `bench-stock.exe` and `bench-layoutctl.exe` differ by **one byte of
code** at `0x264830` inside `.text`, plus the PE timestamp at `0x110` and its
echo at `0x3e2ed4` — and the byte that differs is the immediate in
`ggml_compute_forward_mul_mat_id`, a function `mid.gguf` never enters. Zero
`MUL_MAT_ID` node events appear in any of the nine reference traces, against
2,704 `MUL_MAT` events in the level-3 mid trace alone, so the changed line is
provably dead for this workload.

So every percentage this run reports is a **false positive by construction**.
The point is to find out how large one is, because this project has never
measured that and has been quoting Tier D numbers of 0.5–2% for six sessions.

**Protocol, identical to F33's** so the answer applies to the number that
matters: `mid.gguf`, 16 threads, `tg64` with `pp64` as the second workload,
`--reps 3`, 20 rounds per arm, `--blocks 3`, both arms uninstrumented and built
in this session from one tree.

### The predictions

**P39.1 — the decode `t` interval contains zero.** This is the headline and the
one that decides whether the harness is trustworthy at all. If a pair of
binaries that cannot differ produces a resolved decode result, then F33's
`+1.62% [+1.10, +2.15]` has an unknown false-positive component and the
project's only speedup is in serious doubt. I expect it to hold.

**P39.2 — the decode `t` interval is 1.5–2.5pp wide.** F33's two three-block
runs gave 1.98pp and 2.29pp on the same protocol. Width is a property of the
machine's drift and `t(2) = 4.303`, not of the treatment, so a null run should
reproduce it. If it comes back much narrower, block agreement is being
manufactured (M8) and the width in F33 was doing less work than assumed.

**P39.3 — the decode block spread is 0.4–1.2pp.** F33 measured 0.77pp and
0.82pp. Same reasoning.

**P39.4 — the decode point estimate is within ±0.8pp of zero.** With a ~0.5pp
drift term and three blocks, the mean should sit near zero. This is the number
that literally *is* the false-positive magnitude. If it lands at ±1.5pp, then a
one-percent effect is not measurable on this machine with this protocol at all,
and several published numbers are inside the noise.

**P39.5 — `pp64` is NOT persistently negative, and its point estimate is
above −0.8%.** This is the prediction worth running the experiment for, and it
is a partial test of **M6**. F33's prefill control came back negative in every
single run: −0.81%, −0.43%, −2.18%. F38 read that as the signature of a layout
effect, since prefill cannot see the chunking change but can see 1.1 MB of
relocated `.text`. Here there is no layout change — one byte, same address — so
if the negative prefill is layout, it must vanish. **If prefill comes back
negative here too, the layout explanation loses its main piece of evidence and
M6 weakens**, because the negativity would then be a property of the harness or
the machine rather than of F24's binaries. Either outcome moves M6.

**P39.6 — at least one of the two bootstrap intervals excludes zero while both
`t` intervals contain zero.** M1 says the bootstrap is roughly half as wide as
the truth; on a genuine null that should show up directly as a raised
false-positive rate. This is close to a coin flip on two workloads and is
recorded as a prediction rather than an expectation — but it is the sharpest
available test of M1, because a null is the only place a false positive can be
identified as one.

**P39.7 — the A-arm `tg64` median is 36–44 tok/s and the three blocks' A-arm
medians agree within 2 tok/s.** The gate-first rule (F34): this identifies which
of M10's states the machine was in and whether it stayed there. F33's run sat at
a 39.09 median at these settings. If the blocks disagree by more than that, the
run is describing the machine's drift and not its own arms, and nothing below it
counts.

**P39.8 — added after run A, before run B.** One three-block run is exactly
what F31 condemned, so the false-positive rate gets the treatment F33 gave
F24: a second independent three-block run, pooled to six. **Prediction: run B's
`t` interval on `tg64` overlaps run A's**, which is the property the bootstrap
failed. Run A read `-0.04% [-1.26, +1.18]`. I also predict run B's `pp64` point
estimate is again **above** `-0.8%`, since P39.5's whole content is that the
persistent negativity belongs to F24's binaries and not to the harness.

### What would make this run void rather than informative

Per F34's ordering — gate first, physical plausibility second, interval third.
Free RAM was 5,505 MB against an 840 MiB model (6.5×, well clear of the 1.5×
floor), and nothing else will run on the machine while it is in flight.

---

## F39 — the harness's false-positive rate, measured for the first time: clean on decode, and a resolved false positive on prefill

**Workload:** `mid.gguf`, **16 threads**, `tg64` with `pp64`, both arms
uninstrumented and built in this session from one tree, `--reps 3`, 20 rounds
per arm, **`--blocks 3`, run twice** (six blocks, 240 rounds per arm) —
protocol identical to [`F33`](#f33--f24-survives-at-162-the-interval-that-can-see-drift-works-and-the-control-stopped-being-clean) so the answer applies to the number
that matters. [`P39`](#p39--what-a-null-control-should-look-like-written-before-the-run)
holds the predictions. Raw data: `data/overhead/f39a.json`, `f39b.json`.

This is `HANDOFF.md` section 5 item A1 and it is the first time this project has
measured **its own false-positive rate**. Every percentage below is a false
positive by construction.

### The two binaries cannot differ

`bench-stock.exe` and `bench-layoutctl.exe`
(`patches/04-layout-control.patch`) differ by **three bytes**: the PE timestamp
at `0x110`, its echo at `0x3e2ed4`, and **one byte of code** at `0x264830`
inside `.text` — the immediate in `ggml_compute_forward_mul_mat_id`. F38
measured 5 bytes for this pair; this session's two builds landed 16 seconds
apart, so less of the difference is timestamp and the code difference isolates
to a single byte at a single address.

**The changed line is provably dead for this model.** Across all nine reference
traces there are **zero** `MUL_MAT_ID` node events, against **2,704** `MUL_MAT`
events in the level-3 `mid.gguf` trace alone. Level 3 records every node on
every worker thread, so this is a structural (Tier A) fact, not a sampling
argument. A dense model never enters the patched function.

Same size, same addresses, one dead immediate. Nothing measurable can differ.

### The result

| | tg64 (decode) | pp64 (prefill) |
|---|---|---|
| run A | -0.04% [-1.26, +1.18] | +0.33% [-0.35, +1.01] |
| run B | -0.04% [-1.22, +1.14] | +0.85% [-0.86, +2.55] |
| **pooled, six blocks** | **-0.04% [-0.49, +0.42]** | **+0.59% [+0.01, +1.16]** |

Block estimates, decode: `+0.53 -0.37 -0.27 | -0.58 +0.33 +0.13`.
Block estimates, prefill: `+0.03 +0.38 +0.57 | +0.14 +0.89 +1.51`.

**On decode the harness is clean, and better than it had any right to be.** Two
independent three-block runs returned the *same point estimate to two decimal
places*, with intervals that overlap almost exactly. Pooled to six blocks the
decode false-positive floor is **±0.5pp**. That is the number this project has
needed for six sessions.

**On prefill the six-block `t` interval excludes zero.** `+0.59% [+0.01, +1.16]`
on a pair of binaries that cannot differ. The lower bound clears zero by
0.01pp — one hair, and over the line is over the line. **The project's most
careful interval has a demonstrated false positive**, and it is on the workload
F24/F33 used as its control.

### The prefill false positive is a between-block drift that blocks do not remove

The prefill block estimates rise **monotonically in both runs** — `+0.03 +0.38
+0.57` and `+0.14 +0.89 +1.51`. Three-element monotonicity is a 1-in-6 event
each, so both is roughly 1 in 36 if the blocks were independent draws.

They are not independent, and that is the point:

| where the trend could live | measured | verdict |
|---|---|---|
| inside a block, round to round | per-block `r`(round, b/a) averages **+0.012** (run A) and **−0.045** (run B) | **not there** |
| between blocks, minutes apart | both runs monotonic; dropping each run's first block makes it **worse**, `+0.84% [+0.05, +1.63]` | **here** |

So it is a slow drift at block timescale that moves the two arms unequally — in
run B the A arm is flat over the run (`r = −0.016`) while the B arm rises
(`r = +0.205`). **`--blocks` assumes the per-block point estimates are
independent draws. Consecutive blocks minutes apart are not.** The `t` interval
inherits that, and no number of blocks fixes a trend common to all of them.

This is the same shape as [`F34`](#f34--a-void-run-870-mb-free-against-an-840-mb-model)'s lesson in a new place: F34 found that
sustained contamination makes blocks *agree*, and agreement reads as precision.
F39 finds that a sustained *trend* makes them march, and marching also reads as
precision. **Blocks defend against noise between passes, not against anything
that persists across all of them.** Recorded as **M12**.

### What this does to F24/F33 — mixed, and the honest version is worse in one place

**Good for F24's headline.** F33's decode result is `+1.62% [+1.10, +2.15]` over
six blocks. The decode false-positive floor measured here, same protocol, same
block count, is `-0.04% [-0.49, +0.42]`. F24's effect is **about 3× the whole
width of the null's interval** and its lower bound sits 0.68pp above the null's
upper bound. Nothing in F39 threatens the decode number.

**Bad for the argument F24 was built on.** F24's original strength was
structural: *the arm that should move moves, the arm that should not move does
not*. F33 already reported that prefill had stopped being a clean null (−1.30%
[−2.67, +0.06]). F39 shows something worse — **prefill is not a null on binaries
that cannot differ either**. A control that returns a resolved `+0.59%` when
there is nothing to detect cannot certify the comparison when there is, and
cannot discredit it. **The prefill control should stop being cited as evidence
in either direction at the ~1% level.**

**And it does not rescue M6.** The null's prefill bias is **+0.59%**; F33's
prefill control was **−1.30%** — 1.9pp apart and opposite in sign, so the
harness bias measured here does not explain F33's negative control. Layout
therefore remains a live explanation for it. But F39's A arm sits at 42.50/42.51
tok/s against F33's 39.09, which by this project's own **M5** rule means the two
runs were in different machine states and must not be compared directly. So the
comparison is suggestive and barred at the same time. **M6 is untouched by this
run and still needs the real layout arm (section 5 item A2).**

### Scoring the predictions

| | claim | outcome |
|---|---|---|
| **P39.1** | the decode `t` interval contains zero | **held**, in run A, in run B and pooled. The harness does not invent decode effects |
| **P39.2** | decode `t` width 1.5–2.5pp | **held.** 2.44pp and 2.36pp, against F33's 1.98 and 2.29 |
| **P39.3** | decode block spread 0.4–1.2pp | **held.** 0.89pp and 0.90pp, against F33's 0.77 and 0.82 |
| **P39.4** | decode point estimate within ±0.8pp of zero | **held, at −0.04% in both runs.** This is the false-positive magnitude and it is small |
| **P39.5** | prefill not persistently negative, point above −0.8% | **held in letter (+0.33%, +0.85%) and the inference it was written for is void.** It was designed to separate "layout" from "harness artifact" by seeing whether prefill drifts negative without a layout change. It does not — it drifts *positive*, far enough to resolve. So the workload cannot arbitrate the question in either direction, which is [`F25`](#f25--the-static-overhead-certifies-and-the-shared-builds-answer-is-eaten-by-its-noise-floor)'s **unanswerable design**, and worse than a wrong prediction |
| **P39.6** | at least one bootstrap excludes zero while both `t` intervals contain zero | **failed.** None of the four bootstrap intervals excluded zero, and the interval that produced the run's one false positive was the **`t`**, not the bootstrap |
| **P39.7** | A-arm `tg64` median 36–44 tok/s, blocks agreeing within 2 tok/s | **held.** 42.50 and 42.51; block spreads 0.96 and 0.09 tok/s |
| **P39.8** | run B's `t` interval overlaps run A's | **held, as strongly as it can be.** Identical point estimates and near-identical intervals on decode; overlapping on prefill |

**P39.6 is the one worth more than the seven that held.** M1's charge is that the
bootstrap is roughly half as wide as the truth, so on a genuine null it should
show a raised false-positive rate. It showed none, and the `t` interval — the
fix built to replace it — produced the only false positive in the run. That is
not a rehabilitation of the bootstrap: it is the second confirmation of
[`F33`](#f33--f24-survives-at-162-the-interval-that-can-see-drift-works-and-the-control-stopped-being-clean)'s
corollary, that **the bootstrap is not reliably narrower than the truth, it is
reliably *wrong about* the truth**, and which direction it errs in depends on the
run. On this clean null it erred wide.

### The gate, and a defect it was hiding

`ab_throughput.py` has no baseline gate, so F34's gate-first rule was applied by
hand — which is how **D9** surfaced. `bench_overhead.py` computed its baseline
IQR floor from data **pooled across every block**, so the gate absorbed the
between-block drift that the `t` interval already exists to carry, and charged a
`--blocks` run twice for the same variance. Measured on run A's 60 decode
samples: **2.02% pooled against 1.46% with each block re-centred on its own
median**, the difference being a 2.25% spread in the A-arm block medians. Fixed
in `c18a580`: the gate now uses the worst single block and prints the pooled
figure beside it. `--blocks 1` is unchanged, so no published number moves.

Under the corrected gate this run still reads dirty — worst-block baseline IQR
2.53% and 2.69% on decode, 3.17% and 5.28% on prefill, against a 2% budget.
**That does not undermine a null result, and the reasoning is worth stating
because it inverts.** The gate protects against believing a *positive* claim
from a machine too noisy to resolve it. Extra baseline noise can only inflate a
false positive; none appeared on decode. A null measured on a noisy machine is
the conservative case, so the decode floor of ±0.5pp is if anything an
overestimate of how badly the harness misbehaves when quiet.

### The floor applies at 16 threads, and only there

F39 ran at **16 threads**, matching F33 so the answer would apply to F24. Every
overhead number in [`F36`](#f36--the-overhead-numbers-settle-and-two-f31-claims-are-retracted)
was taken at **8** (gap **G5**). Reading F39's floor against F36's overheads is
therefore a cross-configuration comparison, which is the kind **M5** forbids —
and worth stating because the comparison is unflattering: F36's three intervals
(`[-0.05, +1.16]`, `[+0.38, +1.46]`, `[+0.05, +1.48]`) **all overlap F39's null
`[-0.49, +0.42]`**. If that survives at 8 threads, F36's numbers are at or below
what the harness can distinguish from nothing.

**Measured immediately afterwards, and it went the other way — see
[`F40`](#f40--the-false-positive-floor-is-a-property-of-the-thread-count-and-at-8-threads-this-harness-is-very-good).**
At 8 threads the null's decode interval is `[-0.21, +0.26]`, less than half as
wide, and **every one of F36's point estimates lands outside it**. The overlap
above is an artifact of comparing against the 16-thread floor, which is exactly
the M5 violation this paragraph names. **F36's overheads are measurements.** The
paragraph is left standing because the error is the instructive part.

### An observation about first blocks, deliberately not given a mechanism

The first block of a run has the worst baseline IQR in **three of the four**
run×workload combinations (2.53/2.69% on decode, 5.28% on prefill), and run A's
first block also has the lowest A-arm median (41.76 against 42.72 and 42.55).
That is consistent with the page-cache and standby-list candidate **M10** names
for an 840 MB model read once per invocation.

It is **not** offered as an explanation. Session 6 formed and refuted three
plausible mechanisms in one sitting, one after publishing it, and the cheap part
was the test. The concrete test this suggests: run the same null with a
deliberate cache-warming invocation discarded before block 1, and separately
with `EmptyStandbyList` between blocks, and see whether the first-block penalty
and the prefill trend survive. Neither is done.

---

## P40 — predictions before the 8-thread null, which decides whether F36 measured anything

`HANDOFF.md` section 5 item 1b, opened by F39 rather than planned. F39 measured
the false-positive floor at **16 threads**, to match F33 and so answer the
question about F24. Every overhead number in **F36** — the settled ones, the
ones section 6 of the audit says are safe to quote — was taken at **8**, and all
three of their intervals overlap F39's null interval. Comparing across thread
counts is what **M5** forbids, so the overlap is currently a suspicion rather
than a finding. This run removes the confound: same null pair, same protocol,
`-t 8`, two independent three-block runs.

**P40.1 — the decode `t` interval contains zero at six blocks.** The 16-thread
null did, twice, and there is still nothing to detect. If this fails, the
harness invents decode effects at 8 threads and F36 is in far more trouble than
an overlap.

**P40.2 — the 8-thread decode interval is *narrower* than the 16-thread one**
(half-width under 0.46pp). F10 established that on this machine nothing beats
2.2× and every thread past four turns into barrier wait; more threads means more
of the measurement is scheduling. Eight threads should be the quieter
configuration. If it comes back *wider*, the floor is worse exactly where F36
lives, and F36's numbers are in more doubt rather than less.

**P40.3 — F36's static overhead, `+0.56% [-0.05, +1.16]`, still overlaps this
null's decode interval.** This is the prediction the run exists for. F36's own
interval already contains zero, so the interesting outcome is not whether it
overlaps but by how much: if the null's interval is `[-0.3, +0.3]`-ish then
`+0.56%` sits just outside and F36's static number survives as a marginal
measurement. If the null is as wide as 16 threads', all three of F36's numbers
are indistinguishable from nothing and section 6's provisional wording becomes
the settled one.

**P40.4 — prefill resolves again, and its block estimates rise monotonically
within at least one of the two runs.** This is the test of **M12**, and it is
the one I am least sure of. F39 saw the trend in both runs at 16 threads and
inferred a property of the harness. If prefill comes back clean at 8 threads,
that inference was too fast: the trend would be a property of *16-thread
prefill* and M12 would be overstated in the form it was just written into the
audit. I would rather find that out now than have it found later, so this
prediction is deliberately the one most likely to embarrass the previous
finding.

**P40.5 — the A-arm `tg64` median is 43–47 tok/s, i.e. *faster* than F39's
16-thread 42.50.** A consistency check on F10 rather than on this run: if 8
threads is not faster than 16 on decode, something is wrong with the setup, not
with the statistics. F36's A arms read 45.98/45.99/46.05.

**P40.6 — the corrected gate (D9) passes on at least one of the two runs.**
F39's runs failed it on worst-block IQR (2.53%, 2.69% decode) despite being
clean nulls. Eight threads should be quieter per P40.2, and a run that both
passes the gate and returns a null is the cleanest possible statement of the
floor. If both runs fail the gate again, the floor is being measured on a
machine that never clears its own bar, which is worth knowing and is an argument
about the bar rather than the floor.

---

## F40 — the false-positive floor is a property of the thread count, and at 8 threads this harness is very good

**Workload:** the same null pair as [`F39`](#f39--the-harnesss-false-positive-rate-measured-for-the-first-time-clean-on-decode-and-a-resolved-false-positive-on-prefill) — `bench-stock.exe` vs
`bench-layoutctl.exe`, three bytes apart, one of them code, in a function
`mid.gguf` never enters — at **8 threads** instead of 16. Two independent
three-block runs, `--reps 3`, 20 rounds per arm. Everything else identical.
[`P40`](#p40--predictions-before-the-8-thread-null-which-decides-whether-f36-measured-anything)
holds the predictions. Raw: `data/overhead/f40a.json`, `f40b.json`.

### The result, against F39's

| threads | decode | prefill | A-arm tok/s | worst-block gate |
|---|---|---|---|---|
| **16** (F39) | −0.04% [−0.49, +0.42] | **+0.59% [+0.01, +1.16]** ← resolved | 42.50 | 5.28% **FAIL** |
| **8** (F40) | **+0.02% [−0.21, +0.26]** | **−0.01% [−0.12, +0.11]** | 46.91 | 1.06% **PASS** |

Six blocks each, same binaries, same model, same protocol, ninety minutes apart.

**At 8 threads this harness is much better than anyone here had established.**
The decode floor is **±0.24pp** and the prefill floor **±0.12pp** — half and a
fifth of their 16-thread values. Both gates pass comfortably. Nothing resolves,
which is correct, because there is nothing to detect.

**So F39's floor was not the harness's floor. It was 16 threads' floor.**

### The mechanism is F10, and it was already in the repo

[`F10`](#f10--nothing-beats-22-and-every-thread-past-four-turns-into-barrier-wait)
found that nothing on this machine beats 2.2× and that every thread past four
turns into barrier wait. A measurement at 16 threads is therefore substantially
a measurement of *scheduling*, and scheduling is the variable part. The A-arm
medians say the same thing directly: **46.91 tok/s at 8 threads against 42.50 at
16** — fewer threads, more throughput, less variance, exactly as F10 predicts.

This is stated as a *consistency* with F10, not as a tested mechanism. What is
established is the floor at two thread counts; that F10 explains it is a reading.

### What this does to F39's conclusions — one stands, one is narrowed, one is withdrawn

**Stands: the decode floor, and F24 clearing it.** F24/F33's `+1.62% [+1.10,
+2.15]` was measured at 16 threads, where the floor is ±0.45pp. It clears it by
3.6×. At 8 threads the floor is tighter still. Nothing about F24's decode number
is threatened by either run.

**Narrowed: M12.** F39 found prefill's block estimates rising monotonically in
both its runs and concluded that block estimates are not independent draws, so
the `t` interval has a general failure mode. **At 8 threads there is no trend at
all** — `+0.19 −0.04 +0.05` and `−0.03 −0.13 −0.07`, a 0.22pp and 0.11pp spread,
neither monotonic. The false positive at 16 threads is a real event and the
mechanism it implies is real, but **the generalisation was too broad**: what is
demonstrated is that blocks *can* trend and manufacture a resolved result, in a
configuration where the underlying measurement is already noisy — not that they
generally do. M12 is rewritten accordingly.

**Withdrawn: the claim that F36 measured nothing.** This is the important one and
it was mine, published an hour before this run.

F39's audit entry said all three of F36's overhead intervals overlap the null's,
so none was distinguishable from measuring nothing. That comparison used the
**16-thread** null against F36's **8-thread** numbers — a cross-configuration
comparison that the same paragraph explicitly labelled as forbidden by **M5**.
It was flagged as provisional and it was still wrong to lean on. At the matched
thread count:

| F36 number | interval | vs the 8-thread null `[−0.21, +0.26]` |
|---|---|---|
| ninja-shared | +0.92% [+0.38, +1.46] | **does not overlap the null at all** — a real measurement |
| MSBuild-shared | +0.77% [+0.05, +1.48] | point far outside; intervals touch |
| static | +0.56% [−0.05, +1.16] | point far outside; intervals touch |

All three point estimates sit **outside** the null's whole interval, at 2.3× to
3.8× its half-width. The two marginal rows are marginal because *F36's own*
intervals are wide at three blocks, not because the effect is indistinguishable
from nothing. **F36's overheads are measurements.** The provisional wording in
section 6 of the audit is withdrawn and replaced.

The lesson is the one this project keeps relearning, and it caught the session
that had just written it down: **a comparison you have labelled as forbidden
does not become usable by labelling it.** F39 named the M5 violation, called it
provisional, and drew a conclusion from it anyway. The fifty-minute run that
settles it was available the whole time.

### Scoring the predictions

| | claim | outcome |
|---|---|---|
| **P40.1** | decode `t` contains zero at six blocks | **held.** +0.02% [−0.21, +0.26] |
| **P40.2** | 8-thread decode interval narrower than 16-thread's, half-width under 0.46pp | **held, decisively.** 0.24pp against 0.45pp, and prefill 0.12pp against 0.57pp |
| **P40.3** | F36's static still overlaps the null | **held in letter, and the letter was the wrong question.** The intervals do touch, but every F36 point estimate lands outside the null's whole interval and ninja-shared's interval misses it entirely. The prediction was framed to confirm F39's reading and the data refutes it |
| **P40.4** | prefill resolves again, blocks rise monotonically in at least one run | **failed, and this is the one that earns its keep.** Prefill at 8 threads is the cleanest measurement in either session: ±0.12pp, no monotonicity in either run. Written as the prediction most likely to embarrass F39, and it did |
| **P40.5** | A-arm median 43–47 tok/s, faster than 16 threads | **held.** 46.91 and 46.79 against 42.50, and inside F36's 45.98–46.05 band, so this run and F36 are in comparable machine states — which is what makes the correction above legitimate |
| **P40.6** | the corrected gate passes on at least one run | **held, on both, on both workloads.** Worst-block IQR 0.94/1.06% on decode and 0.79/0.65% on prefill, against a 2% budget. F39's runs failed at 2.53–5.28% |

**P40.4 and P40.3 together are worth more than the four that held**, and they
point the same way: F39 over-read two runs in the noisiest configuration this
project measures in. Both were caught by a fifty-minute control that needed no
judgement call.

### What this changes for how measurements should be taken here

**Measure at 8 threads unless the question is specifically about thread count.**
The floor is half as wide on decode, a fifth as wide on prefill, the gate passes,
and the machine is *faster*. There is no cost to it. Every Tier D number in this
repo was taken at 8 threads except F24/F33, which used 16 because
[`F23`](#f23--ggml-already-solves-core-heterogeneity-for-big-matmuls-and-a-thread-count-can-turn-it-off)
predicted the chunking threshold flips there.

That last point is a genuine constraint, not an oversight: F24's patch changes
`nth * 4` to `nth * 2`, so **the effect itself depends on `nth`** and cannot
simply be re-measured at 8 threads as if it were the same quantity.
`tools/mulmat_chunking.py -t 8,16` says which matmuls flip mode at each count
and should be consulted before assuming the 8-thread number is even comparable.
Recorded as a next step, not done.

---

## P41 — predictions before re-measuring F27's one Tier D row, which is also a test of F40's floor

`HANDOFF.md` section 5 item A3. [`F27`](#f27--ggmls-own-barrier-is-not-a-cheaper-spin-wait-than-openmps-on-this-machine-at-28-threads-it-costs-54-of-decode)
reported four throughput rows. Three (−54.17%, −78.64%, −13.07%) are 13–78× the
drift term and safe at any interval width. The fourth, **−2.06% [−2.42, −1.56]
on `mid.gguf` at 8 threads**, is Tier D, was a single-run bootstrap, and has
never been re-measured — the audit filed the whole finding as Tier B for a
session because of its headline.

**Arms**, both uninstrumented, both relinked from one tree in this session:
`bench-omp.exe` (default, imports `VCOMP140.DLL`) against `bench-noomp.exe`
(`GGML_OPENMP=OFF`, imports none — checked via the PE import table, since
`strings` does not exist in this environment and silently reported zero for
both). `mid.gguf`, 8 threads, `--reps 3`, 20 rounds, `--blocks 3`, run twice.

**This run tests two things at once**, which is why it is worth more than a
tidy-up. `-2.06%` is about **8.5× F40's measured 8-thread decode floor of
±0.24pp**. So it should resolve easily — and if it does not, the floor is not
what F40 says it is, and F40's headline claim ("at 8 threads this harness is
very good") is wrong.

**P41.1 — decode resolves: the six-block `t` interval excludes zero, and stays
negative.** The effect is 8.5× the floor. This is the prediction that would
falsify F40 if it failed.

**P41.2 — the point estimate lands within ±0.8pp of −2.06%**, i.e. inside
[−2.86, −1.26]. The precedent is F24 → F33, where re-measuring with blocks moved
+1.95% to +1.62%, a 0.33pp move. A larger move than 0.8pp would mean the
original number was wrong rather than merely over-precise.

**P41.3 — the six-block `t` interval is *narrower* than F27's bootstrap width of
0.86pp.** This is the interesting one and it runs against the project's
house reading of M1. The naive expectation from F31/F33 is that a `t` interval is
about twice the bootstrap. But F40 measured the 8-thread machine as quiet — the
null's whole six-block interval was 0.47pp wide, and block spreads were 0.53 and
0.44pp — while F27's bootstrap came from a *single* 20-round run and inherited
its outliers. F33's P32.5 already found the bootstrap coming out *wider* than
the `t` on a dirty run. If this fails, M1's factor-of-two holds even on a quiet
configuration and I have over-read F40's floor.

**P41.4 — prefill resolves this time, reversing F27.** F27 reported prefill at
**−1.15% [−1.62, +0.12]**, which did not clear zero. F40 put the 8-thread
prefill floor at **±0.12pp**, so −1.15% is roughly **10× the floor** and should
resolve comfortably. A barrier change has every reason to move prefill — F27
measured −45% there at 28 threads. If prefill comes back *unresolved* again, the
prefill floor from F40 is too optimistic.

**P41.5 — the OpenMP arm's median is 45.5–47.5 tok/s**, comparable to F40's
46.91 and 46.79. This is the M5 check that makes comparing against F40's floor
legitimate at all, and it is the mistake F39 made. If it comes back at 42 or 39,
the floor comparison is void and this run says nothing about F40.

**P41.6 — the corrected gate passes on both runs**, as it did on both of F40's
(worst-block IQR 0.65–1.06% against a 2% budget).

### What each failure would mean

- P41.1 fails → F40's floor is wrong, and the 8-thread configuration is not the
  quiet one it appeared to be.
- P41.3 fails → the bootstrap-vs-`t` relationship is stable after all, and F40's
  "quiet machine" reading was over-read from a null.
- P41.4 fails → the prefill floor specifically is too optimistic, which matters
  because M13 currently says prefill is a good control at 8 threads.
- P41.5 fails → nothing else in this run is interpretable, and it should be
  re-run rather than reported.

---

## F41 — F27's last Tier D row survives at −2.15%, its prefill row was wrong by half, and a three-block interval's *width* is unstable by 7×

**Workload:** `mid.gguf`, **8 threads**, tg64 and pp64, `--reps 3`, 20 rounds per
arm, **`--blocks 3` run twice** (six blocks). Arms both uninstrumented and
relinked from one tree in this session: `bench-omp.exe` (default) against
`bench-noomp.exe` (`GGML_OPENMP=OFF`).
[`P41`](#p41--predictions-before-re-measuring-f27s-one-tier-d-row-which-is-also-a-test-of-f40s-floor)
holds the predictions. Raw: `data/overhead/f41a.json`, `f41b.json`.

`HANDOFF.md` item A3. This was the one row of
[`F27`](#f27--ggmls-own-barrier-is-not-a-cheaper-spin-wait-than-openmps-on-this-machine-at-28-threads-it-costs-54-of-decode)
that had never been re-measured, and the reason the audit had the whole finding
mis-filed as Tier B for a session.

### Arms verified before measuring, and one check that silently did not run

`bench-omp.exe` imports **`VCOMP140.DLL`**; `bench-noomp.exe` imports **none** —
18 imports against 17, read out of the PE import table.

The first attempt at this check used `strings`, **which does not exist in this
environment**, and the shell guard turned the missing binary into "0 VCOMP
mentions" for *both* arms. That reads exactly like a passing check. **A check
that silently passes for the wrong reason is worse than no check**, and this one
would have let a mislabelled pair of arms through. It is recorded because F27's
own arm-verification was the thing being reproduced.

### The result

| | decode (tg64) | prefill (pp64) |
|---|---|---|
| **F27** (single-run bootstrap) | −2.06% [−2.42, −1.56] | −1.15% [−1.62, +0.12], *unresolved* |
| run A (3 blocks) | −2.16% [−2.23, −2.10] | −0.49% [−0.67, −0.31] |
| run B (3 blocks) | −2.14% [−2.62, −1.67] | −0.37% [−1.14, +0.40], *unresolved* |
| **pooled, six blocks** | **−2.15% [−2.28, −2.02]** | **−0.43% [−0.65, −0.21]** |

Both gates pass (worst-block IQR 0.50% and 0.72%). The OpenMP arm reads **46.78
tok/s in both runs** against F40's 46.91/46.79, so the machine is in the same
state and the comparison against F40's floor is legitimate.

**Decode: F27's row survives, essentially unchanged.** −2.06% becomes −2.15%,
a 0.09pp move, with F27's point estimate inside the new interval. The effect is
**9.0× F40's measured 8-thread decode floor**. F27's last Tier D row is now
Tier D, properly measured, and the finding no longer carries a row that has
never been checked.

**Prefill: F27's row was wrong by more than half.** −1.15% becomes **−0.43%
[−0.65, −0.21]**, and F27's point estimate sits *outside* the new interval. The
direction was right and the effect is real — it is 3.6× the prefill floor and
resolves where F27's did not — but the magnitude was overstated by 2.7×. This is
not over-precision, which is what M1 predicts; it is a **wrong point estimate**,
and it came from the same single-run bootstrap that produced the decode row that
reproduced fine.

### The finding nobody predicted: a three-block interval's width is unstable by 7×

The two runs are the same comparison, the same binaries, the same protocol,
minutes apart. Their decode intervals:

```
  run A   -2.16%  [-2.23, -2.10]     0.14pp wide   block spread 0.05pp
  run B   -2.14%  [-2.62, -1.67]     0.95pp wide   block spread 0.37pp
```

The point estimates agree to **0.02pp**. The interval widths differ by **7×**,
and run A's interval sits entirely *inside* run B's.

Run A's three blocks agreed to **0.05pp — the tightest agreement in this
project's history**. Had this session run only run A, it would have published
±0.07pp precision on a quantity whose honest six-block interval is ±0.13pp, and
the tightness would have looked like a triumph of method.

**This is [`F34`](#f34--a-void-run-870-mb-free-against-an-840-mb-model)'s lesson
in its benign form.** F34 found blocks agreeing on a physically impossible
result and warned that agreement reads as precision. F41 shows the same thing
with nothing wrong at all: **block agreement is itself a random variable**, and
three draws can happen to land on top of each other. The `t` interval is
computed *from* that spread, so when the spread is small by luck the interval is
tight by luck.

**So `--blocks 3` from a single invocation is not enough**, and the practical
rule the project has been following by habit should be stated: **two runs of
three blocks, pooled to six.** F33, F39, F40 and F41 all did this; **F36 did
not** — its three settled overhead numbers come from a single three-block run
each, so their *widths* carry this instability even though their point estimates
are the best available. Recorded as **M14**.

The gate-first ordering is what kept run A honest. Tight blocks plus a
physically expected result and a matching baseline median is a real measurement;
tight blocks plus a physically impossible result is F34. **Agreement alone
distinguishes neither.**

### Scoring the predictions

| | claim | outcome |
|---|---|---|
| **P41.1** | decode resolves and stays negative | **held.** −2.15% [−2.28, −2.02], 9.0× the floor. F40's floor is not contradicted, which was the other thing this run was for |
| **P41.2** | point estimate within ±0.8pp of −2.06% | **held** at 0.09pp. F27's decode row was over-precise, not wrong |
| **P41.3** | the six-block `t` is *narrower* than F27's 0.86pp bootstrap | **held.** 0.26pp, 3.3× narrower — and run A's alone was 0.14pp, six times narrower. The house reading of M1 ("`t` is about twice the bootstrap") is a property of noisy configurations, not of the estimators |
| **P41.4** | prefill resolves, reversing F27 | **held, and it exposed something the prediction did not ask about.** It resolves at −0.43%, less than half F27's −1.15%, with F27's estimate outside the new interval |
| **P41.5** | OpenMP arm median 45.5–47.5 tok/s | **held.** 46.78 in both runs, against F40's 46.91/46.79 |
| **P41.6** | the corrected gate passes on both runs | **held.** 0.50% and 0.72% against a 2% budget |

**Six of six is not a good sign on its own** — it mostly means the predictions
were made after F40 had measured the floor, so they were cheap. The two things
worth keeping came from outside the prediction set: prefill's magnitude being
wrong rather than imprecise, and the 7× width instability.

### What this does to F27

F27's four throughput rows are now: three large ones untouched (−54.17%,
−78.64%, −13.07%, all 13–78× the drift term), **decode at 8 threads confirmed at
−2.15%**, and **prefill at 8 threads corrected from −1.15% to −0.43%**. The
finding's conclusion — that ggml's own spin-wait barrier is not the cheap one on
this machine — is unaffected and better supported than before. Its Tier D row is
no longer unchecked.

---

## F42 — F24's 1.1 MB of "regenerated" code is a uniform 16-byte shift, and M6 becomes a question about one number

**Artefacts, not a workload.** Fresh `build-ts-off` builds of stock and
`patches/03-mulmat-chunk-threshold.patch`, plus a `/MAP` build of each. No
throughput measured. This corrects
[`F38`](#f38--f24s-two-arms-differ-in-31-of-their-code-section-and-the-control-built-to-test-it-does-not).

### F38's number reproduces exactly

Stock vs F24's patched arm: **1,137,994 bytes differ** — the same figure F38
reported, from independently rebuilt binaries. The fact is solid. F38's
*interpretation* of it is not.

### What the linker map says

A `/MAP` build was added (`build-map`) and verified image-neutral first: the
`/MAP` binary differs from the ordinary one in **4 bytes**, the PE timestamp and
its echo, so the map describes the binary being measured.

Both arms contain **14,419 functions** in `.text`. Comparing every function's
address between them gives **exactly two distinct deltas**:

| delta | functions |
|---|---|
| **0** | 6,698 |
| **−16** | 7,721 |

And the pivot is the patched function itself:

```
  ggml_compute_forward_mul_mat      0x262e20 -> 0x262e20   (delta 0)
    size, stock                       3664 bytes
    size, patched                     3648 bytes           (-16)
  everything at a higher address    shifted by exactly -16
```

**`mul_mat` shrinks by 16 bytes and every function after it moves down by 16.**
That is the whole of the 1.1 MB. There is no regeneration, no reordering, and
nothing else changed anywhere in the image.

### Why F38 concluded otherwise, and why its test could not have worked

F38 wrote:

> No constant shift explains it (`p[base+k] == s[base]` fails for every `k` in
> ±4096 at a probe inside the block), so this is not simple relocation of an
> otherwise-identical image.

**That inference is wrong, and the method could not have found the shift it was
looking for.** Under a uniform relocation, code bytes move *and* every absolute
address embedded in them changes too, because the targets moved as well. A
256-byte window containing any relocated absolute address will not match
byte-exactly at any offset, including the correct one. F38's own probe results
say this in hindsight — 14 matched (windows with no embedded absolute address)
and 22 did not (windows with one). It read 22 misses as evidence of
regeneration when they are the expected signature of relocation.

**The lesson is about tools, not about this patch.** A byte-window search cannot
distinguish relocation from regeneration in code containing relocations. The
linker map answers in one command what the probe method got backwards, and it
was available the whole time.

### What M6 actually is now

M6 has been "an unknown fraction of F24's +1.62% may be layout, and a third of
the image differs so nobody can reason about it." It is now a precise question:

> Does moving every function after `mul_mat` down by **16 bytes** change decode
> throughput?

That is a far smaller claim to test, and it is testable, because a control that
produces a 16-byte downstream shift with no behavioural change is
constructible — which is what F38 concluded did not exist.

**And the hot path is genuinely affected**, so this is not a question that can be
waved away:

| function | stock | patched | |
|---|---|---|---|
| `ggml_vec_dot_f32` | `0x290fc0` | `0x290fb0` | **−16** |
| `ggml_graph_compute_thread` | `0x265e00` | `0x265df0` | −16 |
| `ggml_compute_forward_mul_mat` | `0x262e20` | `0x262e20` | 0 |
| `ggml_barrier` | `0x2620a0` | `0x2620a0` | 0 |

`ggml_vec_dot_f32` is the hottest function in an F32 decode, and it goes from
`0xc0` (**64-byte aligned**) to `0xb0` (**48 mod 64**). Loop alignment inside it
shifts correspondingly. That is a textbook mechanism for a sub-percent
throughput change, and it is exactly the thing F24's design cannot separate from
the scheduler change.

Recorded as a mechanism that is **plausible and untested** — naming it is not
measuring it, which is the mistake session 6 made three times. The control is
the next step.

---

## P43 — predictions before the layout arm, the experiment M6 has needed since session 4

`HANDOFF.md` section 5 item 2, and the last open item in group A.
[`F42`](#f42--f24s-11-mb-of-regenerated-code-is-a-uniform-16-byte-shift-and-m6-becomes-a-question-about-one-number)
turned M6 from "a third of the image differs" into one question:

> Does moving the hot code by **16 bytes** change decode throughput?

**Arms:** `bench-stock.exe` against `bench-pad.exe`
(`patches/05-layout-arm.patch`), both uninstrumented, both built from one tree
in this session. `mid.gguf`, **8 threads** (where F40 put the floor at ±0.24pp),
`--reps 3`, 20 rounds, `--blocks 3`, run twice.

**Verified before measuring**, which is the step F38's control skipped: exactly
two address deltas across all 14,419 `.text` functions (0 for 6,689, **+16** for
7,730); `mul_mat` the same size in both and 98.6% byte-identical after its move,
the 51 differing bytes being relocated addresses; `ggml_vec_dot_f32` moved +16,
from 0 to 16 mod 64.

**P43.1 — decode does not resolve: the six-block `t` interval contains zero.**
This is the prediction I am least confident in and the whole point of running
it. The reasoning for it: F14 established that decode on this machine is
**bandwidth-bound**, and a bandwidth-bound streaming loop is far less sensitive
to instruction alignment than a short compute-bound one. The reasoning against
it: alignment effects of around a percent are a well-documented hazard in
exactly this kind of measurement, and `ggml_vec_dot_f32`'s alignment mod 64 does
change.

**P43.2 — |point estimate| < 0.5pp.** Even if something moves, I expect it small.

**P43.3 — whatever decode does, |layout| < 1.0pp**, so layout cannot account for
all of F24's +1.62%. This is the prediction that matters for M6: F24 survives as
a real effect if this holds, and is in serious trouble if the layout arm returns
something like +1.5%.

**P43.4 — prefill does not resolve either.** At 8 threads F40 put the prefill
floor at ±0.12pp and prefill was the cleanest thing measured. A layout effect
would have no obvious reason to prefer one phase.

**P43.5 — the gate passes on both runs and the A-arm median is 46–47.5 tok/s**,
matching F40's 46.91/46.79 and F41's 46.78. Without this the comparison against
F40's floor is void, which is the mistake F39 made.

### What each outcome means for M6

- **Unresolved, small** → a 16-byte shift is not worth a percent here, F24's
  +1.62% is attributable to the scheduler change, and M6 closes as far as this
  machine can close it.
- **Resolved and small** (say +0.3%) → layout is real but minor; F24 keeps most
  of its effect and gains a stated uncertainty.
- **Resolved and large** (approaching +1.6%) → F24's number may be substantially
  layout, and the finding needs restating rather than caveating.

**The asymmetry that limits all three:** F24's arm shifts by **-16** and this one
by **+16**. Same magnitude, same hot function, opposite direction, and an
alignment effect need not be symmetric. A null here bounds the effect for this
perturbation; it does not prove layout never matters.

---

## P44 — a prediction about the machine, written while it is visibly slow

Unplanned, and written before the probe loop that tests it. F43's run A came
back with an A-arm median of **40.37 tok/s** where F40 and F41 had **46.78-46.91
four times in a row**, and with a failing gate (worst-block IQR 2.65%). By the
rule P43.5 stated in advance, that run is not comparable to F40's floor and is
void. Nothing else was running: CPU load 1%, 6.2 GB free.

**What is different from every previous slow reading:** this session has just put
the machine through **six full or near-full rebuilds** of llama.cpp, including a
258-target configure-and-build for the `/MAP` configuration. [`F37`](#f37--four-hypotheses-for-m10-all-refuted-including-one-this-session-published)
refuted "thermal" using a 7-8 minute cooldown after *measurement* load and a
30-run curve that was flat to -1.16%. Sustained compilation load is a different
and much heavier thing, and F37 did not test recovery from it.

A single probe taken immediately after the void run read **42.76 tok/s** —
already up from 40.37, which is why this is worth a curve rather than a guess.

**P44.1 — throughput recovers toward ~46.8 tok/s over the next ten minutes**,
and the curve is monotonic-ish rather than flat. If it holds, M10 gains its
first confirmed component: *recovery from sustained compilation load, on a
timescale of minutes*, which is a mechanism F37's design could not have seen.

**P44.2 — it does not fully reach 46.8 within 12 minutes.** The recovery from
40.4 to 42.8 took roughly two minutes, so the remaining 4 tok/s should take
longer than the window if the process is asymptotic.

**P44.3 — if instead the curve is flat at 42-43**, then this is not
time-recovery at all and the slow state is something the machine entered and is
holding. That would make it look much more like the page-cache/standby-list
candidate the audit names, since the builds wrote hundreds of megabytes and
would have pushed the 840 MB model out of cache.

**The measurement is the only thing running.** Each probe is one `llama-bench`
invocation, `-p 0 -n 64 -t 8 -r 3`, 55 seconds apart, 13 of them.

---

## F43 — void: the layout arm's first run

**Recorded rather than discarded**, per the project's habit of keeping failed
runs. Raw: `data/overhead/f43a-void.json`.

`bench-stock.exe` vs `bench-pad.exe` (`patches/05-layout-arm.patch`), 8 threads,
three blocks, the protocol [`P43`](#p43--predictions-before-the-layout-arm-the-experiment-m6-has-needed-since-session-4)
specified. It reported `tg64 +0.14% [-0.31, +0.59]` and `pp64 +0.35% [-0.35,
+1.06]`, neither resolved.

**It is void on two independent counts, both checked before the numbers were
read:**

1. **P43.5, written before the run.** The A-arm median is **40.37 tok/s**, where
   F40 and F41 read 46.78–46.91 across four consecutive runs. P43.5 said in
   advance that without a matching median the comparison against F40's floor is
   void. It is.
2. **The gate.** Worst-block baseline IQR **2.65%** on decode and 2.51% on
   prefill, against a 2% budget.

The A-arm was stable *within* the run (40.49, 40.32, 40.37), so this was a
sustained state rather than drift — and [`F44`](#f44--m10-caught-in-the-act-the-machine-holds-two-regimes-for-minutes-at-a-time-and-switches-between-them-unprompted)
then identified what it was.

**The unresolved result is probably right**, since interleaving protects the
internal A/B, and it agrees with P43.1. **It is still not evidence.** A run whose
baseline is 14% off and whose gate failed does not get to contribute because its
answer looks agreeable — that is the failure mode this whole apparatus exists to
prevent, and the temptation was real.

The layout arm is re-measured in the fast regime instead.

---

## F44 — M10 caught in the act: the machine holds two regimes for minutes at a time and switches between them unprompted

**Workload:** thirteen identical `llama-bench` probes, `mid.gguf`, `-p 0 -n 64
-t 8 -r 3`, **55 seconds apart, nothing else running on the machine**. Provoked
by [`F43`](#f43--void-the-layout-arms-first-run)'s void run.
[`P44`](#p44--a-prediction-about-the-machine-written-while-it-is-visibly-slow)
holds the predictions, all of which were essentially wrong.

### The curve

```
   t(s)   tok/s
      0   39.99      <- immediately after six rebuilds
     61   45.46
    120   46.05
    180   46.16      fast regime, four consecutive probes
    240   46.13
    299   41.00      <- switches, unprompted
    359   41.15
    420   40.56
    480   40.63
    540   41.04      slow regime, eight consecutive probes
    600   38.52
    661   40.84
    721   41.05
```

| regime | probes | median | range |
|---|---|---|---|
| fast (t=61–240) | 4 | **46.09** | 45.46–46.16 |
| slow (t=299–721) | 8 | **40.92** | 38.52–41.15 |

**A 12.6% gap between two states the machine holds for minutes at a time**, with
an unprompted transition between them, under a workload that never varied.

### What this adds to M10, and what it takes away

The audit says M10 is "bimodal rather than smooth" and that "the full range
appears **inside one 4.5-minute window**, so it is run-to-run variance whose
median moves, **not two stable regimes**."

**That last clause is now doubtful.** This is two stable regimes: four
consecutive probes at 46.09 ± 0.35 and then eight at 40.92 ± 1.3, with a clean
step between them. The 46.09 band is exactly where F40 and F41 sat for four
consecutive 25-minute runs, and the 40.92 band is exactly where F43's void run
sat for all three of its blocks. **The regimes are the same ones the measurement
runs live in**, which is why an A-arm median identifies them so reliably.

What it does *not* explain is the switch. Nothing changed: same binary, same
model, same thread count, same 55-second cadence, CPU otherwise at 1%, 6 GB
free. The machine simply stopped being fast at around t=290 and stayed slow for
the next seven minutes.

### Scoring the predictions, all three of which missed

| | claim | outcome |
|---|---|---|
| **P44.1** | recovery toward ~46.8 over ten minutes, monotonic-ish | **failed.** It recovered in **two** minutes, not ten, and then went *back down* and stayed. Monotonic recovery is the wrong shape entirely |
| **P44.2** | it does not fully reach 46.8 within 12 minutes | **held in letter, and the letter is worthless.** It reached 46.16 and the reason it stopped there is that 46.1 *is* the fast regime's ceiling, not that recovery was still in progress |
| **P44.3** | a flat 42–43 curve would point at page-cache | **failed** — the curve is not flat, and the disjunction the prediction set up (recovery *or* stuck) did not contain the answer |

**All three predictions shared one wrong assumption**: that the machine was
*recovering from* something I had done to it, so the curve should have a
direction. It does not have a direction. It has **states**. The six rebuilds
probably did cause the initial 40.0, since the first probe recovered within a
minute — but that recovery finished long before the interesting thing happened.

This is the most useful prediction failure of the session, because the whole
prediction set was built on a frame that the data discards.

### What it means for every measurement here

**A 25-minute run can span a regime switch**, and the two regimes differ by 12.6%
— roughly **twenty times** the effects being measured. Three consequences:

1. **The A-arm median is doing more work than the gate.** It identifies which
   regime a run lived in. F40/F41 at 46.78–46.91 were entirely in the fast
   regime; F43's void run at 40.32–40.49 was entirely in the slow one.
2. **Interleaving is what makes any of this work.** A regime switch mid-run hits
   both arms nearly equally because they alternate every round, which is why
   F43's *internal* A/B (+0.14%) is probably sound even though the run is not
   comparable to F40's floor. The baseline-IQR gate is what catches the
   inflation a switch causes, and it did.
3. **M14 may have this as its mechanism.** F41's run A blocks agreed to 0.05pp
   and run B's to 0.37pp. A run wholly inside one regime should produce the
   former; a run containing a switch, the latter. Stated as a candidate, **not
   tested** — it needs per-round timestamps that the harness does not currently
   record.

### The cheap thing this suggests, and which is now done

`ab_throughput.py --json-out` recorded measurements in order but **no
timestamps**, so no past run can be checked for a regime switch. Every run from
here on carries a `timeline` array — wall-clock offset, block, round, arm, test
and value for every measurement — so the question "did this run span a switch"
is answerable from the raw file.

This was first written up as deliberately *not* done, on the grounds that the
session had already published enough untested mechanisms. That was the wrong
call and it is corrected here: **adding a clock is not publishing a mechanism,
it is building the instrument that can test one.** The M14 hypothesis above
stays untested either way — it needs timestamps on *past* runs, which do not
exist — but every future run can now settle it.

---

## Not yet measured

Listed so the gaps are explicit rather than implied:

- larger real models — the biggest measured is now 8.19 B (F19, F21);
  nothing above that, and **no MoE model at all**, which is the most obvious
  gap in the byte law's coverage
- server workloads with real arrival and eviction patterns (F17 covers
  `llama-batched` only, up to 16 sequences)
