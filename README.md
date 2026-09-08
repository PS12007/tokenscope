<h1 align="center">tokenscope</h1>

<p align="center">
  <b>A per-token inference profiler for llama.cpp.</b><br>
  Attributes the wall-clock time of <i>every generated token</i> to specific phases of the
  forward pass, and emits Chrome Trace Event Format JSON you can open in Perfetto.
</p>

<p align="center">
  <a href=".github/workflows/ci.yml"><img alt="ci" src="https://github.com/PS12007/tokenscope/actions/workflows/ci.yml/badge.svg"></a>
  <a href="#status"><img alt="status" src="https://img.shields.io/badge/status-work%20in%20progress-orange"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <img alt="c++17" src="https://img.shields.io/badge/C%2B%2B-17-informational">
  <img alt="deps" src="https://img.shields.io/badge/runtime%20deps-none-success">
  <img alt="patch size" src="https://img.shields.io/badge/upstream%20patch-3%20files-success">
</p>

---

## The problem

Your llama.cpp run is producing 18 tok/s. You expected 30. Why?

The honest answer most people give is a shrug and a guess about memory bandwidth.
The built-in timings help a little:

```
llama_perf_context_print: prompt eval time =    412.11 ms /    18 tokens
llama_perf_context_print:        eval time =  10233.40 ms /   255 runs
```

Two aggregate numbers for an entire run. They cannot tell you that:

- token 340 stalled because the KV cache was reallocated mid-generation,
- sampling is quietly eating 8% of your decode budget,
- layer 17's attention is disproportionately slow because of how the cache is laid out,
- or that 40% of your "compute" time is threads parked at a spin barrier.

Worse, that split is a heuristic. llama.cpp decides "prefill vs decode" by asking
whether the batch had more than one token, and there is a `FIXME` in the source
acknowledging it misattributes when a caller does not synchronize per token
([`llama-context.cpp:714`](docs/00-architecture-map.md#1-the-prefill--decode-boundary)).
The numbers everyone quotes are aggregates over however many tokens happened
between synchronization points.

## What it does

A build-time-optional instrumentation layer that records explicitly placed timing
scopes into lock-free per-thread buffers and flushes them as
[Chrome Trace Event Format](https://ui.perfetto.dev) JSON — the format Perfetto
and `chrome://tracing` consume. You get a flamegraph-style timeline of a real
inference run, with zero custom UI work.

No tool needs modifying. Set two environment variables:

```console
$ TOKENSCOPE_LEVEL=1 TOKENSCOPE_OUT=run.trace.json \
    llama-bench -m model.gguf -p 512 -n 256 -t 8

tokenscope: wrote run.trace.json (316763 bytes, 259 tokens, 1 threads, 0 dropped)
```

Then ask it questions:

```console
$ python tools/trace_analyze.py run.trace.json

=== run.trace.json ===
    clock=steady_clock  level=1  threads=1  dropped=0

prefill: 2 batch(es), 1681.4 ms

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

  p50  23.92 ms   p95  26.25 ms   p99  28.75 ms   max  33.05 ms
```

That first row is a finding, not a formality: **host-side control-plane work is
0.4% of decode.** The stall hypotheses this project was pitched on — KV
reallocation, logits buffer reserve, graph rebuild — are each instrumented, and
each is a rounding error on a steady-state generation loop. `graph-build` totals
206 µs across 257 tokens, which says llama.cpp's graph reuse essentially never
misses. More in [`docs/FINDINGS.md`](docs/FINDINGS.md).

At `TOKENSCOPE_LEVEL=3` the same command breaks the graph open:

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

  11.2% of worker thread time is barrier wait, not compute.
  99.3% of available thread time is inside a node scope.
```

**11.2% of the CPU budget is threads spinning at a barrier, not computing.**
`ggml_barrier` runs after every graph node — roughly 700 times per token, on
each of 8 threads. Without a work/wait split that time is indistinguishable from
compute, which is why "attention took X ms across 8 threads" is a sentence that
can hide seven idle threads.

And `--layers` groups the same data by layer:

```
  layer         ffn    attn.qkv    attn.out  attn.score        norm       total
  ---------------------------------------------------------------------------
      0      5362.4       808.8       456.8        66.6         8.1      6706.9
      1      5457.4       796.3       450.4        58.8         7.6      6774.6
    ...
     23      5291.3       790.6       437.5        51.2         8.5      6591.4

  median layer 6668.7 us/tok   slowest L5 6824.6 (1.02x)   fastest L15 (0.98x)
  Layers are uniform to within 10%.
```

`--barriers` answers the question the 11.2% raises but cannot settle on its
own: how much of that wait is recoverable?

```
  barrier decomposition -- 8 threads, 824 barriers over 2 decode tokens

  total barrier wait      62.31 ms   thread-time
    arrival imbalance     34.59 ms    55.5%   threads idle, waiting for the last
    after last arrival    27.72 ms    44.5%   release latency and spin-up

  worst nodes by imbalance

  node                   work   imbalance     n  threads busy
  -------------------------------------------------------------
  ffn_out            88.93 ms    10.19 ms    48          8.0
  ffn_swiglu         267.0 us     1.75 ms    48          1.8
  l_out              204.3 us     1.30 ms    48          2.0
  attn_norm          211.9 us    657.9 us    48          1.8
```

Read the last column. **The tiny elementwise nodes run on 1.4–2.0 threads out
of 8** — because ggml partitions over rows, and at batch size 1 a hidden state
is a single row, so thread 0 takes it and the other seven fall through to the
barrier. Ten node types cost more in other threads' waiting than in their own
arithmetic: 0.32% of the work causing 14% of all the imbalance.

That is a mechanism, so it makes a prediction: the same nodes should
parallelize normally when there *are* many rows. Prefill is that workload, and
they do — 1.4 busy threads becomes 7.7. The upper bound on fixing it is a
deliberately unflattering **1.34% of graph wall time** ([`F9`](docs/FINDINGS.md)).

F9 also guessed at a fix — fuse the elementwise chain so it pays one barrier
instead of five — and [`F15`](docs/FINDINGS.md) tested that guess and killed
it. ggml already fuses exactly this kind of chain, removing 49 of 461 barriers
per token; turning that fusion off changes throughput by nothing measurable,
even on a model where barrier wait is 54% of thread time. The barriers a fusion
removes are the ones threads arrive at together. **Barrier count and barrier
cost are different quantities**, and the waiting that has real wall-clock cost
is somewhere else — see the thread sweep below.

Sweeping thread count turns that into advice you can act on today:

```
              synthetic F32 (220M)       Qwen2.5-0.5B Q4_K_M (630M)
 thr   tok/s  speedup  par.eff       tok/s  speedup  par.eff
   1   20.25    1.00x     100%       27.06    1.00x     100%
   4   42.34    2.09x      52%       77.49    2.86x      72%
   6   44.82    2.21x      37%       88.88    3.28x      55%   <- peak, both
   8   44.59    2.20x      28%       88.20    3.26x      41%
  16   41.50    2.05x      13%       68.91    2.55x      16%
  28   39.33    1.94x       7%       66.07    2.44x       9%
```

**Nothing beats 2.2× on the F32 model and 3.3× on the quantized one.** Four
threads already reach most of it; 28 threads is *slower* than six on both,
while occupying seven times the cores. Parallel efficiency ends at 7-9%.

The barrier is where the wasted parallelism becomes visible rather than where
it is created — decode is weight-streaming, so once memory bandwidth saturates
the extra threads cannot go faster, and the difference is paid at the next
rendezvous. Barrier wait rises monotonically with thread count, to 22.9% of
worker time on the F32 model and 42.5% on the quantized one.

The two columns are also a prediction and its test: [`F10`](docs/FINDINGS.md)
argued from the F32 numbers that a model reading fewer bytes per parameter
should scale further, and [`F12`](docs/FINDINGS.md) measured it doing exactly
that — 2.21× to 3.28×. The wall moved up. It did not move out.

At 8B the wall moves back down, to **3.01×**, because absolute traffic is what
sets it — 12.4× more bytes per token beats the model being more compressed. The
same sweep on prefill, which is compute-bound rather than bandwidth-bound, is
the control:

```
 threads    decode    prefill        same model, same weights, same cores
       1     1.00x      1.00x
       6     2.93x      5.97x
       8     3.01x      7.38x
      28     2.95x     11.07x
```

Prefill scales to 11× and decode flatlines at 3×. The only difference is how
many tokens share each weight read ([`F19`](docs/FINDINGS.md)).

Across three models the answer to "where does decode time go" is a different
phase every time:

```
model                 params      ffn  lm_head  barrier
tiny   8L F32 synth     8.9 M   27.6%     0.9%    54.0%
mid   24L F32 synth     220 M   70.1%     2.6%    10.1%
Qwen2.5-0.5B Q4_K_M     630 M   49.2%    29.6%    11.3%
Qwen3-8B     Q4_K_M    8.19 B   63.3%     9.9%     5.6%
```

Barrier on the smallest, the FFN stack in the middle, and on a real 0.5B model
the **output projection at 29.6%** — a 151,936-token vocabulary against
`n_embd` 896, stored at higher precision than anything else in the file. There
is no model-independent answer, which is the argument for measuring rather than
reasoning ([`F13`](docs/FINDINGS.md)).

The 8B row is the same vocabulary — **151,936, unchanged** — against a model
15× larger everywhere else, and `lm_head` falls to 9.9%. Barrier wait falls too,
because bigger matmuls amortize the same barriers over more work. A small model
with a big vocabulary is its own regime, and the 0.5B row does not generalize
([`F19`](docs/FINDINGS.md)).

And serving concurrently changes the answer again. At `-np 16` on the same
model, sixteen sequences cost nothing like sixteen times:

```
phase        np=1 %   np=16 %   work x
ffn           47.8%     58.4%     6.1x
lm_head       29.0%     21.5%     3.7x
barrier       13.6%      7.2%     2.7x
attn.score     1.1%      2.2%     9.8x
```

Weight-bound phases amortize hard — a weight matrix is read once per step
however many sequences ride along, so `lm_head` does 16× the arithmetic for
3.7× the time. Sequence-bound work does not: `attn.score` is 9.8×, because
every sequence has its own KV. **Batching does not just make decode faster, it
changes what decode is** — and it dissolves the single-threaded-node problem
above entirely, 1.0 busy threads becoming 6.0
([`F17`](docs/FINDINGS.md)).

Because the phases are separated, the profile answers a question the aggregate
timers cannot: **what predicts where decode time goes?** Weight *bytes* do, and
parameter counts do not — a distinction with no meaning on an F32 model and a
large one on a quantized file, where llama.cpp stores different tensors at
different precisions.

The clean test is inside the quantization recipe. Q4_K_M stores `ffn_down` at
Q6_K in 18 of Qwen3-8B's 36 layers and Q4_K in the other 18 — the same tensor,
the same shape, the same op, differing only in dtype. `ffn_gate` and `ffn_up`
are Q4_K in every layer and act as controls:

```
                Q6_K layers    Q4_K layers    ratio
  ffn_out        38516.6 us     27138.5 us    1.419     <- the test
  ffn_gate       26735.9 us     26485.5 us    1.009     <- control
  ffn_up         26681.6 us     26506.4 us    1.007     <- control

  predicted from bytes alone:  6.5625 / 4.5 = 1.458
```

1.419 measured against 1.458 predicted, with both controls flat. Across the
whole model, predicting each phase's share of time from its share of bytes is
accurate to **0.3 points**; predicting from parameter counts is wrong by 4.5
([`F19`](docs/FINDINGS.md)). `tools/model_bytes.py` prints that table for any
GGUF without running it.

`--outliers` ranks the slowest tokens and attributes each one's *excess over
median* to a category — because on a slow token everything is large, and the
question is which thing is large **for that token**. `--diff` compares two
traces, because during optimization work the question is always *did my change
help, and where*.

## Design constraints

Hard rules, not aspirations. They are what separates a profiler from a lie about
where your time goes.

| Constraint | How it is enforced | Status |
|---|---|---|
| **Zero overhead when disabled** | Everything behind `TOKENSCOPE_ENABLED`. Off ⇒ macros expand to nothing; no symbol, no branch, no storage. | ✅ verified against the symbol table |
| **Under 2% when enabled** | Measured with interleaved arms and bootstrap CIs, not assumed. | ✅ all levels; level 3 is +0.67% [+0.12, +1.67] at 8 threads. Re-measured at 28 threads: +0.66% [-1.85, +4.18], which the harness declined to certify — see [F10](docs/FINDINGS.md) |
| **No new dependencies** | C++17 standard library on the engine side. Python stdlib for analysis. | ✅ |
| **No locks in the hot path** | Thread-local buffers, merged at flush. | ✅ |
| **Deterministic, not sampled** | Explicitly placed scopes, so the trace is *interpretable* rather than statistical. | ✅ |
| **Never silently drop data** | Bounded budget, loud drop counter in every trace's provenance record. | ✅ |
| **Shared and static builds both** | MSVC refuses to `dllexport` the raw `__declspec(thread)` pointer the hot path reads (C2492), so each module caches its own and they all resolve to one registry-owned buffer. | ✅ was broken — [F18](docs/FINDINGS.md) found it, [F22](docs/FINDINGS.md) fixed it. Two-module test on MSVC, GCC and Clang; llama.cpp shared on Linux still untested |

### Measured cost

24-layer / 768-embd / 220 M-param model, 8 threads, MSVC Release, 15 interleaved
repetitions per arm. Full method and caveats in
[`docs/02-overhead-methodology.md`](docs/02-overhead-methodology.md).

```
decode (tg256, 8 threads, 15 interleaved reps per arm)
  arm                       median tok/s     IQR   overhead vs A
  --------------------------------------------------------------------
  A: compiled out                  42.41    0.7%                  -
  B: in, level 0                   42.28    1.6%    +0.31%  [-1.13, +0.79]
  C1: active level 1               42.26    2.3%    +0.36%  [-0.61, +1.80]
  C2: active level 2               42.12    1.7%    +0.69%  [-0.01, +1.86]
  C3: active level 3               42.13    1.1%    +0.67%  [+0.12, +1.67]
```

Levels 0–2 have intervals containing zero, so the honest reading there is "not
distinguishable from zero". **Level 3 — every graph node event on every worker
thread, ~2.9 million records over the run — is the first arm with a measurable
effect: +0.67%, CI [+0.12, +1.67].** Both ends of that interval are inside the
2% budget, which is the part that matters; the claim survives the pessimistic
end of the measurement, not just the point estimate.

And the number is not an artifact of a full buffer, which would make recording
look cheap by doing less of it:

```
tokenscope: wrote full3.json (228,723,230 bytes, 259 tokens, 8 threads, 0 dropped)
```

And the claim that cannot be tested statistically, tested structurally instead:

```
build-off/tokenscope.lib      864 bytes    0 tokenscope symbols
build-on/tokenscope.lib   992,690 bytes   68 exported symbols
```

## Try it

```bash
git clone https://github.com/PS12007/tokenscope && cd tokenscope

# clone llama.cpp at the pinned commit, copy in the sources, apply the patch
python scripts/bootstrap.py --dest ../llama.cpp

cmake -S ../llama.cpp -B ../llama.cpp/build-ts-on -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF -DGGML_TOKENSCOPE=ON
cmake --build ../llama.cpp/build-ts-on --target llama-bench

# a random-weight model, so you can try this without a 400MB download
python tools/make_tiny_model.py --llama-cpp ../llama.cpp -o models/tiny.gguf

TOKENSCOPE_LEVEL=1 TOKENSCOPE_OUT=run.trace.json \
  ../llama.cpp/build-ts-on/bin/llama-bench -m models/tiny.gguf -p 256 -n 128
python tools/trace_analyze.py run.trace.json

# the work/wait split needs level 3, and a token window to keep the trace small
TOKENSCOPE_LEVEL=3 TOKENSCOPE_TOKENS=8-13 TOKENSCOPE_OUT=l3.trace.json   ../llama.cpp/build-ts-on/bin/llama-bench -m models/tiny.gguf -p 0 -n 20
python tools/trace_analyze.py l3.trace.json --layers --barriers
```

Then drop `run.trace.json` onto [ui.perfetto.dev](https://ui.perfetto.dev).

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `TOKENSCOPE_LEVEL` | `0` | `0` off · `1` host scopes · `2` + per-node aggregates · `3` + every node event |
| `TOKENSCOPE_OUT` | — | trace path; flushed at exit. Unset ⇒ nothing written |
| `TOKENSCOPE_BUDGET_MB` | `256` | total record budget across all threads. Allocated in **1 MiB chunks per thread**, so a budget below `threads x 1 MiB` leaves some threads with no buffer at all — they record nothing and every attempt is counted as a drop |
| `TOKENSCOPE_RING` | `0` | keep the *last* events instead of the first (drops are still counted). **Only takes effect once a thread has filled a whole 1 MiB chunk (~43,700 records).** Below that, or if the budget was too small for the thread to get its first chunk, this flag does nothing — verified, and worth knowing because a tight budget is exactly when you would reach for it |
| `TOKENSCOPE_TOKENS` | all | capture window, e.g. `340-345`. Token slices are still recorded for the whole run; only the scopes inside them are windowed. |

## How it is put together

llama.cpp is a **graph builder plus a graph executor**, and almost every function
whose name suggests it does work does not do the work — `cpy_k()` returns a
`ggml_set_rows` node and takes ~200 ns; the copy happens later, on another
thread. That single fact drives the whole design:

- **Tier 1 — host scopes.** RAII scopes on genuinely serial work: KV slot search,
  graph build/alloc, input setup, logits readback, sampling, tokenization.
- **Tier 2 — op scopes.** Two scopes inside `ggml_graph_compute_thread`, which is
  where every tensor op actually executes. llama.cpp already names every tensor
  `<role>-<layer>`, so the full per-layer breakdown falls out of a string parse
  at flush time rather than instrumentation threaded through the model code.

Two structural decisions do most of the performance work:

- **Collapse the clock reads.** In the node loop, the end of work and the start
  of barrier-wait are the same instant. Four naive timestamp reads per node
  become one. The clock read is ~90% of a scope's cost (26 ns of 52.8 ns
  measured), so this halves the dominant term rather than shaving the store.
- **Move all string work off the worker threads.** Node events store the graph
  node index, not the name. Names are resolved once per graph epoch on the main
  thread — and because llama.cpp reuses graphs across decode steps, that is once
  per *run*, not once per token.

Full write-ups: [`docs/00-architecture-map.md`](docs/00-architecture-map.md) (where
the time actually goes, with file:line references for 21 instrumentation sites)
and [`docs/01-design-scope-timing.md`](docs/01-design-scope-timing.md) (the
mechanism, and the two overhead risks that a naive cost model misses).

## Status

Built in the open. `docs/` is the engineering log, in order — and
[`docs/HANDOFF.md`](docs/HANDOFF.md) is the current state plus what comes next.

- [x] Toolchain + repo bootstrap
- [x] [Map of llama.cpp's inference path](docs/00-architecture-map.md)
- [x] [Scope-timing design](docs/01-design-scope-timing.md)
- [x] Core mechanism + self-test + zero-overhead-when-off proof
- [x] PoC: prefill/decode split, end-to-end, on a real build
- [x] [Overhead measured for level 1](docs/02-overhead-methodology.md)
- [x] Tier 1: host instrumentation across `decode` and `process_ubatch`
- [x] [First findings from real traces](docs/FINDINGS.md)
- [x] Tier 2: per-node work/wait split, per-layer breakdown
- [x] [Barrier decomposition: imbalance vs release, and what it is worth](docs/FINDINGS.md)
- [x] [Thread-count sweep, 1 to 28](docs/FINDINGS.md)
- [x] [Thread pinning: core heterogeneity confirmed as the mechanism](docs/FINDINGS.md)
- [x] [Sampling and tokenizer scopes](docs/FINDINGS.md) — `llama-cli`, not `llama-bench`
- [x] [Three-model decode table](docs/FINDINGS.md)
- [x] [Context shift and the KV cell search](docs/FINDINGS.md)
- [x] [Concurrent sequences, 1 to 16](docs/FINDINGS.md)
- [ ] Perfetto screenshots
- [x] [Real quantized model](docs/FINDINGS.md) — Qwen2.5-0.5B Q4_K_M
- [x] [An 8B model, and five of six predictions](docs/FINDINGS.md) — Qwen3-8B Q4_K_M
- [x] [A defect found in llama.cpp's graph naming](docs/FINDINGS.md), fixed and measured
- [x] [Shared-library builds](docs/FINDINGS.md) — broken in F18, fixed in F22, now tested on three platforms
- [x] [ggml's matmul already load-balances, until a thread count takes it away](docs/FINDINGS.md)
- [ ] Linux/GCC
- [ ] [Upstream issue](docs/03-upstream-issue-draft.md), then a PR

## Repository layout

```
src/tokenscope.h        the mechanism: record, buffer, macros — header-only hot path
src/tokenscope-ggml.h   the only part that knows about ggml, kept separate
src/tokenscope.cpp      cold path: arena, interning, graph epochs, Chrome Trace emit
src/ts_selftest.cpp     8-thread self-test, layout assertions, per-scope cost
src/ts_dllmod.cpp       a second module for the shared-library test, nothing more
src/ts_dlltest.cpp      two binaries, one registry — the F18 regression test
patches/                01: the instrumentation. 02: the F20 naming fix, standalone
examples/               committed reference traces (level 1 and level 3), used by CI
scripts/bootstrap.py    clone at the pin, copy sources, apply patches
tools/trace_analyze.py  summary · per-token · outliers with cause · diff
tools/bench_overhead.py interleaved A/B/C arms, medians, bootstrap CIs
tools/make_tiny_model.py synthesize a random-weight GGUF so tests need no network
tools/model_bytes.py     per-phase weight bytes from a GGUF, to score the byte law
tools/mulmat_chunking.py which matmuls ggml load-balances, and at which thread counts
tools/imbalance_repeat.py N identical runs, because one trace is one draw (F23)
docs/                   the engineering log
```

## License

MIT. See [LICENSE](LICENSE).
