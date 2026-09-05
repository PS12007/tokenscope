<h1 align="center">tokenscope</h1>

<p align="center">
  <b>A per-token inference profiler for llama.cpp.</b><br>
  Attributes the wall-clock time of <i>every generated token</i> to specific phases of the
  forward pass, and emits Chrome Trace Event Format JSON you can open in Perfetto.
</p>

<p align="center">
  <a href="#status"><img alt="status" src="https://img.shields.io/badge/status-work%20in%20progress-orange"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <img alt="c++17" src="https://img.shields.io/badge/C%2B%2B-17-informational">
  <img alt="deps" src="https://img.shields.io/badge/runtime%20deps-none-success">
  <img alt="patch size" src="https://img.shields.io/badge/upstream%20patch-24%20lines-success">
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

tokenscope: wrote run.trace.json (30192 bytes, 259 tokens, 0 dropped)
```

Then ask it questions:

```console
$ python tools/trace_analyze.py run.trace.json

=== run.trace.json ===
    clock=steady_clock  level=1  threads=8  dropped=0

prefill: 2 batch(es), 1654.5 ms

decode: 257 tokens, 6068.8 ms total, 23.61 ms/tok (42.3 tok/s)

  p50  23.46 ms   p95  25.72 ms   p99  27.39 ms   max  29.67 ms
```

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
| **Under 2% when enabled** | Measured with interleaved arms and bootstrap CIs, not assumed. | ✅ for level 1; levels 2–3 not built yet |
| **No new dependencies** | C++17 standard library on the engine side. Python stdlib for analysis. | ✅ |
| **No locks in the hot path** | Thread-local buffers, merged at flush. | ✅ |
| **Deterministic, not sampled** | Explicitly placed scopes, so the trace is *interpretable* rather than statistical. | ✅ |
| **Never silently drop data** | Bounded budget, loud drop counter in every trace's provenance record. | ✅ |

### Measured cost

24-layer / 768-embd / 220 M-param model, 8 threads, MSVC Release, 15 interleaved
repetitions per arm. Full method and caveats in
[`docs/02-overhead-methodology.md`](docs/02-overhead-methodology.md).

```
decode (tg256)
  arm                       median tok/s     IQR   overhead vs A
  --------------------------------------------------------------------
  A: compiled out                  42.23    0.4%                  -
  B: in, level 0                   42.19    0.8%    +0.11%  [-0.25, +0.76]
  C1: active level 1               42.25    0.7%    -0.04%  [-0.48, +0.45]
```

Every interval contains zero, so the correct claim is **not** "0.11% overhead".
It is: *at level 1, the cost is not distinguishable from zero on this system, at
a resolution of about ±0.5%.* Level 1 currently means one scope per
`llama_decode` call; the levels that record every graph node are where the real
risk lives and they will be measured the same way before any number about them
appears here.

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
```

Then drop `run.trace.json` onto [ui.perfetto.dev](https://ui.perfetto.dev).

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `TOKENSCOPE_LEVEL` | `0` | `0` off · `1` host scopes · `2` + per-node aggregates · `3` + every node event |
| `TOKENSCOPE_OUT` | — | trace path; flushed at exit. Unset ⇒ nothing written |
| `TOKENSCOPE_BUDGET_MB` | `256` | total record budget across all threads |
| `TOKENSCOPE_RING` | `0` | keep the *last* N events instead of the first (drops are still counted) |
| `TOKENSCOPE_TOKENS` | all | capture window, e.g. `340-345` |

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

Built in the open. `docs/` is the engineering log, in order.

- [x] Toolchain + repo bootstrap
- [x] [Map of llama.cpp's inference path](docs/00-architecture-map.md)
- [x] [Scope-timing design](docs/01-design-scope-timing.md)
- [x] Core mechanism + self-test + zero-overhead-when-off proof
- [x] PoC: prefill/decode split, end-to-end, on a real build
- [x] [Overhead measured for level 1](docs/02-overhead-methodology.md)
- [ ] Tier 1: full host instrumentation (KV, sampling, tokenizer)
- [ ] Tier 2: per-node work/wait, per-layer breakdown
- [ ] Perfetto screenshots + three-model decode table
- [ ] [Findings from real traces](docs/FINDINGS.md)
- [ ] Upstream issue, then a PR

## Repository layout

```
src/tokenscope.h        the mechanism: record, buffer, macros — header-only hot path
src/tokenscope.cpp      cold path: arena, interning, graph epochs, Chrome Trace emit
src/ts_selftest.cpp     8-thread self-test, layout assertions, per-scope cost
patches/                24 lines of surgical edits to upstream llama.cpp
scripts/bootstrap.py    clone at the pin, copy sources, apply patches
tools/trace_analyze.py  summary · per-token · outliers with cause · diff
tools/bench_overhead.py interleaved A/B/C arms, medians, bootstrap CIs
tools/make_tiny_model.py synthesize a random-weight GGUF so tests need no network
docs/                   the engineering log
```

## License

MIT. See [LICENSE](LICENSE).
