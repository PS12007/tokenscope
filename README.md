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

Two aggregate numbers. They cannot tell you that:

- token 340 stalled for 90 ms because the KV cache was reallocated mid-run,
- sampling is quietly eating 8% of your decode budget,
- layer 17's attention is disproportionately slow because of how the cache is laid out,
- or that 40% of your "compute" time is actually threads parked at a barrier.

`tokenscope` makes all of that visible.

## What it does

A build-time-optional instrumentation layer for llama.cpp that records explicitly
placed timing scopes into lock-free per-thread buffers and flushes them as
[Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview)
JSON — the format [Perfetto](https://ui.perfetto.dev) and `chrome://tracing` consume.

You get a flamegraph-style timeline of a real inference run, with zero custom UI work.

Plus a Python analysis layer that turns a trace into answers:

```
$ tokenscope-analyze run.trace.json
decode: 255 tokens, 10233.4 ms total, 40.1 ms/tok (24.9 tok/s)

  category           total ms     %      per-tok ms
  ─────────────────────────────────────────────────
  attn                4021.7    39.3%       15.77
  ffn                 3894.2    38.1%       15.27
  barrier-wait        1502.9    14.7%        5.89
  sample               512.1     5.0%        2.01
  kv-cache             211.4     2.1%        0.83
  other                 91.1     0.9%        0.36

  p50 38.9 ms   p95 44.2 ms   p99 61.0 ms   max 131.7 ms
  outliers: tok 340 (131.7 ms, kv-cache-realloc 89.9 ms)
```

...and a `diff` mode, because the question during optimization work is always
*did my change help, and where*.

> The numbers above are an illustration of the output format, not measured results.
> Real measured numbers land in [`docs/FINDINGS.md`](docs/FINDINGS.md) as they are produced.

## Design constraints

These are hard rules, not aspirations. They are what separates a profiler from a
lie about where your time goes.

| Constraint | How it is enforced |
|---|---|
| **Zero overhead when disabled** | Everything behind `TOKENSCOPE_ENABLED`. Off ⇒ the macros expand to nothing; no symbols, no branches, no storage. |
| **< 2% overhead when enabled** | Measured, not assumed. See [`tools/bench_overhead.py`](tools/bench_overhead.py) and [`docs/02-overhead-methodology.md`](docs/02-overhead-methodology.md). |
| **No new dependencies** | C++17 standard library only on the engine side. Python stdlib only for analysis. |
| **No locks in the hot path** | Thread-local ring buffers, merged at flush. |
| **Deterministic, not sampled** | Explicitly placed scopes, so the trace is *interpretable* rather than statistical. |

## Status

Work in progress, built in the open. See [`docs/`](docs/) for the engineering log.

- [x] Toolchain + repo bootstrap
- [ ] Map of llama.cpp's inference path ([`docs/00-architecture-map.md`](docs/00-architecture-map.md))
- [ ] Scope-timing design ([`docs/01-design-scope-timing.md`](docs/01-design-scope-timing.md))
- [ ] Proof of concept: prefill/decode split + overhead benchmark
- [ ] Full per-layer / per-phase instrumentation
- [ ] Python analysis + diff tooling
- [ ] Findings writeup

## License

MIT. See [LICENSE](LICENSE).
