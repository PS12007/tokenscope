# 01 — Design: the scope-timing mechanism

Prerequisite reading: [`00-architecture-map.md`](00-architecture-map.md). The design
below is a direct consequence of three facts established there:

1. The math happens inside `ggml_graph_compute_thread()`, not where the API names suggest.
2. Node names already encode `<role>-<layer>`, so layer attribution is a string parse at flush time.
3. Volume is the binding constraint: ~500 nodes/token x 2 events x N threads.

---

## 1. What we are building

Five pieces, in dependency order:

```
ts_now()          timing source          -> a monotonic tick
ts_record         24-byte event record   -> what we store
ts_buffer         per-thread chunked arena
TS_SCOPE(...)     the RAII macro         -> what instrumentation sites use
ts_flush()        merge + emit           -> Chrome Trace Event JSON
```

Everything is behind `TOKENSCOPE_ENABLED`. When it is off, `TS_SCOPE` expands to
nothing at all — not to an empty inline function, not to a disabled branch, to
literally no tokens. The preprocessor removes it before the compiler sees it.

---

## 2. Timing source

### The choice

Default: `std::chrono::steady_clock::now()`.
Opt-in: raw TSC via `__rdtsc()` / `__builtin_ia32_rdtsc()`, behind `TOKENSCOPE_TSC`.

### Why steady_clock is the default

On Windows, `steady_clock` is `QueryPerformanceCounter`, which since Windows 8
is a user-mode read off the invariant TSC through a shared page — no syscall,
roughly 15-25 ns. On Linux, `clock_gettime(CLOCK_MONOTONIC)` goes through the
vDSO, similarly ~20 ns. That is fast enough, portable, and — the important part —
**already coherent across cores**. The OS has done the per-core TSC offset
correction for us.

### Why TSC is opt-in rather than the default

`rdtsc` is ~6-10 ns, which is genuinely better. But it costs three things:

- **Not serializing.** The CPU may reorder `rdtsc` relative to the work you are
  trying to measure. `rdtscp` or an `lfence` fixes it and gives back most of the
  speed advantage.
- **Cross-core coherence is an assumption, not a guarantee.** Invariant TSC is
  near-universal on modern x86, but "near-universal" is not "checked", and a
  profiler that silently produces negative durations on the one machine without
  it is worse than a slower profiler.
- **Calibration.** Ticks must become microseconds for the trace file. That means
  measuring the TSC frequency at startup and carrying the error.

So: TSC is available for people who have measured that they need it, and the
default is the boring correct thing. The tool records which clock produced a
trace in the trace metadata, because a trace whose provenance you cannot
establish is not evidence.

### Reading fewer clocks than there are scopes

The single best overhead optimization is structural rather than micro. In the
Tier 2 loop, the end of the work scope and the start of the wait scope are
**the same instant**:

```cpp
t0 = now();  ggml_compute_forward(...);
t1 = now();  ggml_barrier(...);
t2 = now();
// work = [t0, t1), wait = [t1, t2), and t2 becomes t0 of the next node
```

Four naive clock reads per node collapse to **one per node plus one per graph**.
That is a 4x reduction on the hottest path in the system, achieved by noticing
that the scopes are adjacent rather than by making the clock faster. This is why
Tier 2 uses a purpose-built loop-level construct (`TS_NODE_LOOP`) rather than two
independent `TS_SCOPE`s.

---

## 3. The record

```cpp
struct ts_record {
    uint64_t t0;     // 8  ticks since the process-wide epoch
    uint32_t dur;    // 4  ticks, saturating at UINT32_MAX
    uint32_t ref;    // 4  interned name id (host) | graph node index (node)
    uint32_t token;  // 4  token ordinal this event belongs to
    uint16_t graph;  // 2  graph epoch, resolves `ref` for node events
    uint8_t  kind;   // 1  TS_HOST | TS_NODE_WORK | TS_NODE_WAIT | TS_MARK
    uint8_t  depth;  // 1  nesting depth, for host scopes
};                   // 24 bytes, no padding, 2.67 records per cache line
```

Deliberate omissions, each of which costs bytes on the hot path and buys nothing:

- **No string.** Host scopes intern their name once (section 4). Node scopes store
  the graph node index and resolve the name at flush.
- **No thread id.** It is implied by which buffer the record is in.
- **No end timestamp.** `dur` as a 32-bit tick count covers ~4.3 s at 1 GHz-equivalent
  ticks, which is longer than any single scope we care about; longer scopes saturate
  and are flagged in the output rather than silently wrapping.

### Why node events store an index instead of a name

Interning `node->name` at record time means a hash of a short string plus a table
probe — call it 25-50 ns — on the hottest path in the program, executed ~4,000
times per token per thread. That is the difference between a 0.5% profiler and a
3% profiler.

Instead: store `node_n` (the graph position, already in a register in that loop)
and a `graph` epoch counter. At the end of each `ggml_graph_compute`, if this
graph epoch has not been seen before, the *main* thread walks `cgraph->nodes[]`
once and emits a name table. Thanks to llama.cpp's graph reuse
(`llama-context.cpp:1348`), the same graph is used for essentially every decode
step, so this walk happens roughly **once per run**, not once per token.

All string work is thereby moved off the hot path and out of the worker threads
entirely. This is the second structural win, after collapsing the clock reads.

---

## 4. Name interning for host scopes

Host scope names are compile-time literals. The macro does:

```cpp
static const uint32_t _ts_id = tokenscope::intern("kv.find-slot");
```

A function-local static gets a guard variable: one relaxed atomic load on every
execution after the first, ~1 ns. Host scopes fire on the order of 20 times per
token, so the total cost of interning across an entire token is under 20 ns.
Not worth optimizing further.

---

## 5. Per-thread storage

### Chunked arena, with an optional ring

The natural first instinct is a ring buffer. The right default is an arena, and
the difference matters:

- A **ring** silently discards the oldest records when it wraps. For a profiler
  that is a correctness problem disguised as a memory optimization: you get a
  trace that looks complete and is missing the beginning of the run.
- An **arena** with a hard budget stops recording when the budget is exhausted
  and reports exactly how many events it dropped. The trace is shorter, and it
  tells you it is shorter.

So the default is a chunked arena: a singly-linked list of 1 MiB chunks
(43,690 records each), allocated on demand up to `TOKENSCOPE_BUDGET_MB`
(default 256 MiB across all threads). Ring mode remains available via
`TOKENSCOPE_RING=1` for the legitimate case it serves: *"run the server for ten
minutes and give me the last five seconds."*

Both modes report drops. A profiler that quietly loses data is a profiler that
will eventually make you chase a bug that is not in your code.

### The hot-path write

```cpp
ts_buffer * b = ts_tls;              // thread-local pointer load
if (b->n == b->cap) [[unlikely]] b = ts_grow(b);
ts_record * r = &b->data[b->n++];
r->t0 = t0; r->dur = dur; ...        // one cache line touched, sequential
```

Three instructions of bookkeeping plus a 24-byte sequential store. No atomics, no
locks, no allocation in the common case.

### The Windows TLS trap

This is the platform-specific overhead risk, and it is not hypothetical.

MSVC implements `thread_local` for objects with dynamic initializers via a call
to `__dyn_tls_on_demand_init` on *every access* from a DLL that was not loaded at
process start. llama.cpp builds `ggml` and `llama` as shared libraries by
default. A `thread_local std::vector` or `thread_local std::shared_ptr` in that
position turns a 1 ns TLS read into a ~20 ns guarded function call, on the
hottest path in the program.

Mitigation, applied from the start rather than discovered later:

- `ts_tls` is a **plain POD pointer** (`thread_local ts_buffer * ts_tls = nullptr;`)
  with a constant initializer. No dynamic init, no guard, no on-demand call.
- Registration is a null check: `if (!ts_tls) ts_tls = ts_thread_init();`. The
  branch predicts perfectly after the first node.
- The buffer itself is heap-allocated and owned by a global registry via
  `shared_ptr`, so a worker thread that exits before flush does not take its
  records with it.

### The single-instance problem

Tier 1 scopes live in `llama` (`libllama.so` / `llama.dll`). Tier 2 scopes live in
`ggml-cpu`. Two different shared libraries. If each got its own copy of the
registry, the trace would be split in half.

`ggml-base` is the lowest common dependency of both, so the tokenscope
translation unit lives there with exported symbols, and both consumers resolve to
one instance. The alternative — forcing `BUILD_SHARED_LIBS=OFF` — works but is a
constraint on the user's build we do not need to impose.

---

## 6. Granularity levels

Volume is the binding constraint, so granularity is a first-class runtime knob,
not a debug afterthought. `TOKENSCOPE_LEVEL`:

| Level | What is recorded | Events/token (32L, 8T) | Intended use |
|---|---|---|---|
| **0** | nothing; buffers never allocated | 0 | runtime-disabled baseline |
| **1** | host scopes only (Tier 1) | ~20 | "is my stall in control-plane work?" |
| **2** | Tier 1 + per-node **aggregates** | ~20 + one flush record per node per token | "where does decode time go, by layer and phase?" |
| **3** | Tier 1 + every node event (Tier 2 full) | ~8,000 | "show me the timeline of token 340" |

Level 2 deserves emphasis because it is the one most people should use.
Instead of appending a record per node event, each thread keeps a
`uint64_t acc[n_nodes]` array and does `acc[node_n] += dur`. One add, no
allocation, no growth, and the result is an *exact* per-node total — you lose the
timeline, not the numbers. Memory is `n_nodes * 8` bytes per thread (~16 KB),
constant regardless of run length.

Level 3 exists because "show me the timeline of token 340" is the question that
sells the tool, and it is answerable at level 3 with `TOKENSCOPE_TOKENS=340-345`
to bound the volume.

---

## 7. Flush

```
ts_flush(path)
  |- stop recording (relaxed store to the global enable flag)
  |- lock the registry
  |- for each registered thread buffer:
  |     for each chunk: convert ticks -> microseconds, resolve names, emit
  |- emit the graph name tables as metadata events
  |- emit process/thread name metadata (M events)
  |- write "]" and close
```

Output is Chrome Trace Event Format: a JSON array of objects with
`name`, `cat`, `ph`, `ts`, `dur`, `pid`, `tid`. Complete events use `ph: "X"`.

Two details that are easy to get wrong and expensive to discover later:

- **`ts` is microseconds, as a floating-point number.** Perfetto accepts
  fractional microseconds; emitting integers throws away most of the resolution
  we worked to preserve. We emit 3 decimal places (nanosecond resolution).
- **Nesting must be consistent.** Perfetto renders `ph: "X"` events as a stack
  per `tid`, and a child that outlives its parent by even one microsecond
  produces a visibly broken flamegraph. Since we compute `dur` from real
  timestamps, a parent scope must *strictly* contain its children. The RAII
  ordering guarantees this within a thread; the flush sorts by `(tid, t0, -dur)`
  to make the containment explicit rather than relying on insertion order.

We do not stream JSON during the run. Writing to disk on the hot path would
dominate everything else in this document.

---

## 8. Where the overhead is, and the plan to stay under 2%

### The naive cost model, which is not the interesting part

| Component | Cost |
|---|---|
| `steady_clock::now()` | ~20 ns |
| TLS pointer load | ~1 ns |
| capacity check (predicted) | ~1 ns |
| 24-byte sequential store | ~2 ns |
| **per node, after clock collapsing (1 read + 2 stores)** | **~25 ns** |

At level 3, ~500 nodes/token/thread x 25 ns = **~12.5 us per token per thread**.
Against a 40 ms token that is 0.03%. Against a fast 4 ms token on a small model,
0.3%. Comfortably inside budget.

If the analysis stopped there it would be wrong, because the two mechanisms that
actually threaten the budget are not in that table.

### Risk 1 — cache pollution

Each thread writes ~12 KB of records per token. That is 12 KB of L2 that is no
longer holding model weights, in a workload whose entire performance character is
memory-bandwidth-bound weight streaming. The cost does not show up as time spent
in our code; it shows up as **the matmuls getting slower**, which is precisely
the measurement we are trying to make.

Mitigations, in order of application:

- 24-byte records rather than the 48 a naive layout would produce.
- Level 2 (aggregate) mode has a **constant** ~16 KB working set instead of a
  growing one, and touches it in a strided pattern that matches the node order.
- Level 3 is documented as "the mode with real overhead", and the benchmark
  reports levels 1, 2 and 3 separately rather than quoting one number.

### Risk 2 — the barrier turns per-thread noise into whole-graph latency

This is the one that makes CPU inference profiling different from profiling
almost anything else.

`ggml_barrier` runs after **every node**. All threads rendezvous, several hundred
times per token. At a barrier, the graph waits for the *slowest* thread. So the
overhead the graph experiences is not the mean of per-thread overhead —

```
graph overhead per node  =  max over threads (per-thread overhead)
```

— it is the **max**. Any jitter (a chunk allocation, a page fault on a fresh
arena page, one thread's clock read landing on a cache miss) is paid in full by
every thread, at every one of the hundreds of barriers per token. Overhead that
would be invisible in a single-threaded profiler compounds here.

Mitigations:

- **Pre-touch chunks.** A fresh 1 MiB chunk means 256 page faults spread across
  the next 43,690 events. Allocate and `memset` the next chunk *at flush-safe
  points* (between graphs), never mid-graph.
- **Allocate the level-2 accumulator once**, at graph epoch change, not lazily.
- **Never allocate inside the node loop.** The grow path must be provably
  unreachable during a graph if the budget allows, and the benchmark verifies it
  by asserting zero mid-graph allocations.
- **Measure with thread counts that stress it** — 1, 4, 8, and oversubscribed —
  because a 1-thread overhead number would hide this entire class of problem.

### Risk 3 — the disabled-but-compiled-in case

There are three configurations, and conflating them is how profilers end up
quoting a fictional overhead number:

| Config | Build | Runtime | What it measures |
|---|---|---|---|
| **A** | `TOKENSCOPE_ENABLED=OFF` | n/a | true baseline |
| **B** | `ON` | `TOKENSCOPE_LEVEL=0` | cost of the residual enable-flag branch |
| **C** | `ON` | `TOKENSCOPE_LEVEL=1/2/3` | real profiling overhead |

**A vs B** is the honesty check on "zero overhead when disabled" — and it is a
real risk, because a `if (ts_enabled)` branch inside the node loop is a load and
a predicted branch several hundred times per token. If B is measurably slower
than A, the claim in the README is false and the README changes.

**B vs C** is the number people actually care about.

The benchmark reports all three. Additionally, A is verified structurally, not
just statistically: `dumpbin /symbols` (or `nm`) must show **zero** `tokenscope`
symbols in the `OFF` build, and the binary size delta must be zero. A statistical
test cannot prove the absence of overhead; a symbol table can.

---

## 9. Statistical method for the overhead claim

A single before/after pair is not evidence. `tools/bench_overhead.py`:

- runs `llama-bench`-style fixed-token generation, `n >= 20` repetitions per config
- **interleaves** configurations (ABCABC..., not AAA BBB CCC) so thermal drift
  and background load hit every arm equally
- discards the first repetition of each arm as warm-up
- reports median and interquartile range, not mean and standard deviation,
  because tok/s distributions have a hard ceiling and a long slow tail — the mean
  is dragged by outliers that are not the effect being measured
- reports the overhead as a **confidence interval**, via bootstrap over the
  paired repetitions, so the claim is "1.2% [0.9, 1.6]" rather than "1.2%"
- fails loudly if the baseline arm's own IQR exceeds the effect being measured,
  because on a noisy machine the honest output is "this machine cannot resolve
  2%" rather than a number

The overhead number goes in the README only once it survives that.

---

## 10. Build integration

```cmake
option(TOKENSCOPE_ENABLED "build with tokenscope instrumentation" OFF)
```

`OFF` by default, always. The instrumentation is a debugging build, and a
profiler that requires people to opt out is a profiler that will be quietly
patched out downstream.

Delivery is by patch files against a pinned upstream commit, plus a
`scripts/bootstrap` that clones, checks out the pin, and applies them. Not a
fork: a fork rots, and the upstream diff is the artifact that matters for the
eventual PR conversation.

---

## 11. Order of construction, and why

1. **`ts_now`, `ts_record`, `ts_buffer`, `TS_SCOPE`, `ts_flush`** — standalone,
   with a unit test that runs them under 8 threads and validates the JSON.
2. **Tier 1, prefill/decode split only** — the smallest patch that produces a
   real trace from a real model.
3. **`bench_overhead.py`, configs A/B/C** — before adding anything else.
4. **Everything else**, only if step 3 says there is room.

Step 3 comes before step 4 deliberately. The alternative — instrument everything,
then measure — means that when the number comes back at 6% there is no way to
know which of twenty scopes is responsible. Measuring the mechanism in isolation
first is what makes the eventual number attributable.

Proof of concept and its measured cost: [`02-overhead-methodology.md`](02-overhead-methodology.md).
