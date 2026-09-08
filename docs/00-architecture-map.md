# 00 — Where the time actually goes: a map of llama.cpp's inference path

**Upstream pinned at:** `ggml-org/llama.cpp` @ `4d91760` (shallow clone, 2026-09-04)

**Purpose:** before writing a single line of instrumentation, establish *exactly*
where each phase of inference happens, so that every scope we later place is
placed on purpose.

All line numbers below are against the pinned commit. They will drift. The
*structure* is what matters, and the structure has been stable for a long time.

---

## 0. The one thing that changes the whole design

llama.cpp is not an eager framework. It is a **graph builder plus a graph executor**,
and those two things live in different files, run at different times, and — critically —
**almost every function whose name suggests it does work, does not do the work.**

```
llama_decode(ctx, batch)                        <- host, eager
  |- llama_context::decode()                    <- host, eager
       |- process_ubatch()
            |- model.build_graph()              <- BUILDS a DAG. No math happens.
            |    |- llama_kv_cache::cpy_k()     <- returns a ggml_set_rows NODE
            |    |- llama_kv_cache::get_k()     <- returns a ggml_view NODE
            |- res->set_inputs(&ubatch)         <- host, eager: fills input tensors
            |- graph_compute()                  <- THIS is where all the math happens
                 |- ggml_backend_sched_graph_compute_async()
                      |- ggml_graph_compute()   <- ggml/src/ggml-cpu/ggml-cpu.c:3391
                           |- N x ggml_graph_compute_thread()   <- :3101
                                |- per node: ggml_compute_forward() + ggml_barrier()
```

**Consequence:** you cannot time "KV cache write" by putting an RAII scope around
`llama_kv_cache::cpy_k()`. That call takes ~200 ns and allocates a graph node.
The actual copy happens microseconds later, inside `ggml_compute_forward`, on
whichever worker thread grabs it.

This forces a **two-tier design**, which is the central architectural decision of
this project:

| Tier | What it times | Where the scope goes |
|---|---|---|
| **Tier 1 — host scopes** | Real host-side serial work: batch prep, graph build, graph alloc, input setup, logits readback, sampling, tokenization, KV slot search, KV shift. | RAII scopes in `src/llama-context.cpp`, `src/llama-kv-cache.cpp`, `src/llama-sampler.cpp`, `src/llama-vocab.cpp` |
| **Tier 2 — op scopes** | The actual tensor math, per node, per worker thread, plus barrier wait. | Two scopes inside `ggml_graph_compute_thread()`, `ggml/src/ggml-cpu/ggml-cpu.c:3147` |

Tier 2 is where 90%+ of decode time lives. Tier 1 is where the *surprises* live.

---

## 1. The prefill / decode boundary

### Where it is

There is **no explicit prefill/decode branch in llama.cpp.** Both go through the
same entry point:

- `llama_decode()` -> `llama_context::decode()` — **`src/llama-context.cpp:1644`**

The distinction is emergent: prefill is a `decode()` call with many tokens,
decode is a `decode()` call with one. The only place the engine itself draws
the line is a heuristic in the perf accounting:

**`src/llama-context.cpp:714`** — `llama_context::synchronize()`

```cpp
    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) { t_eval_us += ggml_time_us() - t_compute_start_us; }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) { t_p_eval_us += ggml_time_us() - t_compute_start_us; }
        n_p_eval += n_queued_tokens;
    }
```

This is the entire basis of the `prompt eval time` / `eval time` split that every
llama.cpp user reads off the console. It is:

- **heuristic** — "one token means decode" is an assumption, not a fact;
- **self-admittedly wrong** in the batch-size-1 case (see the `FIXME` the
  maintainers left in place);
- **coarse** — two scalars for an entire run;
- and **only updated on `synchronize()`**, so if the caller does not synchronize
  per token, several tokens get merged into one accounting bucket.

That last point is worth dwelling on, because it means the numbers people quote
are not per-token measurements at all. They are aggregates over however many
tokens happened between synchronization points.

### Where tokenscope draws the line instead

Explicitly, at the top of `decode()`, using the ubatch token count — not a global
counter that a caller can desynchronize:

- scope opens at `src/llama-context.cpp:1644` (entry to `decode`)
- phase = `prefill` if `n_tokens_all > 1` else `decode`
- token index = a monotonically increasing counter incremented once per decode
  call with `n_tokens_all == 1`

This gives one trace slice per generated token, with a stable id we can attach
every nested scope to.

### Relevant sub-boundaries inside `decode()`

| Line | What happens | Worth a scope? |
|---|---|---|
| `1705` | `balloc->init(...)` — batch validation and split planning | yes, `batch-init` |
| `1737` | `sched_reserve()` | yes, occasionally huge |
| `1740` | `memory_update(false)` — pending KV shifts/copies | **yes — outlier source** |
| `1745` | `memory->init_batch(...)` — KV slot search, may retry after `memory_update(true)` | **yes — outlier source** |
| `1793` | `output_reserve(n_outputs_all)` — may reallocate the logits buffer | **yes — outlier source** |
| `1806` | `do { ... } while` — the ubatch loop | one scope per ubatch |
| `1826` | `process_ubatch(...)` | the big one |
| `1874` | `ggml_backend_tensor_get_async(...)` — logits readback | yes, `logits-readback` |

The three lines marked "outlier source" are the leading candidates for
*"why did token 340 take 130 ms when its neighbours took 40"*. They are all
control-plane work that happens on some tokens and not others, which is exactly
the shape of a latency spike that aggregate timers cannot show you.

---

## 2. Per-layer compute

### Graph construction (NOT where time is spent)

- `llama_model::build_graph()` — `src/llama-model.cpp`, dispatching to per-arch
  builders in `src/models/` (`llm_build_llama`, `llm_build_qwen2`, and so on).
- Each builder loops `for (int il = 0; il < n_layer; ++il)` and emits nodes for
  norm -> QKV projection -> RoPE -> KV store -> attention -> output projection ->
  residual -> norm -> FFN -> residual.

Timing this loop measures *how long it takes to describe the model*, not how long
it takes to run it. On a warm run it is often skipped entirely — see the graph
reuse path at `src/llama-context.cpp:1348` (`res->can_reuse(gparams)`), which is
taken on essentially every decode step after the first.

> That graph-reuse fast path is itself a thing worth a scope: when it *misses*,
> you pay a full graph rebuild in the middle of your decode loop.

### The naming hook that makes per-layer attribution free

**`src/llama-context.cpp:2521`** — `llama_context::graph_get_cb()`:

```cpp
return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);   // e.g. "attn_norm-17", "ffn_up-17"
    } else {
        ggml_set_name(cur, name);
    }
    ...
```

Every tensor in the graph is named `<role>-<layer index>`. So at execution time,
a node's `->name` field already carries both the phase and the layer. We do not
need to thread any extra context down into the compute loop — **we parse the
name we are already given.** This is the single highest-leverage fact in this
document: it means Tier 2 instrumentation is two scopes total, not one per phase
per layer.

Node name prefixes we care about, and the categories they map to:

| Prefix | Category |
|---|---|
| `attn_norm`, `ffn_norm`, `norm`, `result_norm` | `norm` |
| `Qcur`, `Kcur`, `Vcur`, `attn_q`, `attn_k`, `attn_v` | `attn.qkv` |
| `k_cache_view`, `v_cache_view`, `cache_k`, `cache_v` | `attn.kv_rw` |
| `kq`, `kq_soft_max`, `kqv`, fused flash-attn nodes | `attn.score` |
| `attn_out`, `kqv_out` | `attn.out` |
| `ffn_up`, `ffn_gate`, `ffn_down`, `ffn_*` | `ffn` |
| `result_output` | `lm_head` |
| everything else | `other` |

---

## 3. KV cache: read, write, allocate, shift

Four genuinely different things live under the name "KV cache work", and
conflating them is exactly the mistake this profiler exists to prevent.

### (a) Slot search — host, eager, per ubatch

**`src/llama-kv-cache.cpp:898`** — `llama_kv_cache::find_slot()`

Called via `prepare()` (`:751`) from `init_batch()` (`:702`).
Linear scan for a contiguous or scattered run of free cells. Cost grows with
cache occupancy. **This is a real host-side cost and a real latency-spike source
when the cache is nearly full.** Scope it.

### (b) Cell bookkeeping — host, eager, per ubatch

**`src/llama-kv-cache.cpp:1097`** — `llama_kv_cache::apply_ubatch()`

Writes position/sequence metadata into `llama_kv_cells`. Cheap, but scope it to
prove it is cheap.

### (c) The actual K/V write — graph node, executed on worker threads

**`src/llama-kv-cache.cpp:1318`** — `llama_kv_cache::cpy_k()`

**`src/llama-kv-cache.cpp:1353`** — `llama_kv_cache::cpy_v()`

```cpp
    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
```

Returns a node. Does nothing. **Tier 2 territory** — caught by node name at
execution time.

### (d) The actual K/V read — graph node

**`src/llama-kv-cache.cpp:1266`** — `get_k()` / **`:1286`** — `get_v()`

Return `ggml_view_*` nodes over the cache tensors. Views are free at execution;
the cost shows up folded into the attention matmul that consumes them. This is
an important honesty constraint on the tool: **"KV cache read time" is not a
separately measurable quantity on the CPU backend.** We will report it as part of
`attn.score` rather than inventing a number for it.

### (e) Shift / defrag / stream copy — host-triggered, graph-executed

**`src/llama-kv-cache.cpp:817`** — `llama_kv_cache::update()`, reached from
`llama_context::memory_update()` at **`src/llama-context.cpp:792`**, called at
`src/llama-context.cpp:1740` on every `decode()`.

Contains: cross-stream `ggml_backend_tensor_copy` (synchronous, blocking), and
the RoPE-shift graph built by `build_graph_shift()` (`:2001`) and executed
inline. **This is the classic mid-generation stall.** It happens on the tokens
where a context-shift is triggered and nowhere else, which is precisely why an
aggregate timer averages it into invisibility.

Scope both the outer `memory_update` and the inner `update(do_shift=true)` branch.

---

## 4. Sampling

Two distinct paths in current llama.cpp, and the profiler must not assume one:

### CPU sampling chain

**`src/llama-sampler.cpp:895`** — `llama_sampler_sample(smpl, ctx, idx)`

- pulls logits via `llama_get_logits_ith()`
- builds a `llama_token_data_array` of `n_vocab` candidates (an `n_vocab`-sized
  fill loop — for a 150k-token vocab this is not free)
- `llama_sampler_apply()` (`:382`) walks the chain: penalties, top-k, top-p,
  temperature, and so on
- `llama_sampler_accept()` (`:372`) updates sampler state

Each link in the chain is a separate `llama_sampler` with its own `apply`. Scoping
`llama_sampler_apply` once, tagged with `smpl->iface->name`, gives per-stage
sampling attribution for free.

### Backend sampling (short-circuit)

Lines `896-906` of the same function: if a backend sampler already produced a
token during graph execution, the CPU chain is skipped entirely. The profiler
must emit a distinguishable scope here (`sample.backend-hit`) or it will report
"sampling is free" and be wrong about why.

Also relevant: `src/llama-context.cpp:1798`, `llama_sampler_backend_begin()` —
a per-logical-batch sampling transaction.

---

## 5. Tokenization

- **`src/llama-vocab.cpp:4415`** — `llama_tokenize()` -> `llama_vocab::tokenize()` (`:4126`)
- **`src/llama-vocab.cpp:4426`** — `llama_token_to_piece()` -> (`:4162`)
- **`src/llama-vocab.cpp:4436`** — `llama_detokenize()` -> (`:4166`)

Encode happens once per request and is usually irrelevant. Decode
(`token_to_piece`) happens **once per generated token, on the hot path**, and for
BPE vocabularies with unicode normalization it is not always as cheap as people
assume. Worth measuring precisely because everyone assumes it is zero.

---

## 6. Thread pool dispatch and barriers

This is where the hardest and most valuable part of the project lives.

### Dispatch

**`ggml/src/ggml-cpu/ggml-cpu.c:3391`** — `ggml_graph_compute()`

- OpenMP build: `#pragma omp parallel num_threads(n_threads)` -> each thread calls
  `ggml_graph_compute_thread()`
- Non-OpenMP build: `ggml_graph_compute_kickoff()` wakes persistent workers
  (`:3242`, `ggml_graph_compute_secondary_thread`), and the calling thread
  becomes worker 0.

Which one you get is a build flag, and it changes the shape of the trace. The
profiler must record which.

**It does now, since session 5** — `threading` in every trace's provenance
record, reported by the translation unit that contains `ggml_barrier` rather
than inferred anywhere else. This paragraph was written in session 1 and was
right; nothing acted on it until [`F26`](FINDINGS.md) found four sessions of
findings labelled with the wrong branch. **A design note that names a risk and
is never implemented is a prediction, and this one came true.**

### The per-node execution loop — **the primary Tier 2 site**

**`ggml/src/ggml-cpu/ggml-cpu.c:3129-3159`**

```cpp
for (int node_n = 0; node_n < cgraph->n_nodes && ...; node_n++) {
    struct ggml_tensor * node = cgraph->nodes[node_n];
    if (ggml_op_is_empty(node->op)) continue;
    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;

    const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
    if (n_fused > 0) { node_n += n_fused; }
    else { ggml_compute_forward(&params, node); }      // <- SCOPE A: work

    ...
    if (node_n + 1 < cgraph->n_nodes) {
        ggml_barrier(state->threadpool);               // <- SCOPE B: wait
    }
}
```

Everything we need is in scope at that point:

- `node->name` -> phase + layer index (section 2)
- `node->op` -> `ggml_op_name(node->op)`
- `state->ith` -> worker thread index, which is our trace `tid`
- `node_n` -> graph position, for ordering

Two scopes here produce the entire per-layer breakdown **and** the work-vs-wait
split. That is the whole ballgame.

### The barrier itself

**`ggml/src/ggml-cpu/ggml-cpu.c:576`** — `ggml_barrier()`

```cpp
    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }
```

A **spin barrier**, executed after *every single node* in the graph. For a 32-layer
model that is roughly 300-600 barriers per token, per thread.

Two consequences that shape the tool:

1. **Spinning burns wall-clock and CPU while doing no work.** A profiler that
   reports "attention took 15 ms across 8 threads" without splitting out wait
   time is actively misleading — most of that may be seven threads spinning while
   one finishes a badly-partitioned op. Attributing work vs wait correctly is
   the difference between a useful tool and a decorative one.
2. **The barrier is the natural cheap sync point.** Because all threads rendezvous
   after every node, we do not need any cross-thread coordination of our own.
   Per-thread buffers with no synchronization are automatically consistent at
   node granularity.

Early back-of-envelope: at ~500 nodes/token x 2 scopes x 8 threads = ~8,000
scope records per token. At 32 bytes/record that is 256 KB/token, ~64 MB for a
250-token run. **Buffer sizing and record width are therefore first-order design
concerns, not implementation details.** This directly motivates the ring-buffer
sizing and the aggregation modes in doc 01.

---

## 7. Existing timing facilities, and why they are not enough

| Facility | What it gives | Why it is insufficient |
|---|---|---|
| `llama_perf_context_print` (`src/llama-context.cpp:4278`) | 2 aggregate scalars | no per-token, no per-phase, heuristic split, admitted `FIXME` |
| `ggml_backend_sched_set_eval_callback` (used by `examples/eval-callback`) | per-node callback | **runs on the scheduler thread, serializes execution, and forces a sync per node** — changes the thing it measures. Fine for correctness debugging, useless for timing. |
| `GGML_PERF` legacy counters | per-op totals | removed/degraded in current ggml; no per-token attribution |
| External samplers (VTune, perf, Nsight) | true hardware attribution | statistical, no notion of "token 340", requires a heavy toolchain |

The gap tokenscope fills is narrow and specific: **deterministic, per-token,
per-phase attribution, with work and wait separated, behind a dependency-free
build flag.**

---

## 8. Instrumentation site summary

| # | Site | File:line | Tier | Category |
|---|---|---|---|---|
| 1 | `llama_context::decode` entry | `llama-context.cpp:1644` | 1 | `token` (prefill/decode root) |
| 2 | `balloc->init` | `llama-context.cpp:1705` | 1 | `batch-init` |
| 3 | `sched_reserve` | `llama-context.cpp:1737` | 1 | `sched-reserve` |
| 4 | `memory_update` | `llama-context.cpp:1740` | 1 | `kv.update` |
| 5 | `memory->init_batch` | `llama-context.cpp:1745` | 1 | `kv.slot-search` ✅ — **covers batch splitting *and* `find_slot`; 97% of it is the splitting (F16)** |
| 6 | `output_reserve` | `llama-context.cpp:1793` | 1 | `output-reserve` |
| 7 | `process_ubatch` | `llama-context.cpp:1826` | 1 | `ubatch` |
| 8 | graph build (reuse miss) | `llama-context.cpp:1366` | 1 | `graph-build` |
| 9 | `ggml_backend_sched_alloc_graph` | `llama-context.cpp:1378` | 1 | `graph-alloc` |
| 10 | `res->set_inputs` | `llama-context.cpp:1389` | 1 | `set-inputs` |
| 11 | `graph_compute` | `llama-context.cpp:1394` | 1 | `graph-compute` |
| 12 | logits readback | `llama-context.cpp:1874` | 1 | `logits-readback` |
| 13 | `find_slot` | `llama-kv-cache.cpp:898` | 1 | `kv.find-slot` ✅ — nested inside site 5, so the two are separable (F16) |
| 14 | `apply_ubatch` | `llama-kv-cache.cpp:1097` | 1 | `kv.apply` |
| 15 | `kv_cache::update` | `llama-kv-cache.cpp:817` | 1 | `kv.shift` |
| 16 | `llama_sampler_sample` | `llama-sampler.cpp:895` | 1 | `sample` ✅ |
| 17 | `llama_sampler_apply` | `llama-sampler.cpp:382` | 1 | `sample` ✅ — this, not 16, is the entry point `common_sampler_sample` uses |
| 18 | `llama_token_to_piece` | `llama-vocab.cpp:4426` | 1 | `tok.decode` ✅ |
| 19 | `llama_tokenize` | `llama-vocab.cpp:4415` | 1 | `tok.encode` ✅ |
| 20 | **`ggml_compute_forward`** | `ggml-cpu.c:3147` | **2** | derived from `node->name` |
| 21 | **`ggml_barrier`** | `ggml-cpu.c:3157` | **2** | `barrier-wait` |

Sites 20 and 21 are the ones that matter most, and are the ones added last,
because they are the ones with the overhead risk.

---

## 9. Benchmark harness

`tools/llama-bench` is the right driver for overhead measurement, because it
already separates the two phases into clean functions with no I/O, no chat
template, and no sampling:

- `test_prompt()` — `tools/llama-bench/llama-bench.cpp:2190` — pure prefill
- `test_gen()`    — `tools/llama-bench/llama-bench.cpp:2212` — pure decode,
  one `llama_decode` + one `llama_synchronize` per token

`test_gen` in particular is exactly the loop we want to characterise: a
synchronize per token means the per-token boundary is unambiguous, which removes
the aggregation artifact described in section 1.

---

## 10. What this changes about the plan

Three things I did not expect going in, all of which alter the design:

1. **The prefill/decode split everyone quotes is a heuristic with a known bug.**
   Not a bad heuristic — but the tool should draw the line itself rather than
   inherit it, and it should be able to *show* the misattribution.

2. **Per-layer instrumentation is nearly free, because llama.cpp already names
   every tensor `<role>-<layer>`.** I was budgeting a week for threading layer
   context through the compute path. It is two scopes and a string parse.

3. **The record volume is the real constraint, not the timer cost.** ~8,000
   records per token means the design question is not "is `steady_clock` fast
   enough" but "what do we do with 64 MB of records". That pushes toward
   configurable granularity levels and an online-aggregation mode as a
   first-class feature rather than an afterthought.

Design doc: [`01-design-scope-timing.md`](01-design-scope-timing.md).
