# X / Twitter post drafts

Drafts for posting as the project progresses. Grouped by what they need to be
true before you can post them.

**House rules for this file**

- Never post a number that is not in `docs/` with a method behind it.
- No "excited to announce". Lead with the finding, not the feeling.
- A screenshot beats a paragraph. A Perfetto screenshot beats both.
- If a thread's payoff is "and here's the repo", the thread was an ad. The
  payoff should be something the reader can use even if they never click.

Repo: https://github.com/PS12007/tokenscope

---

## READY NOW — mechanism validated, PoC working

### A1. The opener (single post)

> llama.cpp tells you "prompt eval time" and "eval time".
>
> Two numbers. For the entire run.
>
> They can't tell you that token 340 stalled on a KV cache realloc, or that 40%
> of your "compute" is threads spinning at a barrier.
>
> So I'm building a per-token profiler that can.

---

### A2. The finding that changed the design (single post)

> Spent a day reading llama.cpp before writing any instrumentation code.
>
> Best thing I found: it already names every tensor `<role>-<layer>`.
> `attn_norm-17`. `ffn_up-17`.
>
> I'd budgeted a week to thread layer context through the compute path.
> It's two scopes and a string parse.

---

### A3. The one that will resonate with anyone who has profiled C++ (thread)

**1/**
> The thing that makes profiling llama.cpp hard isn't the timing.
>
> It's that almost every function whose name suggests it does work, doesn't do
> the work.

**2/**
> `llama_kv_cache::cpy_k()` sounds like it copies K into the cache.
>
> ```cpp
> return ggml_set_rows(ctx, k, k_cur, k_idxs);
> ```
>
> It returns a graph node. Takes ~200ns. The actual copy happens later, on a
> different thread.

**3/**
> Same for the per-layer build loop. Timing it measures how long it takes to
> *describe* the model, not run it.
>
> On a warm decode step llama.cpp skips it entirely via graph reuse.

**4/**
> So you can't just wrap RAII scopes around the obvious functions and call it a
> profiler. You get a beautiful flamegraph of graph construction and learn
> nothing.

**5/**
> It forces a two-tier design:
>
> — host scopes for real serial work (KV slot search, sampling, logits readback)
> — op scopes inside `ggml_graph_compute_thread` for the actual math
>
> Tier 2 is where 90% of the time is. Tier 1 is where the surprises are.

---

### A4. The overhead-method post

> Measured what my llama.cpp profiler costs before adding a single feature to it.
>
> Three arms: compiled out / compiled in but off / active. Interleaved, so
> thermal drift hits every arm equally. Bootstrap CIs, not point estimates.
>
> Also: the harness refuses to print a number when the machine is too noisy to
> support one.

---

### A5. The honesty one — this is the post that gets quote-tweeted by people who
### actually build things

> My benchmark harness's most useful feature is that it sometimes refuses to
> give me an answer:
>
> ```
> baseline IQR is 3.85% of median.
> NOTE: that is wider than the 2% budget being tested.
> This machine cannot resolve a 2% effect right now.
> ```
>
> A profiler that lies about its own overhead is worse than no profiler.

---

### A6. The diff-size post

> The entire patch to instrument llama.cpp with per-token, per-node,
> per-layer profiling:
>
> **104 changed lines across 4 files.**
>
> Everything else lives in its own translation unit that gets copied in.
>
> If you want a maintainer to read your patch, the first requirement is that
> your patch is readable.

---

### A7. Zero-overhead, proven the only way it can be

> "Zero overhead when disabled" is a claim people make and never check.
>
> You can't prove an absence with a benchmark. So:
>
> ```
> build-off/tokenscope.lib      864 bytes,  0 symbols
> build-on/tokenscope.lib   992,690 bytes, 68 symbols
> ```
>
> A symbol table can prove it. A stopwatch can't.

---

### A8. Small, practical, high-utility

> Wrote a script that synthesizes a random-weight llama GGUF so my profiler's
> tests don't need a 400MB download.
>
> Weights are gibberish. Doesn't matter — instrumentation overhead depends on
> the *shape* of the compute, not the values.
>
> Tests you can run offline are tests that get run.

---

## READY NOW — Tier 2 working, real numbers

### B1. The barrier number — strongest single post in this file

> Profiled llama.cpp CPU decode down to individual graph nodes.
>
> **11.2% of worker thread time is spent spinning at a barrier, not computing.**
>
> ```
> ffn         4249.84 ms   69.0%
> barrier      692.53 ms   11.2%   <--
> attn.qkv     640.73 ms   10.4%
> attn.out     359.43 ms    5.8%
> lm_head      160.83 ms    2.6%
> attn.score    42.35 ms    0.7%
> ```
>
> `ggml_barrier` runs after every node. ~700 per token, on each of 8 threads.

**follow-up (important — do not post B1 without it)**
> To be clear about what that does and doesn't mean: 11.2% isn't 11.2% of
> recoverable time.
>
> Some barrier wait is structurally required — the graph has real serial
> dependencies. Separating "required" from "bad partitioning" needs per-node
> arrival spread, which I have in the trace and haven't reduced yet.

*(That reduction is done — see the F-series below. If you post B1 now, F1 is
the follow-up thread, and the honest answer is that most of it is imbalance
and very little of it is recoverable.)*

---

### B2. The validation post — my favourite result

**1/**
> Ran a check on my llama.cpp profiler that I expected to fail.
>
> If CPU decode is bandwidth-bound on streaming weights, then time per phase
> should be proportional to *parameter count*. That's falsifiable.

**2/**
> Predicting every phase from the FFN measurement alone:
>
> ```
> phase          params      pred    meas    delta
> ffn       169,869,312   4249.8  4249.84   (ref)
> attn.qkv   23,592,960    590.3   640.73   +8.6%
> attn.out   14,155,776    354.2   359.43   +1.5%
> lm_head     6,291,456    157.4   160.83   +2.2%
> ```

**3/**
> Three phases, 27x range in weight volume, predicted to within a few percent.
>
> Two things follow.

**4/**
> One: decode is almost purely weight streaming. The actual attention math —
> scores, softmax — is **0.7%**.
>
> If you're optimizing attention arithmetic for batch-1 CPU decode, you're
> optimizing 0.7% of the workload.

**5/**
> Two, and this is why I care more: the tool is measuring what it says it is.
>
> Misattributed nodes, or double-counting across threads, or clock drift would
> not reproduce a 27x spread to within a few percent by accident.

**6/**
> The +8.6% on attn.qkv is the interesting residual, not noise to smooth over.
> That bucket carries RoPE and the KV cache write on top of the projection.
>
> It *should* be the one phase that overshoots. It is.

---

### B3. The measured-overhead post

> Full per-node profiling of llama.cpp: every graph node, every worker thread,
> ~2.9M records over a 256-token run.
>
> Cost: **+0.67%**, 95% CI [+0.12, +1.67].
>
> First arm in five measurement sessions whose interval excludes zero. Both ends
> inside the 2% budget, which is the part that matters.

**follow-up**
> Also checked the obvious way that number could be a lie: a profiler that fills
> its buffer and stops recording gets cheaper.
>
> `229 MB, 259 tokens, 8 threads, 0 dropped.`
>
> The full run was recorded. 0.67% is the cost of recording all of it.

---

### B4. The design-vs-reality post

> My design doc predicted ~0.1% overhead from a per-scope cost model, and
> explicitly warned the model might not hold — cache pollution, and barriers
> turning per-thread jitter into whole-graph latency.
>
> Measured: 0.67%.
>
> Right risks. Magnitude wrong by 6x. That's the whole argument for measuring.

---

### B5. Perfetto screenshot

> [PERFETTO SCREENSHOT of a level-3 trace]
>
> Every generated token, every graph node, every worker thread. 24 layers.
>
> Chrome Trace Event JSON, so Perfetto does all the UI work.

> **Note to self:** needs an actual screenshot. Zoom to 2-3 tokens so the
> per-node structure and the barrier gaps are both visible.

---

## READY NOW — barrier decomposition (F9)

These discharge the promise made in B1's follow-up. **Post B1 before these, or
post F1 standalone — it works either way, but the sequence is the story.**

### F1. The payoff thread — strongest thing in this file

**1/**
> Earlier I posted that 11.2% of llama.cpp's worker thread time is spent
> spinning at a barrier, and said I couldn't yet tell you how much of that was
> recoverable.
>
> Now I can. It's less than you'd hope, and the reason is more interesting than
> the number.

**2/**
> First: what kind of waiting is it?
>
> ```
> barrier wait, steady state    41.33 ms
>   arrival imbalance           34.59 ms   83.7%
>   release latency              6.74 ms   16.3%
> ```
>
> (Total is 62.31 ms; 20.98 of it is one barrier — the thread pool spinning up
> on the first traced token. Excluded, because it isn't a property of the graph.)
>
> 83.7% is threads that finished early, standing around waiting for a straggler.
> That's a partitioning problem, not a serial dependency.

**3/**
> So which nodes make the other threads wait?
>
> Not the big matmuls. They use all 8 threads and leak 4-9% to skew — normal.
>
> It's these:
>
> ```
> node          work      imbalance   threads busy
> ffn_swiglu   267.0 us     1.75 ms       1.8
> l_out        204.3 us     1.30 ms       2.0
> attn_norm    211.9 us    657.9 us       1.8
> ffn_norm     184.4 us    386.1 us       1.4
> ```

**4/**
> Read that column again. **1.4 to 2.0 threads busy, out of 8.**
>
> Ten node types cost more in other threads' waiting than in their own
> arithmetic. Together: 0.32% of the work, 14% of all the imbalance.
>
> The cheapest nodes in the graph are the most expensive barriers.

**5/**
> The mechanism is four lines of ggml, and it's not a bug — it's a default
> meeting an edge case.
>
> ```c
> const int nr  = ggml_nrows(src0);
> const int dr  = (nr + nth - 1)/nth;
> const int ir0 = dr*ith;
> const int ir1 = MIN(ir0 + dr, nr);
> ```
>
> Elementwise ops split over rows.

**6/**
> At batch size 1, a hidden state is **one row**.
>
> nr=1, nth=8 → dr=1 → thread 0 gets [0,1). Every other thread gets
> ir0 = ith ≥ 1 = ir1. An empty range.
>
> One thread works. Seven fall straight through to the barrier.

**7/**
> `n_tasks` doesn't save them. It's read only by `ggml_graph_plan`, for
> work-buffer sizing. It never gates dispatch.
>
> All 8 threads enter every node. All 8 pay the barrier after it. 412 barriers
> per token, 120 of them for nodes only one thread worked on.

**8/**
> Here's the part that makes it a measurement instead of a story.
>
> If the cause is really nrows==1, the same nodes should parallelize fine when
> there are many rows. That's falsifiable. Prefill is exactly that workload.

**9/**
> Same model, same 8 threads, decode vs pp64:
>
> ```
> node          decode busy   pp64 busy
> ffn_swiglu        1.79        7.67
> l_out             2.04        7.88
> attn_norm         1.75        8.00
> ffn_norm          1.44        7.71
> ffn_out           8.00        8.00   <- already parallel, doesn't move
> ```
>
> Confirmed.

**10/**
> So what's it worth? Here's where I have to argue against my own headline.
>
> ```
> graph wall time                 25.207 ms/tok
> near-serial nodes                0.385 ms/tok
> perfect 8-way parallelization
>   would save                     0.337 ms/tok  = 1.34%
> ```
>
> **1.34%. And that's an upper bound that won't be reached.**

**11/**
> These tensors are 1×768. Three microseconds of work.
>
> Split that eight ways and each thread has less work than the barrier that
> follows it. You'd spend more on dispatch than you'd save.
>
> The fix isn't "parallelize them". It's **fuse them** — pay one barrier
> instead of five.

**12/**
> The thread-time number (11.2%) is the flattering one. The wall-time number
> (1.34%, optimistic) is the true one.
>
> A profiler that only reports the first would have me writing a patch that
> makes things slower.
>
> Method + traces: github.com/PS12007/tokenscope

---

### F2. The standalone version (single post)

> Profiled llama.cpp CPU decode per graph node, per thread.
>
> The nodes that waste the most time aren't the matmuls. They're the tiny
> elementwise ops — swiglu, residual add, rmsnorm.
>
> They run on **1.4 of 8 threads**, because at batch size 1 a hidden state is
> one row, and ggml splits work by rows.
>
> 0.32% of the work. 14% of all the barrier waiting.

---

### F3. The honesty post — companion to A5

> Found a real inefficiency in llama.cpp: 29% of decode barriers are paid for
> nodes only one thread worked on.
>
> Thread-time cost: sounds huge.
> Wall-clock upper bound: **1.34%**.
>
> Posting the second number, because the first one would have had me optimizing
> something that can't move.

---

### F4. For the "measure, don't guess" crowd

> My profiler said tiny elementwise ops were serialized at batch size 1.
>
> Instead of shipping that, I turned it into a prediction: if the cause is
> nrows==1, they should parallelize during prefill, where nrows = n_tokens.
>
> Ran it. 1.4 busy threads → 7.7.
>
> A finding you can't falsify is a story.

---

### F5. The tooling detail (niche, but the tool people will like it)

> Nice property of instrumenting a graph executor: node scopes and barrier
> scopes strictly alternate on every thread, and every thread walks the same
> node list in the same order.
>
> So the k-th barrier on thread 3 *is* the k-th barrier on thread 6. Attribution
> is exact, not inferred.
>
> The analyzer refuses to report anything if that ever stops holding.

---

## NEEDS: three model sizes measured

### C0. The per-model table

> Decode time breakdown, three model sizes, same machine:
>
> [TABLE]
>
> [ONE SENTENCE ABOUT WHAT CHANGES WITH SIZE]

---

## NEEDS: a real quantized model, on Linux

### C1. The finding post (thread) — reserve for something better than B2

**1/**
> Built a per-token profiler for llama.cpp mostly to have one.
>
> Then it showed me something I'd have bet against.

**2/**
> [THE FINDING — specific, numeric, surprising]

**3/**
> [WHY IT HAPPENS — the mechanism, not the observation]

**4/**
> [WHAT IT MEANS — what someone should do differently]

**5/**
> Tool's here if you want to look at your own traces: [link]
>
> It's a build flag, no new dependencies, and it writes Chrome Trace JSON so
> Perfetto does all the UI work.

> **Note to self:** do not post this until C1/2 is a real, verified,
> counterintuitive result. A thread built around "and it turns out FFN is
> expensive" is worse than no thread. If the traces never produce a genuine
> surprise, say that instead — "I built this expecting to find X and the data
> says the boring answer is right" is a real post, and an honest one.

---

## NEEDS: upstream conversation started

### D1.

> Opened an issue on llama.cpp describing the profiler before proposing a PR.
>
> Patch is 24 lines. Overhead is measured, not asserted. Off by default.
>
> Asking whether they want it at all before writing code they didn't ask for
> seems like the minimum.

---

### D2. If it lands

> [merged/closed + what was learned either way]

> **Note to self:** post this even if it's closed. "Here's what a maintainer
> said no to and why they were right" is more useful to more people than another
> merge screenshot.

---

## EVERGREEN — usable any time, low risk

### E0. The findings-file post

> My profiler's findings doc opens with: the three things I built it to catch
> aren't happening.
>
> KV realloc, logits buffer reserve, graph rebuild — all instrumented, all
> ~0.4% combined.
>
> That's the tool working. A profiler that only confirms your hypotheses isn't
> measuring anything.

---

### E1.

> Perfetto's UI is free.
>
> Chrome Trace Event Format is a JSON array of `{name, cat, ph, ts, dur, pid,
> tid}`.
>
> An afternoon of work gets you a professional trace viewer for your tool.
> Stop building custom dashboards.

---

### E2.

> Profiler design rule I keep relearning:
>
> Never silently drop data.
>
> A ring buffer that wraps gives you a trace that *looks* complete and is
> missing the beginning of the run. You'll spend a day chasing a bug that isn't
> in your code.
>
> Bound it, drop loudly, print the count.

---

### E3.

> The biggest speedup in my profiler's hot path wasn't a faster clock.
>
> It was noticing that the end of "work" and the start of "wait" are the same
> instant.
>
> 4 timestamp reads per node → 1. The clock read was 90% of the cost.

---

### E4.

> If you're building dev tooling: measure your own overhead **before** you add
> features.
>
> Instrument everything first, and when the number comes back at 6% you have
> twenty suspects and no way to tell them apart.
