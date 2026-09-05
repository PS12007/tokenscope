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
> **123 changed lines across 6 files.**
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

## READY NOW — the thread sweep (F10)

The strongest *practically useful* result in the project: it changes what a
reader does this afternoon. F1 is the better tooling story; G1 is the better
advice.

### G1. The headline thread

**1/**
> Swept llama.cpp CPU decode from 1 to 28 threads on a 20-core machine.
>
> Nothing beats **2.2×**.
>
> Not 8 threads. Not 16. Not 28. Four threads gets you 2.09× and after that
> you're buying rounding error with cores.

**2/**
> ```
> thr   tok/s  speedup  par.eff  barrier%
>   1   20.25    1.00x     100%      0.1%
>   2   35.12    1.73x      87%      2.4%
>   4   42.34    2.09x      52%      5.7%
>   6   44.82    2.21x      37%      8.4%   <- peak
>   8   44.59    2.20x      28%     12.2%
>  12   40.59    2.00x      17%     21.2%
>  16   41.50    2.05x      13%     22.0%
>  28   39.33    1.94x       7%     22.9%
> ```
>
> 28 threads is **12% slower** than 6, using 7× the cores.

**3/**
> Parallel efficiency: 100% → 7%.
>
> If you're running CPU inference on a big box and you set `-t` to your core
> count because that seemed like the obvious thing, you are probably paying for
> a lot of electricity to go slower.

**4/**
> Where does the time go? Straight into the barrier.
>
> Barrier wait as a share of worker thread time climbs monotonically:
> 0.1% → 12.2% at 8 threads → **22.9%** at 20.
>
> By 12 threads more than a fifth of all worker CPU is spinning in
> `ggml_thread_cpu_relax()`.

**5/**
> But — and this is the part I want to get right — **the barrier isn't the
> cause.**
>
> I'd measured earlier that decode on this model is almost pure weight
> streaming. The time is memory traffic, not arithmetic.

**6/**
> So bandwidth saturates around 4 threads. After that an extra thread physically
> cannot go faster. It just finishes its slice out of step with the others, and
> the difference gets paid at the next barrier.
>
> The barrier is where wasted parallelism becomes **visible**. Not where it's
> created.

**7/**
> Which is the whole argument for splitting work from wait.
>
> A profiler that reports "ffn: 69% across 8 threads" cannot tell you that a
> fifth of that is spinning — or that it's spinning *because you asked for too
> many threads*.
>
> Same number. Opposite advice.

**8/**
> One more thing fell out. Per-thread compute spread is 0-2% up to 8 threads,
> then jumps to 13-29%.
>
> 8 is the P-core count on this chip. ggml hands every thread an **equal number
> of rows** — which stops being a fair split the moment the cores aren't equal.

**9/**
> I predicted that would show up as two tight clusters, one per core type.
>
> It doesn't. At 16 threads the spread is continuous: 30.0, 30.2, 30.5, 33.3,
> 33.7, 35.1 … 44.9 ms.
>
> So: symptom confirmed, mechanism not. Proving it needs thread pinning, which
> I haven't done.

**10/**
> Caveat that matters more than usual here: synthetic F32 weights.
>
> This entire result turns on the workload being bandwidth-bound. Q4_K_M reads
> ~4× fewer bytes per parameter and should scale further before hitting the same
> wall.
>
> This is a fact about arithmetic intensity, not about llama.cpp's threading.

**11/**
> Method, traces, and the analysis that produced the barrier column:
> github.com/PS12007/tokenscope
>
> The table is reproducible from the repo. The uninstrumented build produced the
> throughput numbers — I'm not quoting tok/s measured by my own profiler.

---

### G2. The practical single post

> PSA for anyone running llama.cpp on CPU:
>
> more threads than ~4-6 probably makes you slower.
>
> Measured 1→28 threads on a 20-core box. Peak was 6 threads at 2.21×. 28
> threads came in **below** 6, at 7% parallel efficiency.
>
> Decode is memory-bound. You can't thread your way past DRAM.

---

### G3. The one for people who like being wrong in public

> Predicted: per-thread times on a P-core/E-core CPU would split into two tight
> clusters.
>
> Measured: 30.0, 30.2, 30.5, 33.3, 33.7, 35.1, 37.2, 39.0 … 44.9. Continuous.
>
> The step at 8 threads is real and lines up with the P-core count. The clean
> bimodal signature that would *prove* it isn't there.
>
> Symptom confirmed. Mechanism: still a guess.

---

### G4. The negative result (pairs with A4/A5)

> My design doc predicted profiler overhead would compound with thread count —
> a barrier makes the graph pay the *max* of per-thread overhead, not the mean.
>
> Tested it at 8 and 28 threads. Both CIs span zero. The harness refused to
> certify either.
>
> Not confirmed. Not refuted. Posting it as neither.

**follow-up**
> The tempting move is to read "no measurable overhead" as "my mitigations
> worked."
>
> But the baseline IQR was 2-4% and the effect I was hunting is smaller than
> that. The honest version is: if it compounds, it doesn't compound enough to
> see at 28 threads on this machine.

---

### G5. The energy angle (different audience, same data)

> Same llama.cpp workload, same tokens generated:
>
> 4 threads → 2.09× speedup
> 28 threads → 1.94× speedup, 7× the cores
>
> Seven times the CPU time to go slightly slower.
>
> Thread count defaults are an energy decision and almost nobody measures them.

---

## READY NOW — sampling measured, and the boundary bug (F11)

### H1. The bug thread — the best "my tool caught my tool" story in the project

**1/**
> Added sampling scopes to my llama.cpp profiler.
>
> The summary immediately reported **100.1%** of decode time attributed.
>
> Not 99.8. Not 101. A tenth of a percent over — which is the most annoying
> possible amount, because it's too small to be a stupid bug and too large to
> be float noise.

**2/**
> Every timestamp was correct. Every scope was associated with the right token.
>
> The bug was that I'd asked a question with two halves and only answered one.
>
> "Per token" and "inside what" are not the same question.

**3/**
> `common_sampler_sample` runs **after** `llama_context::decode` returns.
>
> So the sampling scopes belong to token 12 — and they happen outside token
> 12's slice.
>
> Charging them against it adds time the slice never contained. 341 of 341
> sampling scopes were outside. 16 of 31 tokens went over.

**4/**
> The fix isn't an epsilon. It's a distinction the report now makes out loud:
>
> - per-token work **inside** the decode slice → charged against it
> - per-token work **outside** it → its own section, labelled
>
> Both are real costs of producing a token. Only one is part of decode.

**5/**
> This is the second time a "these percentages cannot exceed 100" check found
> something inspection wouldn't have.
>
> First time it was scope misparenting. This time a conceptual boundary error.
>
> Neither looked like a bug in the code. Both were bugs in the *claim*.

**6/**
> If you build measurement tools, put in the checks that can only fail if
> you're wrong about something structural.
>
> Not "is this number plausible." Something more like a conservation law.
>
> Plausible numbers are exactly the ones that hide this.

---

### H2. The finding itself (single post)

> Finally measured the two parts of llama.cpp's token loop that `llama-bench`
> never touches, because it never samples and never tokenizes.
>
> ```
> sample        28.6 us/token   0.129% of decode
> tok.decode     0.62 us/token  0.003%
> ```
>
> **99.87% of a generated token is `llama_context::decode`.** And 99.5% of
> *that* is one call: `graph_compute`.

**follow-up**
> Caveat that does real work here: 8192-token synthetic vocabulary. A real 128k
> vocab makes every softmax and sort ~16× bigger, and a grammar-constrained
> sampler is a different workload entirely.
>
> 0.13% is a fact about this run, not about sampling.

---

### H3. The one about instrumenting the wrong function

> Added a scope to `llama_sampler_sample`. Ran it. Zero samples recorded.
>
> Turns out llama-cli never calls it — `common_sampler_sample` calls
> `llama_sampler_apply` once per sampler in the chain instead.
>
> The API function with the obvious name is not the one on the hot path. Again.

**follow-up**
> This is the third time on this project that the function whose name describes
> the work was not the function that does the work.
>
> `cpy_k` doesn't copy. The per-layer build loop doesn't run the model.
> `llama_sampler_sample` doesn't sample.

---

### H4. Tooling honesty (small, evergreen)

> Found a message in my own profiler that was confidently wrong.
>
> "8.2% attributed — the remainder is time no scope covers yet."
>
> No. The remainder was 23 tokens that a capture window deliberately skipped.
> The tool knew that and said something else.
>
> Fixed. But: check what your tool says when it's *partly* out of data.

---

## READY NOW — first real quantized model (F12)

This retires the "synthetic weights" caveat that every earlier post had to
carry. Post I1 after G1; it is the payoff to G1's closing caveat.

### I1. Three predictions, tested (thread)

**1/**
> Every number I'd posted about llama.cpp came from synthetic F32 weights, and
> every post said so.
>
> Finally ran a real quantized model — Qwen2.5-0.5B Q4_K_M.
>
> Three standing predictions. Two held. One held only in a form I hadn't been
> using.

**2/**
> **Prediction 1.** I'd argued decode is bandwidth-bound, so a model reading
> fewer bytes per parameter should scale to more threads.
>
> ```
>        F32 synth      Q4_K_M real
> thr    speedup        speedup
>   2      1.73x          1.95x
>   4      2.09x          2.86x
>   6      2.21x          3.28x  <- peak, both
>  28      1.94x          2.44x
> ```
>
> Held.

**3/**
> But note what the prediction *didn't* say: the peak is still at six threads.
>
> The wall moved **up**, not **out**.
>
> More headroom per thread. Same number of useful threads.

**4/**
> **Prediction 2** is the one I got to be wrong about in an interesting way.
>
> I'd predicted phase time tracks *bytes of weights read*, then wrote "and for a
> uniform dtype that's just parameter count."
>
> Then used parameter count for everything. Because on F32 they're identical.

**5/**
> A real Q4_K_M file is **not** uniform. llama.cpp quantizes tensors differently
> by role.
>
> In this one the output projection is Q8_0 — 8.5 bits/weight — while everything
> else sits near 5.5.
>
> So for the first time the two versions of my prediction disagree. And they can
> be told apart.

**6/**
> ```
> phase      bits/w  params%  bytes%  time%   err(param)  err(byte)
> ffn          5.51    63.5%   55.2%  56.4%      -7.2       +1.2
> lm_head      8.50    27.6%   36.9%  34.0%      +6.4       -2.9
> attn.qkv     5.70     5.0%    4.5%   6.0%      +1.0       +1.5
> attn.out     5.50     3.9%    3.4%   3.6%      -0.3       +0.2
> ```
>
> Bytes: within 2.9 points. Parameters: off by 7.2.

**7/**
> And look at *how* it's wrong — ffn −7.2, lm_head +6.4. Equal and opposite.
>
> That's the signature of a share being moved from one phase to another by
> nothing but dtype.
>
> The law survives. The shortcut doesn't. Read the tensor types, not the config.

**8/**
> **Prediction 3** I got backwards, and the finding now says so.
>
> I'd written that quantization should make my "tiny serial nodes" effect
> *smaller*, since Q4 is more compute-bound.
>
> It got bigger. 0.32% → 0.74% of work; 14% → 19% of all barrier imbalance.

**9/**
> Obvious in hindsight. Those nodes are a fixed cost — a single-row elementwise
> op doesn't care what dtype the matmuls are.
>
> Make the matmuls cheaper and the fixed cost becomes a *larger* share.
>
> I had the direction of the ratio backwards.

**10/**
> Bonus finding that only a real model could produce:
>
> **`lm_head` is 34% of decode time.**
>
> 152k vocabulary against n_embd 896 — the output projection is 28% of
> everything streamed. And it's stored at the highest precision in the file.

**11/**
> Second bonus, this one about GQA.
>
> Qwen2.5-0.5B has 14 query heads and 2 KV heads. So K and V projections are
> 128 wide against Q's 896.
>
> `Kcur` uses 2.5 of 6 threads. `Vcur` 3.0. The FFN matmuls use all 6.
>
> GQA shrinks your KV cache and narrows your matmuls.

**12/**
> Method, traces, every caveat: github.com/PS12007/tokenscope
>
> One model, one machine, 630M params. "Q4_K_M" is a recipe, not a dtype —
> this file is 133 Q5_0 tensors, 121 F32, 13 Q8_0, 12 Q6_K, 12 Q4_K.
>
> Which is rather the point of finding #2.

---

### I2. The standalone (best single post here)

> "Q4_K_M" is not a dtype. It's a recipe.
>
> The Qwen2.5-0.5B Q4_K_M GGUF contains 133 Q5_0 tensors, 121 F32, 13 Q8_0, 12
> Q6_K and 12 Q4_K.
>
> Its output projection is Q8_0 — the *least* compressed thing in the file, and
> **34% of decode time.**
>
> Predicting performance from parameter counts misses this by 7 points. Bytes
> get it to 3.

---

### I3. The being-wrong post

> Wrote a caveat predicting my own finding would get smaller on a quantized
> model.
>
> Measured it. Got bigger. 14% → 19%.
>
> Obvious afterwards: single-row elementwise ops are a fixed cost. Make the
> matmuls cheaper and the fixed cost is a *larger* share, not a smaller one.
>
> The caveat now says it was backwards.

---

### I4. Practical, for people running small models

> If you run a small model with a big vocabulary on CPU, check what your output
> projection is quantized to.
>
> On Qwen2.5-0.5B Q4_K_M it's Q8_0 while the rest of the model is ~5.5 bits —
> 28% of the parameters, 37% of the bytes, **34% of decode time**.
>
> Vocabulary doesn't shrink when your model does.

---

## READY NOW — thread pinning settles it (F14)

Post after G1/G3, which set up the unconfirmed mechanism this pays off.

### J1. The payoff thread

**1/**
> Earlier I posted that llama.cpp CPU decode wastes a growing share of thread
> time at barriers, guessed it was P-cores and E-cores being handed equal work,
> and said I couldn't prove it.
>
> Proved it. The fix I expected turns out to be the wrong fix.

**2/**
> The test: take the heterogeneity away and see if the waste follows.
>
> 12 threads, unpinned, mixed cores → **13% spread, 21.2% barrier**
> 12 threads pinned to 12 identical E-cores → **2% spread, 11.1% barrier**
>
> Same model. Same graph. Same thread count. Same code.

**3/**
> The per-thread compute times on twelve identical cores:
>
> ```
> 134.8 134.9 134.9 135.0 135.2 135.3
> 135.3 135.4 136.6 136.9 136.9 137.0
> ```
>
> That's what "equal work to equal workers" looks like. Barrier waste halves.

**4/**
> So the mechanism is confirmed. ggml splits work by rows:
>
> ```c
> const int dr = (nr + nth - 1)/nth;   // same count for every thread
> ```
>
> Optimal when every worker is equally fast. On this chip P-cores are **2.88×**
> faster than E-cores.

**5/**
> Now the part I got wrong.
>
> ```
>  6 thr unpinned          45.15 tok/s
>  8 thr unpinned          43.60
> 12 thr unpinned          40.89
>  8 thr P-cores only      39.21   -10%
> 12 thr E-cores only      37.51    -8%
> ```
>
> **Every pinned config is slower than letting the scheduler choose.**

**6/**
> Including the homogeneous one that halves its own barrier waste.
>
> Twelve slow equal cores beat neither eight fast ones nor the scheduler's mix.
>
> Removing the heterogeneity costs more than the heterogeneity did.

**7/**
> So: the waste is real. The mechanism is confirmed. And pinning — the obvious
> remedy, the one I set out to validate — makes things worse.
>
> What the data actually argues for is **proportional** row assignment. Give a
> P-core more rows than an E-core. Keep the fast cores *and* close the gap.

**8/**
> Near-miss worth confessing. My first version of this experiment pinned single
> threads and compared decode:
>
> P-core: 28.0 tok/s. E-core: 28.2 tok/s.
>
> Which reads exactly like a broken CPU mask. I nearly threw the experiment out.

**9/**
> Decode is memory-bound. Both core types sit waiting on the same DRAM, so the
> core doesn't matter.
>
> Ran the same masks on *prefill*, which is compute-bound: **265.95 vs 92.39.**
>
> The mask had been working the entire time. I'd picked a workload that couldn't
> see the thing I was measuring.

**10/**
> Also recorded, not explained: 8 threads pinned to what should be 8 distinct
> P-cores leave exactly one thread 12-20% slower, across three runs, and it's
> not consistently the main thread.
>
> So the confirmation rests on the clean 12-core run, and the writeup says so.
>
> github.com/PS12007/tokenscope

---

### J2. The standalone

> llama.cpp splits graph work by rows: every thread gets the same number.
>
> That's optimal when every core is the same speed. On a modern hybrid CPU my
> P-cores are **2.88× faster** than my E-cores.
>
> Pin 12 threads to 12 *identical* cores and barrier waste halves — 21.2% →
> 11.1%.
>
> Equal work to unequal workers is the whole bug.

---

### J3. The one that saves someone an afternoon

> "Pin your threads to P-cores" is advice I've seen a lot and just measured.
>
> 8 threads on P-cores: 39.21 tok/s
> 8 threads unpinned:   43.60 tok/s
>
> The scheduler beat me by 10%. It beat me at 12 threads too.
>
> Pinning fixed the imbalance I was chasing and lost more than it saved.

---

### J4. The methodology one (pairs with F4/G4)

> Measured P-core vs E-core on llama.cpp decode: 28.0 vs 28.2 tok/s. Concluded
> my CPU affinity mask was broken.
>
> It wasn't. Decode is memory-bound — both core types wait on the same DRAM.
>
> Same masks on compute-bound prefill: 265.95 vs 92.39. **2.88×.**
>
> A null result is also a claim about your workload.

---

## READY NOW — three model sizes (F13)

### C0. The per-model table

> Decode time breakdown, three models, same machine, same binary:
>
> ```
> model                 params     ffn   lm_head   barrier
> tiny   8L F32         8.9 M    27.6%      0.9%     54.0%
> mid   24L F32         220 M    70.1%      2.6%     10.1%
> Qwen2.5-0.5B Q4_K_M   630 M    49.2%     29.6%     11.3%
> ```
>
> The biggest consumer is a **different thing in every row.**
>
> "Where does CPU decode time go" has no model-independent answer.

**follow-up**
> The 54% on the tiny model isn't llama.cpp being slow. It's a graph executor
> run far below its design size: ~200 nodes/token, and each node's arithmetic
> finishes before the barrier protecting it does.
>
> Two things stayed stable across all three: attention *math* is 0.7-2.7%, and
> `norm` never exceeds 0.4%.

---

## NEEDS: Linux  (the quantized-model half is done -- see F12 / the I-series)

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
