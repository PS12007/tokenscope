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

### A4. The overhead post (single post, has the strongest credibility signal)

> Measured what my llama.cpp profiler costs before adding a single feature to it.
>
> Three arms: compiled out / compiled in but off / active. Interleaved.
> Bootstrap CIs, not point estimates.
>
> Result: not distinguishable from zero, ±0.5%.
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

> The entire patch to instrument llama.cpp with per-token profiling:
>
> **24 changed lines across 3 files.**
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

## NEEDS: full per-layer instrumentation working (level 2/3)

### B1. The screenshot post — this is the one that travels

> Every generated token, broken down by phase, in Perfetto.
>
> [PERFETTO SCREENSHOT]
>
> That wide bar is not compute. It's seven threads waiting at a barrier for the
> eighth.

---

### B2. The work-vs-wait post

> `ggml_barrier` runs after *every node*. ~500 barriers per token, per thread.
>
> Which means a profiler that reports "attention took 15ms across 8 threads"
> without splitting work from wait is actively lying to you.
>
> Most of that can be 7 threads spinning.

**follow-up**
> It also means instrumentation overhead isn't the *mean* across threads.
>
> It's the **max**. Every barrier waits for the slowest thread.
>
> Jitter that's invisible in a single-threaded profiler compounds here, hundreds
> of times per token.

---

### B3. The per-model table

> Decode time breakdown, three model sizes, same machine:
>
> [TABLE]
>
> [ONE SENTENCE ABOUT WHAT CHANGES WITH SIZE]

---

## NEEDS: a genuine non-obvious finding from a real trace

### C1. The finding post (thread) — the highest-value post in this file

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
