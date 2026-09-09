# 04 — The whole project, and everything wrong with it

**A single reference: what tokenscope is, what it has established, how much each
number is actually worth, and every known defect, gap and trap.**

Written during session 6 (2026-09-08). It exists because the other documents are
organised by *when* things were learned ([`FINDINGS.md`](FINDINGS.md)) or by
*what to do next* ([`HANDOFF.md`](HANDOFF.md)), and neither answers the question
a newcomer or a reviewer actually asks: **which of these numbers can I believe,
and what is broken?**

Sections 2 and 4 are the ones that did not exist anywhere before. If you read
two sections, read those.

> **Updated after F33 landed.** F24 has been re-measured: the effect is real at
> **+1.62% [+1.10, +2.15]** over six blocks, and its prefill control is no
> longer a clean null, which promotes M6 from caveat to live suspicion.
>
> **Updated again after F39 and F40 (session 7).** The harness's own
> false-positive rate is measured for the first time, on two binaries that
> differ by one byte of dead code — and **it depends on the thread count**:
>
> | threads | decode | prefill |
> |---|---|---|
> | 8 | **+0.02% [−0.21, +0.26]** | **−0.01% [−0.12, +0.11]** |
> | 16 | −0.04% [−0.49, +0.42] | **+0.59% [+0.01, +1.16]** — resolved, false by construction |
>
> **At 8 threads this harness is very good.** F24 clears its own 16-thread floor
> by 3.6×, and F36's overheads clear the 8-thread floor. What does not survive
> is F24/F33's prefill *control* at 16 threads (M13). **F39 over-read its own
> two runs** and F40 corrected it fifty minutes later — see section 6 for the
> error, which is the instructive part.

---

## 1. What this is, and what it is not

**tokenscope** is a deterministic scope-timing profiler compiled into
`llama.cpp`, plus a body of measurements taken with it about where per-token
time actually goes on CPU inference.

- **Deterministic, not sampled.** Explicitly placed scopes with a 24-byte
  record, thread-local chunked arenas, no locks in the hot path, Chrome Trace
  Event output. A trace is *interpretable* — you can point at a node and a
  thread — rather than statistical.
- **Three levels.** 1 = host scopes, 2 = per-node accumulators, 3 = every node
  event on every worker thread.
- **Compiled out by default.** `GGML_TOKENSCOPE=OFF` leaves no symbol, branch or
  storage, verified against the symbol table rather than asserted.
- **~174 changed lines across 7 files** in the upstream patch
  (`patches/01-instrument.patch`).

**What it is not:**

- Not a sampling profiler, so it cannot find cost in code with no scope in it.
- Not cross-platform-validated. **Everything is MSVC on one Windows machine.**
  No Linux, no GCC, no clang on a real workload, no ARM, no NUMA hardware.
- Not upstreamed. Nothing has been filed. `llama.cpp`'s `AGENTS.md` forbids an
  agent writing issue or PR text, so [`03`](03-upstream-issue-draft.md) holds an
  evidence pack rather than a draft to paste.
- Not a general benchmark harness. The tools in `tools/` are built around one
  machine's failure modes.

### The hardware every number comes from

| | |
|---|---|
| CPU | i7-14700HX — 8 P-cores + 12 E-cores, **28 logical**, heterogeneous |
| P vs E | P-cores 2.88× faster on compute-bound work; **indistinguishable on decode**, which is bandwidth-bound (F14) |
| RAM | 15.7 GB total, typically 5–7 GB free |
| Toolchain | MSVC 19.44, cmake 4.4.3, ninja 1.13.2 (both pip-installed) |
| llama.cpp | pinned at `4d91760`, **patched in place** |
| Threading | **OpenMP (`vcomp`) by default** — see F26, this was misdocumented for four sessions |

**Core heterogeneity is not a detail here.** It is the mechanism behind F10 and
F14, and it means a thread count on this machine is not a thread count on a
homogeneous server part.

---

## 2. How to read a number in this repo

This is the section session 6 exists to write. The project spent five sessions
with a single word — *certified* — doing work it had not earned, and the repair
is a ladder rather than a badge.

### The trust ladder

| Tier | What it means | How to tell | Examples |
|---|---|---|---|
| **A — structural** | Not statistical at all. True by construction, code reading, symbol table or a CI assertion that fails the build | No interval quoted, because none is needed | zero-overhead-when-off; node/barrier alternation; F22's two-module buffer identity; F8/F20 naming coverage |
| **B — large effect** | Effect is ≫10× the machine's drift term (~0.5pp) | Interval width is irrelevant to the conclusion | F27's −54% / −78% / −13% rows (**its −2.06% row is Tier D and now re-measured — F41**), F28 (76–83% of release latency), F10 (2.2× ceiling), F14 (2.88× P/E), F7/F19/F21 byte law ratios |
| **C — medium, structured** | Effect is several × drift, and supported by a *structure* (control arms, non-overlapping ranges, n≥12) rather than by one interval | Quote with its n and its control | F9 imbalance/release split, F23 chunking-mode ratios (n=12, two non-overlapping comparisons) |
| **D — near the floor** | Effect is **1–3×** the drift term. This is where every overhead and throughput percentage lives | **Needs a between-block `t` interval** and must clear the measured false-positive floor, which **depends on thread count** (F39, F40): decode **±0.24pp at 8 threads**, **±0.45pp at 16**; prefill **±0.12pp at 8** and *unusable at 16*. A bootstrap interval here is roughly half as wide as the truth | F24 (now +1.62%), F25 (+1.16%), F30, F31's three overheads |

**Everything in Tier D published before session 6 was measured with a
within-run bootstrap and is over-precise.** Not necessarily wrong in sign — just
narrower than the evidence supports.

**Tier is a property of a number, not of a finding.** F27 is the trap: its
headline is −54% (Tier B, safe at any interval width) but the same table's
8-thread row was −2.06% (Tier D). A finding can carry rows from two different
tiers, and this document filed F27 under B for a session before anyone noticed.
**F41 closed it**, and the closure vindicated the worry: the decode row survived
at −2.15%, and the *prefill* row on the same line turned out **wrong by 2.7×**
(−1.15% → −0.43%, the old estimate outside the new interval). The row nobody had
checked was the row that was wrong.

### Why Tier D is broken, precisely

`bench_overhead.py`'s bootstrap resamples the runs *inside one invocation*. It
answers "if I redrew these 19 runs from the same afternoon, how far would the
median move." Nobody is asking that. The question is "what would I get if I ran
this again", and until session 6 nothing in the project estimated it.

F31 demonstrated the consequence directly: **the same quantity, the same
unrebuilt binaries, two hours apart, produced two intervals that do not
overlap** — `+0.87% [+0.55, +1.19]` and `+0.31% [+0.01, +0.52]`. Both cleared
the harness's gate.

**Four candidate explanations were tested against the raw data of both runs, and
three of them are not it:**

| candidate | measured | verdict |
|---|---|---|
| position within the round | 0.214pp spread, no monotonic trend | too small |
| lag-1 autocorrelation | +0.136 mean (F31), −0.008 (F30) | too small |
| the unpaired estimator | pairing reconciles F30's two pairs (0.50/0.87 → 0.70/0.72) and does nothing for F31's | not the estimator |
| **splitting one run's own reps in half** | the halves disagree by **up to 0.47pp** | **this is it** |

The instability is *inside* a single invocation. That is why no amount of care
about ordering fixes it, and why the fix has to be an interval that contains it
rather than a better point estimate.

### The fix, and how to use it

`--blocks N` runs the whole round-robin N times and reports

```
mean ± t(N−1) · s / √N
```

over the per-block point estimates, with a real small-sample `t` table. At N=3,
`t = 4.303` — deliberately unflattering, because three passes do not pin a
number down. Both `bench_overhead.py` and `ab_throughput.py` print it and say to
quote it over the bootstrap.

**Rule: no Tier D number leaves this repo without a `--blocks ≥ 3` interval** —
and after F41, **without two such runs pooled to six blocks**. F41 ran one
comparison twice and got interval widths 7× apart (0.14pp against 0.95pp) with
point estimates agreeing to 0.02pp: three blocks can be tight by luck, because
the interval is computed from a block spread that is itself a random variable
(M14).

### But the interval is third in the order of trust, not first

**F34 is the correction to everything above.** A run that started with 870 MB of
free RAM against an 840 MB model reported every instrumented arm as *faster*
than its own compiled-out baseline, and the shared pair's three blocks agreed to
within 0.55pp — **the tightest spread in the run** — with a `t` interval
excluding zero. Sustained contamination does not widen the blocks; it makes them
agree, and their agreement reads as precision.

Blocks defend against drift **between** passes. They do nothing about
contamination spanning **every** pass. So the order is:

1. **The baseline-IQR gate.** If the baseline cannot resolve the effect, nothing
   below it is worth reading, whatever its interval says.
2. **Physical plausibility.** Negative overhead, or a control that moves as much
   as the treatment, voids the run regardless of statistics.
3. **The `t` interval**, meaningful only once 1 and 2 pass — and only above
   **F39's false-positive floor**, which is ±0.5pp on decode over six blocks and
   is *not established at all on prefill*, where the same null resolved at
   +0.59%. An interval that excludes zero is not by itself a result (M12).

Two further guards now enforce this: `preflight_ram()` refuses to start below
1.5x the model size, and a failed gate makes the tool refuse its own block table
rather than printing "quote this" underneath a refusal.

### Harness version matters when reading old numbers

| Since | Change | Affects |
|---|---|---|
| `9ec7c82` | arm order **rotates** each round | everything before it used a fixed order; F30 shows a fixed order costing a pair its certification |
| `6d02d82` | `between_block_ci`, `--blocks` in both tools, "CERTIFIED" removed from `ab_throughput.py` | every published Tier D number predates this |
| `57e3b46` | the gate prints *resolved within this run*, not *certified* | wording only |

---

## 3. The findings register

Trust column uses the ladder above. "⚠" marks a finding whose *stated precision*
is known to be wrong even though its direction may stand.

### The core picture of where time goes

| | Finding | Trust |
|---|---|---|
| **F1** | Host-side control-plane work is not where decode time goes, at all | B |
| **F2** | `kv.slot-search` is the only host-side cost with a real shape — and **both its onward predictions later failed** (F16, F17) | B, superseded in part |
| **F3** | The outliers are all inside the graph — a finding about the tool's own blind spot | A |
| **F4** | Where you draw the slice boundary changes the number by 3% | A |
| **F6** | 11.2% of worker thread time is barrier wait | B |
| **F7** | Phase times track parameter counts to a few percent — the "byte law" | B |
| **F13** | Three models, three different answers to "where does decode time go" | B |
| **F11** | Sampling + detokenization are 0.13% of a token; finding that broke the attribution check | B |

### Barriers and threading

| | Finding | Trust |
|---|---|---|
| **F9** | Most barrier wait is arrival imbalance; a third of barriers are paid for nodes only one thread worked on | C |
| **F10** | Nothing beats 2.2×; every thread past four turns into barrier wait | B |
| **F14** | Core heterogeneity is the mechanism. Pinning proves it and does **not** fix it | B |
| **F15** | Fusion removes 10.6% of barriers and buys nothing — **F9's recommendation was wrong** | B |
| **F23** | ggml already solves core heterogeneity for big matmuls via work stealing, and a thread count can turn it off | C (n=12) |
| **F26** | **Every measurement in the project was on the OpenMP path, and five documents said the opposite** | A |
| **F27** / **F41** | ggml's own barrier is not the cheap one: −54% of decode at 28 threads | **B for the large rows, D re-measured for the rest.** −54.17%, −78.64% and −13.07% are 13–78× the drift term and safe. **F41 re-measured the 8-thread rows over six blocks: decode confirmed at −2.15% [−2.28, −2.02]** (9× the floor), **prefill corrected from −1.15% to −0.43% [−0.65, −0.21]** — F27's prefill estimate is *outside* the new interval, so that row was wrong rather than merely over-precise |
| **F37** | Four candidate causes for M10 tested and all refuted — thermal, thread placement, CPU frequency, workload. Confirms the logical-processor numbering is interleaved, so `0x5555` **is** one thread per P-core | A (they are facts about tests that were run). **Corrects a claim this document published**, and eliminates F14's first candidate |
| **F28** | The barrier this profiler blamed on ggml was **its own allocator**, 76–83% of all release latency | B |

### Models and scaling

| | Finding | Trust |
|---|---|---|
| **F12** | First real quantized model; three standing predictions tested | B |
| **F19** | The 8B run — five of six predictions hold; the byte law gets a controlled experiment via mixed-precision `ffn_down` | B |
| **F21** | The byte law across three architectures and a 13.7× range of `lm_head` share | B |
| **F16** | Context shift is real and cheap — **F2's prediction was wrong** | B |
| **F17** | Concurrent sequences — **F2's last prediction also fails**; the workload broke tokenscope's own prefill/decode boundary | B |

### The profiler auditing itself

| | Finding | Trust |
|---|---|---|
| **F5** | A profiler can misparent its own scopes with correct timestamps | A |
| **F8** | ~15% of ggml graph nodes have no meaningful name | A |
| **F18** | The shared-library build does not link, and MSVC says the obvious fix is illegal (C2492) | A |
| **F22** | The shared build works — the thread-local was never the thing that had to be shared | A |
| **F29** | `TOKENSCOPE_TOKENS=10:11` captured **one** token, silently, since session 4 | A |
| **F20** | The attention output projection is anonymous in every llama.cpp graph, and it is 7% of decode — **a defect in upstream, not here** | A |

### Overhead and throughput — the Tier D block

| | Finding | Trust |
|---|---|---|
| **F24** / **F33** | One line (`nth*4`→`nth*2`), **+1.62% [+1.10, +2.15]** decode over six blocks — the only speedup this project claims. F24's `+1.95% [+1.59, +2.35]` is superseded | **D, re-measured.** Effect real and **~3× the whole width of F39's decode null** (`-0.04% [-0.49, +0.42]`), which is the strongest thing that has ever been said for it. But **its prefill control is now worthless in both directions** (F39/M13), so the structural argument is gone and only the layout question (M6) can settle what the +1.62% is made of |
| **F25** | Static overhead +1.16% [+0.67, +1.87]; shared build's answer eaten by its noise floor | ⚠ D |
| **F30** | Shared overhead +0.87% [+0.55, +1.19] | ⚠ **D, corrected by F31** |
| **F31** | Two bootstrap intervals for one quantity that do not overlap | **A for that** (it is a fact about two runs). **Its "level 3 is a leveller" claim and its generator effect are both RETRACTED by F36** |
| **F34** | A void run: 870 MB free against an 840 MB model produced *negative* overhead in all three pairs, and the tightest block agreement in the run | A (a fact about a failure). Source of M8 and the gate-first ordering |
| **F36** | The overhead numbers settled: static **+0.56% [-0.05, +1.16]**, ninja-shared **+0.92% [+0.38, +1.46]**, MSBuild-shared **+0.77% [+0.05, +1.48]**, three blocks each. Linkage null, generator null | **D, properly measured.** The interval contains every earlier estimate of the same quantity |
| **F39** | **The false-positive rate, measured for the first time.** Two binaries differing in one dead code byte: decode **-0.04% [-0.49, +0.42]** over six blocks (clean, and the same point estimate in two independent runs), prefill **+0.59% [+0.01, +1.16]** (**resolved — a false positive**) | **A for the decode floor** (it is a fact about a null pair, and the number to read every Tier D decode result against). Source of M12, M13 and D9 |
| **F41** | F27's last unchecked row re-measured: **decode −2.15% [−2.28, −2.02]** (survives), **prefill −0.43%** against F27's −1.15%. And the unpredicted part: **two runs of the same comparison gave `t` intervals 7× different in width** (0.14pp vs 0.95pp) with point estimates agreeing to 0.02pp | **D, properly measured.** Source of **M14** |
| **F40** | **The floor is a property of the thread count.** The same null at **8** threads: decode **+0.02% [-0.21, +0.26]**, prefill **-0.01% [-0.12, +0.11]**, both gates passing, A arm *faster* at 46.91 vs 42.50 tok/s. Consistent with F10 — past four threads the measurement is largely scheduling | **A** (a fact about a null pair at two configurations). **Narrows M12, qualifies M13, and withdraws F39's claim that F36 measured nothing** |

**That paragraph used to say F31's "leveller" claim was the durable, non-Tier-D
part. F36 retracted it, and the mistake is the most instructive one in this
document.** The claim compared how far the instrumented arms spread against how
far the compiled-out arms spread — six numbers within half a percent of each
other. In F36 the ordering *reverses* on decode (A 0.16% vs C3 0.32%, against
F31's 0.58% vs 0.27%). It was noise given a mechanism.

**Rule: a claim built from differences between Tier D numbers inherits Tier D,
however structural it sounds.** Being a comparison of spreads rather than a
percentage is what disguised it, and what let it be marked exempt from the
re-measurement that would have caught it.

---

## 4. Known issues register

### 4.1 Methodological — the serious ones

| # | Issue | Status |
|---|---|---|
| **M1** | **Bootstrap CIs are within-run and understate the truth by roughly 2×** in Tier D | Diagnosed and fixed in tooling (§2); **published numbers not yet all re-measured** |
| **M2** | The word "certified" promised reproducibility the method never tested | **Done.** Removed from tool output and from README, `02` and `03` entirely. In `FINDINGS.md` the historical uses are **deliberately left** — they are what each session believed at the time — under a banner at the top of the file defining what the word actually meant and pointing at this ladder |
| **M3** | The instability is *within* one invocation (0.47pp between halves of the same run) and its cause is unidentified — not position, not autocorrelation, not the estimator | **Open.** Blocks contain it; nothing explains it |
| **M8** | **Blocks manufacture confidence under sustained contamination.** F34's three blocks agreed to 0.55pp and excluded zero while reporting a physically impossible result | Guarded by the gate + plausibility check (F34); **the underlying limitation is permanent** |
| **M9** | The machine is shared with whatever else the user is running. A 5.17 GiB `javaw` process voided a 44-minute run | `preflight_ram()` refuses to start; **it cannot detect load that arrives mid-run** |
| **M10** | **Throughput swings between ~38.6 and ~46 tok/s, with no identified cause. F44 caught it switching**: thirteen identical probes 55s apart, nothing else running, gave four consecutive readings at **46.09** then eight at **40.92** — a **12.6% gap between two states each held for minutes**, with an unprompted transition. That contradicts this document's own earlier "not two stable regimes". Eight candidates now refuted: F37's four, plus battery/AC (on AC), the Windows power cap (`PROCTHROTTLEMAX` 100%), memory pressure and background load | F37 killed four candidates: not thermal (more cooling gave a *lower* result; the decay curve is flat), not thread placement (pinning is worse; the best mask is no mask), not CPU frequency (Pearson **r = -0.423**, the wrong sign), not the `-p 0`/`-p 512` workload difference (-0.91%). The full range appears **inside one 4.5-minute window**, so it is run-to-run variance whose median moves, not two stable regimes. ~10x the effects being measured, and it decides whether a run passes the gate | **Open, and the largest unknown in the project.** Live candidates: page-cache/standby-list state for an 840 MB model, per-process power throttling |
| **M11** | A claim built from *differences between* Tier D numbers inherits Tier D. F31's "level 3 is a leveller" was marked non-Tier-D and exempt from re-measurement; F36 reversed it | Recorded; the exemption was the error |
| **M12** | **Blocks can trend rather than scatter, and a trend manufactures a resolved result.** F39's null resolved prefill at **+0.59% [+0.01, +1.16] on binaries that cannot differ**, its six-block estimates rising monotonically in both runs, with the trend absent round-to-round *inside* a block. **Narrowed by F40:** the same null at **8 threads** shows no trend at all (±0.12pp, neither run monotonic), so this is not a general property of `--blocks` — it appears where the underlying measurement is already noisy. F39 generalised from two runs in the noisiest configuration the project measures in | **Open, and configuration-specific.** The failure mode is real and its trigger is unknown. No number of blocks removes a trend common to all of them, but at 8 threads there is no trend to remove |
| **M14** | **A three-block `t` interval's *width* is unstable by up to 7×.** F41 ran one comparison twice: block spreads 0.05pp and 0.37pp, intervals 0.14pp and 0.95pp wide, point estimates agreeing to 0.02pp. Block agreement is itself a random variable and the `t` interval is computed from it, so three blocks can be tight by luck. F34's lesson in benign form — agreement alone distinguishes a real measurement from a contaminated one in neither direction | **Open, cheap fix: two runs of three blocks, pooled to six.** F33, F39, F40, F41 do this; **F36 does not**, so its three overhead numbers' widths carry the instability even though their point estimates stand |
| **M13** | **The prefill control cannot arbitrate anything at the ~1% level *at 16 threads*, which is where F24/F33 used it.** F39 measured it returning a resolved `+0.59%` with nothing to detect; F33 had already found it drifting. **F40 qualifies this:** at 8 threads the same control is clean to ±0.12pp and would be an excellent control — but F24's effect depends on `nth` (F23), so it cannot simply be re-measured there | **Open.** Do not cite F24/F33's prefill control in either direction. M6 needs the layout arm |
| **M4** | F30 and F31 differ in arm order, round length **and** time simultaneously, so the rotation explanation is a story that fits, not evidence | **Open.** The same disease F30 diagnosed in F25 |
| **M5** | Cross-session comparison of absolute throughput is worthless — the same compiled-out binary read 42.94 and 45.92 tok/s in two sessions (**+6.9%**, ~6× the effects being resolved) | Documented; a standing rule |
| **M6** | Code layout is not held constant in F24's A/B. **Quantified in F38:** the 1,137,994 differing bytes are one contiguous block inside `.text`, ~94% dense — **about 31% of the entire code section** — and **0 of 36 code probes appear at the same address** in both binaries | **Open, and the strongest live threat to F24.** F33's prefill control (a workload the patch cannot reach) is negative in every run. **The control built for it does not work** (F38): the same edit in `mul_mat_id` perturbs 5 bytes, so layout cannot be dialled in by choosing a small edit |
| **M7** | The static pair builds under Ninja and the shared pair under MSBuild, so early cross-pair numbers confound linkage with generator | Fixed by adding a Ninja shared pair (F31); the generator turned out to matter (+0.40% [+0.14, +0.69] on decode) |

**M6 deserves emphasis.** It is the strongest live threat to F24, and F33, F38
then F39 each made it worse rather than better. **F39 removed the instrument the
question was being argued with**: the prefill control returns a resolved +0.59%
between binaries that cannot differ, so neither its flatness nor its drift means
anything at this scale. What survives is that F39's decode null is clean at
±0.5pp while F24 measures +1.62%, so *something* real moves decode — but nothing
short of the layout arm can say how much of it is the scheduler constant. F38 is the one to read: a third of
the code section is relocated or regenerated between the two arms, and the
asymmetry it found — one byte of immediate moves nothing in `mul_mat_id` and a
third of the image in `mul_mat` — is why a cheap layout control does not exist. The original argument against it was that the
prefill control was flat; six blocks later the control is drifting negative,
which is exactly what a layout effect looks like on a workload that cannot see
the scheduler change. Decode still moves ~3x further and in the opposite
direction, which layout alone has no reason to produce — so F24 stands, with an
unknown fraction of it unattributed.

### 4.2 Coverage gaps

| # | Gap | Why it matters |
|---|---|---|
| **G1** | **No Linux, no GCC, ever.** All numbers are MSVC/Windows | The blocker for the main upstream conversation. F26 narrowed it: the barrier path measured here *is* Linux's default (OpenMP), so what is untested is `libgomp` and GCC, not a different algorithm |
| **G2** | **No MoE model at all** | The clearest hole in the byte law. MoE is exactly where "bytes streamed per token" stops being a property of the file and starts depending on the router — the law as stated should be **wrong** there, which makes it the most informative test available. Needs a download; **ask first** |
| **G3** | No NUMA hardware | `nth*4` was tuned for NUMA (PR #6915). F24 argues about a constant whose justifying case cannot be tested here |
| **G4** | `mul_mat_id` untested and unchanged | Identical threshold, MoE path. Tied to G2 |
| **G5** | Overhead measured at **8 threads only** (plus one uncertified attempt at 28) | F10 predicts overhead grows with thread count |
| **G6** | SMT is not separated from core heterogeneity | Needs 16 threads without SMT; this machine has 8 P-cores, so the no-SMT arm must pull in E-cores and the arms then differ in core type too. **The confound is in the hardware** |
| **G7** | Server workloads with real arrival/eviction patterns | F17 covers `llama-batched` to 16 sequences only |
| **G8** | Largest model is 8.19 B | Nothing above; RAM-bound at ~7 GB free |
| **G9** | llama.cpp shared build on Linux/ELF untested | F22's fix is verified on three toolchains but only via the two-module test, not a real llama.cpp shared build outside Windows |

### 4.3 Defects found *in tokenscope* (all fixed)

| | Defect | Found by |
|---|---|---|
| **D1** | Scope misparenting with correct timestamps | F5 |
| **D2** | Shared build did not link at all — went undetected for a session because nothing here ever built a DLL | F18 → fixed F22 |
| **D3** | Attribution exceeded 100% by charging post-`decode` work against the decode slice | F11 |
| **D4** | A dead entry in the category table | F19-adjacent |
| **D5** | `TOKENSCOPE_TOKENS=10:11` parsed as one token, silently, for two sessions — and `imbalance_repeat.py` had defaulted to it | F29 |
| **D6** | **Lazy buffer allocation charged to ggml's barrier**, at 76–83% of all release latency, blamed on "thread-pool spin-up" by a sentence nobody had tested | F28 |
| **D7** | Five documents stated the barrier path was ggml's own when every measurement was OpenMP | F26 |
| **D8** | Fixed arm order in the overhead harness, biasing whichever arms ran first | F30 → fixed `9ec7c82` |
| **D9** | **The baseline gate was computed on data pooled across blocks**, so it absorbed the between-block drift the `t` interval already carries and charged a `--blocks` run twice for the same variance. 2.02% pooled against 1.46% within-block on the same 60 samples | F39 → fixed `c18a580`; gates on the worst single block, prints the pooled figure beside it, `--blocks 1` unchanged |

**D6 and D7 are the instructive pair.** Both are the profiler being wrong *about
itself* in a way that looked like a finding about llama.cpp. D6 in particular
means every `--barriers` figure before session 5 attributed the profiler's own
allocator to ggml.

### 4.4 Defects found *outside* tokenscope

| | Defect | Status |
|---|---|---|
| **U1** | **F20** — `build_attn`'s output projection is unnamed in every llama.cpp graph; it is 7% of decode and shows up as `~attn` | Fixed locally in `patches/02-name-attn-output.patch`, applied to the working tree. **Not filed upstream** |
| **U2** | **F24** — `nth*4` chunking threshold; adding threads can disable ggml's own load balancer for a model's largest matmuls | `patches/03-mulmat-chunk-threshold.patch`, **not applied**. Not filed |

Neither can be filed by an agent — `AGENTS.md` forbids it and requires the
contributor be able to defend the change unaided. That constraint turned out to
be **protective**: F24 was the queued submission and its interval is exactly the
thing session 6 found to be over-precise.

### 4.5 Environment and reproduction traps

Each of these cost real time at least once.

- **An A/B where one arm is a binary built earlier is not an A/B.** `build-ts-off`'s
  binary was three days stale and produced **+2.71% where the truth was +1.95%**
  — with cleanly non-overlapping ranges, which made it *more* convincing.
  Session 6 hit the same trap again: `build-ts-shared` predated F28 and relinked
  when rebuilt.
- **MSVC builds *are* reproducible here** — two stock builds of one tree differ
  by **4 bytes** (PE timestamp `0x110`, one word at `0x3e2ed4`). So byte
  comparison is a valid check. Use it.
- **Rebuild before trusting anything.** `ninja: no work to do` is the check that
  the binary matches the tree.
- **Bash heredocs fail on large documents** in this environment; use the file
  tools. `git commit -m` with multi-line bodies is unreliable — use `-F`.
- **`cmd /c` under Git Bash gets path-mangled** — MSYS rewrites `/c` to `C:/`.
  Use `cmd //c`.
- **Every build needs `vcvars64.bat`.** The VS-bundled CMake is not installed.
- **A stalled `curl` holds its output file open**; `Stop-Process -Name curl` first.
- **Hugging Face downloads truncate silently** with `curl` still exiting 0.
  Verify size against the HF API and check the GGUF magic.
- **Do not run anything on the machine during a measurement.** Session 6
  contaminated one run by executing a smoke test alongside it, and then
  discarded that run.
- **Three trees, one source of truth.** `tokenscope/src/tokenscope.*` is
  authoritative; `llama.cpp/ggml/src/tokenscope/` are copies. `diff` them before
  believing a build.

---

## 5. The self-audit record

Four of session 5's five findings, and both of session 6's, are the project
auditing itself. That is either the most valuable thing here or a warning,
depending on taste — but the pattern is consistent enough to name:

**Every one of these was found by looking at something adjacent, not by looking
for the bug.**

- F26 — noticed `-DGGML_USE_OPENMP` on a compile line while configuring an
  unrelated build.
- F28 — found while extending a tool for F27, by reading what "release latency"
  actually contained.
- F29 — found while checking F28's window.
- F30's ordering flaw — found in the raw data while sanity-checking F30 itself.
- F31 — found because P31.4 was written specifically so a published result
  *could* be falsified.

**The recurring failure is the same one, in four costumes:** being statistically
careful *inside* a sample while silent about the sample being one sample.

| | The careless step | Caught by |
|---|---|---|
| F23 | trace numbers treated as exact because the tracing is exact; per-node imbalance actually spreads 8.5× across identical runs | rerunning at n=12 |
| F25 | two arms compared across two invocations, when the whole design exists to avoid comparing across runs | the comparison spanning zero |
| F30 | a fixed arm order inside a correct interleave | reading the raw reps |
| F31 | a bootstrap interval read as a reproducibility claim | re-measuring the same binaries |

**Three wrong turns are also recorded rather than tidied away**, all from
trusting too few runs: a single trace inverted a conclusion; a six-run median
invented a "28-thread anomaly" that does not exist; another six-run median put a
prediction outside a band that twelve runs put inside it.

### The prediction discipline

Predictions are written and **committed before** the run that tests them (P19,
P23–P25, P27, P28, P30–P32). The scoring is kept even when it is unflattering —
session 6 alone recorded two held / three failed (P31) and expects worse.

F25 named the outcome worse than a wrong prediction: one that is
**unanswerable** with the design that was run, because a wrong prediction
teaches something and an unanswerable one teaches nothing.

---

## 6. What is safe to say outside this repo

**Safe as stated** — Tier A and B, with the one-machine caveat attached:

- The shape of per-token time on CPU: control plane is negligible, the graph is
  everything, barrier wait is a large minority of worker time.
- F27's barrier comparison (−54% at 28 threads) *as a result about this
  machine* — and its 8-thread decode row, **−2.15% [−2.28, −2.02]**, re-measured
  over six blocks in F41. **Do not quote F27's −1.15% prefill row**; it is
  superseded by F41's −0.43% [−0.65, −0.21].
- F28, F26, F29, F5, F8, F18/F22 — all facts about code.
- F20 as a defect report about llama.cpp naming.

**Safe only with a `t` interval attached, and only above F39's floor.** Both
outstanding items are done: **F24** (F33: +1.62% [+1.10, +2.15], with M6
attached) and **the overheads** (F36: every build under 1%, three blocks).
F25's, F30's and F31's own percentages are superseded by F36 and should not be
quoted — cite F36.

**F39 and F40 add a floor test that comes before the interval**, and the floor
**depends on the thread count**. A pair of binaries that *cannot* differ
produced:

| threads | decode | prefill |
|---|---|---|
| 8 | **+0.02% [−0.21, +0.26]** | **−0.01% [−0.12, +0.11]** |
| 16 | −0.04% [−0.49, +0.42] | **+0.59% [+0.01, +1.16]** — resolved |

So:

- **compare a number against the floor at its own thread count.** An earlier
  revision of this section compared F36's 8-thread overheads against the
  16-thread floor, concluded none of them was distinguishable from nothing, and
  was **wrong** — refuted by F40 fifty minutes later. The paragraph even named
  the M5 violation it was committing. *A comparison you have labelled as
  forbidden does not become usable by labelling it.*
- **F36's overheads are measurements.** Against the matched 8-thread floor
  `[−0.21, +0.26]`, ninja-shared `+0.92% [+0.38, +1.46]` does not overlap the
  null at all, and static `+0.56%` and MSBuild-shared `+0.77%` have point
  estimates 2.3–3.8× the null's half-width outside it. They read marginal only
  because F36's own three-block intervals are wide.
- **F24's +1.62% clears its own (16-thread) floor by 3.6×**, its lower bound
  +1.10 sitting 0.68pp above the null's +0.42. That is the strongest statement
  ever made for F24 and it did not exist before F39.
- a **prefill** effect at ~1% **at 16 threads** should not be quoted; the null
  resolved there. At 8 threads prefill is the cleanest thing measured here.
- **prefer 8 threads for any Tier D measurement** that is not specifically about
  thread count. The floor is half as wide on decode, the gate passes, and the
  machine is faster (46.91 vs 42.50 tok/s).

**Not safe because it has been retracted:** F31's generator effect (+0.40%) and
its "level 3 is a leveller" structural claim. Both are gone (F36).

**Not safe because the instrument is discredited:** F24/F33's prefill *control*,
in either direction, and any argument built on it (M13) — at 16 threads, which
is where it was used.

**Not safe to generalise at all:**

- Anything about thread scaling, to a homogeneous or NUMA machine.
- Any barrier *cost* number, to `libgomp`.
- The byte law, to MoE.
- F24's constant, to NUMA — which is the case it was tuned for.

---

## 7. Reproduction reference

```bash
python scripts/bootstrap.py --dest ../llama.cpp     # clone, copy, apply patch 01

cmake -S ../llama.cpp -B ../llama.cpp/build-ts-on  -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF -DGGML_TOKENSCOPE=ON  -DLLAMA_BUILD_TESTS=OFF
cmake -S ../llama.cpp -B ../llama.cpp/build-ts-off -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF -DGGML_TOKENSCOPE=OFF -DLLAMA_BUILD_TESTS=OFF
```

### Build directories and what each is for

| dir | generator | linkage | tokenscope | purpose |
|---|---|---|---|---|
| `build-ts-on` | ninja | static | ON | traces; `llama-bench`, `llama-cli`, `llama-batched` |
| `build-ts-off` | ninja | static | OFF | **all throughput numbers** |
| `build-ts-shared` / `-shared-off` | **VS 2022** | shared | ON / OFF | F22, F25, F30 |
| `build-ts-nshared-on` / `-off` | ninja | shared | ON / OFF | **new in session 6** — holds the generator constant against the VS pair (F31) |
| `build-ts-noomp-on` / `-off` | ninja | static | ON / OFF | F27; verify by grepping the exe for `VCOMP` and finding nothing |

`models/` and `llama.cpp/` are gitignored. The llama.cpp clone is **modified in
place**; `git -C ../llama.cpp status --short` should show exactly 8 modified
files plus `?? ggml/src/tokenscope/`.

### Tools

```bash
python tools/trace_analyze.py T.json [--tokens N|--outliers N|--layers N|--barriers N|--diff B.json]
python tools/trace_svg.py T.json -o docs/token.svg          # README figure, no Perfetto
python tools/mulmat_chunking.py M.gguf -t 8,16,28           # which matmuls flip mode
python tools/imbalance_repeat.py -m M.gguf -n 12 [--metric release]
python tools/spinup_probe.py -m M.gguf --windows 1:2,40:41
python tools/model_bytes.py M.gguf                          # the byte law
python tools/bench_overhead.py --pair NAME=OFF,ON [--pair ...] --blocks 3
python tools/ab_throughput.py --a X.exe --b Y.exe --blocks 3
```

`--barriers` requires `TOKENSCOPE_LEVEL=3`. It matches the k-th barrier across
threads and **refuses to report anything** rather than guess if node/barrier
alternation ever stops holding.

### What CI mechanizes

Three platforms × three configurations, and **five claims that fail the build if
they stop being true**: attribution never exceeds 100% (inside-slice only);
every barrier is matched by `--barriers`; F9 (elementwise nodes serial in
decode, parallel in prefill); F16 (shifts visible, `find_slot` small); and F22
(two modules, one buffer per thread).

**Nine reference traces** in `examples/`, 1164 decode tokens, ~21 MB, all
exercised by CI.

---

## 8. Open questions, ranked

1. **Explain what varies between ~40 and ~46 tok/s on this machine.** Not
   thermal — an 8-minute cooldown produced a *lower* result than a 7-minute one,
   and a 30-run curve after cooling was flat to -1.16%. Not scheduler
   migration — pinning made it worse. It is bimodal rather than smooth, it is
   ~10x the effects being measured, and it decides whether a run passes the
   gate. **This is now the largest unknown in the project.**

   **Correction (F37):** an earlier revision of this document claimed `0x5555`
   is *not* one thread per P-core. That was published before it was tested and
   it is **wrong**. Prefill at two threads gives `0x0101` (logical 0,8) 1.87x
   the throughput of `0x0003` (logical 0,1), so logical 0 and 1 share a core,
   the numbering is interleaved, and `0x5555` is exactly what F14 and the
   handoff always said. F14's first candidate for its anomaly is thereby
   eliminated, leaving `--cpu-strict` bit assignment as the survivor.
2. **Separate layout from chunking in F24 (M6), and it is now the only route
   left.** F39 discredited the prefill control that the question was being
   argued with (M13), so neither its flatness nor its drift is evidence any
   more. Harder than it looked — F38
   showed the obvious control perturbs 5 bytes where the real patch perturbs
   1.1 MB. A real layout arm must change `mul_mat`'s *size* without changing
   what it does (padding, a retained uncalled function, forced alignment) and
   then survive the objection that the padding itself costs something. **Still
   the most valuable experiment left**, because F24 is the one result a
   maintainer might act on.
   **Done (F39).** `patches/04-layout-control.patch` was measured against stock
   over six blocks: decode **−0.04% [−0.49, +0.42]** (clean), prefill **+0.59%
   [+0.01, +1.16]** (**resolved, and a false positive by construction**). The
   decode floor is what F24 should be read against; the prefill result is M12
   and M13. Note the binaries are **not** kept — `bench-layoutctl.exe` lives in
   a session temp directory and an earlier revision of this document wrongly
   said it "is built". Rebuild both arms in one session, always.
3. **Explain M3** — why one invocation's own halves disagree by 0.47pp. Blocks
   contain the symptom; nothing explains it. Candidates not yet tested: Windows
   scheduler migration, page-cache state, SMT partner activity, turbo residency.
   **F39 adds a sibling, M12:** the blocks themselves march rather than scatter,
   monotonically and in both runs, on prefill only. Same suspects, and now two
   symptoms to explain with one mechanism.
3b. ~~**Repeat F39's null at 8 threads.**~~ **Done (F40).** The floor is half as
   wide there and F36's numbers survive it. The live remainder is *why* 16
   threads is so much worse, which is M12's trigger and probably F10's mechanism.
4. ~~**Sweep "certified" out of the prose**~~ **Done (M2).** Gone from the tools,
   README, `02` and `03`; annotated rather than erased in `FINDINGS.md`.
5. **Linux + GCC** (G1) — the blocker for the upstream conversation, and smaller
   than it looked once F26 corrected the barrier-path premise.
6. **An MoE model** (G2) — the most informative single test available, because
   the byte law should *fail* there. Needs a download and a RAM check; **ask.**
7. **Decide whether to raise F20 and F24 upstream.** Needs a person, not an
   agent. F20 is the smaller, unblocked one and should go first.
8. **F27 on a second machine** — the biggest unexploited result in the repo, and
   meaningless as a general claim until someone runs the same protocol on a
   homogeneous part with `libgomp`.
9. **A Perfetto screenshot**, the one thing `trace_svg.py` cannot replace, for a
   post that wants to show the UI.

---

## 9. Document map

| Document | What it is for |
|---|---|
| [`README.md`](../README.md) | The pitch, the figure, the claims table |
| [`00-architecture-map.md`](00-architecture-map.md) | Where the scopes live in llama.cpp |
| [`01-design-scope-timing.md`](01-design-scope-timing.md) | Why the mechanism is shaped the way it is |
| [`02-overhead-methodology.md`](02-overhead-methodology.md) | How overhead is measured, and the rules learned the hard way |
| [`03-upstream-issue-draft.md`](03-upstream-issue-draft.md) | Evidence pack for F20/F24 — **read the box at the top before touching it** |
| [`FINDINGS.md`](FINDINGS.md) | Every finding and prediction, chronological, with scoring |
| [`HANDOFF.md`](HANDOFF.md) | Cold-start checklist and what to do next |
| **`04-project-audit.md`** | **This document — trust ladder, issue register, everything broken** |
