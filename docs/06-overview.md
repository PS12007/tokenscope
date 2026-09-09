# 06 — The whole project in one place

**What tokenscope is, what it found, what each result is worth, and what to do
with it.** Written for someone arriving cold, or for the author deciding what to
do next.

If you want the auditor's version — every defect, every caveat, the trust
ladder — read [`04-project-audit.md`](04-project-audit.md). This document is the
readable one.

---

## 1. What it is, in one paragraph

**tokenscope is a profiler you compile into llama.cpp to find out where the time
goes when a language model runs on a CPU.** It places explicit timers ("scopes")
around specific pieces of work — building the computation graph, searching the
KV cache, each individual tensor operation, each thread's wait at a barrier —
and writes a trace you can open in a viewer or feed to the analysis tools here.
It is *deterministic*: it records exactly the things it was told to record,
rather than sampling the stack periodically and inferring. When compiled out, it
leaves no symbol, branch or byte behind.

Around the profiler is the part that turned out to matter more: **seven sessions
of measurements about CPU inference, and a methodology for making those
measurements trustworthy on a noisy laptop.**

| | |
|---|---|
| Instrumentation | ~164 added lines across 7 files (`patches/01`) |
| Analysis tools | ~4,900 lines of Python in `tools/` |
| Findings | **50**, with 20 committed prediction sets scored against them |
| Commits | 162 across 7 sessions |
| CI | 3 platforms x 3 configurations, 5 claims that fail the build if they stop being true |
| Reference traces | 9, ~21 MB, 1,164 decode tokens, all exercised by CI |

---

## 2. What it found — the results that matter

### About CPU inference

**The control plane is irrelevant; the graph is everything.** Host-side work —
batch setup, KV-cache bookkeeping, sampling, detokenization — is **under 0.5%**
of a token. 99.6% is inside `graph-compute`. Sampling plus detokenization
together are **0.13%**. If you are optimising CPU inference, nothing outside the
graph is worth your time. *(F1, F11)*

**Barrier wait is a large minority of worker-thread time — 11.2%** — and most of
it is *arrival imbalance*, threads waiting for the slowest peer, not the barrier
mechanism itself. A third of barriers are paid for nodes only one thread worked
on. *(F6, F9)*

**Thread scaling dies at about 4 threads.** Nothing on this machine beats
**2.2x** speedup no matter how many threads you add; past four, additional
threads convert almost entirely into barrier wait. The mechanism is core
heterogeneity — this CPU has 8 fast P-cores and 12 slow E-cores, and the graph
runs at the speed of the slowest participant. Pinning threads to fast cores
*proves* this is the cause and does **not** fix it. *(F10, F14)*

**Which barrier implementation you compile in is worth up to 54% of decode.**
llama.cpp has a CMake flag, `GGML_OPENMP`, that reads like a build convenience.
Turning it off swaps OpenMP's barrier for ggml's own spin-wait, and at 28
threads that costs **-54% decode** on this machine (**-78%** on a small model).
The cause is measured, not guessed: a spin-wait barrier's release is a
cache-coherence broadcast to every waiting core, and on a CPU whose P-cores and
E-cores sit in different cache clusters that is expensive. *(F27, F41)*

**Phase time tracks bytes streamed, not FLOPs.** Attention, FFN and output
projection times track their parameter counts to within a few percent, across
three model architectures and a 13.7x range of output-layer share. Decode on CPU
is memory-bandwidth-bound, and the useful mental model is "how many bytes must I
stream per token". *(F7, F19, F21)*

**One line of ggml is worth +1.62% decode.** See section 3.

### About the profiler itself

Four findings are tokenscope catching its **own** bugs, and they are the ones a
sceptical reader should weigh most:

- **A barrier this project blamed on ggml for four sessions was tokenscope's own
  memory allocator** — 76-83% of all after-arrival wait time. Every barrier
  figure before session 5 was partly measuring the profiler. *(F28)*
- **Every barrier measurement had come from OpenMP's code path**, while five
  documents said it came from ggml's own. *(F26)*
- **A profiler can misparent its own scopes** while every timestamp is correct.
  *(F5)*
- **`TOKENSCOPE_TOKENS=10:11` captured one token, not two, silently, for two
  sessions.** *(F29)*

### About measuring anything on a laptop

This became the project's real contribution.

- **Confidence intervals computed within one run are about half as wide as the
  truth.** Two such intervals, for the same quantity on the same unrebuilt
  binaries two hours apart, **did not overlap**. The fix is to run the whole
  comparison several times and take the interval across runs. *(F31, F33)*
- **The machine's own false-positive rate is now measured** — the first time
  this project checked what its harness reports when there is *nothing to find*.
  Between two binaries that differ by one byte of dead code: **+0.02% [-0.21,
  +0.26]** at 8 threads. At 16 threads the same null **resolves a +0.59% effect
  that cannot exist** on the prefill workload. *(F39, F40)*
- **This machine's throughput wanders between ~39 and ~46 tok/s** and holds a
  value for minutes at a time, switching unprompted. That is roughly **twenty
  times** the size of the effects being measured. Eleven candidate causes have
  been refuted — thermal, pinning, CPU frequency, workload shape, battery, the
  Windows power cap, memory pressure, background load, sustained load, idling,
  and the page cache. It cannot be steered, only detected. *(F37, F44, F46)*
- **Tight agreement is not evidence of precision.** A contaminated run produced
  the *tightest* block agreement in the project's history while reporting a
  physically impossible result; a clean run produced interval widths **7x apart**
  on the same comparison minutes apart. *(F34, F41)*

---

## 3. The one change that could go upstream

`ggml/src/ggml-cpu/ggml-cpu.c`, in `ggml_compute_forward_mul_mat`:

```c
if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {   // -> nth * 2
```

**What it does.** ggml has two ways to split a matrix multiply across threads:
*work-stealing chunks*, where threads pull work from a shared counter so a slow
core naturally takes less, and a *static split* giving every thread an equal
slice. The threshold that chooses between them contains `nth`, the thread count
— so **adding threads can push a matmul out of the self-balancing mode**, which
is backwards. At 16 threads on a 24-layer model that happens to `ffn_up` and
`ffn_gate`, the two largest nodes in the graph.

**What it is worth.** **+1.62% [+1.10, +2.15]** on decode, `t` interval over six
blocks across two independent runs, 240 rounds per arm. The mechanism was
predicted before it was measured: `ffn_up`'s arrival imbalance per unit work
falls **0.210 -> 0.087**.

**The objection, and its answer.** The two binaries differ by 1,137,994 bytes,
so "you measured code layout, not scheduling" is the obvious challenge. The
linker map shows that difference is **one uniform 16-byte shift** — `mul_mat`
shrinks by 16 bytes and everything after it moves down by exactly that.
Reproducing that shift with *no behaviour change* moves decode by **-0.08%** and,
at a 48-byte shift that puts the hottest function at the patch's exact
alignment, **+0.06%**. Four control runs, all passing the harness gate, spanning
0.14pp. **The +1.62% is the scheduler change.** *(F42, F49, F50)*

**Its real weakness**, which no amount of local measurement fixes: the comment
next to that constant says `nth * 4` was tuned on **NUMA hardware**, and there
is no NUMA hardware here. Also one machine, one compiler, one model family.

---

## 4. Everything you need to file it

**There is deliberately no draft anywhere in this repo.** llama.cpp's
`AGENTS.md` marks agent-written PR descriptions, comments and reviewer responses
**non-overridable**, with immediate PR closure and a possible contributor ban.
Every sentence that goes upstream has to be yours.
[`05-f24-filing-kit.md`](05-f24-filing-kit.md) is the working checklist; this is
the summary.

### Before you write anything

1. **Search first.** `gh search issues --repo ggml-org/llama.cpp "mul_mat chunk"`
   and the same for PRs. If it is already filed, you are done.
2. **Check the constant is still `nth * 4` at current master.** This repo is
   pinned at `4d91760`, where it is. If upstream already changed it, the finding
   is historical.
3. **Re-read `CONTRIBUTING.md` and `AGENTS.md`** in case they have moved.

### What goes in it

- **Open an issue, not a PR.** AGENTS.md asks for this explicitly for behaviour
  changes: discuss the idea and gauge interest first.
- **The problem, before the fix.** The threshold contains `nth`, so adding
  threads can disable ggml's own load balancer for a model's largest matmuls.
  That asymmetry is the finding; the constant is just one way to poke it.
- **The measurement**, with its method: +1.62% [+1.10, +2.15], six blocks, two
  runs, both arms uninstrumented and built from one tree in one session.
- **The mechanism**, measured not asserted: `ffn_up` arrival imbalance
  0.210 -> 0.087, non-overlapping ranges over 12 runs per arm.
- **The layout control**, pre-empted rather than waited for — it is the first
  thing a careful reviewer will ask.
- **The limits, stated by you before anyone asks**: one machine, MSVC/Windows,
  16 threads, two small dense models, **no NUMA**, and `mul_mat_id` (the MoE
  path) deliberately unchanged because no MoE model was available to test it.
- **Not a number for the constant.** `2` was not tuned. Do not defend it.
- **AI disclosure** per the PR template if it reaches PR stage.

### Style rules from AGENTS.md

ASCII only — no `-` em-dashes, no `->` arrows as unicode, no `x` multiplication
signs. Concise. Direct. "Verbose, AI-sounding responses will not be
well-received."

### What you must be able to explain without help

A reviewer can ask you anything. Be able to answer, cold: what `nchunk0`/
`nchunk1` are; which of the two paths is the load-balancing one; why a threshold
containing `nth` is non-monotonic in thread count; why `ffn_up` and `ffn_gate`
and not `ffn_down`; what `ggml_is_numa()` does to that branch; and what "arrival
imbalance per unit work" means and how it was measured.

### The framing that works

*"This constant is load-balancing-relevant and reachable at ordinary thread
counts. Here is a measurement on one machine and the control for the obvious
confound. Is this worth pursuing on hardware that matters?"*

Not *"here is a 1.6% speedup, please merge."* The first invites someone with
NUMA hardware to test it. The second invites them to point out that you cannot.

### The second, easier one

**F20** is a smaller and completely unblocked contribution: llama.cpp's
`build_attn` does not name the attention output projection, so it appears as
`~attn` in every graph and every profiler. It is **7% of decode time** and it is
anonymous. That is a defect report, not a behaviour change, and it is a better
first submission than F24. `patches/02` fixes it in 8 lines.

---

## 5. What is still open

| | |
|---|---|
| **No Linux, no GCC** (G1) | The single biggest gap. Every number here is MSVC on one Windows laptop. Blocks the main upstream conversation. **A second machine is now available** (dual-boot Arch/Windows i7-1255U) — [`07-linux-bringup.md`](07-linux-bringup.md) is the step-by-step |
| **No MoE model** (G2) | The most informative test available, because the byte law should *fail* there — MoE is where "bytes per token" stops being a property of the file and starts depending on the router. Needs a multi-GB download |
| **No second machine** | F27's barrier result (-54%) is the biggest unexploited finding here and means nothing as a general claim until someone reproduces it on a homogeneous CPU |
| **M10** | Why the machine has throughput levels. Eleven candidates refuted; the survivor is firmware/EC policy, which this chassis will not report |
| **M3, M12, M14** | Three open questions about why measurements on this machine are unstable in ways blocks do not fix |

---

## 6. Where to go next, in the order I would do it

1. **File F20.** Small, unblocked, real, and it is a defect in upstream rather
   than a proposal. Good first contact with the project.
2. **Linux + GCC.** Turns a one-machine curiosity into a result. Everything here
   is reproducible from `scripts/bootstrap.py` plus the patches; the harness is
   platform-independent Python, and **no model download is needed** —
   `tools/make_tiny_model.py` generates the test model locally.
   **[`07-linux-bringup.md`](07-linux-bringup.md) is a step-by-step for this**,
   with a verification ladder and a ready-to-paste session prompt.
3. **Decide about F24** with section 4 in hand.
4. **An MoE model**, if you are willing to spend the download. It is the one
   experiment here whose *predicted outcome is failure* of an existing claim,
   which makes it the most informative single thing left.
5. **F27 on any second machine**, even a cheap one. A homogeneous CPU would
   isolate core heterogeneity from the barrier implementation, which this
   machine structurally cannot do.
