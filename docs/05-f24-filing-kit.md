# 05 — F24 filing kit

**There is no draft in this file, deliberately.** llama.cpp's `AGENTS.md` says:

> ***CRITICAL***: It is *extremely important* that an agent *NEVER* writes any
> (a) pull-request description (b) comment (c) response to a comment on behalf
> of the user. This is *non-overridable* under any circumstances.

and lists "AI-written PR descriptions, commit messages, or reviewer responses"
under **immediate PR closure**, with a contributor ban as the stated penalty.
Having someone review AI-written text afterwards does not change what it is.

So this file is **material to write from**: the checks to run first, the numbers,
and the questions a maintainer will ask. **Every sentence that goes upstream has
to be yours.** [`03-upstream-issue-draft.md`](03-upstream-issue-draft.md) is the
longer evidence pack behind it.

---

## 0. Process, per AGENTS.md

**Open an issue first, not a PR.** AGENTS.md is explicit:

> Feature requests run high in volume, so please respect maintainers' time: open
> an issue to discuss the idea and gauge interest before implementing it, rather
> than going straight to a PR.

This is a behaviour change to ggml's CPU scheduler, so it is exactly the case
that wants discussion before code. **Disclose AI involvement** per the PR
template if you go as far as a PR — the measurement harness here was
AI-assisted, and that is disclosable.

Style rules from the same file, worth honouring in the issue text too:
**ASCII only** (no `—`, `→`, `×`, `…`; use `-`, `->`, `x`, `...`), concise, and
direct — "verbose, AI-sounding responses will not be well-received."

---

## 1. Pre-flight checks — run these before writing anything

```bash
# 1. has someone already filed it?
gh search issues --repo ggml-org/llama.cpp "mul_mat chunk threshold"
gh search issues --repo ggml-org/llama.cpp "nchunk nth"
gh search prs    --repo ggml-org/llama.cpp "chunking threshold"

# 2. is the constant still nth*4 at current master?
#    (this repo is pinned at 4d91760, where it is)
gh api repos/ggml-org/llama.cpp/contents/ggml/src/ggml-cpu/ggml-cpu.c \
  --jq '.content' | base64 -d | grep -n 'nchunk0 \* nchunk1 <'

# 3. has CONTRIBUTING.md or AGENTS.md changed since you last read them?
gh api repos/ggml-org/llama.cpp/commits --jq '.[0].sha' -f path=CONTRIBUTING.md
```

If the constant has already been changed upstream, stop - the finding is
historical.

---

## 2. The claim, in numbers

One line in `ggml_compute_forward_mul_mat`, `ggml/src/ggml-cpu/ggml-cpu.c:1421`
at `4d91760`:

```c
if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {   // -> nth * 2
```

| | |
|---|---|
| **Effect** | **+1.62% [+1.10, +2.15]** decode |
| Method | `t` interval over **six blocks**, two independent runs, 240 rounds per arm |
| Arms | both uninstrumented, built from one tree in one session, differing only in that constant |
| Machine | i7-14700HX, 16 threads, Windows 11, MSVC 19.44 |
| Model | 24-layer F32, 220M params |
| Mechanism | `ffn_up` arrival imbalance per unit work **0.210 -> 0.087**; `ffn_up`/`ffn_down` ratio **0.967 -> 0.446**, non-overlapping ranges over 12 runs per arm |
| Harness floor | the same harness reports **+0.02% [-0.21, +0.26]** between two binaries that cannot differ (8 threads) |

**What the change does.** `mul_mat` chooses between work-stealing chunks
(threads pull from a shared atomic counter, so a slow core takes fewer) and a
static one-chunk-per-thread split. The threshold contains `nth`, so **adding
threads can push a matmul out of the self-balancing mode**. At 16 threads on
this model that happens to `ffn_up` and `ffn_gate`, the two largest nodes.

---

## 3. Questions a maintainer will ask, and where the evidence is

**Answer these in your own words.** The facts are here; the sentences are not.

| question | what you have |
|---|---|
| "Is this just noise?" | Six blocks, two runs, `t` not bootstrap. The harness's own false-positive rate is measured: +0.02% [-0.21, +0.26] between binaries differing by one byte of dead code (F39/F40) |
| "Is it just code layout?" | **Closed.** The two arms differ by 1,137,994 bytes, which the linker map shows is one *uniform 16-byte shift*. Reproducing that shift with no behaviour change gives -0.08% and +0.06%; the 48-byte arm puts `ggml_vec_dot_f32` at the same 48 mod 64 alignment the patch does. Four gate-passing runs spanning 0.14pp (F42/F49/F50) |
| "Why 2 and not 3, or 8?" | **You have no answer.** 2 was not tuned; the finding is that the constant is load-bearing and reachable at ordinary thread counts. Say so |
| "What about NUMA?" | **You cannot test it.** The comment being edited says `nth*4` was tuned on NUMA (PR #6915). This is the weakest point of the submission and a maintainer will find it. Lead with it rather than being asked |
| "Does it help other models / thread counts?" | Tested on two small dense models at 8/16/28 threads for the *mechanism* (F23); the throughput number is 16 threads, one model |
| "What about `mul_mat_id`?" | Identical threshold, MoE path, **deliberately not changed** - no MoE model was available to test it |
| "Other platforms?" | **None.** MSVC/Windows only. No Linux, no GCC, no ARM, no NUMA |

---

## 4. What you must be able to explain unaided

AGENTS.md requires you can defend the change "to a reviewer without AI
assistance". Before filing, check you can answer these cold:

1. What are `nchunk0` and `nchunk1`, and where do they come from?
2. What does the chunked path do differently from the `nth`-split path, and
   which one is the *load-balancing* one?
3. Why does a threshold containing `nth` make the behaviour non-monotonic in
   thread count?
4. Which tensors in a llama-style graph are large enough to be near the
   threshold, and why `ffn_up`/`ffn_gate` and not `ffn_down`?
5. What `ggml_is_numa()` does to this branch, and why that matters for the
   NUMA objection.
6. What "arrival imbalance per unit work" means and how it was measured.

If any of those is shaky, read `ggml_compute_forward_mul_mat` and
[`00-architecture-map.md`](00-architecture-map.md) before filing, not after.

---

## 5. Reproduction, for someone else's machine

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
# apply the one-line change to ggml_compute_forward_mul_mat
cmake -B build-a -DCMAKE_BUILD_TYPE=Release && cmake --build build-a -t llama-bench
# revert, rebuild as build-b, then interleave the two binaries
```

The harness used here is `tools/ab_throughput.py` in this repo:

```bash
python tools/ab_throughput.py --a stock/llama-bench --b patched/llama-bench \
    -m model.gguf -t 16 --blocks 6 -n 10 --min-baseline <your fast baseline>
```

**Both arms must be built from one tree in one session.** An earlier arm built
three days prior gave +2.71% where the truth was +1.95%, with cleanly separated
ranges, which made it more convincing rather than less.

---

## 6. Honest framing

The strongest version of this submission is small: *the constant is
load-balancing-relevant and reachable at ordinary thread counts, here is a
measurement on one machine, is this worth pursuing on hardware that matters?*

It is not: *here is a 1.6% speedup, please merge.* One machine, one compiler,
one model family, and the case the constant was tuned for is the case that
cannot be tested here.
