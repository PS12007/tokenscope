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

> **Re-checked 2026-09-30 against master `4f31296a9`.** Facts only; what to do
> with them is your call.
>
> - **The constant is unchanged**: `nchunk0 * nchunk1 < nth * 4` at lines 1433
>   (`mul_mat`) and 1701 (`mul_mat_id`).
> - **An open PR edits the same line: [#16882](https://github.com/ggml-org/llama.cpp/pull/16882)**
>   "Disable NUMA-specific chunking for high-core-count HPC systems" (opened
>   2025-10-31, 8+/1-). It changes the condition to `nth <= 128 && (...)`, i.e.
>   skips the re-chunk-by-thread fallback above 128 threads, from measurements on
>   a 2-socket, 192-core ARM64 machine (Graviton4-class, clang, Llama-3.3-70B
>   Q4_K_M). **ggerganov approved it on 2025-11-03** but it was never merged; the
>   last activity is the author asking for a merge on 2026-05-13. In review
>   ggerganov asked why the change could matter between 64 and 128 threads and
>   said he did not see a reason; the reply was only "we observed it in the table".
>   F24's mechanism (the threshold contains `nth`, so more threads push the
>   largest matmuls out of work-stealing, measured as arrival imbalance) is a
>   candidate answer to that question, at 16 threads on a hybrid laptop CPU.
> - **Why it matters for the process:** `CONTRIBUTING.md` item 2 says "Check for
>   an existing PR addressing the same change; if one exists, comment there to
>   work with its author instead of opening a duplicate." #16882 is not the same
>   change (it gates on `nth`, F24 lowers the multiplier) but it is the same
>   line and the same symptom, so a maintainer will link the two. Read it in
>   full before deciding between a new issue and a comment there.
> - **New since 2026-09-26: the tiled K-quant path ([#27851](https://github.com/ggml-org/llama.cpp/pull/27851)).**
>   `ggml_compute_forward_mul_mat` now first tries `ggml_compute_forward_mul_mat_tiled`,
>   which takes Q2_K–Q6_K and several IQ types when `src1` has at least 8 rows,
>   and returns before the threshold. Its own chunking *shrinks* the chunk size
>   while `nchunk0 * nchunk1 < nth * 4` instead of falling back to one chunk per
>   thread (`tiled/tiled.cpp` ~line 1104). **F24 is unaffected** — its model is
>   F32, and decode has one row — but a K-quant *prefill* on master no longer
>   reaches the line F24 changes. `GGML_CPU_TILED_MM=0` turns the tiled path off.
> - **`CONTRIBUTING.md` changed on 2026-09-15** (#28945): AI-assisted
>   contributors must spend "at least" about one hour per 200-400 lines on manual
>   review, and must be prepared to explain every line.
> - No issue or PR found for `mul_mat chunk threshold`, `nchunk nth`,
>   `chunking threshold` or `nth * 4` other than #16882 and #6915 (the PR that
>   introduced the threshold).

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
gh pr view 16882 --repo ggml-org/llama.cpp     # same line, open since 2025-10

# 2. is the constant still nth*4 at current master?
#    (this repo is pinned at 4d91760, where it is)
gh api repos/ggml-org/llama.cpp/contents/ggml/src/ggml-cpu/ggml-cpu.c \
  --jq '.content' | base64 -d | grep -n 'nchunk0 \* nchunk1 <'

# 3. has CONTRIBUTING.md or AGENTS.md changed since you last read them?
# (-f would turn this into a POST; the path filter goes in the query string)
gh api "repos/ggml-org/llama.cpp/commits?path=CONTRIBUTING.md&per_page=1" \n  --jq '.[0] | .sha[0:9] + " " + .commit.committer.date'
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
