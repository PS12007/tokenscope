# HANDOFF — state of the project, and what to do next

Updated **during session 7 (2026-09-09)**, after F46. Everything here is either
a fact about the current tree or an explicit next step. Read this first when
picking the project back up.

**Read [`04-project-audit.md`](04-project-audit.md) first if you want the whole
picture in one place** -- what every number is worth, every known defect, every
gap and every trap. This file is the "what to do next"; that one is the "what is
actually true and what is broken".

**Cold-start checklist, in order.** Each takes seconds and each has caught
something real:

```bash
cd "C:/-CS/TLI profiler/tokenscope"
git log --oneline origin/main..main     # MUST be empty
git status --short                      # MUST be empty
git -C ../llama.cpp status --short      # MUST match section 3's list exactly
grep -n 'nchunk0 \* nchunk1 <' ../llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c
                                        # two `nth * 4` lines = stock (F24 not applied)
for f in tokenscope.h tokenscope-ggml.h tokenscope.cpp; do
  diff -q src/$f ../llama.cpp/ggml/src/tokenscope/$f; done   # MUST be silent
grep -o 'GGML_USE_OPENMP' ../llama.cpp/build-ts-on/build.ninja | head -1
                                        # MUST print it -- see F26
```

**Before any measurement, in addition:**

```bash
# free RAM must exceed ~1.5x the model. bench_overhead.py now enforces this
# (preflight_ram), ab_throughput.py does not. F34 was voided at 870 MB free.
systeminfo | grep -i "Available Physical"

# and nothing else may run. A 5.17 GiB javaw process voided a 44-minute run;
# this session also contaminated one of its own by writing docs alongside it.
```

**Probe the machine before starting any run (F44/F46).** Throughput here sits on
**discrete levels — 40.9, 43.5 and 46.0 tok/s were all seen in one afternoon** —
each held for minutes, switching unprompted, ~12% apart, and **not forceable**:
sustained load, idling and the page cache are all refuted. A run on the wrong
level is not comparable to a floor measured on another, which voided three runs
in session 7.

```bash
# 8 seconds, and it protects a 25-minute run
python tools/ab_throughput.py --a A.exe --b B.exe -m ../models/mid.gguf     -t 8 --blocks 3 --min-baseline 46 --json-out run.json
```

Then check `timeline` in the JSON afterwards to confirm the run *stayed* there.

**And know the floor before believing a percentage.** F39 and F40 measured what
this harness reports between two binaries that *cannot* differ, and **it depends
on the thread count**:

| threads | decode | prefill |
|---|---|---|
| **8** | **+0.02% [-0.21, +0.26]** | **-0.01% [-0.12, +0.11]** |
| **16** | -0.04% [-0.49, +0.42] | **+0.59% [+0.01, +1.16]** — resolved |

So **an interval excluding zero is not by itself a result**, **measure at 8
threads** unless the question is about thread count, and **compare a number
against the floor at its own thread count** — F39 did not, and F40 refuted it.

**Then read the compiled-out arm's absolute median before believing anything.**
Decode on this machine wanders between **38.6 and 46.0 tok/s** for reasons F37
tested and could not find (M10), and that swing is ~10x any effect measured
here. F36's A arms read 45.98/45.99/46.05; F35's read 43.50/43.65/43.12. **Two
runs with different A-arm medians are not comparable**, whatever their intervals
say.

Then, before trusting any measurement, rebuild rather than assuming the binaries
match the tree — see the A/B trap in section 3.

**New in session 5: the trace tells you how it was built.** Every trace's
provenance record now carries `threading` and `compute_linkage`, and
`trace_analyze.py` prints them on the summary line. If a barrier number ever
looks strange, read that line before theorising — the field exists because four
sessions of documents described every barrier figure here as coming from a code
path none of them came from (**F26**).

**Session 7 in one paragraph.** It ran group A item 1 and got a two-sided
answer. **F39** measured the harness's false-positive rate for the first time,
against a null control whose two binaries differ by **one byte of dead code** —
`mul_mat_id`'s immediate, in a function that never executes here (zero
`MUL_MAT_ID` events across all nine reference traces against 2,704 `MUL_MAT`
events in the level-3 mid trace). Two independent three-block runs. **Decode is
clean**: `-0.04% [-0.49, +0.42]`, and the two runs returned the same point
estimate to two decimal places — so F24's `+1.62%` clears the floor by 3× and
that is the strongest thing anyone has been able to say for it. **Prefill is
not**: `+0.59% [+0.01, +1.16]`, resolved, on binaries that cannot differ. The
cause is a between-block *trend* — the block estimates rise monotonically in
both runs, the effect is absent round-to-round inside a block, and dropping each
run's first block makes it worse. So `--blocks`, session 6's fix for M1, has its
own failure mode (**M12**): F34 found sustained contamination makes blocks
agree, and a sustained trend makes them march, and both read as precision. The
casualty is F24's *argument* rather than its number — a prefill control that
resolves at +0.59% with nothing to detect cannot certify or discredit anything
at this scale (**M13**), so M6 now has only the layout arm left. Along the way,
**D9**: the baseline gate was computed on data pooled across blocks and so
charged a `--blocks` run twice for the same variance (2.02% pooled against 1.46%
within-block); fixed. And a caveat the session put in writing rather than
leaving implicit — F39 ran at 16 threads, F36's overheads at 8, and **all three
of F36's intervals overlap F39's null**, which the 8-thread null (item 1b) would
settle.

**Session 7's fourth act went at M6 and ran into the machine.** **F42** is the
one that mattered: a linker map — which nobody had made — shows F24's 1,137,994
differing bytes are *one uniform 16-byte shift*, `mul_mat` shrinking by 16 and
all 7,721 functions after it moving down by exactly that. **F38's "relocated,
regenerated, or both" is wrong**, and its byte-window probe could not have found
the shift, because relocated code carries absolute addresses that move too. So
M6 became one question — does moving the hot code 16 bytes change throughput? —
and `patches/05` answers it with a verified +16 shift, checked against the map
*before* measuring, which is the step F38's control skipped. **Then three
measurement attempts were voided by the machine.** **F44** caught M10 in the
act: four probes at 46.09, then eight at 40.92, unprompted, nothing running.
**F46** refuted sustained load, idling, and — from data already collected — **the
page cache**, the audit's most promising candidate since session 6; then
corrected F44's "two regimes" to *at least three levels*, the middle one being
where F35's A arms sat all along. Eleven M10 candidates are refuted and the
survivor is below the OS. Net: M6 is **built and blocked** rather than stuck,
and the tools gained `--min-baseline` and a per-measurement `timeline`.

**Session 7's third act closed A3 and found the limit of `--blocks`.** **F41**
re-measured F27's one unchecked row. Decode survives at **−2.15% [−2.28,
−2.02]**, 9× F40's floor; **prefill did not** — F27's −1.15% becomes −0.43%
[−0.65, −0.21] with the old estimate outside the new interval, so the row nobody
had checked was the row that was wrong. The unpredicted part is **M14**: two
runs of the identical comparison, minutes apart, produced `t` intervals **7×
different in width** (0.14pp and 0.95pp) with point estimates agreeing to
0.02pp. Run A's three blocks agreed to **0.05pp, the tightest in the project's
history** — by luck. Block agreement is itself a random variable and the
interval is computed from it, so `--blocks 3` from one invocation can be tight
for no reason. F34 said agreement reads as precision when something is wrong;
F41 says it does when nothing is. **The rule is now two runs of three blocks,
pooled to six** — which F33, F39, F40 and F41 all did and **F36 did not**. Also
worth carrying forward: the arm check for this run used `strings`, which **does
not exist in this environment**, and the shell guard turned that into "0 VCOMP
mentions" for *both* arms — a check that silently passes for the wrong reason.
The PE import table is the reliable route.

**And session 7's second half is the correction to its first.** **F40** re-ran
the identical null at **8 threads** and the harness is a different instrument
there: decode `+0.02% [-0.21, +0.26]`, prefill `-0.01% [-0.12, +0.11]`, both
gates passing at worst-block IQR ~1%, and the A arm *faster* at 46.91 tok/s
against 42.50 — fewer threads, more throughput, less variance, which is what
F10 has said since session 2. Three consequences, and two of them undo F39.
**The floor is a property of the thread count**, so compare a number against the
floor at its own. **F36's overheads are measurements after all**: against the
matched 8-thread null, ninja-shared `+0.92% [+0.38, +1.46]` does not overlap it
at all and the other two sit 2.3–3.8× the null's half-width outside it — F39 had
compared them against the *16-thread* floor, in a paragraph that explicitly
named the M5 violation it was committing, and that claim is withdrawn. **M12 is
narrowed** from "blocks are not independent draws" to a configuration-specific
effect: at 8 threads prefill's blocks show no trend at all. P40.4 was written to
be the prediction most likely to embarrass F39 and it was, which is the method
working rather than failing. The standing lesson: *a comparison you have
labelled as forbidden does not become usable by labelling it*, and the run that
settles it was fifty minutes away the whole time.

**Session 6 in one paragraph.** It set out to close section 5 item 7 and ended
up auditing the project's statistics. Item 7 *is* closed — `bench_overhead.py`
takes N build pairs (`--pair`) and round-robins every arm of every pair in one
invocation, which is what F25 said the question needed. But the answer kept
moving. **F30** measured the shared build at +0.87% [+0.55, +1.19]; **F31**
re-measured the same quantity on the *same unrebuilt binaries* two hours later
and got +0.31% [+0.01, +0.52] — **non-overlapping, both passing the gate**. The
bootstrap resamples inside one invocation and knows nothing about the next one,
and this project had never tested it across invocations. Diagnosing that
eliminated three tidy explanations (arm position 0.214pp, autocorrelation +0.14,
the unpaired estimator) and found the instability *inside* a single run: split
one run's reps in half and the halves disagree by up to 0.47pp. **The fix is
`--blocks N`**, a `t` interval over per-block point estimates with a real
small-sample table (`t(2) = 4.303`, unflattering on purpose). Then everything
Tier D was re-measured: **F33** repaired F24 to **+1.62% [+1.10, +2.15]** and
**F36** settled the overheads at **under 1% for every build**, with an interval
that contains every one of the five superseded estimates — so those runs never
disagreed, only their intervals did.

**Session 6's other half is what went wrong.** **F34** is a void run: 870 MB
free against an 840 MB model made every *instrumented* arm faster than its own
baseline, and the shared pair's three blocks agreed to 0.55pp — **the tightest
spread in the run** — with an interval excluding zero. Blocks defend against
drift *between* passes and do nothing about contamination spanning *every* pass;
sustained contamination makes them agree, and agreement reads as precision. So
the order of trust is **gate first, physical plausibility second, interval
third**, and two guards now enforce it (`preflight_ram()` refuses to start below
1.5x the model; a failed gate makes the tool refuse its own block table). **F36
then retracted two F31 claims**, including one this project had marked "durable,
not Tier D" and therefore exempt from re-measurement — a comparison of spreads
that was six numbers within half a percent of each other. **F37** tested four
causes for the machine's 38.6–46.0 tok/s swing and refuted all of them,
including a `0x5555` topology claim session 6 had *published* before testing.
Three hypotheses formed and killed in one sitting; the tests cost ten minutes
each and the publishing did not.

**Read [`04-project-audit.md`](04-project-audit.md) before quoting any number.**
It is new in session 6 and carries the trust ladder, the issue register (M1–M11)
and what is safe to say outside the repo.

**Session 5 in one paragraph.** Five findings, **F25 through F29**, and the
theme is that four of the five are the project auditing itself. It began by
closing section 5 item 7 — the `GGML_TOKENSCOPE=OFF` shared build now exists, so
the shared/static overhead comparison could finally run — and while configuring
it, noticed `-DGGML_USE_OPENMP` on the compile line for `ggml-cpu.c`. **Every
barrier number this project has ever taken came from `#pragma omp barrier`**,
and five documents said the opposite (**F26**). That turned into
`-DGGML_OPENMP=OFF` builds and **F27**, the largest result of the session:
ggml's own spin-wait barrier costs **-54% of decode at 28 threads** on this
machine, and -2.06% at 8 (**re-measured in session 7 at -2.15%**, F41), with the
cause measured in the traces rather than inferred. Then, while extending a tool for F27, found that one barrier out of
824 held 76-83% of all after-arrival time in every level-3 trace — and that it
was **tokenscope's own lazy buffer allocation**, which `--barriers` had been
blaming on ggml's thread pool since session 1 (**F28**). Fixed; the
imbalance/release split moves from 55/45 to **84.6/15.4**. **F29** is a parser
bug: `TOKENSCOPE_TOKENS=10:11` captured one token rather than two, silently, and
`imbalance_repeat.py` had defaulted to it since session 4. **F25** is the
overhead run that started everything: the static number certifies for the first
time since session 1 at **+1.16% [+0.67, +1.87]**, and the shared build's answer
was eaten by its own noise floor for a reason worth reading.

Also: the README finally has a picture. `tools/trace_svg.py` renders one token
per-thread as a theme-aware SVG from a committed trace, which closes most of
what the Perfetto item wanted, without a browser.

**Session 4 in one paragraph.** Closed section 5 item 6 — the shared-library
build works (**F22**) — then **re-scoped item 5 out from under itself (F23)**
and **acted on the re-scoped version (F24)**, which produced the first
throughput improvement this project has ever certified. Findings went F21 ->
**F24**. 18 commits, all pushed, CI green on all three platforms. The F20
upstream issue is *not* filed: llama.cpp's `AGENTS.md` forbids an agent writing
issue or PR text, so `docs/03` now holds an evidence pack to write from instead
of a draft to paste.

On F23: item 5 assumed ggml gives every thread an equal share of rows, and for
matmul that is only half true. Above `nchunk0 * nchunk1 >= nth * 4` threads
steal chunks from an atomic counter, and a matmul in that mode carries roughly a
third to a half the arrival imbalance per unit work. So ggml already solves core
heterogeneity for big matmuls, by a method that needs no model of core speed —
but the threshold contains `nth`, so **adding threads can turn it off**. Twelve
runs at each of six points, two models, with a cross-model control at a fixed
thread count; two of the comparisons have non-overlapping ranges.

Then **F24 acted on it**: one line, `nth * 4` -> `nth * 2` in `mul_mat`,
re-measured in session 6 at **+1.62% [+1.10, +2.15] on decode** (F33; F24's
**+1.95% [+1.59, +2.35]** was a single run's bootstrap and too narrow), by
`bench_overhead.py`'s own bootstrap over 20 interleaved rounds. That harness
has declined to certify six times across four sessions; this is the first
thing it has ever passed. Prefill is the control and stays uncertified, which
is what makes the decode number believable. `patches/03-mulmat-chunk-threshold.patch`,
**not applied to the tree** and deliberately kept out of `01-instrument.patch`.

**Read F23's reproducibility section even if you skip the rest.** Getting there
took three wrong turns, each caused by trusting too few runs: a single trace
inverted the conclusion, a six-run median invented a "28-thread anomaly" that
does not exist, and another six-run median put a prediction outside its band
that twelve runs put inside it. Per-node imbalance spreads up to **8.5×** across
identical runs. `bench_overhead.py` has interleaved arms and bootstrapped CIs
since session 1 — that discipline was never applied to numbers read *out of
traces*, which got treated as exact because the tracing is exact. The tracing is
exact. The machine is not.

On F22 specifically: F18 had called it a genuine incompatibility between the hot path's
raw thread-local and the registry's need for one instance across DLLs — both
true, but `ts_tls` is a cache and the buffer is the state, so the cache never
needed to be single-instance. Each module now keeps its own and they all resolve
to one registry-owned buffer; the hot path is unchanged. A two-module test now
guards it on all three CI platforms. The shared build's overhead is still
unmeasured.

**Session 3 in one paragraph.** Ran the 8B that section 5 item 4 was waiting on
— it was already on the machine, pulled by Ollama, no download needed. Wrote six
predictions down and committed them *before* measuring; five held and one failed
on both its specifics. Findings went F18 -> **F21**. The best result is a
controlled experiment the earlier models could not support: llama.cpp's Q4_K_M
recipe stores `ffn_down` at two different precisions in different layers, so the
same node in adjacent layers differs only in dtype, and the byte law predicts
the ratio to 2.7% with two control tensors flat at 1.00. Also found and fixed a
defect in **upstream llama.cpp** (F20) and a dead entry in **tokenscope's own**
category table (found because the first would have exposed the second). All 24
commits from sessions 2 and 3 are pushed.

---

## 1. Where things stand

**Working, measured, committed:**

- Core scope-timing mechanism (24-byte record, thread-local chunked arena,
  RAII scopes, Chrome Trace Event output). Self-test passes under 8 threads.
- Tier 1: 11 host scopes across `llama_context::decode` and `process_ubatch`.
- Tier 2: per-node work and barrier wait, per thread, in
  `ggml_graph_compute_thread`.
- Per-phase and per-layer attribution across all 24 layers.
- Barrier decomposition: arrival imbalance vs release latency, per node, with
  the matching refused rather than guessed if node/barrier alternation breaks
  (FINDINGS F9).
- Sampling, tokenizer and cell-search scopes (sites 13, 16-19), so the whole
  per-token loop is covered and not just the parts `llama-bench` reaches.
- Thread-count sweep, 1 to 28, throughput from the uninstrumented build
  (FINDINGS F10).
- Python analysis: summary, per-token, outliers-with-cause, per-layer, diff.
- Overhead: **settled in session 6 (F36)** at 8 threads, level 3, three blocks
  each — static **+0.56% [-0.05, +1.16]**, ninja-shared **+0.92% [+0.38,
  +1.46]**, MSBuild-shared **+0.77% [+0.05, +1.48]**. Every build under 1% and
  the three cannot be told apart; linkage and generator contrasts both span
  zero. Supersedes F25's +1.16% and F30's +0.87%, whose single-run bootstraps
  were too narrow — F36's intervals contain them. It was +0.67% [+0.12,
  +1.67] in session 1; session 2 tried at 8 and 28 threads and the harness
  **refused to certify either**, because the machine's noise floor had moved
  (baseline IQR 2-4%, wider than the effect). Quote a number with its thread
  count and its build, and expect to have to re-measure on a quiet machine.
  **The two cannot be compared across sessions** — F30 found the compiled-out
  static arm reading 42.94 tok/s in session 5 and 45.92 in session 6, the same
  code on the same machine, which is ~6x the effect being resolved. Within F30's
  single invocation the difference of overheads is **+0.36pp [-0.32, +1.17]**:
  bounded, not resolved.
- Zero-overhead-when-off verified against the symbol table.
- **Shared-library builds work (F22).** Each module keeps its own `ts_tls`
  cache over one registry-owned buffer. `ts_dlltest` builds two binaries plus a
  third opened at runtime and asserts they share one buffer per thread.
- CI for **three platforms x three configurations** (instrumentation on, off,
  and shared), including the over-attribution regression check and the F22
  two-module test.
- **`data/overhead/`** (new, session 6) holds the raw measurement series behind
  F30–F36, including the void run (F34) and the refused one (F35) as failure
  specimens. Its README re-derives F36's published interval in six lines. These
  were in a temp directory until the end of session 6, which would have taken
  them with it.
- **`patches/04-layout-control.patch`** (new, session 6): F24's identical edit
  applied to `mul_mat_id`, a path dense models never enter. **Not a layout
  control** — it perturbs 5 bytes — but a genuine **null** control, and the
  harness's false-positive rate has never been measured.

**The upstream patch is 174 changed lines across 7 files.** F9 and F10 needed
no new instrumentation at all -- only analysis of traces the existing scopes
already produced. The growth from 104/4 is the sampling and tokenizer scopes,
which `llama-bench` never reaches.

**Not done** — the honest list is in [`FINDINGS.md`](FINDINGS.md) under
"Not yet measured" and [`02`](02-overhead-methodology.md) under "Remaining gaps".
The short version:

| Gap | Why it matters |
|---|---|
| Linux / GCC never built or measured | Everything so far is MSVC on Windows. **The barrier-path half of this gap was wrong for four sessions (F26):** F9/F10 measured the *OpenMP* path, which is also Linux's default, so what is untested is GCC and the `libgomp` runtime, not a different barrier algorithm. **F27 then measured the other barrier on this machine** and found it 3-5x more expensive, so every barrier figure here is from the cheaper implementation, not the pessimistic one |
| ~~No real quantized model~~ | **Done (F12).** Qwen2.5-0.5B Q4_K_M is in `models/`, gitignored. Largest real model measured is 630 M params |
| ~~Shared-library build is BROKEN~~ | **Fixed (F22).** F18's option 2, implemented and verified: each module caches its own `ts_tls`, all resolving to one registry-owned buffer. `BUILD_SHARED_LIBS=ON` links and traces correctly. Two-module regression test passes on all three toolchains; **llama.cpp shared on Linux still untested**, and the shared build's overhead has never been measured |
| ~~Sampling / tokenizer scopes not written~~ | **Done (F11).** `llama-cli` is now built in `build-ts-on`. Sampling + detokenization are 0.13% of a token |
| ~~Context-shift behaviour~~ | **Done (F16).** 4 spikes in 699 tokens at `-c 256`, 1.13-1.34x median |
| Concurrent sequences / server workload | The last untested prediction in F2, and F16 says it is still plausible: the head-pointer trick that makes `find_slot` O(1) is much weaker with many streams |
| ~~Larger real model~~ | **Done (F19, F21).** Three more real models measured, to 8.19 B. Biggest is now 8.19 B; **no MoE model at all**, which is the clearest remaining gap. Session 4 asked and was told **not to download one** — it needs several GB and free RAM is ~7 GB against the 8B's 4.86 GiB. Ask again rather than assuming |
| ~~No Perfetto screenshot~~ | **Sidestepped in session 5.** `tools/trace_svg.py` renders one token from a committed trace as a theme-aware SVG, and the README opens with it. That is better than a screenshot for a repo -- it is text, it diffs, and anyone who clones can regenerate it -- but it is **not** the Perfetto UI, and a post that wants to show the UI still wants a screenshot |
| ~~No thread pinning~~ | **Done (F14).** Mechanism confirmed: homogeneous cores drop spread 13%->2% and halve barrier wait. Pinning is not the fix |
| Upstream issue not filed | Two issues now, and [`03`](03-upstream-issue-draft.md) says which goes first. **The F20 naming defect is not blocked on Linux** and should be filed on its own; the instrumentation proposal still is. **An agent must not write or file it** — see the box at the top of `03` |
| ~~Shared build's overhead still unmeasured~~ | **Done (F36).** +0.77% [+0.05, +1.48] at level 3 over three blocks, and **indistinguishable from static** (+0.56% [-0.05, +1.16]); every pairwise difference spans zero. Supersedes F25's "unmeasured" and F30's +0.87%, both single-run bootstraps |
| F24 not raised upstream, and `mul_mat_id` untested | Now **+1.62% [+1.10, +2.15]** (F33), one machine, one thread count, one model, no NUMA — and NUMA is what the constant was tuned for. **F33 also found the prefill control drifting negative**, so some unknown fraction may be code layout rather than chunking (audit M6) |

---

## 2. Push first. Check before anything else.

```bash
git log --oneline origin/main..main    # MUST be empty
git push origin main
```

**Session 3 ended with everything pushed**, including the 15 commits session 2
left stranded. That is the first time this has been true, so do not assume it
stays that way — run the check anyway.

Session 3's outage was total, not GitHub-specific: `ping 1.1.1.1` lost 100% of
packets and DNS to 8.8.8.8 timed out, for roughly an hour. Diagnosing that took
one command and was worth it, because "GitHub is blocked" and "this machine has
no network" call for different responses. Then it came back with no warning and
the first retry succeeded.

**What worked: a retry loop in the background** (`git push` every 45s, up to 40
times) started early and left alone while the real work continued. It landed on
its first attempt after the network returned, with no further attention. Do that
at the *start* of a session rather than pushing by hand between commits.

Networking here is **intermittent, not blocked**. Session 1 concluded GitHub was
specifically unreachable and stopped retrying, which is why 14 commits sat local
for a whole session. Session 2 saw a `git push` succeed while a `curl
https://github.com` seconds later still failed, a `git fetch` die mid-protocol
with `expected flush after ref listing`, a Hugging Face download of 469 MB
complete without a hiccup, and then hours where nothing connected at all.
Session 3 lost all networking for about an hour and got it back without doing
anything.

**The rule: a failed connection says nothing about the next one. Retry, and
retry again later.**

## 3. Environment — what had to be set up, and the traps

### Toolchain

There was no `cmake`, no `ninja`, and nothing on `PATH` at the start. What
worked:

- **MSVC 19.44** exists at
  `C:\Program Files\Microsoft Visual Studio\2022\Community` but the VS-bundled
  CMake is **not** installed (the "C++ CMake tools" component is missing).
- `pip install cmake ninja` supplies both (cmake 4.4.3, ninja 1.13.2). They land
  in `C:\Users\priya\AppData\Local\Programs\Python\Python312\Scripts`, already
  on `PATH`.
- Every build must run under `vcvars64.bat`. There is a helper in the scratchpad:

```bat
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
%*
```

  Recreate it and invoke as `cmd /c "vc.bat cmake --build ..."`.

### Paths

```
C:\-CS\TLI profiler\
├── tokenscope\            the repo (git; origin github.com/PS12007/tokenscope)
│   ├── build-on\          ninja, TOKENSCOPE_ENABLED=ON   -> ts_selftest.exe
│   ├── build-off\         ninja, TOKENSCOPE_ENABLED=OFF  -> the 0-symbol proof
│   └── build-shared\      ninja, ENABLED=ON + BUILD_SHARED_LIBS=ON
│                          -> ts_selftest + ts_dlltest (F22's two-module test)
├── llama.cpp\             upstream clone, pinned at 4d91760, PATCHED IN PLACE
│   ├── build-ts-on\       ninja, static, GGML_TOKENSCOPE=ON
│   │                      bin\llama-bench.exe, llama-cli.exe, llama-batched.exe
│   ├── build-ts-off\      ninja, static, GGML_TOKENSCOPE=OFF
│   │                      bin\llama-bench.exe   <- ALL throughput numbers
│   ├── build-ts-shared\   VS 17 2022, BUILD_SHARED_LIBS=ON, TOKENSCOPE=ON
│   │                      bin\Release\llama-bench.exe  (needs --config Release)
│   ├── build-ts-shared-off\  NEW in session 5. Same as build-ts-shared with
│   │                      GGML_TOKENSCOPE=OFF. Item 7 could not run without it
│   ├── build-ts-nshared-on\   NEW in session 6, F31. ninja, BUILD_SHARED_LIBS=ON
│   │                      + GGML_TOKENSCOPE=ON. Exists to hold the generator
│   │                      constant against build-ts-shared, which is MSBuild
│   ├── build-ts-nshared-off\  NEW in session 6, F31. Same, TOKENSCOPE=OFF
│   ├── build-ts-noomp-on\    NEW in session 5, F27. ninja, static,
│   │                      GGML_OPENMP=OFF + GGML_TOKENSCOPE=ON
│   └── build-ts-noomp-off\   NEW in session 5, F27. GGML_OPENMP=OFF + OFF, the
│                          F27 throughput arm. Verify a noomp build by grepping
│                          the exe for VCOMP -- it must find nothing
└── models\
    ├── tiny.gguf          8L,   34 MB  synthetic F32
    ├── mid.gguf           24L, 840 MB  synthetic F32   <- the workhorse
    └── qwen-q4km.gguf     Qwen2.5-0.5B-Instruct Q4_K_M, 469 MB, REAL
```

### Exactly what is non-stock in `llama.cpp`, as of the end of session 4

`models/` and `llama.cpp/` are gitignored. The clone is **modified in place** and
`git -C ../llama.cpp status --short` should show precisely this:

```
 M ggml/CMakeLists.txt          | patch 01, the instrumentation
 M ggml/src/CMakeLists.txt      |   (regenerate with bootstrap.py --make-patch)
 M ggml/src/ggml-cpu/ggml-cpu.c |
 M src/llama-context.cpp        |
 M src/llama-kv-cache.cpp       |
 M src/llama-sampler.cpp        |
 M src/llama-vocab.cpp          |
 M src/llama-graph.cpp            patch 02, the F20 naming fix -- NOT in patch 01
?? ggml/src/tokenscope/          copies of src/tokenscope.*, build inputs
```

Anything else in that list is something a previous session left behind and did
not write down.

**`patches/03-mulmat-chunk-threshold.patch` (F24) is NOT applied.** The tree is
at stock `nth * 4` at both `ggml-cpu.c:1422` and `:1698`. Check with:

```bash
grep -n 'nchunk0 \* nchunk1 <' ../llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c
```

Two `nth * 4` lines means stock. To reproduce F24, `git -C ../llama.cpp apply`
the patch, rebuild **both** `build-ts-on` and `build-ts-off`, measure, then
reverse-apply it. It is deliberately kept out of `01-instrument.patch` even
though `ggml-cpu.c` is in `bootstrap.py`'s `TOUCHED` list, for the same reason
`llama-graph.cpp` is: it is a behaviour change and has nothing to do with the
instrumentation. **`--make-patch` while it is applied would silently fold a
scheduler change into the instrumentation patch.**

Three trees, one source of truth: `tokenscope/src/tokenscope.*` is authoritative
and `llama.cpp/ggml/src/tokenscope/` are copies. Session 4 ended with them in
sync; verify before believing a build:

```bash
for f in tokenscope.h tokenscope-ggml.h tokenscope.cpp; do
  diff -q src/$f ../llama.cpp/ggml/src/tokenscope/$f
done
```

That drifted once during session 4 — a comment added to `src/tokenscope.h` after
the copy — which is harmless only because it was a comment.

### The models Ollama already has

**A real 8B is already on disk**, pulled by Ollama before this project started,
so nothing about it needs a download. Ollama stores GGUF blobs unmodified and
content-addressed; `llama-bench -m` opens one directly:

```
~/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f
```

That is **Qwen3 8B Q4_K_M**, 4.86 GiB, 8.19 B params, 36 layers, `n_embd` 4096,
`n_ff` 12288, GQA 32/8 — 13x the parameters of the Qwen2.5-0.5B used in F12.
Read the metadata with `gguf-py` rather than trusting the tag; the manifest at
`~/.ollama/models/manifests/registry.ollama.ai/library/qwen3/8b` maps tags to
blobs. `dolphin-llama3` (8B) and `dolphin-mistral` (7B) are there too, and **both
were measured in F21** — blobs `sha256-ea025c10...` and `sha256-11a57a9b...`
respectively. **No MoE model is present**, and session 4 asked and was told not
to download one.

**The constraint is RAM, not disk.** This machine has 15.7 GB total and about
7 GB free, against a 4.86 GiB model. It fits and it does not thrash — measured
38.78 pp32 / 7.39 tg16 tok/s at 8 threads — but the margin is thin enough that
anything else running can page the weights out and quietly corrupt a decode
number, because decode is bandwidth-bound (F14). Check free memory before
trusting a run at this size, and again after.

**One consequence of the F20 naming patch, for reading old traces.** Because
`llama-graph.cpp` carries it, traces from `build-ts-on` are not comparable with
pre-session-3 traces in one respect: `attn.out` appears and `~attn` nearly
vanishes. That is the fix working, not a regression. `git -C ../llama.cpp
checkout src/llama-graph.cpp` reverts it if a comparison ever needs the old
behaviour.

**And a trap that cost real time in session 4, generalised.** An A/B where one
arm is a binary built earlier is not an A/B. `build-ts-off`'s binary was three
days older than the tree, so it silently lacked the F20 patch, and a throughput
comparison came out **+2.71% when the true figure was +1.95%** — with clean
non-overlapping ranges, which made it *more* convincing rather than less. Before
trusting any A/B, check the binary's timestamp against `git -C ../llama.cpp
status`, and build both arms in the same session. (F24.)

**`llama-cli` is required** for anything involving sampling, tokenization or
context shift (F11, F16) -- `llama-bench` calls none of them. It is built in
`build-ts-on`; add it to the `--target` list when rebuilding.

**The machine matters for F10/F14.** i7-14700HX: 8 P-cores + 12 E-cores, 28
logical. P-cores are 2.88x faster than E-cores on compute-bound work (and
indistinguishable on decode, which is bandwidth-bound). Affinity via
`llama-bench -C <hex> --cpu-strict 1` works; masks used were `0x5555` (one
thread per P-core, but see F14's unexplained anomaly) and `0x0fff0000` (the 12
E-cores, which behaves cleanly).

**Env vars found in session 2:** `GGML_CPU_DISABLE_FUSION=1` turns off ggml's
only op fusion (RMS_NORM+MUL) at runtime -- that is what made F15 measurable
without patching anything.

`models/` and `llama.cpp/` are gitignored deliberately. **Both builds are
needed** — the overhead harness compares them, and rebuilding `build-ts-off`
from scratch takes several minutes.

### Traps hit, so they are not hit again

- **Bash heredocs fail on large documents** in this environment. A ~300-line
  markdown heredoc died with `unexpected EOF while looking for matching '`.
  Use the Write tool for long files; heredocs are fine for short commit
  messages.
- **`git commit -m` with multi-line bodies** is unreliable here too. Write the
  message to a file and use `git commit -F`.
- **A stalled `curl` holds its output file open**, so `rm` fails with `Device or
  resource busy`. `Stop-Process -Name curl -Force` first.
- **Hugging Face downloads kept truncating** in session 1 -- a Qwen2.5-0.5B
  GGUF stopped at 86 KB, then 0 bytes, with `curl` still exiting 0. **In
  session 2 the same download succeeded**, complete and byte-exact, using
  `curl -L --fail --retry 5 --retry-delay 3 --retry-all-errors -C -`. Verify
  the size against the HF API tree listing and check the GGUF magic before
  trusting a downloaded model. Pass `-s`; the progress meter floods the
  transcript.
- **CMake 4.x** rejects `cmake_minimum_required(VERSION < 3.5)`. llama.cpp is
  fine; other projects may not be.

---

## 4. Rebuilding from scratch

```bash
python scripts/bootstrap.py --dest ../llama.cpp    # clone, copy sources, apply patch

cmake -S ../llama.cpp -B ../llama.cpp/build-ts-on  -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF -DGGML_TOKENSCOPE=ON -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build-ts-on --target llama-bench

# same with -DGGML_TOKENSCOPE=OFF into build-ts-off

# The shared-library arm (F22). Visual Studio generator, so --build needs
# --config Release; the default is Debug and it will silently give you one.
cmake -S ../llama.cpp -B ../llama.cpp/build-ts-shared -DBUILD_SHARED_LIBS=ON \
      -DGGML_TOKENSCOPE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF
cmake --build ../llama.cpp/build-ts-shared --config Release --target llama-bench

python tools/make_tiny_model.py --llama-cpp ../llama.cpp -o ../models/mid.gguf \
       --layers 24 --embd 768 --heads 12 --heads-kv 4 --vocab 8192
```

And the repo's own three, which need no llama.cpp, no model and no network. The
third is new in session 4 and is what CI runs to guard F22:

```bash
cmake -S . -B build-on     -DCMAKE_BUILD_TYPE=Release -DTOKENSCOPE_ENABLED=ON
cmake -S . -B build-off    -DCMAKE_BUILD_TYPE=Release -DTOKENSCOPE_ENABLED=OFF
cmake -S . -B build-shared -DCMAKE_BUILD_TYPE=Release -DTOKENSCOPE_ENABLED=ON \
      -DBUILD_SHARED_LIBS=ON
cmake --build build-on && cmake --build build-off && cmake --build build-shared
ctest --test-dir build-on          # selftest
ctest --test-dir build-shared      # selftest + dlltest (two modules, one buffer)
```

`build-shared` adds three targets that only exist in that configuration:
`tokenscope_shared` (the DLL with the registry), `ts_dllmod` (a second module,
linked) and `ts_dynmod` (a third, opened at runtime). If `dlltest` ever fails on
its *first* assertion — "each module has its own ts_tls cache" — the modules have
collapsed into one binary and every other assertion in it has gone vacuous;
fix that before reading the rest.

### The edit loop that is easy to get wrong

`src/tokenscope.*` in this repo is the **source of truth**. The copies in
`llama.cpp/ggml/src/tokenscope/` are build inputs. After editing:

```bash
cp src/tokenscope.h src/tokenscope-ggml.h src/tokenscope.cpp \
   ../llama.cpp/ggml/src/tokenscope/
# rebuild, then, if upstream FILES changed (not tokenscope's own):
python scripts/bootstrap.py --make-patch --dest ../llama.cpp
```

Forgetting the `cp` means rebuilding the old code and debugging a fix that is
already correct. It cost time in session 1.

**And if you instrument a file that is not already in `bootstrap.py`'s
`TOUCHED` list, `--make-patch` silently regenerates the *old* patch and reports
success.** That happened twice in session 2. The list is now:
`ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`,
`ggml/src/ggml-cpu/ggml-cpu.c`, `src/llama-context.cpp`,
`src/llama-sampler.cpp`, `src/llama-vocab.cpp`, `src/llama-kv-cache.cpp`.
Add to it the moment you touch a new file.

---

## 4b. What the tools do now

```bash
python tools/trace_analyze.py T.json                 # summary (host / node / outside-slice)
python tools/trace_analyze.py T.json --tokens 40     # per-token table
python tools/trace_analyze.py T.json --outliers 10   # slowest tokens, attributed
python tools/trace_analyze.py T.json --layers 6      # per-layer thread time
python tools/trace_analyze.py T.json --barriers 12   # NEW: imbalance vs release, per node
python tools/trace_analyze.py A.json --diff B.json   # did my change help, and where
python tools/mulmat_chunking.py M.gguf -t 8,16,28    # NEW: matmul partitioning mode
python tools/imbalance_repeat.py -m M.gguf -t 8,16 -n 12 --ratio ffn_up/ffn_out
python tools/mulmat_chunking.py M.gguf -t 16 --mult 2   # model a PATCHED build
python tools/ab_throughput.py --a old.exe --b new.exe -m M.gguf -t 16 -n 20
python tools/imbalance_repeat.py -m M.gguf --metric release   # NEW: the other
                                        # half of --barriers, repeated (P27.2)
python tools/spinup_probe.py -m M.gguf --windows 1:2,40:41    # NEW: is the big
                                        # after-arrival barrier ours? (P28)
python tools/trace_svg.py T.json -o docs/token.svg            # the README
                                        # picture, from a trace, no Perfetto
python tools/bench_overhead.py -m M.gguf -n 20 -t 8 --levels 3     --pair static=../llama.cpp/build-ts-off/bin,../llama.cpp/build-ts-on/bin     --pair shared=../llama.cpp/build-ts-shared-off/bin/Release,../llama.cpp/build-ts-shared/bin/Release
                                        # NEW: N build pairs, every arm of every
                                        # pair in ONE round-robin (F30). This is
                                        # the only way a cross-build comparison
                                        # is interleaved at all
```

**`--blocks N` is the flag that matters now.** It runs the whole round-robin N
times and reports a `t` interval over the per-block point estimates. Quote that,
never the bootstrap: the bootstrap resamples inside one invocation, and F31
found two of its intervals, for one quantity on unrebuilt binaries, that did not
overlap. `ab_throughput.py` takes `--blocks` and `--json-out` too.

Two guards exist because they were needed rather than foreseen.
`preflight_ram()` refuses to start when free memory is under 1.5x the model --
F34 ran at 870 MB against an 840 MB model and reported the instrumented build as
*faster*. And a failed baseline gate now makes the tool refuse its own block
table, instead of printing "quote this" underneath a refusal.

**`--min-baseline` is new in session 7 and you should use it (F46).** This
machine sits on **discrete throughput levels** — 40.9, 43.5 and 46.0 tok/s were
all observed today, each held for minutes, switching unprompted and refusing to
be forced. They are ~12% apart, about twenty times the effects being measured.
`--min-baseline 46` probes the A binary once (~8 seconds) and **refuses to start
a 25-minute run** if the machine is on the wrong level. It exists because F43
and both halves of F45 were voided for exactly that, by hand, after the fact.

    python tools/ab_throughput.py --a A.exe --b B.exe -m M.gguf         -t 8 --blocks 3 --min-baseline 46 --json-out run.json

`--json-out` also writes a **`timeline`** now: one entry per measurement with a
wall clock, so a finished run can be asked whether it *stayed* on one level.
F45's layout run had an 11-measurement excursion invisible in its pooled median,
and the timeline is what found it.

**The gate changed in session 7 (D9).** It used to be computed on the arms'
data pooled across every block, which folded in the between-block drift the `t`
interval already carries -- so a `--blocks` run was charged twice for the same
variance and could be told "this machine cannot resolve a 2% effect" about a run
whose interval was already saying so. It now gates on the **worst single
block** and prints the pooled figure beside it. `--blocks 1` is unchanged and no
published number moves. **`ab_throughput.py` still has no gate at all** -- F39
applied it by hand, which is how D9 was found, and which is worth doing every
time until the tool grows one.

`--pair` is repeatable and each pair is scored against **its own** compiled-out
arm. With more than one pair you also get the difference of overheads with its
own bootstrap CI, and the compiled-out arms against each other -- which measures
DLL cost in llama.cpp, not in tokenscope. `--bin-off`/`--bin-on` still work.
Arm order rotates each round since F30; `--no-rotate` restores the old fixed
order if you need to reproduce a pre-`9ec7c82` number.

`--barriers` needs `TOKENSCOPE_LEVEL=3`. It matches the k-th barrier across
threads, which is exact because node and barrier scopes strictly alternate and
every thread walks the same node list -- and it **refuses to report anything**
rather than guess if that ever stops holding.

`split_totals` now separates per-token work *inside* the decode slice from work
*outside* it (sampling, detokenization run after `decode` returns). Charging the
latter against the former is what pushed attribution over 100% in F11.

**Nine reference traces** in `examples/`, **1164 decode tokens**, ~21 MB, all
exercised by CI. (The table said five and 1012 until session 4 counted them; the
three 8B/batched ones were added in sessions 2-3 without the table being
updated.)

| trace | level | decode tok | what it covers |
|---|---|---|---|
| `mid-24L-tg256` | 1 | 257 | host scopes, the F1 control-plane result |
| `mid-24L-L3-tok10-11` | 3 | 25 | the F9 barrier data, and F23's baseline |
| `mid-24L-L3-pp64` | 3 | 0 | level-3 prefill, the other half of F9's test |
| `mid-24L-cli-sampling` | 1 | 31 | `llama-cli`, sampling + detokenization (F11) |
| `qwen-ctxshift-c256` | 1 | 699 | real model, context shift, `find_slot` (F16) |
| `qwen-batched-np16` | 1 | 92 | 16 concurrent sequences (F17) |
| `qwen3-8b-L3-tok8-13` | 3 | 17 | the 8B, before the F20 naming fix |
| `qwen3-8b-named-attnout` | 3 | 17 | the 8B, after it — the second `trace_svg.py` figure comes from here |
| `mid-24L-L3-tok10-11-f28` | 3 | 26 | **new in session 5.** Same model, window and thread count as `mid-24L-L3-tok10-11`, taken after the F28 fix. Median of seven candidate runs by total imbalance, not the best one. The README's barrier split and its timeline figure both come from here |

**CI mechanizes five claims**, and each fails the build if the finding stops
being true: attribution never exceeds 100% (inside-slice only), every barrier is
matched by `--barriers`, F9 (elementwise nodes serial in decode, parallel in
prefill), F16 (shifts visible, `find_slot` a small part of its scope), and
**F22 (two modules, one buffer per thread)**.

The F22 one is `ctest --test-dir build-shared`, built on all three platforms
from `BUILD_SHARED_LIBS=ON`. It needs no llama.cpp, no model and no network, and
runs in 0.04 s. It exists because F18 — the shared build not linking at all —
went undetected for a session purely because nothing here ever built a DLL.

---

## 5. Next steps, in the order I would do them

**Session 7 closed all of group A except item 2.** The null control was
measured (F39), repeated at 8 threads where it corrected F39 (F40), and F27's
last unchecked row was re-measured (F41). What is left in group A is **item 2,
the layout arm** — and it is now the *only* route to M6, because F39 discredited
the prefill control the question used to be argued with.

**Session 6 rewrote this list** before that. It closed item 7 (the shared
build's overhead, F36) and then spent most of its length on something that was
not on the list at all: the intervals every Tier D number in this project was
quoted with were **within-run intervals**, roughly half as wide as the truth.
**Read [`04-project-audit.md`](04-project-audit.md) first** — it has the trust
ladder and the issue register (M1–M14) that the items below refer to.

### A. Do these first — they need only a quiet machine and no judgement call

1. ~~**Measure the null control.**~~ **DONE — F39, session 7.** Two runs of
   three blocks each, protocol identical to F33. **Decode `-0.04% [-0.49,
   +0.42]`** — clean, and the two runs agreed to two decimal places. **Prefill
   `+0.59% [+0.01, +1.16]` — resolved, and false by construction.** Three
   consequences, all live: F24's +1.62% clears the decode floor by 3× (the
   strongest thing ever said for it); the prefill control is discredited in both
   directions (**M13**); and `--blocks` has its own failure mode because
   consecutive blocks are not independent draws (**M12**). Also produced **D9**,
   a gate computed on pooled data, fixed in `c18a580`.

1b. ~~**Repeat that null at 8 threads.**~~ **DONE — F40, session 7**, and it
   corrected F39. **The floor depends on the thread count**: decode `+0.02%
   [-0.21, +0.26]` and prefill `-0.01% [-0.12, +0.11]` at 8 threads, against
   `[-0.49, +0.42]` and a *resolved* `+0.59%` at 16. Both gates pass; the A arm
   is *faster* (46.91 vs 42.50 tok/s), consistent with F10. Consequences:
   **F36's overheads are measurements** after all — F39's contrary claim
   compared them against the wrong thread count's floor and is withdrawn; **M12
   is narrowed** to a configuration-specific effect rather than a general
   property of `--blocks`; and **8 threads is the configuration to measure in**
   unless the question is about thread count.

2. **The layout arm for M6 — BUILT, and blocked on the machine.**
   `patches/05-layout-arm.patch` exists and is verified against the linker map:
   exactly two address deltas (0 and **+16**) across all 14,419 `.text`
   functions, `mul_mat` the same size in both arms and 98.6% byte-identical
   after its move, `ggml_vec_dot_f32` shifted with everything else. **F42** is
   why this was possible at all — F24's 1.1 MB of differing bytes turned out to
   be a *uniform 16-byte shift*, not the third of a regenerated image F38
   reported, so the perturbation is reproducible without touching behaviour.

   **What it needs is one 50-minute window on the 46 tok/s level** (F44/F46),
   plus a matched null in the same window. Three attempts in session 7 were
   voided: F43 on the wrong level, both halves of F45 on the gate. Run it with
   `--min-baseline 46` and check the `timeline` afterwards. **Everything except
   the machine is ready.**

   Design caveat, stated in the patch header rather than buried: F24's arm
   shifts **−16** and this one **+16**. Same magnitude, same hot function,
   opposite direction, and an alignment effect need not be symmetric. If the
   +16 arm comes back null, a −16 arm is the obvious follow-up and is a far
   smaller job than building the first one was.

3. ~~**Re-measure F27's `-2.06%` row.**~~ **DONE — F41, session 7.** Six blocks
   over two runs: **decode confirmed at −2.15% [−2.28, −2.02]**, 9× F40's floor,
   with F27's estimate inside the new interval. **Prefill was the one that was
   wrong**: −1.15% becomes **−0.43% [−0.65, −0.21]**, F27's estimate *outside*
   the new interval — wrong by 2.7×, not merely over-precise. F27's conclusion is
   unaffected and better supported. Also produced **M14**: the two runs' `t`
   intervals differ in width by **7×** (0.14pp against 0.95pp) while their point
   estimates agree to 0.02pp, so **three blocks can be tight by luck** and the
   standing rule becomes *two runs of three blocks, pooled to six*.

### B. The standing scientific gaps, unchanged by session 6

4. **Linux + GCC.** Still the blocker for the main upstream conversation, and
   smaller than it looked: F26 found that the barrier path measured here *is*
   Linux's default (OpenMP), so what is untested is `libgomp` and GCC, not a
   different algorithm. F27 then showed the worry is not small — swapping
   barrier implementations changes decode by up to 54% here.

5. **An MoE model.** The clearest hole in the byte law, and the most informative
   test available, because the law as stated should be **wrong** there: MoE is
   where bytes-streamed-per-token stops being a property of the file and starts
   depending on the router. Needs a download of several GB and a RAM check.
   **Ask before downloading** — sessions 4 and 5 both asked and were told not to.
   It would also make `mul_mat_id` testable, which is item 2's other half.

6. **F27 on a second machine.** The biggest unexploited result in the repo and
   meaningless as a general claim until someone runs the same protocol on a
   homogeneous part with `libgomp`. The protocol is `ab_throughput.py --blocks 3`
   plus `imbalance_repeat.py --metric release`, n≥12, on two builds differing
   only in `GGML_OPENMP`.

### C. Needs a person, not an agent

7. **File the F20 naming issue**, then decide about F24. `AGENTS.md` forbids an
   agent writing issue or PR text and requires the contributor be able to defend
   the change unaided; [`03`](03-upstream-issue-draft.md) holds an evidence pack
   to write *from*, not a draft to paste. **That constraint was protective**:
   F24 was the queued submission and session 6 found its interval too narrow and
   its control unclean. `03` now tells a maintainer both things up front.
   Re-check before filing: that `build_attn` still has seven overloads and still
   does not name the output projection at current `master`, that
   `CONTRIBUTING.md` has not changed, and that nobody has filed it already.

### D. The one that would change how everything else is measured

8. **Explain M10** — decode on this machine wanders between **38.6 and 46.0
   tok/s** with no identified cause, which is ~10× every effect measured here
   and decides whether a run passes the gate. **Four causes tested and refuted
   in F37:** not thermal (8 minutes of cooling gave a *lower* result than 7, and
   a 30-run post-cooldown curve was flat to −1.16%), not thread placement
   (pinning is worse; the best mask is no mask), not CPU frequency (Pearson
   **r = −0.423**, the wrong sign), not the `-p 0`/`-p 512` workload difference
   (−0.91%). Untested and still live: page-cache and standby-list state for an
   840 MB model read once per invocation, and per-process power throttling.

   Until it is explained, the practical rule stands and needs no mechanism:
   **the compiled-out arm's absolute median identifies which state the machine
   was in, and two runs with different A-arm medians are not comparable.**

### Closed, for the record

- ~~item 3, a Perfetto screenshot~~ — sidestepped in session 5 by `trace_svg.py`.
  What remains is only the part that genuinely needs a browser.
- ~~item 6, the shared-library build~~ — fixed in F22, overhead measured in F36.
- ~~item 7, the shared build's overhead~~ — **F36**: every build under 1% and
  indistinguishable.
- ~~sweeping "certified" from the prose~~ — done; annotated rather than erased
  in `FINDINGS.md`, gone everywhere else.

## 6. Things I would tell myself

From session 1, still true:

- **The consistency checks earned their keep.** Two real bugs were caught by
  "these percentages cannot exceed 100". Keep adding checks of that shape.
- **The benchmark harness refusing to answer is a feature.** It declined four
  more times in session 2 and was right each time.
- **Do not run builds while benchmarking.**
- **Check documented features actually work.** A feature is not done when the
  code exists, only when something calls it.

Added by session 2, in rough order of how much time they would have saved:

- **Write predictions down before you can test them, and date them.** Four were
  tested in session 2. Two were wrong. Without the written version I would have
  remembered predicting whichever turned out right. This is now the single most
  valuable habit in the project.
- **A scope's name is a claim about what it measures** (F16). `kv.slot-search`
  wrapped `init_batch`, and the search was 2.7% of the number. F2 then reasoned
  from the name rather than the code and got two things wrong for one reason.
  When a scope's number looks interesting, re-read what it actually wraps.
- **Barrier count and barrier cost are different quantities** (F15). Removing
  10.6% of the barriers changed throughput by nothing measurable. The ones you
  can cheaply remove are the ones nobody was waiting at.
- **A null result is a claim about your workload.** P-core vs E-core on decode
  reads 28.0 vs 28.2 tok/s and looks like a broken CPU mask. It is not; decode
  is bandwidth-bound. The same masks on prefill give 265.95 vs 92.39. Nearly
  threw out F14 over this.
- **Trace-derived throughput is not throughput.** A traced token carries the
  recording cost on exactly the token being measured. Every tok/s number in
  FINDINGS comes from the *uninstrumented* build for this reason; a single r=1
  traced run briefly showed a 21% degradation that was pure noise.
- **The Bash tool mangles `
` inside heredocs.** Backslash escapes in Python
  written via `<<'EOF'` came out as literal newlines and broke the file three
  times. Use the Write tool for anything with escapes, or build strings with
  `chr(92)`. Line-range edits located by content are more reliable than
  `str.replace` on text containing em-dashes or `×`.
- **`bootstrap.py --make-patch` silently regenerates the old patch** if a newly
  touched file is not in its `TOUCHED` list. It reported success while ignoring
  two edited files. Add the file to the list *when you first edit it*.
- **curl progress output floods the transcript.** Use `-s` on any large
  download.
- **A feature can look dead because your test is too small.** `TOKENSCOPE_RING`
  produced byte-identical output to non-ring mode three times running, which
  reads exactly like the `TOKENSCOPE_TOKENS` bug from session 1. It works.
  Records come in 1 MiB chunks *per thread*, so ring only engages after a thread
  fills one (~43,700 records) -- all three tests were under that. Before
  declaring a feature inert, check its precondition is actually met. (It is
  genuinely inert at small budgets though, and the README now says so.)

Added by session 3:

- **A prediction can be right for a reason worth half of what it claimed.**
  P19.3 predicted `attn_v` would cost less than its 3.56x byte ratio because
  Q4_K needs dequantizing and F16 does not. It came in at 2.911, below, as
  predicted. But `Qcur`/`Kcur` — same dtype, differing only in size — undershoots
  *its* byte ratio by 10% too, so roughly half the effect was a size artifact
  present in both. **Check whether your mechanism is the only thing producing
  the sign you predicted**, using a pair where it cannot be operating.
- **Two mechanisms can move at once and you will model one.** P19.4 reasoned
  correctly about memory traffic, got the direction right and both specifics
  wrong, because F14's core-heterogeneity penalty stopped applying at the same
  time — at 8B every thread waits on memory, so slow cores cost nothing. When a
  prediction fails, check whether a *second* known mechanism changed regime.
- **The strength of a test is not the separation between the hypotheses.** I
  predicted the 8B would test the byte law weakly, because bytes and parameters
  only disagreed by 4.2 points there against 8.3 on the 0.5B. It was the
  sharpest result in the project — 0.3 points of residual. Separation and
  precision are different axes.
- **Look for the controlled experiment inside the data you already have.** The
  best result of the session cost nothing to produce: `ffn_down` is Q6_K in 18
  layers and Q4_K in 18, which is a paired experiment with controls sitting
  inside a file that had already been traced. Ask what varies *within* a
  workload before running another one.
- **A dead branch in a lookup table has no symptom.** `{ "kqv_out", ... }` was
  unreachable behind `{ "kq", ... }` for the whole project. Nothing failed,
  because no model had ever produced that node name. Found only by asking where
  a *hypothetical* name would land. The self-test now checks the table's shape
  rather than its behaviour on the inputs that happen to exist.
- **Verify a check by breaking the thing it checks.** After adding the shadow
  test I reintroduced the bug, confirmed it failed, and restored it. A check
  that has never failed is a check nobody has tested — this is the "check
  documented features actually work" lesson applied to the checks themselves.
- **`cmd /c "vc.bat <command>"` from the Bash tool silently opens an
  interactive shell** instead of running the command, and produces no error.
  Use the **PowerShell tool** with
  `cmd.exe /c "call `"$vc`" >nul 2>&1 && cd /d ... && <command>"` instead; that
  works reliably. This cost a wasted 7-minute background build.
- **The heredoc backslash trap from session 2 is still live and still bites.**
  A `\n` inside a `<<'EOF'` Python heredoc reached the file as a real newline
  and broke `ts_selftest.cpp` mid-build. The rule stands: **use the Write or
  Edit tool for anything containing escapes.**

Added by session 4:

- **When two requirements conflict, check whether they are really about the same
  object.** F18 stated a genuine incompatibility: the hot path needs a raw
  thread-local, the registry needs one instance across DLLs, and MSVC will not
  let one variable be both. Both halves were true and the conclusion did not
  follow, because `ts_tls` is a *cache* and the registry-owned buffer is the
  *state*. Nothing ever required the cache to be single-instance. The fix took
  an hour; the framing was the whole problem, and it sat unexamined for a
  session because "genuine incompatibility" reads like a finished thought.
- **The dangerous part of a small change is what it makes load-bearing.** The
  edit is ~40 lines. It also silently turned `ts_thread_init` into something
  that must be idempotent, made every caller responsible for writing the result
  back, and — worst — would have made level 2 record *no node data at all* in
  shared builds, with a valid trace, zero drops and no warning. That last one
  was found by asking "who initializes this, and in which module?", not by any
  test. **After changing where state lives, re-derive the initialization order
  for every reader of it.**
- **Reproduce the failure before fixing it.** One command, and it converts "it
  builds" from a hope into evidence. The `LNK1120` is in F22 for the same
  reason.
- **Verify a shared-state fix by counting, not by looking.** The trace says four
  worker threads at `-t 4`. Had the DLLs each built their own per-thread state,
  it would say eight — the records would all still be present, the trace would
  still parse, and every per-thread percentage would be computed on half a
  thread. A count that *could* have come out wrong is worth more than a table
  that looks right.
- **`Select-String` on a here-string of Python is not worth it.** Inline Python
  through the PowerShell tool mangled quotes twice. The session-2 rule
  generalizes: **anything with escapes or quoting goes in a file via the Write
  tool**, then gets run. This applies to PowerShell here-strings as much as bash
  heredocs — and to *backticks*, which bash command-substitutes inside double
  quotes: `python -c "...markdown with \`code\` spans..."` silently deleted two
  spans from HANDOFF and reported success, exactly the shape of the session-2
  backslash trap. Markdown is full of backticks. Use the Edit tool for it.
- **An A/B where one arm is a binary you saved earlier is not an A/B.** F24's
  first throughput run said **+2.71% with non-overlapping ranges** and was wrong.
  `build-ts-off` had last been built on 5 Sept; `llama-graph.cpp` changed on
  6 Sept for the F20 naming patch. So the "stock" binary predated a change the
  patched one contained, and the arms differed by more than the line under test.
  Rebuilding both from one tree took it to +1.49% at n=12 and +1.95% at n=20.
  Nothing looked wrong — the clean separation made it *more* convincing. This
  project already knew to interleave arms in **time**; it had not written down
  that they have to be matched in **version**. Check binary timestamps against
  `git status` before trusting any comparison.
- **Read the target project's `AGENTS.md` before writing anything aimed at it.**
  llama.cpp's forbids an agent writing PR descriptions, issue comments or
  reviewer responses, non-overridably, and lists `gh issue create` among things
  not to run for a user. Session 4 wrote a finished, paste-ready issue body for
  F20 before reading it, and had to reframe the whole section as an evidence
  pack. The measurements were the valuable part anyway; the prose was the part
  that was not wanted. `docs/03` now leads with that rule.
- **One trace is one draw, and six are not many more.** This cost three wrong
  turns in one session. A single trace inverted F23's conclusion. Six runs then
  produced a "28-thread anomaly" that does not exist — twelve runs give 1.041
  where six gave 0.558 — and **a whole prediction (P23.4) was written, committed
  and tested to explain it** before re-measuring showed there was nothing there.
  Six runs also put P23.4's own answer outside its predicted band at 0.747 where
  twelve give 0.944. Per-node imbalance spreads up to **8.5×** between identical
  runs. This project already knew not to trust one throughput number —
  `bench_overhead.py` has interleaved arms and bootstrap CIs, and has refused to
  certify six times — and then trusted trace-derived ratios at n=1 and n=6
  anyway, because the tracing is exact so the numbers *looked* exact.
  **Before explaining a surprising number, re-measure it.** That is cheaper than
  the prediction it saves you writing.
- **Predict ratios against a control, not levels.** P23.1 predicted a number
  would rise, in a regime where every comparable number also rose; it held and
  meant almost nothing. P23.2 predicted a ratio against a node that did not
  change mode, and that one carried the entire finding. When drafting a
  prediction, ask what else moves at the same time and divide by it.
- **Check the instrument can return the quantity you are predicting.** P23.3
  named `lm_head` as its control. `--barriers` matches a node to the barrier
  *after* it, and `lm_head` is the last node of the graph, so the quantity does
  not exist for it. That is a cheaper check than the measurement it wasted.

Added by session 5, in rough order of how much they would have saved:

- **A claim about how your code was BUILT is checkable in one command, and a
  caveat is a claim.** Five documents said every barrier number here came from
  ggml's spin-wait threadpool. `grep GGML_USE_OPENMP build.ninja` would have
  cost four seconds in session 1 and the answer was the opposite (F26). It
  survived four sessions because it lived only in *caveats* — the part of a
  document that exists to say what a result does not cover, which is exactly the
  part nobody re-derives. Every `#ifdef` a finding's scope depends on is worth
  one grep.
- **An anomaly detector that also explains the anomaly has two outputs, and
  usually only one of them was measured.** `--barriers` correctly flagged a
  barrier holding three quarters of all release latency, correctly excluded it
  from the corrected split, and then explained it as "thread-pool spin-up". It
  was tokenscope's own allocator (F28). The detection was real; the attribution
  was prose in an authoritative voice. F16's lesson — a scope's *name* is a
  claim — applies to diagnostic *messages* too.
- **Predict the scaling of the quantity you will actually read.** P28.2
  predicted 1.5-2.5x going 8 -> 16 threads, arguing that a mutex serialises the
  work so the total cannot double. Measured 4.0-4.5x. The mutex does serialise,
  so the *wall* duration is linear — but `--barriers` reports **thread time**,
  and n threads each sit through the whole linear stall. Linear duration summed
  over n threads is quadratic. Right physics, wrong denominator.
- **"Interleave the arms" means the arms of the comparison you are making.**
  `bench_overhead.py` has interleaved within one invocation since session 1.
  F25's question was shared-versus-static, which is a comparison *between* two
  invocations, and the machine's noise floor moved by 2x between them. The
  prediction came back neither confirmed nor falsified, which is worse than
  wrong. Same trap as F24 in a different disguise: there the arms differed in
  version, here in time.
- **Real asymmetries are not automatically the relevant ones.** P27.4 found two
  genuine differences between the threading paths, wrote them down in advance,
  and both were microseconds — and both were paid by the arm that *won*. Writing
  a mechanism down early is still right; it just does not make the mechanism
  load-bearing.
- **Prediction and control are different jobs, and a control has to be able to
  not move.** `ab_throughput.py` warns that a certified control means the
  comparison is broken. In F27 the control certified because `GGML_OPENMP`
  changes every barrier in every graph including prefill's. The tool was right to
  complain and the complaint did not apply — which means the honest report is
  "this comparison has no control", not "the warning is spurious".
- **sscanf does not care what it leaves behind.** `"%u-%u"` then `"%u"` made
  `10:11` mean token 10 alone, silently, for two sessions (F29). The cases worth
  asserting in a parser are the ones it must *refuse*; the accepted ones are the
  ones somebody already tried by hand.
- **Pick a committed reference artifact by the median, not by eye.** The new
  reference trace is the median of seven candidate runs by total imbalance. F23
  paid to learn that one trace is one draw; a repo that quotes its luckiest run
  is the same mistake with a longer half-life.
- **The blocked item may be blocked on the wrong thing.** "Perfetto screenshot"
  sat at the top of the list for four sessions needing a browser and a person.
  What the README actually needed was a picture, and a picture generated from a
  committed trace is *better* for a repo than a screenshot: it is text, it
  diffs, and anyone who clones can regenerate it. Ask what the item is for
  before assuming its stated form.


---

## 7. Numbers to quote (all reproducible from the committed code)

| Claim | Value | Source |
|---|---|---|
| Upstream patch size | 174 lines, 7 files | `patches/01-instrument.patch` |
| F20 naming fix size | 8 added, 13 removed, 1 file | `patches/02-name-attn-output.patch` |
| Per-scope cost | 52.8 ns (2 clock reads + 1 store) | `ts_selftest` |
| Level 3 overhead, static, 8 threads | **+0.56% [-0.05, +1.16]**, `t` over 3 blocks. Spans zero, so bounded rather than resolved | [`FINDINGS`](FINDINGS.md) F36 |
| ...same, session 1 | +0.67% [+0.12, +1.67] | [`02`](02-overhead-methodology.md) |
| Level 3 overhead, ninja-shared | **+0.92% [+0.38, +1.46]**, `t` over 3 blocks | [`FINDINGS`](FINDINGS.md) F36 |
| Level 3 overhead, MSBuild-shared | **+0.77% [+0.05, +1.48]**, `t` over 3 blocks | [`FINDINGS`](FINDINGS.md) F36 |
| Do the three builds differ? | **No.** Every pairwise difference spans zero; linkage -0.03% [-0.45, +0.24], generator +0.07% [-0.40, +0.55] | [`FINDINGS`](FINDINGS.md) F36 |
| Superseded overhead figures | +0.67%, +1.16%, +0.50%, +0.87%, +0.31% — **all single-run bootstraps, all too narrow.** F36's interval contains every one | F25, F30, F31 |
| Zero-overhead-when-off | 0 symbols, 864-byte archive | [`02`](02-overhead-methodology.md) §2 |
| Shared-library build | links and traces correctly on MSVC; **overhead measured and indistinguishable from static** | F22, F36 |
| Machine state swing | decode wanders **38.6–46.0 tok/s** with no identified cause; ~10x the effects measured. Not thermal, not placement, not frequency, not workload | [`FINDINGS`](FINDINGS.md) F37 |
| Matmul arrival imbalance, work-stealing vs equal-slice | **0.34-0.56x** against 0.94-1.51; two comparisons with non-overlapping ranges, n=12 per arm | [`FINDINGS`](FINDINGS.md) F23 |
| Per-node imbalance, spread over 12 identical runs | up to **8.5x** at 8 threads, 1.3-1.5x at 16 | [`FINDINGS`](FINDINGS.md) F23 |
| Decode speedup from `nth*4` -> `nth*2` in mul_mat | **+1.62% [+1.10, +2.15]**, t over 6 blocks in 2 runs, 240 rounds/arm. **The prefill control is no longer a clean null** | [`FINDINGS`](FINDINGS.md) F33 |
| ...same patch on prefill (control) | -0.81% [-2.44, +0.35], **not** certified | [`FINDINGS`](FINDINGS.md) F24 |
| Barrier wait | 11.2% of worker thread time (post-F28 trace reads 11.5%) | [`FINDINGS`](FINDINGS.md) F6 |
| ...of which arrival imbalance | **84.6%**, on a post-F28 trace with no artifact to exclude | [`FINDINGS`](FINDINGS.md) F28 |
| Decode cost of `GGML_OPENMP=OFF`, 28 threads | **-54.17% [-55.05, -53.31]** certified; -2.06% at 8 threads | [`FINDINGS`](FINDINGS.md) F27 |
| ...same, `tiny.gguf` | **-78.64%** at 28 threads, -13.07% at 8 | [`FINDINGS`](FINDINGS.md) F27 |
| Release latency per unit work, ggml pool vs OpenMP | 1.95x at 8 threads, 4.9x at 16, both non-overlapping, n=12 | [`FINDINGS`](FINDINGS.md) F27 |
| Total barrier wait per unit work at 28 threads | 0.752 (ggml pool) vs 0.249 (OpenMP) | [`FINDINGS`](FINDINGS.md) F27 |
| tokenscope's own first-touch artifact, before F28 | 76-83% of all after-arrival barrier time, scaling as thread count SQUARED | [`FINDINGS`](FINDINGS.md) F28 |
| Barriers behind single-threaded nodes | 120 of 412 per token | [`FINDINGS`](FINDINGS.md) F9 |
| Upper bound on fixing that | 1.34% of graph wall time, and F15 found no reachable part | F9, F15 |
| Barriers removed by ggml's one fusion | 49 of 461 per token, for no measurable throughput | [`FINDINGS`](FINDINGS.md) F15 |
| Best speedup at any thread count | 2.21x F32 / 3.28x Q4_K_M, both at 6 threads | F10, F12 |
| `lm_head` share, Qwen2.5-0.5B Q4_K_M | 34% of decode thread time | [`FINDINGS`](FINDINGS.md) F12 |
| ...same tensor, Qwen3-8B, identical vocabulary | 10.5% | [`FINDINGS`](FINDINGS.md) F19 |
| ...dolphin-mistral-7B, 32k vocabulary | 2.8% | [`FINDINGS`](FINDINGS.md) F21 |
| Byte law, max error over 4 models | 0.3-0.5 points (0.5B: 2.9) | F19, F21 |
| ...predicting from parameter counts instead | wrong by 1.1-7.2 points | F12, F19, F21 |
| `ffn_down` Q6_K vs Q4_K layers, same shape | **1.419x measured, 1.458x predicted**, controls at 1.007/1.009 | [`FINDINGS`](FINDINGS.md) F19 |
| `attn_v` (F16) vs `attn_k` (Q4_K), same shape | 2.911x, byte ratio 3.556x | [`FINDINGS`](FINDINGS.md) F19 |
| Anonymous attention output projection | 7.1% of decode thread time, in all 7 `build_attn` overloads | [`FINDINGS`](FINDINGS.md) F20 |
| Peak decode speedup, Qwen3-8B | 3.01x at 8 threads (0.5B: 3.28x at 6) | [`FINDINGS`](FINDINGS.md) F19 |
| Prefill speedup, same model, 28 threads | **11.07x**, against 2.95x on decode | [`FINDINGS`](FINDINGS.md) F19 |
| Barrier wait, Qwen3-8B at 6 threads | 5.6% (F6 measured 11.2% on the F32 model) | [`FINDINGS`](FINDINGS.md) F19 |
| Phase time predicted from weight BYTES | within 2.9 points on a real quantized model | [`FINDINGS`](FINDINGS.md) F12 |
| ...predicted from parameter counts | wrong by 7.2 points on the same model | [`FINDINGS`](FINDINGS.md) F12 |
| Sampling + detokenization | 0.13% of a token | [`FINDINGS`](FINDINGS.md) F11 |
| Context shift, `-c 256` | 4 spikes in 699 tokens, 1.13-1.34x median | [`FINDINGS`](FINDINGS.md) F16 |
| KV cell search (`find_slot`) | 1.17 us/token, *falls* as the cache fills, 1.55 us at 16 sequences | F16, F17 |
| Batching 16 sequences | 3.06x aggregate throughput | [`FINDINGS`](FINDINGS.md) F17 |
| Parallel efficiency at 28 threads | 7% | [`FINDINGS`](FINDINGS.md) F10 |
| P-core vs E-core, compute-bound | 2.88x (265.95 vs 92.39 tok/s) | [`FINDINGS`](FINDINGS.md) F14 |
| Barrier wait, mixed vs homogeneous cores | 21.2% -> 11.1% at 12 threads | [`FINDINGS`](FINDINGS.md) F14 |
| Host control-plane work | 0.4% of decode | [`FINDINGS`](FINDINGS.md) F1 |
| Attention math (`attn.score`) | 0.7% of thread time | [`FINDINGS`](FINDINGS.md) F7 |
| Phase time vs parameter count | within 1.5–8.6% over a 27× range — **F32 only**, see the F12 rows | [`FINDINGS`](FINDINGS.md) F7 |

Everything above is measured on **synthetic F32 weights unless the row names a
real model**, MSVC Release, Windows 11, on an i7-14700HX (8 P-cores + 12
E-cores). Say so whenever quoting them.

**Four real models have now been measured** (Qwen2.5-0.5B, Qwen3-8B,
dolphin-llama3-8B, dolphin-mistral-7B) and the F19/F21 rows come from those.
The three 7-8B ones live in Ollama's blob store; paths are in section 3.

All of it is 8 threads except the F10 and F12 rows, which are the sweeps
themselves. The level-3 overhead figure is an 8-thread number too, which
matters because docs/01 predicts it should move with thread count: at 28
threads it re-measured as +0.66% [-1.85, +4.18], which the harness declined to
certify.
