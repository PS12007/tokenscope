# 10 — Taking F51 and F52 to llama.cpp: a guide for a person, from zero

**Who this is for:** you, if you have never filed a GitHub issue and do not yet
understand what F51 and F52 are. It explains both findings in plain terms, tells
you how GitHub issues work, lists what an issue must contain with every number
and where it came from, and ends with questions you should be able to answer
before posting anything.

**What it deliberately does not contain: any text to paste.** llama.cpp's
`AGENTS.md` says an AI agent must *never* write an issue, pull-request
description or comment on a contributor's behalf, calls that rule
non-overridable, and says the penalty is a ban — for your account, not the
agent's. There is also a practical reason that matters more to you: maintainers
reply with questions, and if the words were not yours you cannot answer them.
So this file gives you understanding and facts. The sentences have to be yours.
Short and plain is better than polished — `AGENTS.md` says "verbose,
AI-sounding responses will not be well-received".

> **Checked on 2026-09-10 against llama.cpp master `df03399b8`** (same day), so
> you know where things stood when this was written — re-check before posting:
>
> - **F52 is still present on master.** In `ggml/src/ggml-cpu/ggml-cpu.c`, the
>   OpenMP `else` branch (lines 3439–3441 on that commit) still goes straight to
>   `ggml_graph_compute_thread` without calling `ggml_thread_apply_priority`.
> - **F51's path is unchanged on master**: `ggml_barrier` is still a bare
>   `#pragma omp barrier` (line 583) and `GGML_OPENMP` still defaults to `ON`.
> - **#26200 is still open with 0 comments.**
> - **There is no built-in escape for GCC users.** llama.cpp has a
>   `GGML_OPENMP_FETCH` option that downloads LLVM's OpenMP (which spins), but
>   its CMake refuses to run unless the compiler is Clang
>   (`ggml/src/CMakeLists.txt`: "GGML_OPENMP_FETCH currently requires Clang on
>   Windows"). So for a GCC/MinGW build the only fix today is
>   `-DGGML_OPENMP=OFF` — a fact worth knowing for the #26200 comment.
>
> Our own measurements were taken on the older pinned commit `4d9176092`. Say so.

---

## Part 1 — What you would be reporting, in plain language

### First, four ideas you need

**Threads.** llama.cpp speeds up the maths by splitting each step of the model
across several CPU threads (`-t 8` means eight). Each thread does a slice.

**A barrier.** After each step, every thread has to wait until *all* threads
have finished their slice, because the next step needs the whole result. That
waiting point is a barrier. A decode token on our test model goes through **412
barriers**. So how fast a barrier is matters a lot.

**Two ways to wait.** A thread at a barrier can either *spin* — keep checking
"are we done yet?" in a tight loop, which is instant to wake but burns CPU — or
*sleep* — ask Windows to put it to sleep and wake it later, which saves CPU but
takes microseconds each time, because the operating system has to get involved.
For hundreds of barriers per token, sleeping is very expensive.

**OpenMP, and who provides it.** llama.cpp can use a standard library called
OpenMP to run its threads, and it does by default (`GGML_OPENMP=ON`). But
"OpenMP" is only a specification — each compiler ships its own implementation.
Microsoft's compiler (MSVC) ships `vcomp`. GCC ships `libgomp`. They behave
differently, and that difference is the whole of F51. llama.cpp also has its
**own** thread pool, used when you build with `GGML_OPENMP=OFF`.

### F51 — GCC's OpenMP on Windows makes llama.cpp much slower

**What happens.** If you build llama.cpp on Windows with GCC (the "MinGW" or
"MSYS2" or "w64devkit" way), the default build uses `libgomp`. The Windows
version of `libgomp` has no spinning barrier at all — every thread sleeps at
every barrier and Windows has to wake each one. On our laptop that makes decode
**2.65× slower** at 8 threads than the same GCC build using llama.cpp's own
thread pool. With Microsoft's compiler, the two are the same speed.

**Why we think so.** Three separate kinds of evidence:
1. **The source.** In GCC's source, the Windows build of `libgomp` has no
   barrier of its own and falls back to a generic one made of a lock plus two
   semaphores, with no spin phase (`libgomp/config/posix/bar.c`).
2. **The numbers behave like that.** The extra cost per barrier grows in a
   straight line with the thread count — about 9 µs per extra thread — which is
   what "wake every thread one at a time" predicts.
3. **The knobs do nothing.** `OMP_WAIT_POLICY=ACTIVE` and
   `GOMP_SPINCOUNT=INFINITE` normally tell `libgomp` to spin. Here they change
   nothing, because this version never reads them.

**The important catch: someone already reported this.** Issue
[ggml-org/llama.cpp#26200](https://github.com/ggml-org/llama.cpp/issues/26200)
(opened 2026-07-27) describes exactly this mechanism, with a 4-core CPU and a
Mixture-of-Experts model, and measured **+40%**. It has **no replies** and has
been labelled `stale` (GitHub's bot marks issues nobody has touched). So F51 is
**not a new issue**. What you would add is a **comment** on #26200 with data it
does not have:

- it expects dense models to suffer "less so" — ours is a dense model and loses
  far more (2.65×, against their +40%)
- two different laptops with hybrid CPUs (P-cores + E-cores), both much worse
- the cost grows with thread count, which explains why their 4-core number is
  smaller

**Do not open a new issue for F51.** A duplicate wastes maintainers' time, and
`AGENTS.md` asks you to check existing issues first. A comment on #26200 is the
right shape.

### F52 — OpenMP builds on Windows run at half speed with one thread

**What happens.** Windows 11 can "power throttle" a thread — run it slowly, or
move it to a slow E-core — to save battery. llama.cpp knows this, and has a
function, `ggml_thread_apply_priority()`, that tells Windows "do not throttle
this thread". It calls it in two of its three code paths. The third — an
**OpenMP build asked to use exactly one thread** (`-t 1`) — skips the call. So
that one thread is left throttleable, and on our laptop Windows throttles it:
single-thread decode drops from ~21 tok/s to ~9, erratically.

**The fix is one line** — call that function in the missing branch. It is in
`patches/07-omp-single-thread-prio.patch`. With it, the OpenMP build at one
thread runs at 20.74 tok/s, the same as the thread-pool build (20.95), and
nothing changes at 8 threads.

**The catch here.** Our second laptop does **not** show the slowdown. We do not
know why. A maintainer will almost certainly ask, and "I don't know, it
reproduces on one of my two machines" is an honest and acceptable answer — but
you have to say it up front, not have it discovered.

**Nobody has reported F52** as far as we could search (issues and PRs, several
phrasings). One related open PR, **#16014**, goes the other way — it would
compile the whole "do not throttle" code out of GCC/MinGW builds. Our
measurement is evidence of what that would cost. Mentioning it is optional.

**Is `-t 1` important?** Honestly, not very — most people use more threads. So
F52 is a real but small-reach bug. That is fine for an issue; say it plainly.

---

## Part 2 — Which to do, in which order

1. **Comment on #26200 with the F51 numbers.** Easiest: no new issue, no fix to
   defend, just data added to someone else's report. Good first contact.
2. **Then, if you want, open a new issue for F52.** It is new, has a clear
   cause and a one-line fix — but you must be ready for "can't reproduce".
3. **Do not open a pull request** for patch 07 yet. `AGENTS.md` asks for an
   issue first, to see whether maintainers want the change at all.

There are other candidates in this repo (F20's naming defect — `ROADMAP.md`
0.1 calls it the easiest first contact — and F24, which has its own kit in
[`05-f24-filing-kit.md`](05-f24-filing-kit.md)). One thing at a time.

---

## Part 3 — How GitHub issues work, step by step

**Before anything:** you need a GitHub account (you have one: PS12007). Issues
are public and permanent — anyone can read them, and deleting later does not
really remove them.

### Searching first (do this every time, just before posting)

1. Go to <https://github.com/ggml-org/llama.cpp/issues>.
2. Clear the search box's default `is:issue is:open` so closed issues show too,
   then search a few phrasings: `openmp mingw`, `libgomp`, `power throttling`,
   `single thread windows`.
3. Also check <https://github.com/ggml-org/llama.cpp/pulls> the same way — a fix
   may already be in flight.
4. Check #26200 has not been closed or answered since 2026-09-10.

### Commenting on an existing issue (for F51)

1. Open <https://github.com/ggml-org/llama.cpp/issues/26200>.
2. Read the whole thing, including any comments added since.
3. Scroll to the bottom; there is a comment box. It accepts **Markdown** (see
   below). Use the **Preview** tab before posting.
4. Post. That is it. You can edit your own comment later.

### Opening a new issue (for F52)

1. Go to the issues page and click **New issue**.
2. llama.cpp uses **templates** — you will be asked to pick a type (bug report
   categories and so on). Pick the closest bug type; it gives you a form with
   labelled boxes to fill in. Fill in every box it asks for.
3. Read [`CONTRIBUTING.md`](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
   first. `AGENTS.md` asks first-time contributors to confirm they have.
4. **Title:** name the symptom and the condition in one line — what is slow,
   when, on what. No "Bug:" prefixes beyond what the template adds; no hype.
5. Preview, then submit.

### Markdown in two minutes

- a line starting with `- ` is a bullet
- wrap commands and output in three backticks on their own lines, so they show
  as code:
  ````
  ```
  llama-bench -m model.gguf -t 1
  ```
  ````
- a table is rows of `| a | b |`, with a `|---|---|` line under the header
- `**bold**` for the one number that matters, sparingly

---

## Part 4 — What each must contain (checklists and facts, not text)

Tick every item. If you cannot explain an item in your own words, leave it out
rather than paste something you do not understand.

### For the #26200 comment (F51)

- [ ] **What you ran it on.** Both machines:
  - machine 1: Intel i7-14700HX (8 P-cores + 12 E-cores, 28 threads), 16 GB, Windows 11 build 26200
  - machine 2: Intel i7-1255U (2 P-cores + 8 E-cores, 12 threads), 16 GB, Windows 11 build 26200
  - compiler: GCC 16.1.0 (MSYS2 UCRT64, "Rev5") on both; MSVC 14.44 on machine 1 for comparison
  - llama.cpp commit `4d9176092` — **and say it is older than current master**
- [ ] **What model.** A synthetic dense F32 model, 220 M parameters, 24 layers,
  made with `tools/make_tiny_model.py` (`--layers 24 --embd 768 --heads 12
  --heads-kv 4 --vocab 8192`). Say it is synthetic.
- [ ] **The command**, exactly: `llama-bench -m mid.gguf -p 0 -n 64 -t 8`,
  built twice, once with `-DGGML_OPENMP=OFF`. Matches the issue's own repro.
- [ ] **The headline numbers** (decode, tokens/s — pick the few that matter):

  | | machine 1 | machine 2 | source |
  |---|---|---|---|
  | GCC `GGML_OPENMP=OFF` vs ON, 8 threads | **+165%** [+158, +173] | **+205%** [+200, +210] | FINDINGS F51; `results/10-machine1-gcc/f51-t8.json`; `results/08-windows-gcc/f27-t8.json` |
  | same, all threads | 4.5× (28 threads) | +217% (12 threads) | `sweep.json`; `f27-t12.json` |
  | MSVC `GGML_OPENMP=OFF` vs ON, 8 threads | −0.5% / +1.4% (a tie) | — | `sweep.json` |
  | prefill, 8 threads | +79% | +114% | same files |

- [ ] **The two things the issue does not already say**: dense models are hit
  *harder* here, not less; and the extra cost per barrier grows with thread count
  (machine 1: ~26 µs at 2 threads, ~50 at 4, ~89 at 8, ~155 at 16, ~252 at 28,
  from 412 barriers per token).
- [ ] **That the spin settings did nothing** (`OMP_WAIT_POLICY=ACTIVE`,
  `GOMP_SPINCOUNT=INFINITE`, all within ~3% of default) — confirming the issue's
  symbol-level claim with a measurement.
- [ ] **Short.** The issue already explains the mechanism; do not re-explain it.
  Data plus one or two sentences of what it shows.
- [ ] **AI disclosure.** `AGENTS.md`: "Disclose when AI meaningfully
  contributed." It did — the measurements and analysis were done with an AI
  coding assistant. Say so in one line.

### For a new F52 issue

- [ ] **Search first** (Part 3). If someone has reported it since, comment there
  instead.
- [ ] **Check it still exists on current master.** Open
  `ggml/src/ggml-cpu/ggml-cpu.c` on GitHub, find `ggml_graph_compute`, and look
  for the `#ifdef GGML_USE_OPENMP` block: the `n_threads > 1` branch calls
  `ggml_thread_apply_priority`, the `else` branch does not. If master has fixed
  it, stop.
- [ ] **What you ran it on** — machine 1 only reproduces; say machine 2 does not.
- [ ] **The symptom**, with numbers: OpenMP build at `-t 1`, repeated runs, 9–17
  tok/s and erratic (±3 to ±5 within a run); the thread-pool build 20–21 tok/s
  and steady. **Both MSVC and GCC OpenMP builds show it** — that matters,
  because it means it is not a GCC problem.

  | `-t 1`, decode, six runs each | tok/s | source |
  |---|---|---|
  | OpenMP build, unmodified | 9.55, 17.45, 9.34, 9.28, 9.08, 9.26 | `results/10-machine1-gcc/p07-t1.json` |
  | OpenMP build + one-line fix | mean 20.74, all within ±0.4 | same |
  | `GGML_OPENMP=OFF` build | mean 20.95 | same |
  | at `-t 8`, fix vs unmodified | −2.0%, −0.2% (no change) | same |

- [ ] **The cause, pointing at code**: which function, which branch, and that
  the other two paths do call it (at the pinned commit: `ggml-cpu.c` lines
  ~3384 and ~3440 call it; the `else` at ~3446 does not — re-find these on
  master, line numbers move).
- [ ] **The fix**, as a code snippet: the one added line from
  `patches/07-omp-single-thread-prio.patch`. Offer it; do not open a PR.
- [ ] **What you do not know**: why machine 2 does not reproduce; that `-t 1` is
  an uncommon configuration.
- [ ] **Optional:** that PR #16014 would remove this protection from MinGW
  builds entirely, and your numbers show what it is worth.
- [ ] **AI disclosure**, one line.

---

## Part 5 — Can you answer these without help? If not, do not post yet

These are the questions a maintainer might plausibly ask. Answer each out loud
in your own words.

**F51**
1. What is a barrier, and why does a decode token go through hundreds of them?
2. What is the difference between a spinning and a sleeping barrier, and why
   does that matter more with more threads?
3. Why is the same flag a tie with MSVC but 2.65× with GCC, on the same laptop?
4. Why did `OMP_WAIT_POLICY=ACTIVE` not help?
5. Your model is synthetic and F32. Why is that still a fair test of barrier
   cost? (Hint: the barrier count comes from the graph shape, not the weights.)

**F52**
6. What is Windows power throttling / EcoQoS, and what does
   `ggml_thread_apply_priority` do about it?
7. Why does only the one-thread OpenMP case miss the call? Point at the branch.
8. Why does it show up with both MSVC and GCC?
9. Why might your second laptop not show it? (You do not need the answer — you
   need to be able to say you do not know, and name what differs.)
10. Why does the fix not change anything at 8 threads?

If any of these stumps you, the relevant sections are FINDINGS F51/F52 and
Part 1 above. Asking an AI to *explain* them to you is fine and is exactly what
`AGENTS.md` lists as permitted use — "learning, exploration, and understanding
the codebase". Asking it to *write your reply* is not.

---

## Part 6 — After you post

- **Silence is normal.** llama.cpp gets hundreds of issues; #26200 itself got
  none. Do not bump or tag maintainers repeatedly. One polite follow-up after a
  couple of weeks, at most.
- **If someone asks a question**, answer it yourself, briefly. If you need to
  re-run something to answer, the commands are in FINDINGS F51/F52 and
  `tools/runtime_sweep.py`.
- **If someone says "can't reproduce"** (likely for F52): that is information,
  not rejection. Ask what hardware and Windows build they tried.
- **Record what happened** in `docs/HANDOFF.md` (issue link, date, any reply),
  so the next session does not re-file it.

---

## Glossary

| term | meaning |
|---|---|
| decode | generating one new token; the slow, one-at-a-time part |
| prefill | processing the prompt; many tokens at once, so barriers cost less per token |
| tok/s | tokens per second; higher is faster |
| `-t N` | how many CPU threads llama.cpp uses |
| MinGW / MSYS2 / w64devkit | ways of using GCC on Windows |
| MSVC | Microsoft's C/C++ compiler |
| OpenMP | a threading standard; `vcomp` (MSVC) and `libgomp` (GCC) implement it |
| `GGML_OPENMP=OFF` | build flag: use llama.cpp's own thread pool instead of OpenMP |
| P-core / E-core | fast and slow cores on Intel's hybrid CPUs |
| EcoQoS / power throttling | Windows running a thread slower to save power |
| `[+158, +173]` | the interval the measurement is confident the true value lies in |
