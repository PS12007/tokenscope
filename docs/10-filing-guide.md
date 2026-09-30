# 10 — Taking F51 and F52 to llama.cpp: a guide for a person, from zero

**Who this is for:** you, if you have never filed a GitHub issue or opened a
pull request, and do not yet understand what F51 and F52 are. It explains both
findings in plain terms, says exactly what the code change is, walks through
GitHub issues *and* pull requests click by click and command by command, lists
what each post must contain with every number and where it came from, and ends
with questions you should be able to answer before posting anything.

**What it deliberately does not contain: any text to paste.** llama.cpp's
`AGENTS.md` says an AI agent must *never* write an issue, pull-request
description, commit message or comment on a contributor's behalf, calls that
rule non-overridable, and says the penalty is a ban — for your account, not the
agent's. It also forbids an agent running `git push` or creating the PR for you.
There is a practical reason that matters more to you: maintainers reply with
questions, and if the words were not yours you cannot answer them. So this file
gives you understanding, facts and commands. The sentences, and every push and
click that sends something to GitHub, have to be yours. Short and plain is
better than polished — `AGENTS.md` says "verbose, AI-sounding responses will not
be well-received".

> **Last re-checked 2026-09-30 against master `4f31296a9`.** Nothing below
> changed: F52's `else` branch is still at lines 3439–3441 without the call and
> the fix applies cleanly; `ggml_barrier` and the `GGML_OPENMP` default are
> unchanged; #26200 is still closed with only the bot's comment; PR #16014 has
> not moved; no issue or PR mentions F52. **One rule changed:** on 2026-09-15
> `CONTRIBUTING.md` (#28945) added that AI-assisted contributors should spend
> "at least" roughly one hour per 200–400 lines on manual review, and must be
> ready to explain every line they submit. For a one-line change that is not a
> time burden, but it is the bar a reviewer will hold you to.
>
> **Re-checked on 2026-09-14 against llama.cpp master `1bc7a5af0`** (first
> checked 2026-09-10 on `df03399b8`). Re-check again just before posting:
>
> - **F52 is still present on master.** In `ggml/src/ggml-cpu/ggml-cpu.c`,
>   inside `ggml_graph_compute`, the OpenMP `n_threads > 1` branch calls
>   `ggml_thread_apply_priority` (line 3433); the `else` branch (lines
>   3439–3441) goes straight to `ggml_graph_compute_thread` without it.
> - **F51's path is unchanged**: `ggml_barrier` is still a bare
>   `#pragma omp barrier` (line 583) and `GGML_OPENMP` still defaults to `ON`.
> - **Changed since 2026-09-10: #26200 is now closed.** The stale bot closed it
>   on 2026-09-11 as "not planned", after 14 days with no human reply. Its only
>   comment is the bot's. It is **not locked**, so comments are still possible.
>   This changes the F51 plan — see Part 2.
> - **F52 is still unreported.** Searches for `ggml_thread_apply_priority`,
>   `power throttling openmp`, `EcoQoS`, `libgomp windows` turn up nothing about
>   it. PR #16014 (below) is still open and has not moved since 2025-10-22.
> - **There is no built-in escape for GCC users.** llama.cpp has a
>   `GGML_OPENMP_FETCH` option that downloads LLVM's OpenMP (which spins), but
>   its CMake refuses to run unless the compiler is Clang
>   (`ggml/src/CMakeLists.txt`: "GGML_OPENMP_FETCH currently requires Clang on
>   Windows"). So for a GCC/MinGW build the only fix today is
>   `-DGGML_OPENMP=OFF`.
>
> Our own measurements were taken on the older pinned commit `4d9176092`. Say
> so, or re-measure on master first (Part 4 shows how; strongly recommended for
> F52).

---

## The whole path at a glance

```
F51 (no code change)                 F52 (one-line code change)
--------------------                 --------------------------
search issues/PRs                    search issues/PRs
open a new issue that links #26200   reproduce on current master      (Part 4)
(optional) one-line comment on       open a bug issue                 (Part 5)
  #26200 pointing to it              wait for a reply or ~1 week
answer questions                     fork, clone, branch, edit, build (Part 6)
                                     test, commit, push to YOUR fork
                                     open the PR on the web page       (Part 6)
                                     answer review, update, get merged (Part 7)
```

"Pushing upstream" is not a single `git push`. Nobody outside the maintainers
can push to `ggml-org/llama.cpp`. You push to **your own copy** (a *fork*), then
ask the maintainers to pull your change in — that request is the *pull
request*. They review it and, if they agree, merge it.

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

**Is there a code change?** Not from us. F51 is a problem in GCC's library, not
a line in llama.cpp you can fix. What llama.cpp *could* do is one of the things
#26200 suggested: document it in `docs/build.md`, or default `GGML_OPENMP` to
`OFF` for MinGW builds. Those are the maintainers' choice; do not write a PR for
either until a maintainer says which they want.

**The important catch: someone already reported this.** Issue
[ggml-org/llama.cpp#26200](https://github.com/ggml-org/llama.cpp/issues/26200)
(opened 2026-07-27) describes exactly this mechanism, with a 4-core CPU and a
Mixture-of-Experts model, and measured **+40%**. Nobody replied, the bot
labelled it `stale`, and on **2026-09-11 the bot closed it**. So F51 is **not a
new finding** — but the report of it is now sitting closed and unseen. What
you have that it does not:

- it expects dense models to suffer "less so" — ours is a dense model and loses
  far more (2.65×, against their +40%)
- two different laptops with hybrid CPUs (P-cores + E-cores), both much worse
- the cost grows with thread count, which explains why their 4-core number is
  smaller

### F52 — OpenMP builds on Windows run at half speed with one thread

**What happens.** Windows 11 can "power throttle" a thread — run it slowly, or
move it to a slow E-core — to save battery (Microsoft calls this EcoQoS).
llama.cpp knows this, and has a function, `ggml_thread_apply_priority()`, that
tells Windows "do not throttle this thread" (added upstream in PR #12995). It is
called on the thread-pool path and on the OpenMP path when there is more than
one thread. The one case that skips it — an **OpenMP build asked to use exactly
one thread** (`-t 1`) — leaves that thread throttleable, and on our laptop
Windows throttles it: single-thread decode drops from ~21 tok/s to ~9,
erratically.

**Why it is skipped (the code).** In `ggml_graph_compute`, the OpenMP build does
this (master `1bc7a5af0`, lines ~3420–3442, shortened):

```c
if (n_threads > 1) {
    #pragma omp parallel num_threads(n_threads)
    {
        ...
        int ith = omp_get_thread_num();

        ggml_thread_apply_priority(threadpool->prio);      // every thread opts out
        if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
        }
        ggml_graph_compute_thread(&threadpool->workers[ith]);
    }
} else {
    atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
    ggml_graph_compute_thread(&threadpool->workers[0]);     // no opt-out here
}
```

The priority call lives *inside* the parallel region, and the parallel region
only exists when `n_threads > 1`. With one thread the `else` runs on the
caller's own thread, which never got the call.

**The change, exactly.** One added line, the first statement of the `else`:

```diff
     } else {
+        ggml_thread_apply_priority(threadpool->prio);
         atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
         ggml_graph_compute_thread(&threadpool->workers[0]);
     }
```

It is the same call, with the same argument, that the other branch already
makes. It is in `tokenscope/patches/07-omp-single-thread-prio.patch` (written
against `4d9176092`, where the hunk sits at line 3444; on master it is 3439).

**What it does, precisely.** With the default priority
(`GGML_SCHED_PRIO_NORMAL`, what `llama-bench` and `llama-cli` use unless you pass
`--prio`), on Windows the function only turns off power throttling for the
calling thread and returns — it does not raise the thread's priority. On Linux
and macOS it would set the scheduling priority only if you ask for a non-normal
one, so with defaults the line does nothing there. Things worth knowing before
a reviewer asks:

- **It runs on the caller's thread**, not a worker the library created. In
  `llama-bench` that is the main thread. If an application calls
  `ggml_graph_compute` from its own thread, that thread gets the opt-out (and,
  with `--prio` above normal, a raised priority). The `n_threads > 1` branch
  already does exactly this to the caller, because OpenMP's thread 0 *is* the
  caller — so the change makes `-t 1` behave like `-t 2` already does, not
  something new.
- **It is called on every graph compute** — once per token. That is also true of
  the existing `n_threads > 1` branch. It is one system call; on our numbers it
  costs nothing measurable.
- **It does not touch the thread-pool build** (`GGML_OPENMP=OFF`), which is
  inside a different `#ifdef` and already opts out.

**The measured effect.** With it, the OpenMP build at one thread runs at 20.74
tok/s, the same as the thread-pool build (20.95), and nothing changes at 8
threads.

**The catch here.** Our second laptop does **not** show the slowdown. We do not
know why. A maintainer will almost certainly ask, and "I don't know, it
reproduces on one of my two machines" is an honest and acceptable answer — but
you have to say it up front, not have it discovered.

**PR #16014** (open, idle since 2025-10-22) goes the other way — it would compile
the whole "do not throttle" code out of GCC/MinGW builds. A maintainer there
already pushed back on it. Our measurement is evidence of what that code is
worth on MinGW. Mentioning it is optional.

**Is `-t 1` important?** Honestly, not very — most people use more threads. So
F52 is a real but small-reach bug. That is fine for an issue; say it plainly.

---

## Part 2 — Which to do, in which order

1. **F51: open a new issue that links #26200.** Updated plan now that #26200 is
   closed. A comment on a closed issue only notifies its author and anyone
   subscribed; maintainers triaging new issues will not see it. A new issue
   that says, in your words, "#26200 reported this, was auto-closed with no
   reply, here is more data showing it is worse than it said" is not a
   duplicate — it is a re-open with evidence. Optionally, afterwards, leave a
   one-line comment on #26200 linking to your new issue so its author knows.
   No PR.
2. **F52: reproduce on current master (Part 4), then open an issue (Part 5).**
   `CONTRIBUTING.md`: "Bug-fix PRs must include a reproducible issue". The
   issue comes first even though you already have the fix.
3. **F52: then the PR (Part 6).** Open it once a maintainer has replied to the
   issue, or after about a week of silence (a bug fix does not need permission
   the way a feature does). Link the issue from it.

Do them one at a time, a few days apart. As a new contributor, `CONTRIBUTING.md`
asks you to keep **at most one open PR**.

There are other candidates in this repo (F20's naming defect — `ROADMAP.md`
0.1 calls it the easiest first contact — and F24, which has its own kit in
[`05-f24-filing-kit.md`](05-f24-filing-kit.md)). One thing at a time.

---

## Part 3 — How GitHub issues work, step by step

**Before anything:** you need a GitHub account (you have one: PS12007). Issues
and PRs are public and permanent — anyone can read them, and deleting later does
not really remove them. Your GitHub CLI (`gh`) is already logged in as PS12007
on machine 1, which is what `git push` will use; you will not need a password.

### Searching first (do this every time, just before posting)

1. Go to <https://github.com/ggml-org/llama.cpp/issues>.
2. Clear the search box's default `is:issue is:open` so closed issues show too,
   then search a few phrasings: `openmp mingw`, `libgomp`, `power throttling`,
   `single thread windows`, `ggml_thread_apply_priority`.
3. Also check <https://github.com/ggml-org/llama.cpp/pulls> the same way (again
   clear `is:open`) — a fix may already be in flight or already merged.
4. Check #26200 has not been reopened or answered since 2026-09-14.

From a terminal, the same searches are:

```
gh search issues --repo ggml-org/llama.cpp "power throttling"
gh search prs    --repo ggml-org/llama.cpp "ggml_thread_apply_priority"
```

### Opening a new issue (F51 and F52)

1. Go to the issues page and click **New issue**.
2. llama.cpp makes you **pick a template**. The ones that exist today:
   - **Bug (compilation)** — "Compile bug:" — not yours
   - **Bug (model use / results)** — titles start "Eval bug:" — the closest fit
     for both: the build works, the results (speed) are wrong
   - **Bug (misc.)** — "Misc. bug:" — acceptable fallback
   - Enhancement / Research / Refactor — not yours
3. The "Eval bug" form has these boxes. **Bold** ones are required:
   - **Name and Version** — the `build:` line `llama-bench` prints at the bottom
     of its table (commit and build number), plus compiler. Run
     `llama-cli --version` if you have it built, it prints the same.
   - **Operating systems** — tick Windows
   - **GGML backends** — tick CPU
   - **Hardware** — CPU model, cores, RAM (Part 8 has the list)
   - Models — optional; say synthetic, how made
   - **Problem description & steps to reproduce** — the heart of it: what you
     ran, what you saw, what you expected. Commands in code blocks.
   - First Bad Commit — optional; for F52 you can point to PR #12995 as the
     change that added the priority call without covering this branch, if you
     have checked that yourself. Otherwise leave it empty.
   - **Relevant log output** — paste the `llama-bench` tables
4. Read [`CONTRIBUTING.md`](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
   first. `AGENTS.md` asks first-time contributors to confirm they have.
5. **Title:** the template pre-fills the prefix ("Eval bug: "). After it, name
   the symptom and the condition in one line — what is slow, when, on what. No
   hype, no exclamation marks.
6. Click **Preview**, read it top to bottom, then **Create**.
7. Note the issue number GitHub gives you (e.g. #28300). You need it for the PR.

### Commenting on an existing issue

1. Open the issue, read the whole thing including recent comments.
2. The comment box is at the bottom. It accepts **Markdown**. Preview, then
   **Comment**. You can edit your own comment later (the `...` menu on it).

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
- `#26200` on its own becomes a link to that issue; `@name` notifies a person
  (do not @ maintainers to get attention)
- `AGENTS.md` asks for plain ASCII in code and commits: write `x` not `×`, `-`
  not `—`, `->` not `→`. Doing the same in issues and PRs is a good habit.

---

## Part 4 — Reproduce F52 on current master first (about an hour)

Our numbers are from `4d9176092`, which is weeks old and carries tokenscope's
instrumentation. A maintainer's first question is "does it happen on master?".
Answering "yes, here are numbers on `<today's commit>`" before they ask makes
the issue much stronger. It also gives you the clean clone Part 6 needs.

**Do not use `C:\-CS\TLI profiler\llama.cpp` for any of this.** That tree has
tokenscope patches applied to eight files; a PR from it would carry all of them.
Make a separate, clean copy.

### 4.1 Fork and clone (once)

1. Open <https://github.com/ggml-org/llama.cpp> and click **Fork** (top right).
   Owner: PS12007. Name: leave as `llama.cpp`. Leave "Copy the master branch
   only" ticked. Click **Create fork**. You now have
   `https://github.com/PS12007/llama.cpp` — your copy, which you can push to.
2. In Git Bash (the download is a few hundred MB):

   ```
   cd /c/-CS
   git clone https://github.com/PS12007/llama.cpp.git llama.cpp-pr
   cd llama.cpp-pr
   git remote add upstream https://github.com/ggml-org/llama.cpp.git
   git fetch upstream
   git remote -v
   ```

   `git remote -v` should show two names: `origin` = your fork (you push here),
   `upstream` = ggml-org (you only fetch from here).
3. Check your identity, which goes into the commit:

   ```
   git config user.name     # PS
   git config user.email    # the email on your GitHub account, or
                            # <id>+PS12007@users.noreply.github.com to keep it private
   ```

   GitHub shows your noreply address under Settings -> Emails. Set it for this
   repo only with `git config user.email "<address>"`.

### 4.2 Build the two arms, stock (no change yet)

Start from the newest master:

```
git checkout -B repro upstream/master
git log -1 --oneline          # write this commit down; it goes in the issue
```

**MSVC** (from a "Developer PowerShell for VS" or any shell where `cmake` finds
Visual Studio):

```
cmake -S . -B build-omp
cmake --build build-omp   --config Release --target llama-bench -j 8
cmake -S . -B build-noomp -DGGML_OPENMP=OFF
cmake --build build-noomp --config Release --target llama-bench -j 8
```

Binaries land in `build-omp\bin\Release\llama-bench.exe` and
`build-noomp\bin\Release\llama-bench.exe`.

**GCC** (optional second compiler; in the **MSYS2 UCRT64** shell, or Git Bash
with `export PATH="/c/msys64/ucrt64/bin:$PATH"` first — without that, Windows
pops "Entry Point Not Found"):

```
cmake -S . -B build-gcc-omp   -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-gcc-omp   --target llama-bench
cmake -S . -B build-gcc-noomp -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF
cmake --build build-gcc-noomp --target llama-bench
```

Showing it on MSVC alone is enough — MSVC is the default Windows build and the
one maintainers are most likely to try.

### 4.3 Measure

Same conditions for every run: on the charger, same Windows power mode, nothing
else running, don't touch the laptop. Write those conditions down — they are
the first suspects for why machine 2 differs. `powercfg /getactivescheme` prints
the power plan.

Alternate the two builds so drift hits both equally (Git Bash):

```
M="/c/-CS/TLI profiler/models/mid.gguf"
for i in 1 2 3 4 5 6; do
  ./build-omp/bin/Release/llama-bench.exe   -m "$M" -p 0 -n 64 -t 1
  ./build-noomp/bin/Release/llama-bench.exe -m "$M" -p 0 -n 64 -t 1
done 2>&1 | tee repro-t1-stock.txt
```

**What you expect** if master still has the bug on this machine: the OpenMP
build mostly around 9 tok/s with big `±`, the no-OpenMP build steady around 21.
If both are ~21, it no longer reproduces on master — stop and find out why
before filing (Windows update? something upstream changed?).

### 4.4 Apply the change and measure again

Open `ggml/src/ggml-cpu/ggml-cpu.c` in your editor. Search for
`#pragma omp parallel num_threads(n_threads)`. Scroll down to the `} else {`
that closes that `if`. Type the new line as the first line inside the `else`,
indented 8 spaces, like its neighbours:

```c
        ggml_thread_apply_priority(threadpool->prio);
```

Type it yourself rather than applying patch 07 — it is one line, and you should
know exactly where it went. Then check git sees exactly that and nothing else:

```
git diff --stat      # expect: 1 file changed, 1 insertion(+)
git diff             # expect: the one + line, in the else branch
git diff --check     # expect: no output (no trailing spaces)
```

If `--stat` says hundreds of lines changed, your editor rewrote the line
endings to CRLF. Undo with `git checkout -- ggml/src/ggml-cpu/ggml-cpu.c`, set
the editor to keep LF, and redo the edit.

Rebuild only the OpenMP arm and repeat, adding an 8-thread check:

```
cmake --build build-omp --config Release --target llama-bench -j 8
for i in 1 2 3 4 5 6; do
  ./build-omp/bin/Release/llama-bench.exe   -m "$M" -p 0 -n 64 -t 1
  ./build-noomp/bin/Release/llama-bench.exe -m "$M" -p 0 -n 64 -t 1
done 2>&1 | tee repro-t1-fixed.txt
./build-omp/bin/Release/llama-bench.exe -m "$M" -p 0 -n 64 -t 8 -r 10
```

**Expect:** the OpenMP build now ~21 and steady at `-t 1`, and `-t 8` about the
same as before the change. For the before/after at `-t 8`, build a stock copy
too (`git stash`, build into `build-omp-stock`, `git stash pop`), or reuse the
stock `-t 8` number from our F52 data and say so.

Keep `repro-t1-stock.txt` and `repro-t1-fixed.txt` (copy them into
`tokenscope/results/` — they are data, not throwaway). Their tables go in the
issue.

**Optional but strong:** a real model too. `models/qwen-q4km.gguf` is already
here. One synthetic F32 model invites "does it happen with a real model?".

---

## Part 5 — What each issue must contain (checklists and facts, not text)

Tick every item. If you cannot explain an item in your own words, leave it out
rather than paste something you do not understand.

### For the F51 issue (linking #26200)

- [ ] **Link #26200 in the first lines**, and say plainly that it was
  auto-closed with no reply and that you are adding data, not a new mechanism.
- [ ] **What you ran it on.** Both machines:
  - machine 1: Intel i7-14700HX (8 P-cores + 12 E-cores, 28 threads), 16 GB, Windows 11 build 26200
  - machine 2: Intel i7-1255U (2 P-cores + 8 E-cores, 12 threads), 16 GB, Windows 11 build 26200
  - compiler: GCC 16.1.0 (MSYS2 UCRT64, "Rev5") on both; MSVC 14.44 on machine 1 for comparison
  - llama.cpp commit `4d9176092` — **and say it is older than current master**
    (or re-run the 8-thread pair from Part 4 on master and quote that)
- [ ] **What model.** A synthetic dense F32 model, 220 M parameters, 24 layers,
  made with `tools/make_tiny_model.py` (`--layers 24 --embd 768 --heads 12
  --heads-kv 4 --vocab 8192`). Say it is synthetic.
- [ ] **The command**, exactly: `llama-bench -m mid.gguf -p 0 -n 64 -t 8`,
  built twice, once with `-DGGML_OPENMP=OFF`. Matches #26200's own repro.
- [ ] **The headline numbers** (decode, tokens/s — pick the few that matter):

  | | machine 1 | machine 2 | source |
  |---|---|---|---|
  | GCC `GGML_OPENMP=OFF` vs ON, 8 threads | **+165%** [+158, +173] | **+205%** [+200, +210] | FINDINGS F51; `results/10-machine1-gcc/f51-t8.json`; `results/08-windows-gcc/f27-t8.json` |
  | same, all threads | 4.5× (28 threads) | +217% (12 threads) | `sweep.json`; `f27-t12.json` |
  | MSVC `GGML_OPENMP=OFF` vs ON, 8 threads | −0.5% / +1.4% (a tie) | — | `sweep.json` |
  | prefill, 8 threads | +79% | +114% | same files |

- [ ] **The two things #26200 does not already say**: dense models are hit
  *harder* here, not less; and the extra cost per barrier grows with thread count
  (machine 1: ~26 µs at 2 threads, ~50 at 4, ~89 at 8, ~155 at 16, ~252 at 28,
  from 412 barriers per token).
- [ ] **That the spin settings did nothing** (`OMP_WAIT_POLICY=ACTIVE`,
  `GOMP_SPINCOUNT=INFINITE`, all within ~3% of default).
- [ ] **That there is no escape but `GGML_OPENMP=OFF`** for GCC users, because
  `GGML_OPENMP_FETCH` requires Clang.
- [ ] **What you are asking for**, in one sentence: e.g. whether maintainers
  would take a docs note or a MinGW default. Ask, don't propose a PR yet.
- [ ] **Short.** #26200 already explains the mechanism; link it, don't
  re-explain it.
- [ ] **AI disclosure.** `CONTRIBUTING.md`: "Undisclosed AI usage may result in
  your account being permanently banned". The measurements and analysis were
  done with an AI coding assistant; the post is yours. Say so in one line.

### For the F52 issue

- [ ] **Search first** (Part 3). If someone has reported it since, comment there.
- [ ] **Confirmed on current master** (Part 4) — give the commit hash you built.
  If you skipped Part 4, say the numbers are from `4d9176092`.
- [ ] **What you ran it on** — machine 1 reproduces; say machine 2 does not.
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

  (Replace with or add your Part 4 master numbers.)
- [ ] **The exact repro command** and both CMake configure lines.
- [ ] **The cause, pointing at code**: `ggml_graph_compute` in
  `ggml/src/ggml-cpu/ggml-cpu.c`; the `n_threads > 1` branch calls
  `ggml_thread_apply_priority` inside the parallel region; the `else` does not.
  Give line numbers *from the commit you name* and a GitHub permalink (on the
  file page, press `y` to pin the URL to a commit, then click a line number,
  shift-click another to select a range).
- [ ] **The fix**, as a three-line diff snippet (Part 1). Say you are happy to
  open a PR.
- [ ] **The test question.** `CONTRIBUTING.md` wants a regression test with bug
  fixes, but this is a Windows scheduler effect with no automated way to catch
  it. Say that, and ask whether they want one. Do not add a file under `tests/`
  (`AGENTS.md` forbids that without approval).
- [ ] **What you do not know**: why machine 2 does not reproduce; that `-t 1` is
  an uncommon configuration.
- [ ] **Optional:** that PR #16014 would remove this protection from MinGW
  builds, and your numbers show what it is worth there.
- [ ] **AI disclosure**, one line.

---

## Part 6 — The F52 pull request, step by step

Only after the issue exists (Part 2). This continues from the clone in Part 4.

### 6.1 Make a clean branch from fresh master

```
cd /c/-CS/llama.cpp-pr
git stash list                    # if Part 4's edit is stashed, drop it: git stash drop
git checkout -- .                 # throw away the repro edit
git fetch upstream
git checkout -b ggml-cpu-omp-t1-prio upstream/master
```

Branch name: anything short and descriptive. It shows in the PR but does not
matter much.

### 6.2 Make the change again

Same one line, same place as Part 4.4. Then:

```
git diff --stat      # 1 file changed, 1 insertion(+)
git diff --check     # no output
```

### 6.3 Build and test

`CONTRIBUTING.md` lists what it expects before a PR. For this change:

| what it asks | what that means here |
|---|---|
| run the full CI locally (`ci/README.md`) | `ci/run.sh` is a Linux script. On Windows, do the nearest honest thing: build **everything** (not just `llama-bench`) with MSVC, run `ctest`, and say in the PR that's what you ran |
| perplexity and performance not harmed | performance: your Part 4 table. Perplexity: the change cannot alter results (it sets a thread's power mode, not any maths); say that rather than running it — or run `llama-perplexity` on `qwen-q4km.gguf` before/after if you want to be thorough |
| modified ggml source -> `test-backend-ops` | it compares backends' operators; needs two backends, and you changed no operator. Run it on CPU anyway if it builds; say it's not really applicable |
| bug fix -> regression test | not automatable (Part 5); point to the issue where you asked |

Full build and tests (MSVC):

```
cmake -S . -B build
cmake --build build --config Release -j 8
ctest --test-dir build -C Release --output-on-failure
```

Some tests need a model download or network and may skip or fail for reasons
that aren't yours. Run the same `ctest` on a stock build (no change) if anything
fails, to show it failed before too.

It is also worth building with GCC (Part 4.2 commands, whole project), because
the line compiles on a path that has broken MinGW builds before (issue #14953).
A Linux build is not needed — on Linux, with default priority, the line does
nothing, and upstream CI will build it there anyway.

### 6.4 Commit — in your words

```
git add ggml/src/ggml-cpu/ggml-cpu.c
git status            # only that file is staged
git commit            # opens an editor; write the message, save, close
```

What llama.cpp expects of the message (write it yourself; `AGENTS.md` prohibits
AI-written commit messages):

- **first line**: `<module> : <what changed>` — lowercase, short, no full stop.
  The module for this file is `ggml-cpu` (or `ggml`). Look at
  `git log --oneline upstream/master -- ggml/src/ggml-cpu/ggml-cpu.c` for how
  others title changes to this file.
- a blank line, then optionally a sentence or two of why, and `Fixes #<issue>`
- ASCII only
- The maintainers squash-merge and rewrite the title with the PR number, so
  don't polish it.
- AI trailer: `AGENTS.md` asks for `Assisted-by: <assistant>` (never
  `Co-authored-by:`) when an agent commits for you. You're committing yourself,
  so the required disclosure is the PR template's AI line. Adding the trailer as
  well is your call.

Check it: `git log -1` and `git show --stat`.

### 6.5 Push to your fork

```
git push -u origin ggml-cpu-omp-t1-prio
```

This sends the branch to **PS12007/llama.cpp**, not to ggml-org. Nothing is
public on llama.cpp yet. If git asks for credentials, run `gh auth setup-git`
once and push again.

### 6.6 Open the pull request (web page)

1. Open <https://github.com/PS12007/llama.cpp>. A yellow bar says
   "ggml-cpu-omp-t1-prio had recent pushes" — click **Compare & pull request**.
   (If it's gone: go to <https://github.com/ggml-org/llama.cpp/compare>, click
   "compare across forks".)
2. **Check the four dropdowns at the top**, the most common mistake:
   `base repository: ggml-org/llama.cpp` · `base: master` ←
   `head repository: PS12007/llama.cpp` · `compare: ggml-cpu-omp-t1-prio`.
   The page should say "Able to merge".
3. Scroll down to the **Files changed** preview: exactly one file, `+1 −0`. If
   there is anything else, stop and fix the branch.
4. **Title**: same format as the commit (`ggml-cpu : ...`).
5. **Description**: GitHub pre-fills llama.cpp's template. Fill it in yourself:
   - **Overview** — what the PR does and why, in two or three sentences, and a
     link: `Fixes #<issue number>` (that keyword makes GitHub close the issue
     automatically when the PR is merged)
   - **Additional information** — the before/after table, what you built and
     tested (6.3), the machine-2 caveat, the test question. Or point to the
     issue if it's all there.
   - **Requirements** — do **not** delete this section (the template says the PR
     may be rejected). Keep the contributing-guidelines line. On the AI line
     write `YES` and one sentence on how AI was used (e.g. the investigation
     and measurements were done with an AI coding assistant; you reviewed and
     typed the change and wrote the text).
6. Leave **"Allow edits by maintainers"** ticked — `CONTRIBUTING.md` asks for
   it; it lets a reviewer push a small fix to your branch.
7. **Create pull request** (not "draft" — draft means "not ready for review").
   Use **Preview** on the description first.

---

## Part 7 — After you post (issues and PR)

- **Silence is normal.** llama.cpp gets hundreds of issues and PRs; #26200 got
  none and the bot closed it. Don't bump or @ maintainers. One polite follow-up
  after a couple of weeks, at most. The stale bot marks things after about a
  month of inactivity and closes them 14 days later — a real comment resets it.
- **CI.** For a first-time contributor, GitHub holds the PR's CI runs until a
  maintainer clicks "Approve and run workflows". Once they run, some red jobs
  may be unrelated to you (flaky or broken on master). Open a failed job, and if
  it isn't about `ggml-cpu.c`, say so briefly rather than trying to fix it.
- **If a reviewer asks for a change**: edit, then

  ```
  git add ggml/src/ggml-cpu/ggml-cpu.c
  git commit
  git push
  ```

  The PR updates itself. Don't open a new PR. Don't squash your commits
  yourself — they squash on merge.
- **If they ask you to update to latest master** (or the PR sits long enough to
  conflict):

  ```
  git fetch upstream
  git rebase upstream/master
  git push --force-with-lease
  ```

  `--force-with-lease` is needed because rebase rewrites your branch; the
  "with-lease" part refuses to overwrite anything a reviewer pushed that you
  don't have.
- **If someone asks a question**, answer it yourself, briefly. If you need to
  re-run something, the commands are in Part 4, FINDINGS F51/F52 and
  `tools/runtime_sweep.py`.
- **If someone says "can't reproduce"** (likely for F52): that is information,
  not rejection. Ask what CPU, Windows build and power mode they tried.
- **If they close it or prefer another approach**, that is their call; thank
  them and let it go.
- **After a merge**: delete the branch (GitHub offers a button on the PR), and
  locally `git checkout master && git branch -D ggml-cpu-omp-t1-prio`. Keep the
  fork for next time or delete it in its Settings.
- **Record what happened** in `docs/HANDOFF.md` (issue and PR links, dates, any
  reply), so the next session doesn't re-file it.

---

## Part 8 — Can you answer these without help? If not, do not post yet

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
   `ggml_thread_apply_priority` do about it at the default priority?
7. Why does only the one-thread OpenMP case miss the call? Point at the branch.
8. Why does it show up with both MSVC and GCC?
9. Why might your second laptop not show it? (You do not need the answer — you
   need to be able to say you do not know, and name what differs.)
10. Why does the fix not change anything at 8 threads?
11. Which thread does the new call affect, and is that different from what
    `-t 2` already does?
12. Why is there no regression test, and what did you test instead?
13. What does the line do on Linux?

If any of these stumps you, the relevant sections are FINDINGS F51/F52 and
Part 1 above. Asking an AI to *explain* them to you is fine and is exactly what
`AGENTS.md` lists as permitted use — "learning, exploration, and understanding
the codebase". Asking it to *write your reply* is not.

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
| issue | a public report or question on the project's GitHub |
| fork | your own copy of the repository on GitHub, which you can push to |
| `origin` / `upstream` | git's names here for your fork / the real ggml-org repo |
| branch | a named line of commits; one branch per PR |
| pull request (PR) | a request for the maintainers to merge a branch from your fork |
| CI | the automatic builds and tests GitHub runs on every PR |
| rebase | replay your commits on top of newer master |
| squash-merge | the maintainers merge your PR as a single commit |
| stale bot | GitHub automation that labels, then closes, inactive issues and PRs |
