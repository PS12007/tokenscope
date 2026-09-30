# 11 — F52 on current master: the fix verified (2026-09-14)

**Short version: the one-line fix works on current llama.cpp master, with both
MSVC and GCC.** When Windows throttles the thread, stock OpenMP builds at `-t 1`
fall to 8–11 tok/s; with the fix they hold 19–20 tok/s, the same as the
`GGML_OPENMP=OFF` build. `-t 8` is unchanged. The full MSVC build succeeds and
`ctest` passes 43/43. **Nothing is committed or pushed.**

Raw data, the launcher source and the benchmark script:
[`results/11-master-f52/`](../results/11-master-f52/). The draft issue, PR
and commit message, with review notes: [`12-f52-draft-review.md`](12-f52-draft-review.md).

---

## 1. Where things stand

| item | state |
|---|---|
| fork | `github.com/PS12007/llama.cpp`, identical to ggml-org master `1bc7a5a` on 2026-09-14 |
| clone | `C:\-CS\llama.cpp-pr` — shallow (depth 1), 209 MB, remotes `origin` = fork, `upstream` = ggml-org. Builds already deleted |
| the edit | **applied, uncommitted** in that clone: `git diff --stat` = `1 file changed, 1 insertion(+)` |
| commit / push / issue / PR | **none done.** You write the commit message and do these yourself |
| commit identity in the clone | `PS <priyanshcan@gmail.com>` from your global git config. That email becomes public on GitHub once pushed; use your GitHub noreply address if you'd rather it didn't (`git config user.email ...` inside the clone) |
| #26200 (F51) | closed by the stale bot on 2026-09-11, not locked — see doc 10, Part 2 |

### When you are ready

```
cd /c/-CS/llama.cpp-pr
git fetch --depth 1 upstream master        # optional: check master has not moved/fixed it
git diff                                   # still the one line
git checkout -b ggml-cpu-omp-t1-prio
git commit -am "<your message>"
git push -u origin ggml-cpu-omp-t1-prio
```

If master has moved, the simplest route is to delete the clone and redo the
one-line edit on a fresh one (doc 10, Part 4.1). When you are finished:
`rm -rf /c/-CS/llama.cpp-pr` (in Git Bash) — nothing else depends on it.

Order per `CONTRIBUTING.md`: **issue first**, then the PR with `Fixes #<issue>`.

---

## 2. The change

`ggml/src/ggml-cpu/ggml-cpu.c`, `ggml_graph_compute()`, the `#ifdef
GGML_USE_OPENMP` block (master `1bc7a5a`, line 3440 after the edit):

```diff
@@ -3437,6 +3437,7 @@ enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cpl
             ggml_graph_compute_thread(&threadpool->workers[ith]);
         }
     } else {
+        ggml_thread_apply_priority(threadpool->prio);
         atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
         ggml_graph_compute_thread(&threadpool->workers[0]);
     }
```

Also saved as [`results/11-master-f52/fix.diff`](../results/11-master-f52/fix.diff).
`git diff --check` is clean; the file is ASCII with LF endings.

**Checked on master before editing:** the priority call sites are identical to
the pinned `4d9176092` — the OpenMP `n_threads > 1` branch (line 3433), the
threadpool's secondary threads (3246), the threadpool main thread at creation
(3377) and on resume (3298). Only the OpenMP `else` lacks it.

**What the call does on Windows at default priority** (lines 2568–2590): if the
priority is not LOW, it calls `SetThreadInformation(ThreadPowerThrottling)` with
`StateMask = 0` — "do not throttle this thread" — then returns early for
NORMAL without touching `SetThreadPriority`. On Linux/macOS at NORMAL it
returns immediately. So with defaults, the fix only opts the calling thread out
of Windows power throttling (EcoQoS).

---

## 3. How it was tested

Machine 1: i7-14700HX (8P + 12E, 28 threads), 16 GB, Windows 11 Home 10.0.26200,
on AC power, Balanced plan, no power-mode overlay. Model `models/mid.gguf`
(synthetic dense F32, 220 M params, 24 layers). `llama-bench -p 0 -n 64`, 5 reps
per run unless noted.

| arm | how built |
|---|---|
| msvc-stock | MSVC 19.44 (VS 2022 14.44), Ninja, Release, `GGML_OPENMP=ON` (default), before the edit |
| msvc-fixed | same, after the edit |
| msvc-noomp | MSVC, `-DGGML_OPENMP=OFF`, before the edit (the edit is inside `#ifdef GGML_USE_OPENMP`, so it can't affect this arm) |
| gcc-stock / gcc-fixed / gcc-noomp | GCC 16.1.0 (MSYS2 UCRT64 Rev5), Ninja, Release, same three |

MSVC builds run under `vcvars64.bat`; GCC binaries need
`C:\msys64\ucrt64\bin` first on `PATH`.

### 3.1 Forcing the throttle: `ecoqos_run`

The slowdown is **intermittent** — Windows decides when to throttle (see 4.1
and 4.2). To test deterministically, a 40-line launcher
([`results/11-master-f52/ecoqos_run.c`](../results/11-master-f52/ecoqos_run.c))
starts the command suspended, sets process-level EcoQoS
(`SetProcessInformation(ProcessPowerThrottling, EXECUTION_SPEED on)`), resumes
it and waits. That is the throttling state behind Task Manager's "Efficiency
mode" (which additionally lowers base priority; the launcher does not). A
thread-level opt-out overrides the process setting, which is exactly what
`ggml_thread_apply_priority` does — so under the launcher a thread that makes
the call runs normally, and one that doesn't is throttled.

Build: `cl /O2 /W4 ecoqos_run.c` under `vcvars64`. Use:
`ecoqos_run.exe llama-bench.exe -m mid.gguf -p 0 -n 64 -t 1`.

---

## 4. Results

### 4.1 First look: stock, normal conditions — did NOT reproduce

Six interleaved rounds each, before any edit:

| `-t 1` | tok/s per run |
|---|---|
| msvc-stock | 21.14, 21.75, 21.49, 21.54, 21.57, 21.52 |
| msvc-noomp | 21.58, 21.54, 21.51, 21.63, 21.49, 21.55 |
| gcc-stock | 22.37, 22.42, 22.57, 22.46, 22.36, 22.32 |
| gcc-noomp | 22.45, 22.65, 22.47, 22.47, 22.16, 22.35 |

And the **exact old binary** that measured 9.55/9.34/9.28… on 2026-09-10
(`llama.cpp/build-gcc-omp-off`, commit `4d9176092`): 22.26, 22.76, 22.71.

So the absence was conditions, not code: same binary, same machine, different
day. What differed is unknown (the VS Code window was in the foreground and
the Settings app was open; the laptop was in active use shortly before).
Files: `repro-t1-stock.jsonl`, `repro-gcc-t1-stock.jsonl`.

### 4.2 The matrix — it came back, and the fix holds

`matrix.sh`: 3 interleaved rounds under `ecoqos_run`, then 2 rounds normal, then
2 rounds of `-t 8` both ways. File: `f52-matrix.jsonl` (46 runs).

| `-t 1`, decode tok/s | stock | **fixed** | noomp |
|---|---|---|---|
| MSVC, EcoQoS forced | 7.89, 8.06, 8.06 → **8.0** | 20.04, 21.52, 19.44 → **20.3** | 19.69, 20.34, 19.44 → 19.8 |
| GCC, EcoQoS forced | 7.48, 7.54, 7.92 → **7.6** | 19.45, 19.17, 19.14 → **19.3** | 19.96, 19.15, 19.18 → 19.4 |
| MSVC, normal | 8.51 ±2.61, 9.85 ±3.41 → **9.2** | 19.46 ±0.21, 19.11 ±0.44 → **19.3** | 19.66, 19.60 → 19.6 |
| GCC, normal | 9.49 ±3.23, 12.78 ±6.01 → **11.1** | 19.38 ±0.30, 19.39 ±0.24 → **19.4** | 19.72, 19.03 → 19.4 |

(± is llama-bench's within-run standard deviation over 5 reps.)

- **The natural slowdown reappeared** in the "normal" rounds, ~15 minutes into
  an unattended run: stock erratic at 9–11 with large spread, exactly the
  2026-09-10 signature. The fixed build beside it stayed steady.
- **Under forced EcoQoS it's deterministic**: stock always ~8, fixed always
  matches noomp.
- Absolute speed was ~19.5 in this run against ~21.5 in 4.1 — the machine's
  usual drift (M10 in HANDOFF); compare arms within a table, not across.

### 4.3 `-t 8` — no effect

The matrix's two rounds showed MSVC fixed 5–8% *below* stock, but stock always
ran first. Re-run with 16 runs in ABBA order, 10 reps each
(`f52-t8-abba.jsonl`):

| MSVC `-t 8` | runs | mean |
|---|---|---|
| stock | 41.3 39.8 39.3 44.0 39.5 43.2 43.3 42.8 | 41.66 |
| fixed | 43.1 42.6 43.4 43.7 41.9 42.8 42.4 43.0 | 42.87 (+2.9%) |

The ranges overlap and the sign flipped from the first run: noise and order,
as expected — at `-t 8` the edited branch never runs. GCC in the matrix:
normal 16.53 → 16.55, EcoQoS 16.43 → 16.80.

### 4.4 Build and tests

- **MSVC full build** of the fixed tree (all 253 targets: tools, examples,
  tests): success, no errors. One configure-time note unrelated to the change
  (web UI `dist.tar.gz.sha256` download failed; OpenSSL not found).
- **`ctest --test-dir build-omp -j 8`: 100% passed, 43/43**, including
  `test-barrier` and `test-thread-safety`. `test-backend-ops` is not among the
  43. Log: `ctest-omp.log`.
- **GCC**: `llama-bench` target built with the edit (compiles `ggml-cpu.c`) and
  ran — see 4.2. The full project was not built with GCC.
- **Not run:** `test-backend-ops` (compares backends; only CPU here, and no
  operator changed), `llama-perplexity` (the change sets a thread power state
  and cannot alter results), Linux (the call is a no-op there at default
  priority), a real (non-synthetic) model.

---

## 5. What this changes about the F52 story

- **Confirmed on current master**, not just the pinned commit, with both
  compilers. The draft's numbers from `4d9176092` can be replaced by these.
- **Intermittent on the reproducing machine too**, not only "machine 2 does not
  reproduce". Say so in the issue — a maintainer who tries once and sees 21
  tok/s will otherwise close it.
- **There is now a deterministic repro** (3.1's launcher). It turns "sometimes,
  on my laptop" into "every time, on this laptop"; it has not been tried on
  another machine, but nothing in it is specific to this one. It is the single
  strongest thing to put in the issue. Offer the launcher source; it is
  small enough to paste or link.
- Still unknown: what makes Windows throttle a single-threaded console process
  on its own (foreground state, focus, activity, the hybrid scheduler).
  Machine 2 was not re-tested; the launcher would likely reproduce there too.

## 6. Leftovers

- `C:\-CS\llama.cpp-pr` (209 MB) stays until you commit and push, then delete it.
- `results/11-master-f52/` and this doc are uncommitted in the tokenscope repo,
  as are the doc 10 edits and doc 12.
- `results/11-master-f52/llama-bench-runs.txt` (added 2026-09-16) is the
  pasteable log output for the issue's required "Relevant log output" field:
  per-run numbers for every arm, the ABBA re-run, one raw jsonl row, the ctest
  line. Markdown tables would need a rebuild of the MSVC arms (~15 min, ~500 MB).
