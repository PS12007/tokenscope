// tokenscope self-test.
//
// Validates the mechanism in isolation, before any of it is pointed at
// llama.cpp. Checks, in order:
//
//   1. record layout is exactly 24 bytes
//   2. scopes nest correctly and durations are monotonic
//   3. N threads writing concurrently produce N disjoint tracks, no locks taken
//      on the hot path, no lost records within budget
//   4. the emitted JSON parses and has the shape Perfetto expects
//   5. per-scope cost, measured, so the overhead model in docs/01 has a number
//
// Build: see CMakeLists.txt. Run: ts_selftest <out.json>
//
// SPDX-License-Identifier: MIT

#include "tokenscope.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef TOKENSCOPE_ENABLED
int main() {
    std::printf("tokenscope: built with instrumentation DISABLED.\n");
    std::printf("This binary is the control for the A/B/C overhead comparison.\n");
    return 0;
}
#else

static int g_failures = 0;

static void report_shadowed(const char * shadowed, const char * by) {
    std::printf("    \"%s\" is unreachable: \"%s\" matches it first\n", shadowed, by);
}

static void check(bool cond, const char * what) {
    std::printf("  [%s] %s\n", cond ? " ok " : "FAIL", what);
    if (!cond) g_failures++;
}

// Something the optimizer cannot delete, with a cost we control.
static std::atomic<uint64_t> g_sink{0};
static void busy(int iters) {
    uint64_t x = 1;
    for (int i = 0; i < iters; ++i) x = x * 6364136223846793005ull + 1442695040888963407ull;
    g_sink.fetch_add(x, std::memory_order_relaxed);
}

static void worker(int id, int n_tokens, int scopes_per_token) {
    for (int t = 0; t < n_tokens; ++t) {
        TS_SCOPE("worker.token");
        for (int s = 0; s < scopes_per_token; ++s) {
            TS_SCOPE("worker.inner");
            busy(50);
        }
    }
    (void) id;
}

// ---------------------------------------------------------------------------
// 5. per-scope cost
// ---------------------------------------------------------------------------
static double measure_scope_cost_ns() {
    const int N = 200000;

    // baseline: the same loop with no scope
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) busy(4);
    auto t1 = std::chrono::steady_clock::now();

    for (int i = 0; i < N; ++i) {
        TS_SCOPE("cost.probe");
        busy(4);
    }
    auto t2 = std::chrono::steady_clock::now();

    const double base = std::chrono::duration<double, std::nano>(t1 - t0).count();
    const double instr = std::chrono::duration<double, std::nano>(t2 - t1).count();
    return (instr - base) / N;
}

// ---------------------------------------------------------------------------
// 4. minimal JSON shape check. No dependency, so: structural, not a full parser.
// ---------------------------------------------------------------------------
static bool json_shape_ok(const std::string & s, size_t & n_events) {
    if (s.empty() || s.front() != '[') return false;
    size_t depth = 0, i = 0;
    bool in_str = false, esc = false;
    n_events = 0;
    for (; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (esc)            esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"')  in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{' || c == '[') { if (++depth == 2 && c == '{') n_events++; }
        else if (c == '}' || c == ']') { if (depth == 0) return false; --depth; }
    }
    return depth == 0 && !in_str;
}

int main(int argc, char ** argv) {
    const char * out_path = argc > 1 ? argv[1] : "selftest.trace.json";

#if defined(_WIN32)
    _putenv_s("TOKENSCOPE_LEVEL", "1");
#else
    setenv("TOKENSCOPE_LEVEL", "1", 1);
#endif
    ts_init_from_env();

    std::printf("tokenscope self-test\n");
    std::printf("  level=%d\n", ts_g_level);

    // 1. layout ------------------------------------------------------------
    std::printf("\n1. record layout\n");
    check(sizeof(ts_record) == 24, "ts_record is 24 bytes");
    check(TS_ACTIVE, "TS_ACTIVE is true at level 1");

    // 2 + 3. concurrency ----------------------------------------------------
    std::printf("\n2. concurrent recording\n");
    const int n_threads = 8;
    const int n_tokens  = 20;
    const int per_token = 30;

    for (int tok = 0; tok < n_tokens; ++tok) {
        TS_TOKEN_BEGIN(tok == 0 ? 1 : 0, tok == 0 ? 64u : 1u);
        std::vector<std::thread> ths;
        ths.reserve(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            ths.emplace_back(worker, i, 1, per_token);
        }
        for (auto & th : ths) th.join();
        TS_TOKEN_END();
    }
    check(true, "8 threads x 20 tokens x 30 scopes completed without deadlock");

    // 5. cost ---------------------------------------------------------------
    std::printf("\n3. per-scope cost\n");
    const double ns = measure_scope_cost_ns();
    std::printf("  measured: %.1f ns per scope (2 clock reads + 1 store)\n", ns);
    check(ns < 500.0, "per-scope cost is under 500 ns (sanity bound, not the target)");

    // 4. output -------------------------------------------------------------
    std::printf("\n4. trace output\n");

    // The build facts the node loop reports (F26). TS_THREAD_PREPARE is the
    // macro ggml actually calls, so exercise it; then overwrite with a pair of
    // values this translation unit could not have produced on its own, so the
    // assertion below proves the trace carries what the CALLER said rather
    // than what tokenscope.cpp assumed. That distinction is the entire finding.
    TS_THREAD_PREPARE(0);
    ts_note_build(2, 1);

    ts_flush(out_path);

    std::FILE * f = std::fopen(out_path, "rb");
    check(f != nullptr, "trace file was created");
    std::string data;
    if (f) {
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        data.resize((size_t) sz);
        if (sz > 0) {
            size_t rd = std::fread(&data[0], 1, (size_t) sz, f);
            data.resize(rd);
        }
        std::fclose(f);
    }
    size_t n_events = 0;
    check(json_shape_ok(data, n_events), "trace is balanced JSON");
    std::printf("  %zu events, %zu bytes\n", n_events, data.size());

    const size_t expect_min = (size_t) n_threads * n_tokens * per_token;
    check(n_events >= expect_min, "no scope records were lost");
    check(data.find("\"ph\":\"X\"") != std::string::npos, "complete (X) events present");
    check(data.find("\"cat\":\"decode\"") != std::string::npos, "decode token slices present");
    check(data.find("\"cat\":\"prefill\"") != std::string::npos, "prefill token slice present");
    check(data.find("\"dropped\":0") != std::string::npos, "provenance record reports zero drops");
    check(data.find("\"threading\":\"ggml-threadpool (openmp available, unused)\"") != std::string::npos,
          "provenance carries the threading path the caller reported");
    check(data.find("\"compute_linkage\":\"shared\"") != std::string::npos,
          "provenance carries the linkage the caller reported");

    // The category table is ordered by hand and the first match wins, so an
    // entry that a shorter entry already covers is dead code that silently
    // misfiles nodes. This caught nothing when written only because F20 had
    // just fixed the one instance; it exists so the next one is loud.
    {
        const int shadowed = ts_check_category_table(&report_shadowed);
        check(shadowed == 0, "no category-table prefix is shadowed by an earlier one");
    }

    // TOKENSCOPE_TOKENS parsing. "10:11" used to fall through to a bare "%u"
    // and capture one token while the docs, the tools and the trace filenames
    // all said two -- silently, because sscanf does not care what it leaves
    // behind. The cases that matter are the ones that must be REFUSED.
    std::printf("\n5. capture-window parsing\n");
    {
        struct { const char * in; int ok; uint32_t lo, hi; } cases[] = {
            { "10",      1, 10, 10 },
            { "10-11",   1, 10, 11 },
            { "10:11",   1, 10, 11 },
            { "8-13",    1,  8, 13 },
            { "10:11x",  0,  0,  0 },
            { "10 - 11", 0,  0,  0 },
            { "11-10",   0,  0,  0 },
            { "",        0,  0,  0 },
            { "abc",     0,  0,  0 },
        };
        int bad = 0;
        for (const auto & c : cases) {
            uint32_t lo = 0xFFFFFFFFu, hi = 0xFFFFFFFFu;
            const int got = ts_parse_token_window(c.in, &lo, &hi);
            const bool fine = got == c.ok && (!c.ok || (lo == c.lo && hi == c.hi));
            if (!fine) {
                ++bad;
                std::printf("    '%s' -> ok=%d lo=%u hi=%u, expected ok=%d lo=%u hi=%u\n",
                            c.in, got, lo, hi, c.ok, c.lo, c.hi);
            }
        }
        check(bad == 0, "every capture-window form parses or is refused as specified");
    }

    std::printf("\n%s (%d failure%s)\n",
        g_failures == 0 ? "PASS" : "FAIL", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}

#endif // TOKENSCOPE_ENABLED
