// tokenscope shared-library test — the regression test FINDINGS F18 never had.
//
// F18 was found by hand, a session after the code that caused it, because
// nothing ever built tokenscope as a shared library. This test builds two
// separate binaries that each include tokenscope.h, and asserts the invariant
// F22 turns on:
//
//   every module has its OWN `ts_tls` cache, and they all resolve to ONE
//   registry-owned buffer per thread.
//
// Both halves matter. If the caches were shared, the test would be checking
// that a variable equals itself and would pass even if F22 were reverted. If
// the buffers differed, each module would record into its own thread_state and
// the trace would carry one thread's work under several thread ids -- which is
// not a crash, not a dropped record, and not visible in any way except by
// counting. So the test counts.
//
// Run: ts_dlltest <out.json> [path to a runtime-loadable copy of the module]
//
// SPDX-License-Identifier: MIT

#include "tokenscope.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

extern "C" {
void * ts_mod_buffer(void);
void * ts_mod_cache_addr(void);
void * ts_mod_depth_addr(void);
void   ts_mod_work(int iters);
}

#ifndef TOKENSCOPE_ENABLED
int main() {
    std::printf("tokenscope: built with instrumentation DISABLED, nothing to test.\n");
    return 0;
}
#else

static int g_failures = 0;

static void check(bool cond, const char * what) {
    std::printf("  [%s] %s\n", cond ? " ok " : "FAIL", what);
    if (!cond) g_failures++;
}

// Per-thread observations, collected without locks and inspected after join.
struct observation {
    const void * exe_buffer     = nullptr;
    const void * mod_buffer     = nullptr;
    const void * exe_cache_addr = nullptr;
    const void * mod_cache_addr = nullptr;
    const void * exe_depth      = nullptr;
    const void * mod_depth      = nullptr;
};

static void worker(observation * obs, int n_iter) {
    // Interleave the two modules' scopes so the recorded nesting spans the
    // module boundary: exe.outer contains dllmod.outer contains dllmod.inner.
    // Depth is only correct across that boundary because F22 moved the counter
    // into the shared buffer.
    TS_SCOPE("exe.outer");

    obs->exe_buffer     = (const void *) ts_buffer_get();
    obs->exe_cache_addr = (const void *) &ts_tls;
    obs->exe_depth      = (const void *) ts_depth_slot();

    ts_mod_work(n_iter);

    obs->mod_buffer     = ts_mod_buffer();
    obs->mod_cache_addr = ts_mod_cache_addr();
    obs->mod_depth      = ts_mod_depth_addr();
}

// Count distinct thread ids in the trace. This is the check that would catch
// two modules building separate thread_states per thread: the records would all
// be present and the JSON would be valid, but one OS thread's work would be
// split across two trace thread ids.
//
// Negative ids are excluded because they are not threads. Token slices are
// emitted on a synthetic tid of -1 so that Perfetto draws them as their own
// track above the workers rather than interleaved with one thread's scopes.
static size_t count_distinct_tids(const std::string & s) {
    std::set<long long> tids;
    const std::string key = "\"tid\":";
    size_t i = 0;
    while ((i = s.find(key, i)) != std::string::npos) {
        i += key.size();
        const long long tid = std::strtoll(s.c_str() + i, nullptr, 10);
        if (tid >= 0) tids.insert(tid);
    }
    return tids.size();
}

int main(int argc, char ** argv) {
    const char * out_path = argc > 1 ? argv[1] : "dlltest.trace.json";

#if defined(_WIN32)
    _putenv_s("TOKENSCOPE_LEVEL", "1");
#else
    setenv("TOKENSCOPE_LEVEL", "1", 1);
#endif
    ts_init_from_env();

    std::printf("tokenscope shared-library test\n");
    std::printf("  level=%d\n", ts_g_level);
    check(TS_ACTIVE, "instrumentation is active");

    // Give the main thread a buffer unconditionally, so the thread-id count at
    // the end is a fixed number rather than one that depends on which later
    // phases happened to run.
    TS_SCOPE("exe.main");
    bool dyn_thread_ran = false;

    const int n_threads = 4;

    std::printf("\n1. two modules, one buffer\n");

    std::vector<observation> obs((size_t) n_threads);
    {
        TS_TOKEN_BEGIN(0, 1u);
        std::vector<std::thread> ths;
        ths.reserve((size_t) n_threads);
        for (int i = 0; i < n_threads; ++i) {
            ths.emplace_back(worker, &obs[(size_t) i], 200);
        }
        for (auto & th : ths) th.join();
        TS_TOKEN_END();
    }

    // The control. If these are equal, the two "modules" are one binary and
    // every other assertion below is vacuous -- so this is checked first and
    // reported as a distinct failure, not folded into the others.
    bool caches_distinct = true;
    for (const auto & o : obs) {
        if (o.exe_cache_addr == o.mod_cache_addr) caches_distinct = false;
    }
    check(caches_distinct,
          "each module has its own ts_tls cache (the control: without this the "
          "rest is vacuous)");

    // The invariant.
    bool one_buffer = true;
    for (const auto & o : obs) {
        if (!o.exe_buffer || o.exe_buffer != o.mod_buffer) one_buffer = false;
    }
    check(one_buffer, "both modules resolve to the same buffer on each thread");

    // The depth counter lives in that buffer, so it follows -- but it is
    // asserted separately because it is a separate F22 decision, and because a
    // future change could plausibly move one without the other.
    bool one_depth = true;
    for (const auto & o : obs) {
        if (!o.exe_depth || o.exe_depth != o.mod_depth) one_depth = false;
    }
    check(one_depth, "both modules share one host-scope depth counter");

    // Distinct threads must still get distinct buffers. Sharing everything
    // would satisfy every check above and reintroduce the lock the design
    // exists to avoid.
    {
        std::set<const void *> bufs;
        for (const auto & o : obs) bufs.insert(o.exe_buffer);
        check(bufs.size() == (size_t) n_threads,
              "different threads still get different buffers");
    }

    // -----------------------------------------------------------------------
    // 2. A module loaded at RUNTIME, onto a thread that already has a buffer.
    //
    // This is the GGML_BACKEND_DL shape: llama.cpp can load backends with
    // dlopen/LoadLibrary rather than linking them, so a module can arrive after
    // threads exist and have already recorded. It is also the case where a
    // cache-based design is most likely to be wrong, because the newcomer's
    // cache starts null at a point where the registry already has an answer --
    // it must join the existing buffer, not allocate a second one.
    //
    // Skipped rather than failed when no path is given, so the test still runs
    // for anyone invoking the binary by hand.
    // -----------------------------------------------------------------------
    if (argc > 2) {
        std::printf("\n2. a module loaded after the threads already ran\n");
        const char * mod_path = argv[2];

#if defined(_WIN32)
        HMODULE h = LoadLibraryA(mod_path);
        void * sym = h ? (void *) GetProcAddress(h, "ts_mod_buffer") : nullptr;
#else
        void * h = dlopen(mod_path, RTLD_NOW | RTLD_LOCAL);
        void * sym = h ? dlsym(h, "ts_mod_buffer") : nullptr;
#endif
        check(h != nullptr, "the runtime-loaded module opened");
        check(sym != nullptr, "ts_mod_buffer resolved from the loaded module");

        if (sym) {
            void * (*dyn_buffer)(void) = (void * (*)(void)) sym;

            // The main thread has a buffer already: it opened the token slice
            // above, which records.
            const void * mine = (const void *) ts_buffer_get();
            check(mine == dyn_buffer(),
                  "a module loaded mid-run joins this thread's existing buffer");

            // And a thread created after the load still gets exactly one
            // buffer, shared with the newcomer.
            const void * t_exe = nullptr;
            const void * t_dyn = nullptr;
            std::thread([&] {
                TS_SCOPE("exe.after-load");
                t_exe = (const void *) ts_buffer_get();
                t_dyn = dyn_buffer();
            }).join();
            dyn_thread_ran = true;
            check(t_exe != nullptr && t_exe == t_dyn,
                  "a thread created after the load also sees one buffer");
            check(t_exe != mine, "...and it is that thread's own, not the main thread's");
        }
    }

    std::printf("\n3. trace output\n");
    ts_flush(out_path);

    std::string data;
    if (std::FILE * f = std::fopen(out_path, "rb")) {
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz > 0) {
            data.resize((size_t) sz);
            data.resize(std::fread(&data[0], 1, (size_t) sz, f));
        }
        std::fclose(f);
    }
    check(!data.empty(), "trace file was created");

    check(data.find("\"exe.outer\"")    != std::string::npos,
          "the executable's scopes reached the trace");
    check(data.find("\"dllmod.outer\"") != std::string::npos,
          "the shared library's scopes reached the trace");
    check(data.find("\"dllmod.inner\"") != std::string::npos,
          "nested scopes from the shared library reached the trace");
    check(data.find("\"dropped\":0")    != std::string::npos,
          "no records were dropped");

    // Exactly one trace thread id per OS thread that recorded. Asserted as
    // equality against a count this test knows exactly, not as an upper bound:
    // two modules building separate state per thread would double it while
    // leaving every record present and the JSON valid, and a bound generous
    // enough to survive edits to the test is generous enough to miss that.
    {
        const size_t tids     = count_distinct_tids(data);
        const size_t expected = (size_t) n_threads   // the workers
                              + 1                    // main, which records below
                              + (size_t) (dyn_thread_ran ? 1 : 0);
        std::printf("  %zu distinct trace thread ids, expected %zu "
                    "(%d workers + main%s)\n",
                    tids, expected, n_threads,
                    dyn_thread_ran ? " + 1 created after the module load" : "");
        check(tids == expected,
              "one trace thread id per thread, not one per thread per module");
    }

    std::printf("\n%s (%d failure%s)\n",
        g_failures == 0 ? "PASS" : "FAIL", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}

#endif // TOKENSCOPE_ENABLED
