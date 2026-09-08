// A second module, for the shared-library test. Stands in for ggml-cpu.dll.
//
// The point of this file is that it is a DIFFERENT binary from the one running
// ts_dlltest, and includes tokenscope.h independently. It therefore compiles
// its own copy of the `ts_tls` cache -- which is the whole subject of
// FINDINGS F22 -- while linking against the one exported registry in the
// tokenscope shared library.
//
// Nothing here is clever. It records a scope and reports two pointers, and the
// test in ts_dlltest.cpp draws the conclusions.
//
// SPDX-License-Identifier: MIT

#include "tokenscope.h"

#include <atomic>

#if defined(_WIN32)
#  define TS_MOD_API __declspec(dllexport)
#else
#  define TS_MOD_API __attribute__((visibility("default")))
#endif

extern "C" {

#ifdef TOKENSCOPE_ENABLED

static std::atomic<unsigned long long> g_sink{0};

// Record a nested pair of host scopes from inside this module. The names are
// distinct from anything the executable records, so the test can tell whose
// records reached the trace.
TS_MOD_API void ts_mod_work(int iters) {
    TS_SCOPE("dllmod.outer");
    unsigned long long x = 1;
    {
        TS_SCOPE("dllmod.inner");
        for (int i = 0; i < iters; ++i) {
            x = x * 6364136223846793005ull + 1442695040888963407ull;
        }
    }
    g_sink.fetch_add(x, std::memory_order_relaxed);
}

// This module's view of the calling thread's buffer. Must equal the
// executable's view: two caches, one buffer.
TS_MOD_API void * ts_mod_buffer(void) {
    return (void *) ts_buffer_get();
}

// The address of THIS module's cache variable. Must NOT equal the
// executable's, or the test is vacuous -- it would be checking that one
// variable equals itself.
TS_MOD_API void * ts_mod_cache_addr(void) {
    return (void *) &ts_tls;
}

// This module's view of the nesting depth counter, which after F22 lives in the
// buffer rather than in a thread-local of its own.
TS_MOD_API void * ts_mod_depth_addr(void) {
    return (void *) ts_depth_slot();
}

#else  // instrumentation compiled out

TS_MOD_API void   ts_mod_work(int)          { }
TS_MOD_API void * ts_mod_buffer(void)       { return nullptr; }
TS_MOD_API void * ts_mod_cache_addr(void)   { return nullptr; }
TS_MOD_API void * ts_mod_depth_addr(void)   { return nullptr; }

#endif

} // extern "C"
