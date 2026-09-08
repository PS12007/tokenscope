// tokenscope — per-token inference profiler for llama.cpp
//
// Design notes: docs/01-design-scope-timing.md
//
// Contract:
//   - When TOKENSCOPE_ENABLED is not defined, every macro in this header expands
//     to nothing. Not to an empty function, not to a disabled branch — to no
//     tokens at all. There is no symbol, no storage and no branch left behind.
//   - When it is defined, no scope in the hot path takes a lock, allocates in
//     the common case, or touches a string.
//
// SPDX-License-Identifier: MIT

#ifndef TOKENSCOPE_H
#define TOKENSCOPE_H

// ---------------------------------------------------------------------------
// Disabled build: the entire feature is preprocessed away.
// ---------------------------------------------------------------------------
#ifndef TOKENSCOPE_ENABLED

#define TS_SCOPE(name)                  do {} while (0)
#define TS_SCOPE_L(name, layer)         do {} while (0)
#define TS_TOKEN_BEGIN(kind, ntok)      do {} while (0)
#define TS_TOKEN_END()                  do {} while (0)
#define TS_TOKEN_SCOPE(is_prefill, n)   do {} while (0)
#define TS_MARK(name)                   do {} while (0)
#define TS_GRAPH_BEGIN(g, nn)           do {} while (0)
#define TS_GRAPH_END()                  do {} while (0)
#define TS_NODE_LOOP_DECL()             do {} while (0)
#define TS_THREAD_PREPARE(n)            do {} while (0)
#define TS_NODE_WORK_END(node_n)        do {} while (0)
#define TS_NODE_WAIT_END(node_n)        do {} while (0)
#define TS_ACTIVE                       (0)

#else // TOKENSCOPE_ENABLED

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Visibility. tokenscope lives in ggml-base so that the instrumentation in
// libggml-cpu and the instrumentation in libllama resolve to ONE registry.
// See docs/01, section "The single-instance problem".
// ---------------------------------------------------------------------------
#if defined(TOKENSCOPE_STATIC)
#  define TS_API
#elif defined(_WIN32)
#  ifdef TOKENSCOPE_BUILD
#    define TS_API __declspec(dllexport)
#  else
#    define TS_API __declspec(dllimport)
#  endif
#else
#  define TS_API __attribute__((visibility("default")))
#endif

#if defined(_MSC_VER)
#  define TS_INLINE  __forceinline
#  define TS_UNLIKELY(x) (x)
#else
#  define TS_INLINE  inline __attribute__((always_inline))
#  define TS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

// Free functions in this header are included from ggml-cpu.c, which is C, so
// they must be `static inline` -- a bare `inline` in C99 needs an external
// definition somewhere. Member functions must NOT be static, hence two macros.
#define TS_SINLINE static TS_INLINE

// ---------------------------------------------------------------------------
// Record kinds
// ---------------------------------------------------------------------------
enum ts_kind {
    TS_KIND_HOST      = 0,  // ref = interned name id
    TS_KIND_NODE_WORK = 1,  // ref = graph node index
    TS_KIND_NODE_WAIT = 2,  // ref = graph node index (barrier after that node)
    TS_KIND_MARK      = 3,  // instant event, dur = 0
};

// Granularity. Runtime, via TOKENSCOPE_LEVEL. See docs/01 section 6.
enum ts_level {
    TS_LEVEL_OFF   = 0,  // nothing recorded, buffers never allocated
    TS_LEVEL_HOST  = 1,  // host scopes only
    TS_LEVEL_AGG   = 2,  // host scopes + per-node aggregates (constant memory)
    TS_LEVEL_FULL  = 3,  // host scopes + every node event
};

// 24 bytes, no padding. Layout is asserted at compile time in tokenscope.cpp.
struct ts_record {
    uint64_t t0;      // ticks since the process epoch
    uint32_t dur;     // ticks, saturating
    uint32_t ref;     // interned name id, or graph node index
    uint32_t token;   // token ordinal
    uint16_t graph;   // graph epoch, resolves `ref` for node records
    uint8_t  kind;    // enum ts_kind
    uint8_t  depth;   // nesting depth for host scopes
};

// ---------------------------------------------------------------------------
// Per-thread buffer.
//
// The hot path touches ONLY `n`, `cap` and `data`. Everything else is cold.
// `ts_tls` is a plain pointer with a constant initializer, deliberately: an
// MSVC thread_local with a dynamic initializer, accessed from a DLL, compiles
// to a guarded call to __dyn_tls_on_demand_init on EVERY access. See docs/01,
// "The Windows TLS trap".
// ---------------------------------------------------------------------------
struct ts_buffer {
    struct ts_record * data;   // current chunk
    uint32_t           n;      // records used in the current chunk
    uint32_t           cap;    // records available in the current chunk

    // Host-scope nesting depth. It lives HERE, in the registry-owned buffer,
    // rather than in a thread-local of its own, because host scopes nest across
    // module boundaries: a decode scope in libllama contains node scopes
    // recorded from libggml-cpu. Each module has its own `ts_tls` cache (see
    // below) but they all point at this one buffer, so this counter is the
    // thread's, not the module's. Touched only by host scopes, never by the
    // per-node path.
    uint32_t           depth;

    // cold
    void *   chunks;           // opaque: owning chunk list
    uint64_t dropped;          // events lost to the budget
    uint64_t acc_generation;   // level-2 accumulator epoch
    uint64_t * acc_work;       // level-2: per-node work ticks
    uint64_t * acc_wait;       // level-2: per-node wait ticks
    uint32_t acc_n;            // node count the accumulators are sized for
    int32_t  tid;              // trace thread id
};

// ---------------------------------------------------------------------------
// Globals. Read on the hot path, written only at setup/flush.
// ---------------------------------------------------------------------------
TS_API extern int ts_g_level;        // enum ts_level; 0 disables everything
TS_API extern uint32_t ts_g_token;   // current token ordinal
TS_API extern uint16_t ts_g_graph;   // current graph epoch

// Whether the CURRENT token is inside the TOKENSCOPE_TOKENS capture window.
// Updated once per token in ts_token_begin, read on the hot path: one load and
// a branch that predicts perfectly for the whole token. Gating per-event on a
// range comparison would put two more loads in the node loop for no benefit.
TS_API extern int ts_g_capture;

// ---------------------------------------------------------------------------
// The thread-local cache, and why it is `static` rather than exported.
//
// MSVC refuses `__declspec(dllexport)` on data with thread storage duration
// (C2492) -- not "discourages", refuses. So with BUILD_SHARED_LIBS=ON a single
// exported `ts_tls` is not expressible at all, and ggml-cpu.dll failed to link
// against it. See FINDINGS F18.
//
// The way out: `ts_tls` is only ever a CACHE. The buffer it points at is owned
// by the registry in ggml-base, which is exported and therefore single-instance
// across the process. So each module compiles its own copy of the pointer and
// fills it by calling the exported ts_thread_init(), which hands back the
// registry's existing buffer for this thread if there is one. Two modules, two
// pointers, one buffer -- and ts_flush walks the registry, so it finds every
// record regardless of which module's cache recorded it.
//
// This keeps the hot path exactly as docs/01 designed it: one load of a plain
// `__declspec(thread)` pointer with a constant initializer, no
// __dyn_tls_on_demand_init guard, and no cross-DLL call. The cost moves to
// thread setup, which is where it belongs.
//
// Because ts_thread_init() can only ever assign to its OWN module's copy, every
// caller must write the result back to its own. The accessors below do that;
// nothing else should touch ts_tls directly.
//
// `static` in a header means one copy per translation unit, not per module,
// which is a handful of extra pointer-sized TLS slots and one extra cold call
// per TU per thread. That is the entire cost of this scheme.
//
// The number of copies does not affect correctness, which is worth stating
// because it is the obvious objection. An inline function that touches an
// internal-linkage variable is COMDAT-folded by the linker, so several TUs may
// silently end up sharing one slot -- and that is fine here, because the slot
// is a cache whose only possible values are null or this thread's one buffer.
// Fold them all together or keep them all apart and the observable behaviour is
// the same; only the number of cold ts_thread_init() calls changes.
// ---------------------------------------------------------------------------
static
#ifdef _MSC_VER
__declspec(thread)
#else
__thread
#endif
struct ts_buffer * ts_tls = 0;

#define TS_ACTIVE (ts_g_level != TS_LEVEL_OFF)

// ---------------------------------------------------------------------------
// Timing source. docs/01 section 2.
// ---------------------------------------------------------------------------
TS_API uint64_t ts_now_slow(void);          // portable, always correct
TS_API double   ts_ticks_to_us(uint64_t t); // resolved at flush, not on the hot path

#if defined(TOKENSCOPE_TSC) && (defined(__x86_64__) || defined(_M_X64))
#  if defined(_MSC_VER)
#    include <intrin.h>
TS_SINLINE uint64_t ts_now(void) { return __rdtsc(); }
#  else
#    include <x86intrin.h>
TS_SINLINE uint64_t ts_now(void) { return __builtin_ia32_rdtsc(); }
#  endif
#else
TS_SINLINE uint64_t ts_now(void) { return ts_now_slow(); }
#endif

// ---------------------------------------------------------------------------
// Cold-path helpers. None of these are called in steady state.
// ---------------------------------------------------------------------------
TS_API struct ts_buffer * ts_thread_init(void);
TS_API struct ts_record * ts_grow(struct ts_buffer * b);
TS_API uint32_t ts_intern(const char * name);

TS_API void ts_init_from_env(void);
TS_API void ts_token_begin(int is_prefill, uint32_t n_tokens);
TS_API void ts_token_end(void);
TS_API void ts_flush(const char * path);

// Node-name -> phase classification is a hand-ordered prefix table, and the
// first match wins. An entry that another entry is a prefix of is therefore
// unreachable. ts_check_category_table returns how many such entries exist --
// zero, or the classification is silently wrong for some node. `report`, if
// given, is called with (shadowed_prefix, shadowing_prefix) for each. See
// FINDINGS F20, where "kqv_out" turned out to have been dead the whole time.
typedef void (*ts_shadow_report_fn)(const char * shadowed, const char * by);
TS_API int ts_check_category_table(ts_shadow_report_fn report);

// Graph epochs. ts_graph_begin returns non-zero if this cgraph has not been
// seen before, in which case the caller (the main thread, outside the node
// loop) should walk the graph once and register node names. Because llama.cpp
// reuses graphs across decode steps, that happens about once per run.
TS_API int  ts_graph_begin(const void * cgraph, uint32_t n_nodes);
TS_API void ts_graph_set_node(uint32_t node_n, const char * name, const char * op);
TS_API void ts_graph_end(void);

// Sizes this thread's level-2 accumulators. Called once per graph per thread,
// outside the node loop, so that level 2 never allocates mid-graph.
TS_API void ts_thread_prepare(uint32_t n_nodes);

// True when the current token is inside the TOKENSCOPE_TOKENS window.
// ts_token_begin already maintains ts_g_capture from this; exposed for tests.
TS_API int ts_token_selected(void);

// Records how the module running the node loop was built, so the trace says it
// instead of the reader assuming it. See TS_BUILD_THREADING below for why the
// values have to come from the caller and not from this file. Called once per
// thread per graph, off the node path; every caller writes the same constants,
// so the store is deliberately unsynchronised.
TS_API void ts_note_build(int threading, int shared_linkage);

// ---------------------------------------------------------------------------
// The hot path.
// ---------------------------------------------------------------------------
// This module's view of the calling thread's buffer, initialising it on first
// use. The write-back to `ts_tls` is the whole point: ts_thread_init() lives in
// ggml-base and can only assign to ggml-base's copy of the cache, so a caller
// that did not store the result would call it again on every single event --
// correct, but at the cost of a cross-DLL call on the hot path, which is
// exactly what this design exists to avoid.
TS_SINLINE struct ts_buffer * ts_buffer_get(void) {
    struct ts_buffer * b = ts_tls;
    if (TS_UNLIKELY(b == 0)) {
        b = ts_thread_init();
        if (TS_UNLIKELY(b == 0)) return 0;
        ts_tls = b;
    }
    return b;
}

// The host-scope depth counter for this thread, or 0 if there is no buffer
// (level OFF, where depth is never read). Not on the per-node path.
TS_SINLINE uint32_t * ts_depth_slot(void) {
    struct ts_buffer * b = ts_buffer_get();
    return b ? &b->depth : 0;
}

TS_SINLINE struct ts_record * ts_reserve(void) {
    if (TS_UNLIKELY(!ts_g_capture)) return 0;
    struct ts_buffer * b = ts_buffer_get();
    if (TS_UNLIKELY(b == 0)) return 0;
    if (TS_UNLIKELY(b->n == b->cap)) {
        return ts_grow(b);
    }
    return &b->data[b->n++];
}

TS_SINLINE void ts_emit(uint64_t t0, uint64_t t1, uint32_t ref, uint8_t kind, uint8_t depth) {
    struct ts_record * r = ts_reserve();
    if (TS_UNLIKELY(r == 0)) return;
    const uint64_t d = t1 - t0;
    r->t0    = t0;
    r->dur   = d > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t) d;
    r->ref   = ref;
    r->token = ts_g_token;
    r->graph = ts_g_graph;
    r->kind  = kind;
    r->depth = depth;
}

// Level 2: accumulate into a fixed per-node array instead of appending.
// One add, no growth, exact totals. docs/01 section 6.
TS_SINLINE void ts_acc(uint32_t node_n, uint64_t dur, int is_wait) {
    if (TS_UNLIKELY(!ts_g_capture)) return;
    // ts_buffer_get, not a bare `ts_tls` read: ts_thread_prepare() sized the
    // accumulators through ggml-base's copy of the cache, so this module's copy
    // can still be null on the first node even though the thread has a buffer.
    // Reading ts_tls directly here would silently record nothing at level 2.
    struct ts_buffer * b = ts_buffer_get();
    if (TS_UNLIKELY(b == 0 || node_n >= b->acc_n)) return;
    if (is_wait) b->acc_wait[node_n] += dur;
    else         b->acc_work[node_n] += dur;
}

#ifdef __cplusplus
} // extern "C"

// ---------------------------------------------------------------------------
// RAII scope. C++ only; the C sites in ggml use the explicit loop macros below.
// ---------------------------------------------------------------------------
namespace tokenscope {

class scope {
public:
    TS_INLINE scope(uint32_t id) : m_id(id), m_t0(ts_g_level ? ts_now() : 0) {
        // Record nesting depth explicitly rather than leaving the analyzer to
        // infer it from timestamps. Two adjacent scopes touch at exactly one
        // instant, and floating-point microseconds cannot tell "ends where the
        // next begins" from "contains the next" -- which silently reparents a
        // sibling and makes self-time attribution wrong. One increment is a
        // cheaper fix than an epsilon that has to be right on every machine.
        // The slot is cached rather than looked up again in the destructor, so
        // the increment and the decrement provably act on the same counter.
        // The buffer address is stable for the life of the thread -- ts_grow
        // replaces `data`, never the ts_buffer itself -- so holding it is safe.
        if (m_t0) {
            m_slot = ts_depth_slot();
            if (m_slot) m_depth = (*m_slot)++;
        }
    }
    TS_INLINE ~scope() {
        if (m_t0) {
            if (m_slot) --(*m_slot);
            ts_emit(m_t0, ts_now(), m_id, TS_KIND_HOST,
                    m_depth > 255 ? 255 : (uint8_t) m_depth);
        }
    }
    scope(const scope &) = delete;
    scope & operator=(const scope &) = delete;
private:
    uint32_t   m_id;
    uint64_t   m_t0;
    uint32_t   m_depth = 0;
    uint32_t * m_slot  = 0;
};

// A token (or prefill batch) boundary. RAII because llama_context::decode has
// a dozen early returns and a missed end would silently corrupt the timeline.
class token_scope {
public:
    TS_INLINE token_scope(int is_prefill, uint32_t n_tokens) {
        ts_token_begin(is_prefill, n_tokens);
    }
    TS_INLINE ~token_scope() { ts_token_end(); }
    token_scope(const token_scope &) = delete;
    token_scope & operator=(const token_scope &) = delete;
};

} // namespace tokenscope

#define TS_CAT_(a, b) a##b
#define TS_CAT(a, b)  TS_CAT_(a, b)

// A host scope. The name is interned exactly once per site, ever.
#define TS_SCOPE(name)                                                        \
    static const uint32_t TS_CAT(ts_id_, __LINE__) = ts_intern(name);         \
    tokenscope::scope TS_CAT(ts_sc_, __LINE__)(TS_CAT(ts_id_, __LINE__))

#define TS_SCOPE_L(name, layer) TS_SCOPE(name)   // layer folded into ref later

#define TS_TOKEN_SCOPE(is_prefill, ntok)                                      \
    tokenscope::token_scope TS_CAT(ts_tok_, __LINE__)((is_prefill), (ntok))

#endif // __cplusplus

// ---------------------------------------------------------------------------
// Token and graph boundaries.
// ---------------------------------------------------------------------------
#define TS_TOKEN_BEGIN(is_prefill, ntok) ts_token_begin((is_prefill), (ntok))
#define TS_TOKEN_END()                   ts_token_end()
#define TS_GRAPH_BEGIN(g, nn)            ts_graph_begin((g), (nn))
#define TS_GRAPH_END()                   ts_graph_end()

#define TS_MARK(name)                                                         \
    do {                                                                      \
        if (ts_g_level) {                                                     \
            static const uint32_t ts_mk_id = ts_intern(name);                 \
            const uint64_t t = ts_now();                                      \
            ts_emit(t, t, ts_mk_id, TS_KIND_MARK, 0);                         \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// The ggml node loop. C-compatible, and structured so that the end of the work
// scope and the start of the wait scope share one clock read.
//
//   TS_NODE_LOOP_DECL();                 // once, before the loop
//   for (...) {
//       ... ggml_compute_forward(...);
//       TS_NODE_WORK_END(node_n);        // one clock read
//       ... ggml_barrier(...);
//       TS_NODE_WAIT_END(node_n);        // one clock read, becomes next t0
//   }
//
// Four naive reads per node collapse to two. docs/01 section 2.
// ---------------------------------------------------------------------------
// Which threading path the node loop was compiled for, and whether the module
// containing it is a separate shared object. Both are resolved where the macro
// is USED -- inside the translation unit that also contains `ggml_barrier` --
// because that is the only place the answer is authoritative. Deciding it
// inside tokenscope.cpp would answer a question about tokenscope.cpp.
//
// This exists because of FINDINGS F26: for four sessions the docs said every
// barrier number here came from ggml's own spin-wait threadpool, and every one
// of them came from `#pragma omp barrier`. No trace could have contradicted
// that, because no trace recorded it. docs/00 said "the profiler must record
// which" from the start; this is that, three sessions late.
//
// Testing GGML_USE_OPENMP is not a ggml dependency in the linkage sense -- no
// ggml symbol is referenced -- and the `_OPENMP` arm exists to distinguish
// "ggml is not using OpenMP" from "this compiler has no OpenMP", which are
// different build mistakes.
#if defined(GGML_USE_OPENMP)
#  define TS_BUILD_THREADING 1    // ggml_barrier is #pragma omp barrier
#elif defined(_OPENMP)
#  define TS_BUILD_THREADING 2    // OpenMP is available, ggml is not using it
#else
#  define TS_BUILD_THREADING 0    // ggml's own threadpool, atomic spin-wait
#endif

#if defined(TOKENSCOPE_STATIC)
#  define TS_BUILD_LINKAGE 0
#else
#  define TS_BUILD_LINKAGE 1
#endif

#define TS_THREAD_PREPARE(n)                                                  \
    do {                                                                      \
        if (ts_g_level) {                                                     \
            ts_note_build(TS_BUILD_THREADING, TS_BUILD_LINKAGE);              \
            ts_thread_prepare((uint32_t)(n));                                 \
        }                                                                     \
    } while (0)

#define TS_NODE_LOOP_DECL()                                                   \
    uint64_t ts_t_mark = ts_g_level >= TS_LEVEL_AGG ? ts_now() : 0

#define TS_NODE_WORK_END(node_n)                                              \
    do {                                                                      \
        if (ts_t_mark) {                                                      \
            const uint64_t ts_t1 = ts_now();                                  \
            if (ts_g_level == TS_LEVEL_AGG) {                                 \
                ts_acc((uint32_t)(node_n), ts_t1 - ts_t_mark, 0);             \
            } else {                                                          \
                ts_emit(ts_t_mark, ts_t1, (uint32_t)(node_n),                 \
                        TS_KIND_NODE_WORK, 0);                                \
            }                                                                 \
            ts_t_mark = ts_t1;                                                \
        }                                                                     \
    } while (0)

#define TS_NODE_WAIT_END(node_n)                                              \
    do {                                                                      \
        if (ts_t_mark) {                                                      \
            const uint64_t ts_t1 = ts_now();                                  \
            if (ts_g_level == TS_LEVEL_AGG) {                                 \
                ts_acc((uint32_t)(node_n), ts_t1 - ts_t_mark, 1);             \
            } else {                                                          \
                ts_emit(ts_t_mark, ts_t1, (uint32_t)(node_n),                 \
                        TS_KIND_NODE_WAIT, 0);                                \
            }                                                                 \
            ts_t_mark = ts_t1;                                                \
        }                                                                     \
    } while (0)

#endif // TOKENSCOPE_ENABLED
#endif // TOKENSCOPE_H
