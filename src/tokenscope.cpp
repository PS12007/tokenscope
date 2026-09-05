// tokenscope — implementation.
//
// Everything in this file is either cold path (setup, flush) or explicitly
// marked as hot. The hot path is in the header, inlined at the call site.
//
// SPDX-License-Identifier: MIT

#ifdef TOKENSCOPE_ENABLED

#define TOKENSCOPE_BUILD 1
#include "tokenscope.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <atomic>
#include <thread>

#if defined(_WIN32)
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <unistd.h>
#  include <sys/types.h>
#endif

static_assert(sizeof(ts_record) == 24, "ts_record must stay 24 bytes; see docs/01 section 3");

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
extern "C" {
TS_API int      ts_g_level = TS_LEVEL_OFF;
TS_API uint32_t ts_g_token = 0;
TS_API uint16_t ts_g_graph = 0;

#ifdef _MSC_VER
__declspec(thread) ts_buffer * ts_tls = nullptr;
__declspec(thread) uint32_t    ts_depth = 0;
#else
__thread ts_buffer * ts_tls = nullptr;
__thread uint32_t    ts_depth = 0;
#endif
}

namespace {

constexpr size_t CHUNK_BYTES   = 1u << 20;                  // 1 MiB
constexpr size_t CHUNK_RECORDS = CHUNK_BYTES / sizeof(ts_record);

struct chunk {
    std::vector<ts_record> recs;
    explicit chunk(size_t n) { recs.resize(n); }   // resize, not reserve: pre-touch
};

// Owns everything that outlives a worker thread.
struct thread_state {
    ts_buffer                            buf{};
    std::vector<std::unique_ptr<chunk>>  chunks;
    std::vector<uint64_t>                acc_work;
    std::vector<uint64_t>                acc_wait;
    std::string                          name;
};

struct graph_info {
    uint32_t                 n_nodes = 0;
    std::vector<std::string> node_names;
    std::vector<std::string> node_ops;
};

struct registry {
    std::mutex                                  mu;
    std::vector<std::shared_ptr<thread_state>>  threads;
    std::vector<std::string>                    names;      // interned host-scope names
    std::unordered_map<std::string, uint32_t>   name_ids;
    std::unordered_map<const void *, uint16_t>  graph_ids;
    std::vector<graph_info>                     graphs;      // index = epoch
    size_t                                      budget_bytes = 256ull << 20;
    size_t                                      used_bytes   = 0;
    bool                                        ring         = false;
    uint64_t                                    t_epoch      = 0;
    double                                      ticks_per_us = 1000.0;
    std::string                                 clock_name   = "steady_clock";
    // capture window
    uint32_t                                    tok_lo = 0;
    uint32_t                                    tok_hi = 0xFFFFFFFFu;
    // per-token bookkeeping, main thread only
    std::vector<uint64_t>                       tok_t0;
    std::vector<uint8_t>                        tok_is_prefill;
    std::vector<uint32_t>                       tok_ntok;
    std::vector<uint64_t>                       tok_dur;
    uint64_t                                    cur_tok_t0 = 0;
};

registry & reg() {
    static registry r;
    return r;
}

int32_t next_tid() {
    static std::atomic<int32_t> n{0};
    return n.fetch_add(1, std::memory_order_relaxed);
}

const char * env_or(const char * k, const char * dflt) {
    const char * v = std::getenv(k);
    return (v && *v) ? v : dflt;
}

int32_t cur_pid() {
#if defined(_WIN32)
    return (int32_t) GetCurrentProcessId();
#else
    return (int32_t) getpid();
#endif
}

// JSON string escaping, minimal but correct for the characters ggml node names
// and our own scope names can actually contain.
void json_escape(std::string & out, const std::string & s) {
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", (unsigned) (unsigned char) c);
                    out += b;
                } else {
                    out += c;
                }
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
extern "C" TS_API uint64_t ts_now_slow(void) {
    return (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
}

extern "C" TS_API double ts_ticks_to_us(uint64_t t) {
    return (double) t / reg().ticks_per_us;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
static void calibrate_clock(registry & r) {
#if defined(TOKENSCOPE_TSC) && (defined(__x86_64__) || defined(_M_X64))
    r.clock_name = "rdtsc";
    // Calibrate against steady_clock over a short window. Documented as opt-in
    // precisely because this calibration carries error; see docs/01 section 2.
    const auto  w0 = std::chrono::steady_clock::now();
    const uint64_t c0 = ts_now();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const uint64_t c1 = ts_now();
    const auto  w1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(w1 - w0).count();
    r.ticks_per_us = us > 0 ? (double) (c1 - c0) / us : 1000.0;
#else
    r.clock_name = "steady_clock";
    // steady_clock::period is nanoseconds on every implementation we target;
    // derive it rather than assuming, so a surprising platform is a wrong
    // scale factor and not a silently wrong trace.
    using period = std::chrono::steady_clock::period;
    r.ticks_per_us = (double) period::den / ((double) period::num * 1e6);
#endif
}

extern "C" TS_API void ts_init_from_env(void) {
    static std::once_flag once;
    std::call_once(once, [] {
        registry & r = reg();
        calibrate_clock(r);
        r.t_epoch = ts_now();

        ts_g_level = std::atoi(env_or("TOKENSCOPE_LEVEL", "0"));
        if (ts_g_level < 0) ts_g_level = 0;
        if (ts_g_level > TS_LEVEL_FULL) ts_g_level = TS_LEVEL_FULL;

        const int mb = std::atoi(env_or("TOKENSCOPE_BUDGET_MB", "256"));
        r.budget_bytes = (size_t) (mb > 0 ? mb : 256) << 20;
        r.ring = std::atoi(env_or("TOKENSCOPE_RING", "0")) != 0;

        // Flush at exit if an output path was given. This is what lets
        // tokenscope instrument llama.cpp without any tool -- llama-bench,
        // llama-cli, llama-server -- needing a single line changed.
        const char * out = env_or("TOKENSCOPE_OUT", "");
        if (*out && ts_g_level != TS_LEVEL_OFF) {
            static std::string s_out;
            s_out = out;
            std::atexit([] {
                if (ts_g_level != TS_LEVEL_OFF) ts_flush(s_out.c_str());
            });
        }

        const char * win = env_or("TOKENSCOPE_TOKENS", "");
        if (*win) {
            unsigned lo = 0, hi = 0;
            if (std::sscanf(win, "%u-%u", &lo, &hi) == 2) { r.tok_lo = lo; r.tok_hi = hi; }
            else if (std::sscanf(win, "%u", &lo) == 1)    { r.tok_lo = lo; r.tok_hi = lo; }
        }
    });
}

// ---------------------------------------------------------------------------
// Per-thread buffers
// ---------------------------------------------------------------------------
extern "C" TS_API ts_buffer * ts_thread_init(void) {
    ts_init_from_env();
    if (ts_g_level == TS_LEVEL_OFF) return nullptr;

    registry & r = reg();
    auto st = std::make_shared<thread_state>();

    {
        std::lock_guard<std::mutex> lk(r.mu);
        if (r.used_bytes + CHUNK_BYTES > r.budget_bytes) {
            // Out of budget before this thread even starts. Hand back a buffer
            // with cap == 0 so every reserve() drops and counts, rather than
            // silently succeeding.
            st->buf.data = nullptr;
            st->buf.n = st->buf.cap = 0;
        } else {
            st->chunks.push_back(std::make_unique<chunk>(CHUNK_RECORDS));
            r.used_bytes += CHUNK_BYTES;
            st->buf.data = st->chunks.back()->recs.data();
            st->buf.n    = 0;
            st->buf.cap  = (uint32_t) CHUNK_RECORDS;
        }
        st->buf.tid    = next_tid();
        st->buf.chunks = st.get();
        r.threads.push_back(st);
    }

    ts_tls = &st->buf;
    return ts_tls;
}

extern "C" TS_API ts_record * ts_grow(ts_buffer * b) {
    registry & r = reg();
    auto * st = static_cast<thread_state *>(b->chunks);
    if (!st) { b->dropped++; return nullptr; }

    std::lock_guard<std::mutex> lk(r.mu);

    if (r.ring && !st->chunks.empty()) {
        // Ring mode: recycle the oldest chunk instead of allocating. Records are
        // lost, and we say so.
        auto oldest = std::move(st->chunks.front());
        st->chunks.erase(st->chunks.begin());
        b->dropped += oldest->recs.size();
        st->chunks.push_back(std::move(oldest));
    } else {
        if (r.used_bytes + CHUNK_BYTES > r.budget_bytes) {
            b->dropped++;
            return nullptr;
        }
        st->chunks.push_back(std::make_unique<chunk>(CHUNK_RECORDS));
        r.used_bytes += CHUNK_BYTES;
    }

    b->data = st->chunks.back()->recs.data();
    b->cap  = (uint32_t) st->chunks.back()->recs.size();
    b->n    = 1;
    return &b->data[0];
}

// ---------------------------------------------------------------------------
// Interning
// ---------------------------------------------------------------------------
extern "C" TS_API uint32_t ts_intern(const char * name) {
    registry & r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.name_ids.find(name);
    if (it != r.name_ids.end()) return it->second;
    const uint32_t id = (uint32_t) r.names.size();
    r.names.emplace_back(name);
    r.name_ids.emplace(name, id);
    return id;
}

// ---------------------------------------------------------------------------
// Token boundaries
// ---------------------------------------------------------------------------
extern "C" TS_API int ts_token_selected(void) {
    const registry & r = reg();
    return ts_g_token >= r.tok_lo && ts_g_token <= r.tok_hi;
}

extern "C" TS_API void ts_token_begin(int is_prefill, uint32_t n_tokens) {
    ts_init_from_env();
    if (ts_g_level == TS_LEVEL_OFF) return;
    registry & r = reg();
    r.cur_tok_t0 = ts_now();
    r.tok_t0.push_back(r.cur_tok_t0);
    r.tok_is_prefill.push_back((uint8_t) (is_prefill != 0));
    r.tok_ntok.push_back(n_tokens);
    r.tok_dur.push_back(0);
    ts_g_token = (uint32_t) (r.tok_t0.size() - 1);
}

extern "C" TS_API void ts_token_end(void) {
    if (ts_g_level == TS_LEVEL_OFF) return;
    registry & r = reg();
    if (r.tok_dur.empty()) return;
    r.tok_dur.back() = ts_now() - r.cur_tok_t0;
}

// ---------------------------------------------------------------------------
// Graph epochs
// ---------------------------------------------------------------------------
extern "C" TS_API int ts_graph_begin(const void * cgraph, uint32_t n_nodes) {
    if (ts_g_level < TS_LEVEL_AGG) return 0;
    registry & r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.graph_ids.find(cgraph);
    if (it != r.graph_ids.end() && r.graphs[it->second].n_nodes == n_nodes) {
        ts_g_graph = it->second;
        return 0;   // already known; no name walk needed
    }
    const uint16_t epoch = (uint16_t) r.graphs.size();
    r.graphs.push_back(graph_info{});
    r.graphs.back().n_nodes = n_nodes;
    r.graphs.back().node_names.resize(n_nodes);
    r.graphs.back().node_ops.resize(n_nodes);
    r.graph_ids[cgraph] = epoch;
    ts_g_graph = epoch;
    return 1;       // caller should register node names
}

extern "C" TS_API void ts_graph_set_node(uint32_t node_n, const char * name, const char * op) {
    if (ts_g_level < TS_LEVEL_AGG) return;
    registry & r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    if (ts_g_graph >= r.graphs.size()) return;
    auto & g = r.graphs[ts_g_graph];
    if (node_n >= g.node_names.size()) return;
    g.node_names[node_n] = name ? name : "";
    g.node_ops[node_n]   = op   ? op   : "";
}

extern "C" TS_API void ts_graph_end(void) { /* reserved */ }

// Sizes this thread's level-2 accumulators for the current graph. Called once
// per graph per thread, OUTSIDE the node loop: the level-2 mode's whole point
// is that it never allocates while a graph is running. docs/01, "Risk 2".
extern "C" TS_API void ts_thread_prepare(uint32_t n_nodes) {
    if (ts_g_level != TS_LEVEL_AGG) return;
    ts_buffer * b = ts_tls;
    if (!b) { b = ts_thread_init(); if (!b) return; }
    auto * st = static_cast<thread_state *>(b->chunks);
    if (!st) return;
    if (st->acc_work.size() < n_nodes) {
        st->acc_work.assign(n_nodes, 0);
        st->acc_wait.assign(n_nodes, 0);
        b->acc_work = st->acc_work.data();
        b->acc_wait = st->acc_wait.data();
        b->acc_n    = n_nodes;
    }
}

// ---------------------------------------------------------------------------
// Flush
// ---------------------------------------------------------------------------
namespace {

struct out_event {
    double      ts_us;
    double      dur_us;
    int32_t     tid;
    const std::string * name;
    const char *        cat;
    uint32_t    token;
    uint8_t     depth;
    uint8_t     ph_instant;
};

void emit_event(std::string & out, bool & first, const out_event & e, int32_t pid) {
    if (!first) out += ",\n";
    first = false;
    out += "{\"name\":\"";
    json_escape(out, *e.name);
    out += "\",\"cat\":\"";
    out += e.cat;
    out += "\",\"ph\":\"";
    out += e.ph_instant ? "i" : "X";
    out += "\",\"pid\":";
    out += std::to_string(pid);
    out += ",\"tid\":";
    out += std::to_string(e.tid);
    char b[64];
    std::snprintf(b, sizeof(b), ",\"ts\":%.3f", e.ts_us);
    out += b;
    if (!e.ph_instant) {
        std::snprintf(b, sizeof(b), ",\"dur\":%.3f", e.dur_us);
        out += b;
    }
    out += ",\"args\":{\"tok\":";
    out += std::to_string(e.token);
    out += ",\"depth\":";
    out += std::to_string((unsigned) e.depth);
    out += "}}";
}

const char * category_for(const std::string & node_name) {
    // Node names are "<role>-<layer>"; see docs/00 section 2.
    static const struct { const char * p; const char * cat; } table[] = {
        { "attn_norm",   "norm"      }, { "ffn_norm",  "norm"      },
        { "result_norm", "norm"      }, { "norm",      "norm"      },
        { "Qcur",        "attn.qkv"  }, { "Kcur",      "attn.qkv"  },
        { "Vcur",        "attn.qkv"  }, { "attn_q",    "attn.qkv"  },
        { "attn_k",      "attn.qkv"  }, { "attn_v",    "attn.qkv"  },
        { "cache_k",     "attn.kv_rw"}, { "cache_v",   "attn.kv_rw"},
        { "k_cache",     "attn.kv_rw"}, { "v_cache",   "attn.kv_rw"},
        { "kq",          "attn.score"}, { "kqv",       "attn.score"},
        { "attn_out",    "attn.out"  }, { "kqv_out",   "attn.out"  },
        { "ffn_",        "ffn"       },
        { "result_output","lm_head"  },
    };
    for (const auto & e : table) {
        if (node_name.compare(0, std::strlen(e.p), e.p) == 0) return e.cat;
    }
    return "other";
}

} // namespace

extern "C" TS_API void ts_flush(const char * path) {
    if (ts_g_level == TS_LEVEL_OFF) return;

    registry & r = reg();
    const int level = ts_g_level;
    ts_g_level = TS_LEVEL_OFF;          // stop recording before we read

    std::lock_guard<std::mutex> lk(r.mu);

    const int32_t pid = cur_pid();
    std::string out;
    out.reserve(1u << 22);
    out += "[\n";
    bool first = true;

    // --- process / thread metadata -----------------------------------------
    {
        static const std::string mn = "process_name";
        if (!first) out += ",\n";
        first = false;
        out += "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":";
        out += std::to_string(pid);
        out += ",\"tid\":0,\"args\":{\"name\":\"llama.cpp (tokenscope)\"}}";

        for (const auto & st : r.threads) {
            out += ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":";
            out += std::to_string(pid);
            out += ",\"tid\":";
            out += std::to_string(st->buf.tid);
            out += ",\"args\":{\"name\":\"worker ";
            out += std::to_string(st->buf.tid);
            out += "\"}}";
        }
        (void) mn;
    }

    // --- per-token slices on a synthetic track -----------------------------
    static const std::string s_prefill = "prefill";
    static const std::string s_decode  = "decode";
    for (size_t i = 0; i < r.tok_t0.size(); ++i) {
        if (r.tok_dur[i] == 0) continue;
        out_event e{};
        e.ts_us  = ts_ticks_to_us(r.tok_t0[i] - r.t_epoch);
        e.dur_us = ts_ticks_to_us(r.tok_dur[i]);
        e.tid    = -1;      // dedicated track above the workers
        e.name   = r.tok_is_prefill[i] ? &s_prefill : &s_decode;
        e.cat    = r.tok_is_prefill[i] ? "prefill" : "decode";
        e.token  = (uint32_t) i;
        emit_event(out, first, e, pid);
    }

    // --- recorded scopes ----------------------------------------------------
    static const std::string s_unknown = "<unknown>";
    static const std::string s_barrier = "barrier-wait";
    for (const auto & st : r.threads) {
        for (const auto & c : st->chunks) {
            const size_t n = (&c == &st->chunks.back()) ? st->buf.n : c->recs.size();
            for (size_t i = 0; i < n; ++i) {
                const ts_record & rec = c->recs[i];
                out_event e{};
                e.ts_us      = ts_ticks_to_us(rec.t0 - r.t_epoch);
                e.dur_us     = ts_ticks_to_us(rec.dur);
                e.tid        = st->buf.tid;
                e.token      = rec.token;
                e.depth      = rec.depth;
                e.ph_instant = (rec.kind == TS_KIND_MARK);

                if (rec.kind == TS_KIND_HOST || rec.kind == TS_KIND_MARK) {
                    e.name = rec.ref < r.names.size() ? &r.names[rec.ref] : &s_unknown;
                    // Category IS the scope name for host scopes. Bucketing them
                    // all under "host" would collapse the breakdown the tool
                    // exists to produce -- and would make outlier attribution
                    // report "the slow thing was the token", which is not news.
                    e.cat = e.name->c_str();
                } else if (rec.kind == TS_KIND_NODE_WAIT) {
                    e.name = &s_barrier;
                    e.cat  = "barrier";
                } else {
                    const graph_info * g = rec.graph < r.graphs.size() ? &r.graphs[rec.graph] : nullptr;
                    if (g && rec.ref < g->node_names.size()) {
                        e.name = &g->node_names[rec.ref];
                        e.cat  = category_for(g->node_names[rec.ref]);
                    } else {
                        e.name = &s_unknown;
                        e.cat  = "other";
                    }
                }
                emit_event(out, first, e, pid);
            }
        }
    }

    // --- level-2 aggregates -------------------------------------------------
    // Emitted as counter events so Perfetto shows them as tracks, and so the
    // Python analyzer can read exact totals without a timeline.
    if (level == TS_LEVEL_AGG) {
        for (const auto & st : r.threads) {
            for (size_t i = 0; i < st->acc_work.size(); ++i) {
                if (st->acc_work[i] == 0 && st->acc_wait[i] == 0) continue;
                const graph_info * g = ts_g_graph < r.graphs.size() ? &r.graphs[ts_g_graph] : nullptr;
                const std::string & nm = (g && i < g->node_names.size()) ? g->node_names[i] : s_unknown;
                out += ",\n{\"name\":\"agg\",\"ph\":\"i\",\"s\":\"g\",\"pid\":";
                out += std::to_string(pid);
                out += ",\"tid\":";
                out += std::to_string(st->buf.tid);
                out += ",\"ts\":0,\"args\":{\"node\":\"";
                json_escape(out, nm);
                out += "\",\"cat\":\"";
                out += category_for(nm);
                out += "\",\"work_us\":";
                char b[64];
                std::snprintf(b, sizeof(b), "%.3f", ts_ticks_to_us(st->acc_work[i]));
                out += b;
                out += ",\"wait_us\":";
                std::snprintf(b, sizeof(b), "%.3f", ts_ticks_to_us(st->acc_wait[i]));
                out += b;
                out += "}}";
            }
        }
    }

    // --- provenance ---------------------------------------------------------
    // A trace whose provenance you cannot establish is not evidence.
    {
        uint64_t dropped = 0;
        for (const auto & st : r.threads) dropped += st->buf.dropped;
        out += ",\n{\"name\":\"tokenscope\",\"ph\":\"M\",\"pid\":";
        out += std::to_string(pid);
        out += ",\"tid\":0,\"args\":{\"clock\":\"";
        out += r.clock_name;
        out += "\",\"level\":";
        out += std::to_string(level);
        out += ",\"threads\":";
        out += std::to_string(r.threads.size());
        out += ",\"tokens\":";
        out += std::to_string(r.tok_t0.size());
        out += ",\"dropped\":";
        out += std::to_string(dropped);
        out += ",\"budget_mb\":";
        out += std::to_string(r.budget_bytes >> 20);
        out += ",\"ring\":";
        out += r.ring ? "true" : "false";
        out += "}}";
    }

    out += "\n]\n";

    FILE * f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "tokenscope: could not open '%s' for writing\n", path);
        return;
    }
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);

    uint64_t dropped = 0;
    for (const auto & st : r.threads) dropped += st->buf.dropped;
    std::fprintf(stderr,
        "tokenscope: wrote %s (%zu bytes, %zu tokens, %zu threads, %llu dropped)\n",
        path, out.size(), r.tok_t0.size(), r.threads.size(),
        (unsigned long long) dropped);
}

#endif // TOKENSCOPE_ENABLED
