// tokenscope — the small part that knows about ggml.
//
// Kept separate from tokenscope.h so that the core stays dependency-free and
// can be unit-tested without ggml. This header exists so the instrumentation
// site in llama-context.cpp is one line instead of a loop, which keeps the
// upstream patch small enough to read.
//
// SPDX-License-Identifier: MIT

#ifndef TOKENSCOPE_GGML_H
#define TOKENSCOPE_GGML_H

#include "tokenscope.h"

#ifndef TOKENSCOPE_ENABLED

#define TS_GGML_GRAPH(gf) do {} while (0)

#else

#include "ggml.h"

// Registers the graph's node names with tokenscope, once per distinct graph.
//
// Node events on worker threads record only the node INDEX, never a string;
// names are resolved here, on the calling thread, outside the compute loop.
// Because llama.cpp reuses graphs across decode steps this body runs about
// once per run rather than once per token. docs/01 section 3.
static inline void ts_register_ggml_graph(struct ggml_cgraph * gf) {
    if (!gf || ts_g_level < TS_LEVEL_AGG) {
        return;
    }
    const int n = ggml_graph_n_nodes(gf);
    if (!ts_graph_begin(gf, (uint32_t) n)) {
        return;   // already registered
    }
    for (int i = 0; i < n; ++i) {
        const struct ggml_tensor * nd = ggml_graph_node(gf, i);
        ts_graph_set_node((uint32_t) i, ggml_get_name(nd), ggml_op_name(nd->op));
    }
}

#define TS_GGML_GRAPH(gf) ts_register_ggml_graph(gf)

#endif // TOKENSCOPE_ENABLED
#endif // TOKENSCOPE_GGML_H
