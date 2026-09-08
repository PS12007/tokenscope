#!/usr/bin/env python3
"""Which of a model's matmuls self-balance across threads, and which do not.

ggml's `ggml_compute_forward_mul_mat` has TWO partitioning modes, and which one
a given matmul gets is decided at runtime from its shape and the thread count:

    nchunk0 = CEIL(nr0 / chunk_size)          # chunk_size is 64 when nr1 == 1
    nchunk1 = CEIL(nr1 / chunk_size)
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        nchunk0 = nr0 > nr1 ? nth : 1;        # <-- fall back to one chunk/thread
        nchunk1 = nr0 > nr1 ? 1 : nth;
    }

    -- ggml/src/ggml-cpu/ggml-cpu.c, ggml_compute_forward_mul_mat

Above the threshold, threads pull chunks from a shared atomic counter
(`ggml_threadpool_chunk_add`) and a slow core simply takes fewer -- the work
STEALS, so core heterogeneity costs nothing. Below it, every thread is handed
exactly one equal slice and the node finishes when the slowest thread does.

That threshold moves with `nth`, which is the part worth knowing: raising the
thread count can push a matmul OUT of the self-balancing mode. See FINDINGS F23.

This computes the classification from a GGUF's tensor table. Nothing is run and
no weights are read.

    python tools/mulmat_chunking.py MODEL.gguf                 # at 8 threads
    python tools/mulmat_chunking.py MODEL.gguf -t 6,8,16,28    # a sweep
    python tools/mulmat_chunking.py MODEL.gguf --batch 64      # prefill

Caveats, in the same spirit as model_bytes.py:

  - This models the decode case (`--batch 1`) by default, where nr1 == 1 and so
    chunk_size is 64. Prefill takes the same code with a larger nr1.
  - `nr0` is taken as the tensor's output width, which is the first dimension of
    the result for the `mul_mat(W, x)` form llama.cpp builds. Fused or reshaped
    paths (llamafile sgemm, the IQ panel gemm) return before this code and are
    not modelled -- both are noted where they short-circuit.
  - NUMA forces the static path unconditionally. Assumed off.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.join(_HERE, "..", "..", "llama.cpp", "gguf-py"),
    os.path.join(_HERE, "..", "llama.cpp", "gguf-py"),
):
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.abspath(_cand))
        break

try:
    from gguf.gguf_reader import GGUFReader
except ImportError:  # pragma: no cover
    sys.exit(
        "could not import gguf-py.\n"
        "It ships with llama.cpp; this script looks for ../llama.cpp/gguf-py."
    )


# The matmul roles, in the order they run within a layer. Anything else in the
# file (norms, biases, the embedding table, which is a gather and not a matmul)
# is skipped: this is about mul_mat's partitioning, not about all work.
MATMUL_ROLES = [
    "attn_q", "attn_k", "attn_v", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
    "output",
]


def chunking(nr0: int, nr1: int, nth: int, mult: int = 4):
    """ggml_compute_forward_mul_mat's chunk plan. Returns (mode, nchunk, busy).

    `busy` is how many threads can get work at all -- which is the number of
    chunks when there are fewer chunks than threads, and nth otherwise.

    `mult` is the 4 in `nth * 4`. It is a parameter so this can model a patched
    build as well as a stock one -- see P24, which lowers it and measures what
    happens. Anything but 4 is NOT what upstream does.
    """
    chunk_size = 64 if (nr0 == 1 or nr1 == 1) else 16

    nchunk0 = (nr0 + chunk_size - 1) // chunk_size
    nchunk1 = (nr1 + chunk_size - 1) // chunk_size

    if nchunk0 * nchunk1 < nth * mult:
        # The static fallback: one chunk per thread, equal rows each.
        nchunk0 = nth if nr0 > nr1 else 1
        nchunk1 = 1 if nr0 > nr1 else nth
        mode = "static"
    else:
        mode = "dynamic"

    total = nchunk0 * nchunk1

    # A chunk is dr0 rows; threads beyond CEIL(nr0/dr0) get an empty range.
    dr0 = (nr0 + nchunk0 - 1) // nchunk0
    reachable = (nr0 + dr0 - 1) // dr0 if dr0 else 0
    busy = min(nth, total, max(reachable, 1))

    return mode, total, busy


def collect_matmuls(path: str):
    """One row per distinct matmul role, with its output width nr0."""
    reader = GGUFReader(path)

    seen = OrderedDict()
    for t in reader.tensors:
        role = re.sub(r"^blk\.\d+\.", "", t.name)
        role = re.sub(r"\.weight$|\.bias$", "", role)
        if role not in MATMUL_ROLES:
            continue
        if role in seen:
            seen[role]["n"] += 1
            continue
        # GGUFReader reports shape in ggml order: shape[0] is the row length
        # (ne00) and shape[1] the number of rows (ne01). The result's first
        # dimension -- ggml's nr0 -- is ne01.
        dims = [int(d) for d in t.shape if d]
        nr0 = dims[1] if len(dims) > 1 else dims[0]
        seen[role] = {"role": role, "nr0": nr0, "n": 1,
                      "dims": dims,
                      "order": MATMUL_ROLES.index(role)}

    rows = sorted(seen.values(), key=lambda r: r["order"])
    return reader, rows


def report(rows, nth_list, nr1: int, mult: int = 4):
    width = max(len(r["role"]) for r in rows) + 2

    header = f"  {'matmul':<{width}}{'nr0':>8}{'chunks':>9}"
    for nth in nth_list:
        header += f"{('t=' + str(nth)):>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    flips = []
    for r in rows:
        chunk_size = 64 if nr1 == 1 else 16
        nat = (r["nr0"] + chunk_size - 1) // chunk_size
        line = f"  {r['role']:<{width}}{r['nr0']:>8}{nat:>9}"
        modes = []
        for nth in nth_list:
            mode, total, busy = chunking(r["nr0"], nr1, nth, mult)
            modes.append(mode)
            tag = "dynamic" if mode == "dynamic" else f"static/{busy}"
            line += f"{tag:>12}"
        print(line)
        if len(set(modes)) > 1:
            first_static = next(
                nth for nth, m in zip(nth_list, modes) if m == "static")
            flips.append((r["role"], first_static))

    print()
    dyn_at = [nth for nth in nth_list]
    for nth in dyn_at:
        n_dyn = sum(1 for r in rows
                    if chunking(r["nr0"], nr1, nth, mult)[0] == "dynamic")
        print(f"  at {nth:>3} threads: {n_dyn} of {len(rows)} matmul roles "
              f"self-balance, {len(rows) - n_dyn} take equal slices")

    if flips:
        print()
        print("  Roles that LOSE self-balancing as threads are added:")
        for role, nth in flips:
            print(f"    {role:<20} static from {nth} threads up")
        print()
        print("  Those are the ones where core heterogeneity starts to cost,")
        print("  because an equal slice to a slower core delays the barrier.")
    return flips


def main() -> int:
    ap = argparse.ArgumentParser(
        description="classify a model's matmuls by ggml's chunking mode")
    ap.add_argument("model")
    ap.add_argument("-t", "--threads", default="8",
                    help="comma-separated thread counts (default 8)")
    ap.add_argument("--batch", type=int, default=1,
                    help="tokens in the batch; 1 is decode (default 1)")
    ap.add_argument("--mult", type=int, default=4,
                    help="the 4 in ggml's nth*4 threshold. Only 4 is what "
                         "upstream does; other values model a patched build "
                         "(P24)")
    args = ap.parse_args()

    nth_list = [int(x) for x in args.threads.split(",") if x.strip()]
    if not nth_list:
        return 2

    reader, rows = collect_matmuls(args.model)
    if not rows:
        sys.exit("no matmul tensors recognised in that file")

    def meta(key, default="?"):
        f = reader.fields.get(key)
        if f is None:
            return default
        try:
            return f.parts[f.data[0]][0]
        except Exception:
            return default

    arch = reader.fields.get("general.architecture")
    arch_s = "?"
    if arch is not None:
        try:
            arch_s = bytes(arch.parts[arch.data[0]]).decode()
        except Exception:
            pass

    print(f"=== {os.path.basename(args.model)} ===")
    print(f"    arch={arch_s}  batch={args.batch}  "
          f"(nr1={args.batch}, chunk_size={64 if args.batch == 1 else 16})"
          + ("" if args.mult == 4 else f"  THRESHOLD nth*{args.mult} (PATCHED, not upstream)"))
    print()
    report(rows, nth_list, args.batch, args.mult)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
