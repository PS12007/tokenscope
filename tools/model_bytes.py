#!/usr/bin/env python3
"""Per-phase weight bytes and parameters from a GGUF, in tokenscope's phases.

F7 predicted that time per phase tracks *bytes of weights read*, and added the
shortcut "for a uniform dtype that is just parameter count". F12 measured a real
Q4_K_M file where those two disagree and found the byte form right to within 2.9
points and the parameter form wrong by 7.2. That arithmetic was done by hand.

This does it from the file, in the same phase names `trace_analyze.py` reports,
so the byte law can be tested against any model without redoing it. It reads
only the GGUF tensor table -- no weights are loaded and the model is never run.

    python tools/model_bytes.py MODEL.gguf
    python tools/model_bytes.py MODEL.gguf --by-role     # per tensor role
    python tools/model_bytes.py MODEL.gguf --time ffn=56.4,lm_head=34.0

`--time` takes measured per-phase time shares and prints the two prediction
errors side by side, which is the F12 table.

A caveat inherited from F12: these are bytes *stored*, not bytes *fetched*. They
ignore cache reuse, which at batch size 1 with one sequence is close to nil for
weights but is not exactly nil.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

# gguf-py ships inside the llama.cpp checkout, which lives beside this repo.
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
    from gguf.constants import GGML_QUANT_SIZES
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.exit(
        "could not import gguf-py.\n"
        "It ships with llama.cpp; this script looks for ../llama.cpp/gguf-py.\n"
        "Point PYTHONPATH at it if the checkout is elsewhere."
    )


# Tensor role -> tokenscope phase. The phase names and the groupings are the
# ones in `category_for()` in src/tokenscope.cpp; keep the two in step, since
# the whole point is comparing this table against a trace.
ROLE_TO_PHASE = [
    ("attn_q_norm",  "norm"),      # Qwen3-style QK-norm, before the plain
    ("attn_k_norm",  "norm"),      # attn_q / attn_k prefixes below
    ("attn_norm",    "norm"),
    ("ffn_norm",     "norm"),
    ("output_norm",  "norm"),
    ("attn_q",       "attn.qkv"),
    ("attn_k",       "attn.qkv"),
    ("attn_v",       "attn.qkv"),
    ("attn_qkv",     "attn.qkv"),
    ("attn_output",  "attn.out"),
    ("ffn_",         "ffn"),
    ("output",       "lm_head"),
    ("token_embd",   "embed"),
]


def phase_for(role: str) -> str:
    for prefix, phase in ROLE_TO_PHASE:
        if role.startswith(prefix):
            return phase
    return "other"


def bits_per_weight(qtype) -> float:
    """Stored bits per weight for a ggml type, from gguf-py's own block table."""
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    return type_size * 8.0 / block_size


def collect(path: str):
    reader = GGUFReader(path)

    rows = []
    for t in reader.tensors:
        # "blk.17.ffn_down.weight" -> role "ffn_down"
        role = re.sub(r"^blk\.\d+\.", "", t.name)
        role = re.sub(r"\.weight$|\.bias$", "", role)
        n_params = 1
        for d in t.shape:
            if d:
                n_params *= int(d)
        bpw = bits_per_weight(t.tensor_type)
        rows.append(
            {
                "name": t.name,
                "role": role,
                "phase": phase_for(role),
                "params": n_params,
                "bytes": n_params * bpw / 8.0,
                "bpw": bpw,
                "dtype": getattr(t.tensor_type, "name", str(t.tensor_type)),
            }
        )
    return reader, rows


def aggregate(rows, key):
    agg = defaultdict(lambda: {"params": 0, "bytes": 0.0, "dtypes": set(), "n": 0})
    for r in rows:
        a = agg[r[key]]
        a["params"] += r["params"]
        a["bytes"] += r["bytes"]
        a["dtypes"].add(r["dtype"])
        a["n"] += 1
    return agg


def print_table(agg, label, measured=None):
    # `token_embd` is a single-row gather at decode, not a streamed matrix, so
    # counting it against phases that stream every byte would flatter them all.
    # It is reported, then excluded from the shares.
    streamed = {k: v for k, v in agg.items() if k != "embed"}
    tot_p = sum(v["params"] for v in streamed.values())
    tot_b = sum(v["bytes"] for v in streamed.values())
    if tot_p == 0:
        print("no streamed tensors found -- is this a GGUF model file?")
        return

    has_time = bool(measured)
    head = f"  {label:<14}{'n':>4}{'bits/w':>8}{'param share':>13}{'byte share':>12}"
    if has_time:
        head += f"{'time share':>12}{'err(param)':>12}{'err(byte)':>11}"
    print()
    print(head)
    print("  " + "-" * (len(head) - 2))

    err_p_max = err_b_max = 0.0
    for k in sorted(streamed, key=lambda k: -streamed[k]["bytes"]):
        v = streamed[k]
        ps = 100.0 * v["params"] / tot_p
        bs = 100.0 * v["bytes"] / tot_b
        bpw = v["bytes"] * 8.0 / v["params"]
        line = f"  {k:<14}{v['n']:>4}{bpw:>8.2f}{ps:>12.1f}%{bs:>11.1f}%"
        if has_time:
            ts = measured.get(k)
            if ts is None:
                line += f"{'-':>12}{'-':>12}{'-':>11}"
            else:
                # Sign convention is F12's: measured minus predicted, so a
                # negative error means the prediction was too high.
                ep, eb = ts - ps, ts - bs
                err_p_max = max(err_p_max, abs(ep))
                err_b_max = max(err_b_max, abs(eb))
                line += f"{ts:>11.1f}%{ep:>+12.1f}{eb:>+11.1f}"
        print(line)

    print()
    print(f"  streamed per token: {tot_p / 1e9:.3f} G params, "
          f"{tot_b / 2**30:.3f} GiB, {tot_b * 8.0 / tot_p:.2f} bits/weight")
    if "embed" in agg:
        e = agg["embed"]
        print(f"  token_embd (gathered, not streamed): {e['params'] / 1e9:.3f} G "
              f"params, {e['bytes'] / 2**30:.3f} GiB -- excluded above")
    if has_time:
        print()
        print(f"  max error predicting time from parameters: {err_p_max:.1f} points")
        print(f"  max error predicting time from bytes:      {err_b_max:.1f} points")
        verdict = ("bytes win" if err_b_max < err_p_max
                   else "parameters win" if err_p_max < err_b_max else "a tie")
        print(f"  -> {verdict}")


def parse_measured(s: str) -> dict[str, float]:
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            sys.exit(f"--time wants phase=share pairs, got {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to a .gguf file")
    ap.add_argument("--by-role", action="store_true",
                    help="break down by tensor role instead of tokenscope phase")
    ap.add_argument("--time", metavar="P=SHARE,...",
                    help="measured per-phase time shares, to score the two predictions")
    args = ap.parse_args()

    path = os.path.expanduser(args.model)
    if not os.path.isfile(path):
        sys.exit(f"no such file: {path}")

    reader, rows = collect(path)

    def meta(key, default="?"):
        f = reader.fields.get(key)
        if f is None:
            return default
        try:
            v = f.contents()
            return v.decode() if isinstance(v, bytes) else v
        except Exception:
            return default

    arch = meta("general.architecture")
    print(f"{os.path.basename(path)}")
    print(f"  {meta('general.name')}   arch={arch}   "
          f"{len(rows)} tensors, {sum(r['params'] for r in rows) / 1e9:.2f} G params total")

    measured = parse_measured(args.time) if args.time else None
    if args.by_role:
        print_table(aggregate(rows, "role"), "role", measured)
    else:
        print_table(aggregate(rows, "phase"), "phase", measured)

    # Same-shape tensors at different dtypes are the cleanest possible test of
    # the byte law: identical work, different bytes. Point them out.
    by_shape = defaultdict(set)
    for r in rows:
        if r["phase"] in ("attn.qkv", "attn.out", "ffn"):
            by_shape[(r["params"], r["phase"])].add((r["role"], r["dtype"], r["bpw"]))
    pairs = [(k, v) for k, v in by_shape.items() if len({d for _, d, _ in v}) > 1]
    if pairs:
        print("\n  same shape, different dtype -- a controlled test of the byte law:")
        for (nparams, phase), variants in sorted(pairs, key=lambda kv: -kv[0][0]):
            vs = sorted(variants, key=lambda x: x[2])
            desc = "  vs  ".join(f"{role} {dt} ({bpw:.2f} b/w)" for role, dt, bpw in vs)
            ratio = vs[-1][2] / vs[0][2]
            print(f"    {phase:<10} {nparams / 1e6:>7.1f} M params each: {desc}")
            print(f"    {'':<10} {'':>7}  byte ratio {ratio:.2f}x -- "
                  f"time should differ by the same factor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
