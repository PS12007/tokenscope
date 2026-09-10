#!/usr/bin/env python3
"""
runtime_sweep.py - decode throughput by thread count, across several builds.

Written for P51/F51: does F27's reversal follow the OpenMP runtime or the CPU?
It runs `llama-bench -p 0 -n N -r R` for every (arm, thread count), with the
arms interleaved and their order rotated per thread count, ascending then
descending, so a slow drift or a thermal trend lands on every arm alike and
shows up as an order effect rather than as an arm effect.

Then, optionally, the same benchmark on one arm under several environments
(`--env-arm`), for knobs such as OMP_WAIT_POLICY that act at run time.

This is a scaling survey, not an A/B: it reports llama-bench's own mean and
standard deviation per cell and makes no interval claim. Use ab_throughput.py
for a number you intend to quote.

    python tools/runtime_sweep.py -m ../models/mid.gguf \\
        --arm msvc-omp=../llama.cpp/build-ts-off/bin \\
        --arm gcc-omp=../llama.cpp/build-gcc-omp-off/bin \\
        --threads 1,2,4,8 --env-arm gcc-omp --env-threads 8 \\
        --env OMP_WAIT_POLICY=ACTIVE --env GOMP_SPINCOUNT=INFINITE \\
        --json-out sweep.json

GCC/MinGW binaries need the MSYS2 runtime ahead of Git for Windows' older
libstdc++ on PATH, or Windows shows an "entry point not found" dialog;
--prepend-path does that for the child processes only.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def bench(exe: str, model: str, t: int, n_gen: int, reps: int,
          env: dict[str, str]) -> dict:
    cmd = [exe, "-m", model, "-p", "0", "-n", str(n_gen), "-r", str(reps),
           "-t", str(t), "-o", "json"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise SystemExit(f"$ {' '.join(cmd)}\n{p.stderr[-2000:]}")
    row = json.loads(p.stdout)[0]
    return {"avg_ts": row["avg_ts"], "stddev_ts": row["stddev_ts"],
            "samples_ts": row.get("samples_ts"), "n_threads": row["n_threads"],
            "t_start": t0, "elapsed_s": round(time.time() - t0, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=BIN_DIR")
    ap.add_argument("--threads", default="1,2,4,8,16,28")
    ap.add_argument("-n", "--n-gen", type=int, default=64)
    ap.add_argument("-r", "--reps", type=int, default=5)
    ap.add_argument("--no-descending", action="store_true",
                    help="skip the descending pass (the thermal/order control)")
    ap.add_argument("--env-arm", help="arm to re-run under each --env")
    ap.add_argument("--env-threads", type=int, default=8)
    ap.add_argument("--env", action="append", default=[], metavar="K=V[,K=V]")
    ap.add_argument("--env-rounds", type=int, default=3)
    ap.add_argument("--prepend-path", default=r"C:\msys64\ucrt64\bin" if os.name == "nt" else "")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    ex = ".exe" if os.name == "nt" else ""
    arms = {}
    for a in args.arm:
        name, d = a.split("=", 1)
        exe = os.path.join(d, "llama-bench" + ex)
        if not os.path.isfile(exe):
            raise SystemExit(f"{name}: no {exe}")
        arms[name] = exe
    names = list(arms)
    threads = [int(x) for x in args.threads.split(",")]

    base_env = dict(os.environ)
    if args.prepend_path:
        base_env["PATH"] = args.prepend_path + os.pathsep + base_env.get("PATH", "")

    out = {"model": args.model, "arms": arms, "n_gen": args.n_gen,
           "reps": args.reps, "sweep": [], "env_test": []}
    passes = [("asc", threads)] + ([] if args.no_descending else [("desc", threads[::-1])])
    for pname, order in passes:
        for i, t in enumerate(order):
            k = i % len(names)
            for name in names[k:] + names[:k]:
                r = bench(arms[name], args.model, t, args.n_gen, args.reps, base_env)
                r.update({"pass": pname, "arm": name, "threads": t})
                out["sweep"].append(r)
                print(f"{pname:4s} t={t:<3d} {name:14s} {r['avg_ts']:8.2f} +- "
                      f"{r['stddev_ts']:5.2f} tok/s  ({r['elapsed_s']:.0f}s)", flush=True)

    if args.env_arm:
        variants = [("default", {})]
        for e in args.env:
            kv = dict(x.split("=", 1) for x in e.split(","))
            variants.append((e, kv))
        for rnd in range(args.env_rounds):
            k = rnd % len(variants)
            for label, kv in variants[k:] + variants[:k]:
                env = dict(base_env, **kv)
                r = bench(arms[args.env_arm], args.model, args.env_threads,
                          args.n_gen, args.reps, env)
                r.update(round=rnd, env=label, arm=args.env_arm, threads=args.env_threads)
                out["env_test"].append(r)
                print(f"env  r{rnd} {label:28s} {r['avg_ts']:8.2f} +- "
                      f"{r['stddev_ts']:5.2f} tok/s", flush=True)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
