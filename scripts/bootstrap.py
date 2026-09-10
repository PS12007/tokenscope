#!/usr/bin/env python3
"""
bootstrap.py - produce an instrumented llama.cpp checkout.

Deliberately not a fork. A fork rots the moment upstream moves, and the diff
against upstream is the artifact that matters for the eventual PR conversation.
So instead:

    1. clone llama.cpp at a pinned commit (or reuse an existing checkout)
    2. copy tokenscope's sources into ggml/src/tokenscope/
    3. apply the BASELINE patches, which contain ONLY the surgical edits to
       existing upstream files

Step 2 keeps the profiler single-sourced in this repo. Step 3 keeps the patch
small enough that a maintainer can read the whole thing.

Only 01 and 02 are the instrumented tree. Every other patches/*.patch is an
experiment arm (F24's scheduler change, the M6 layout controls) that exists to
be measured and reverted -- 05 and 06 are alternatives on the same region and
cannot even coexist. Applying them all put F24's treatment into what every
later step calls "stock" (docs/09, A1). Pass --arm to apply one on purpose.

Usage:
    python scripts/bootstrap.py [--dest ../llama.cpp] [--no-clone] [--arm 03]
    python scripts/bootstrap.py --make-patch --dest ../llama.cpp

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/ggml-org/llama.cpp.git"
PIN = "4d9176092d00586775af140581bb0b558ddc4389"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "src")
PATCHES = os.path.join(HERE, "patches")

# Files copied verbatim into the upstream tree.
COPY = ["tokenscope.h", "tokenscope-ggml.h", "tokenscope.cpp"]

# The instrumented tree. Anything else in patches/ is an experiment arm.
BASELINE = ["01-instrument.patch", "02-name-attn-output.patch"]

# Stock mul_mat and mul_mat_id both chunk at `nth * 4`. Patches 03 and 04 each
# change one of them; the layout arms 05/06 change neither. This is the check
# that catches a contaminated baseline -- `git status` shows 9 entries either
# way, because the arms edit files 01 already touches (docs/09, A2).
CPU_C = os.path.join("ggml", "src", "ggml-cpu", "ggml-cpu.c")
STOCK_THRESHOLD = "nchunk0 * nchunk1 < nth * 4"

# Upstream files the patches touch. Listed here so --make-patch produces a
# stable, reviewable diff instead of whatever happens to be dirty.
TOUCHED = [
    "ggml/CMakeLists.txt",
    "ggml/src/CMakeLists.txt",
    "ggml/src/ggml-cpu/ggml-cpu.c",   # tier 2: the node loop
    "src/llama-context.cpp",          # tier 1: host scopes
    "src/llama-sampler.cpp",          # tier 1: sampling, via llama-cli
    "src/llama-vocab.cpp",            # tier 1: tokenize / detokenize
    "src/llama-kv-cache.cpp",         # tier 1: the cell search itself
]


def run(cmd: list[str], cwd: str | None = None, check: bool = True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and p.returncode != 0:
        print(f"$ {' '.join(cmd)}\n{p.stdout}\n{p.stderr}", file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(cmd)}")
    return p


def clone(dest: str) -> None:
    if os.path.isdir(os.path.join(dest, ".git")):
        print(f"    reusing existing checkout at {dest}")
        return
    print(f"    cloning {REPO} -> {dest}")
    run(["git", "clone", "--filter=blob:none", REPO, dest])
    run(["git", "checkout", PIN], cwd=dest)


def copy_sources(dest: str) -> None:
    out = os.path.join(dest, "ggml", "src", "tokenscope")
    os.makedirs(out, exist_ok=True)
    for f in COPY:
        shutil.copy2(os.path.join(SRC, f), os.path.join(out, f))
        print(f"    ggml/src/tokenscope/{f}")


def resolve_arm(name: str) -> str:
    """`03`, `03-mulmat-chunk-threshold` or the full filename -> the filename."""
    arms = sorted(f for f in os.listdir(PATCHES)
                  if f.endswith(".patch") and f not in BASELINE)
    hits = [f for f in arms if f == name or f.startswith(name)]
    if len(hits) != 1:
        raise SystemExit(f"--arm {name!r} matches {hits or 'nothing'}; "
                         f"arms are: {', '.join(arms)}")
    return hits[0]


def apply_patches(dest: str, arm: str | None) -> None:
    if not os.path.isdir(PATCHES):
        print("    no patches/ directory")
        return
    files = [f for f in BASELINE if os.path.isfile(os.path.join(PATCHES, f))]
    if arm:
        files.append(arm)
    skipped = sorted(f for f in os.listdir(PATCHES)
                     if f.endswith(".patch") and f not in files)
    if not files:
        print("    no patches to apply")
        return
    for f in files:
        path = os.path.join(PATCHES, f)
        chk = run(["git", "apply", "--check", path], cwd=dest, check=False)
        if chk.returncode != 0:
            rev = run(["git", "apply", "--reverse", "--check", path], cwd=dest, check=False)
            if rev.returncode == 0:
                print(f"    {f}: already applied, skipping")
                continue
            print(f"    {f}: DOES NOT APPLY\n{chk.stderr}", file=sys.stderr)
            raise SystemExit(
                "The pinned commit is " + PIN[:7] + ". If your checkout is at a "
                "different revision the patch will need rebasing.")
        run(["git", "apply", path], cwd=dest)
        print(f"    {f}: applied")
    for f in skipped:
        print(f"    {f}: experiment arm, not applied")


def verify_thresholds(dest: str, arm: str | None) -> None:
    with open(os.path.join(dest, CPU_C), encoding="utf-8") as fh:
        n = fh.read().count(STOCK_THRESHOLD)
    print(f"    `{STOCK_THRESHOLD}` occurs {n}x in {CPU_C.replace(os.sep, '/')}")
    if arm:
        print(f"    (arm {arm} requested -- stock is 2, so read this against it)")
    elif n != 2:
        raise SystemExit(
            f"STOCK IS 2, FOUND {n}. An experiment arm is compiled into what "
            "will be treated as the baseline. Reset the checkout to the pin, "
            "re-run bootstrap, and do not measure anything from this tree.")


def make_patch(dest: str) -> None:
    """Regenerate patches/01-instrument.patch from the working tree."""
    os.makedirs(PATCHES, exist_ok=True)
    p = run(["git", "diff", "--"] + TOUCHED, cwd=dest)
    out = os.path.join(PATCHES, "01-instrument.patch")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(p.stdout)
    n = sum(1 for l in p.stdout.splitlines()
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
    print(f"wrote {out} ({n} changed lines across {len(TOUCHED)} files)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=os.path.join(os.path.dirname(HERE), "llama.cpp"))
    ap.add_argument("--no-clone", action="store_true",
                    help="use an existing checkout at --dest")
    ap.add_argument("--make-patch", action="store_true",
                    help="regenerate the patch from --dest instead of applying it")
    ap.add_argument("--arm", metavar="NAME",
                    help="also apply one experiment arm, e.g. 03 (default: none)")
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    arm = resolve_arm(args.arm) if args.arm else None

    if args.make_patch:
        make_patch(dest)
        return 0

    print(f"tokenscope bootstrap -> {dest}\n")
    print("[1/4] upstream checkout")
    if not args.no_clone:
        clone(dest)
    elif not os.path.isdir(dest):
        raise SystemExit(f"--no-clone given but {dest} does not exist")

    print("[2/4] copying tokenscope sources")
    copy_sources(dest)

    print("[3/4] applying patches")
    apply_patches(dest, arm)

    print("[4/4] checking the scheduler thresholds")
    verify_thresholds(dest, arm)

    ex = ".exe" if os.name == "nt" else ""
    print(f"""
Done. Build two configurations -- you need both to measure honestly:

  # A: baseline, instrumentation compiled out
  cmake -S {dest} -B {dest}/build-ts-off -DCMAKE_BUILD_TYPE=Release \\
        -DBUILD_SHARED_LIBS=OFF -DGGML_TOKENSCOPE=OFF
  cmake --build {dest}/build-ts-off --target llama-bench

  # B/C: instrumented
  cmake -S {dest} -B {dest}/build-ts-on -DCMAKE_BUILD_TYPE=Release \\
        -DBUILD_SHARED_LIBS=OFF -DGGML_TOKENSCOPE=ON
  cmake --build {dest}/build-ts-on --target llama-bench

Then trace a run:

  TOKENSCOPE_LEVEL=1 TOKENSCOPE_OUT=run.trace.json \\
    {dest}/build-ts-on/bin/llama-bench{ex} -m MODEL.gguf -p 256 -n 128
  python tools/trace_analyze.py run.trace.json

And measure what it cost:

  python tools/bench_overhead.py --model MODEL.gguf \\
    --bin-off {dest}/build-ts-off/bin --bin-on {dest}/build-ts-on/bin
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
