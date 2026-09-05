#!/usr/bin/env python3
"""
bootstrap.py - produce an instrumented llama.cpp checkout.

Deliberately not a fork. A fork rots the moment upstream moves, and the diff
against upstream is the artifact that matters for the eventual PR conversation.
So instead:

    1. clone llama.cpp at a pinned commit (or reuse an existing checkout)
    2. copy tokenscope's sources into ggml/src/tokenscope/
    3. apply patches/*.patch, which contain ONLY the surgical edits to
       existing upstream files

Step 2 keeps the profiler single-sourced in this repo. Step 3 keeps the patch
small enough that a maintainer can read the whole thing.

Usage:
    python scripts/bootstrap.py [--dest ../llama.cpp] [--no-clone]
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

# Upstream files the patches touch. Listed here so --make-patch produces a
# stable, reviewable diff instead of whatever happens to be dirty.
TOUCHED = [
    "ggml/CMakeLists.txt",
    "ggml/src/CMakeLists.txt",
    "src/llama-context.cpp",
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


def apply_patches(dest: str) -> None:
    if not os.path.isdir(PATCHES):
        print("    no patches/ directory")
        return
    files = sorted(f for f in os.listdir(PATCHES) if f.endswith(".patch"))
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
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)

    if args.make_patch:
        make_patch(dest)
        return 0

    print(f"tokenscope bootstrap -> {dest}\n")
    print("[1/3] upstream checkout")
    if not args.no_clone:
        clone(dest)
    elif not os.path.isdir(dest):
        raise SystemExit(f"--no-clone given but {dest} does not exist")

    print("[2/3] copying tokenscope sources")
    copy_sources(dest)

    print("[3/3] applying patches")
    apply_patches(dest)

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
