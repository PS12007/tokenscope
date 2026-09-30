#!/bin/bash
# F52 verification matrix on master 1bc7a5a. Interleaved rounds; jsonl, one row per llama-bench run.
cd /c/-CS/llama.cpp-pr || exit 1
export PATH="/c/msys64/ucrt64/bin:$PATH"
S="/c/Users/priya/AppData/Local/Temp/claude/C---CS-TLI-profiler/a9a52bd8-4044-46d0-96cf-1265c87f6a4a/scratchpad"
M="/c/-CS/TLI profiler/models/mid.gguf"
OUT=f52-matrix.jsonl
rm -f "$OUT"

run() { # run <label> <mode normal|ecoqos> <threads> <exe>
    local label=$1 mode=$2 t=$3 exe=$4 pre=""
    [ "$mode" = ecoqos ] && pre="$S/ecoqos_run.exe"
    $pre "$exe" -m "$M" -p 0 -n 64 -t "$t" -o jsonl \
        | sed "s/^{/{\"arm\":\"$label\",\"mode\":\"$mode\",/" >> "$OUT"
}

declare -A EXE=(
    [msvc-stock]=./build-omp-stock-bin/llama-bench.exe
    [msvc-fixed]=./build-omp/bin/llama-bench.exe
    [msvc-noomp]=./build-noomp/bin/llama-bench.exe
    [gcc-stock]=./build-gcc-omp-stock-bin/llama-bench.exe
    [gcc-fixed]=./build-gcc-omp/bin/llama-bench.exe
    [gcc-noomp]=./build-gcc-noomp/bin/llama-bench.exe
)
ARMS="msvc-stock msvc-fixed msvc-noomp gcc-stock gcc-fixed gcc-noomp"

for round in 1 2 3; do
    for a in $ARMS; do run "$a" ecoqos 1 "${EXE[$a]}"; done
done
for round in 1 2; do
    for a in $ARMS; do run "$a" normal 1 "${EXE[$a]}"; done
done
for round in 1 2; do
    for a in msvc-stock msvc-fixed gcc-stock gcc-fixed; do
        run "$a" normal 8 "${EXE[$a]}"
        run "$a" ecoqos 8 "${EXE[$a]}"
    done
done
echo DONE
