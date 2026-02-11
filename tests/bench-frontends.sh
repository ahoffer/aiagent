#!/usr/bin/env bash
# Measures wall-clock latency for each agent frontend across two prompts.
# Runs each prompt 3 times and reports the median. All frontends connect
# to the same Ollama instance via NodePort.
#
# Usage:
#   ./tests/bench-frontends.sh
#   MODEL=qwen3:8b ./tests/bench-frontends.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../defaults.sh"

MODEL="${MODEL:-$AGENT_MODEL}"
TIMEOUT="${TIMEOUT:-120}"
RUNS="${RUNS:-3}"

PROMPT_SIMPLE="What is 2+2?"
PROMPT_CODE="Write a Python function that checks if a string is a palindrome"

echo "================================================"
echo "  Frontend Latency Benchmark"
echo "================================================"
echo ""
echo "Model:      $MODEL"
echo "Ollama:     $OLLAMA_HOST"
echo "Runs:       $RUNS per prompt"
echo "Prompts:    [simple] $PROMPT_SIMPLE"
echo "            [code]   $PROMPT_CODE"
echo "Timestamp:  $(date -Iseconds)"
echo ""

# Time a single invocation. Sets BENCH_ELAPSED and BENCH_EXIT on return.
# Usage: time_once command [args...]
# For stdin-piped frontends: time_once_stdin "input" command [args...]
BENCH_ELAPSED=""
BENCH_EXIT=0

time_once() {
    _time_inner "" "$@"
}

time_once_stdin() {
    local stdinData="$1"
    shift
    _time_inner "$stdinData" "$@"
}

_time_inner() {
    local stdinData="$1"
    shift

    local outFile errFile timeFile
    outFile=$(mktemp)
    errFile=$(mktemp)
    timeFile=$(mktemp)

    BENCH_EXIT=0
    TIMEFORMAT='%3R'
    if [ -n "$stdinData" ]; then
        { time echo "$stdinData" | timeout "$TIMEOUT" "$@" >"$outFile" 2>"$errFile" ; } 2>"$timeFile" || BENCH_EXIT=$?
    else
        { time timeout "$TIMEOUT" "$@" >"$outFile" 2>"$errFile" ; } 2>"$timeFile" || BENCH_EXIT=$?
    fi

    BENCH_ELAPSED=$(cat "$timeFile")
    rm -f "$outFile" "$errFile" "$timeFile"
}

# Pick median from 3 values. Reads newline-separated floats from stdin.
median3() {
    sort -n | sed -n '2p'
}

# Frontend runner functions. Each takes a prompt string and runs the
# frontend once, leaving results in BENCH_ELAPSED and BENCH_EXIT.

run_aider() {
    local prompt="$1"
    export OLLAMA_API_BASE="$OLLAMA_HOST"
    local scratchDir
    scratchDir=$(mktemp -d)
    pushd "$scratchDir" >/dev/null
    time_once aider \
        --model "ollama/$MODEL" \
        --message "$prompt" \
        --yes-always \
        --no-git \
        --no-stream
    popd >/dev/null
    rm -rf "$scratchDir"
}

run_goose() {
    local prompt="$1"
    export OLLAMA_HOST
    time_once goose run \
        -t "$prompt" \
        --no-session \
        --provider ollama \
        --model "$MODEL"
}

run_ollmcp() {
    local prompt="$1"
    time_once_stdin "$prompt" ollmcp \
        -H "$OLLAMA_HOST" \
        -m "$MODEL" \
        -j "$SCRIPT_DIR/../mcp-servers.json"
}

run_opencode() {
    local prompt="$1"
    time_once opencode run \
        -m "ollama/$MODEL" \
        "$prompt"
}

run_openhands() {
    local prompt="$1"
    export LLM_API_KEY="ollama"
    export LLM_BASE_URL="$OLLAMA_HOST"
    export LLM_MODEL="ollama/$MODEL"
    time_once openhands \
        --headless \
        --override-with-envs \
        --task "$prompt"
}

# Mapping of frontend names to their runner functions and executables.
FRONTENDS="aider goose ollmcp opencode openhands"
declare -A FRONTEND_CMD=(
    [aider]=aider
    [goose]=goose
    [ollmcp]=ollmcp
    [opencode]=opencode
    [openhands]=openhands
)

printf '%-12s  %-10s  %-10s  %-10s  %s\n' "FRONTEND" "SIMPLE" "CODE" "AVG" "STATUS"

for fe in $FRONTENDS; do
    cmd="${FRONTEND_CMD[$fe]}"
    if ! command -v "$cmd" &>/dev/null; then
        printf '%-12s  %-10s  %-10s  %-10s  %s\n' "$fe" "-" "-" "-" "not installed"
        continue
    fi

    echo "--- $fe ---" >&2
    status="ok"
    simpleMedian=""
    codeMedian=""

    # Simple prompt
    simpleTimes=""
    for (( i=1; i<=RUNS; i++ )); do
        echo "  [$fe] simple run $i/$RUNS" >&2
        "run_$fe" "$PROMPT_SIMPLE"
        if [ "$BENCH_EXIT" -ne 0 ] && [ "$BENCH_EXIT" -ne 124 ]; then
            status="exit=$BENCH_EXIT"
        elif [ "$BENCH_EXIT" -eq 124 ]; then
            status="timeout"
        fi
        simpleTimes+="${BENCH_ELAPSED}"$'\n'
    done
    if [ "$status" = "ok" ]; then
        simpleMedian=$(echo -n "$simpleTimes" | median3)
    fi

    # Code prompt
    codeTimes=""
    for (( i=1; i<=RUNS; i++ )); do
        echo "  [$fe] code run $i/$RUNS" >&2
        "run_$fe" "$PROMPT_CODE"
        if [ "$BENCH_EXIT" -ne 0 ] && [ "$BENCH_EXIT" -ne 124 ]; then
            status="exit=$BENCH_EXIT"
        elif [ "$BENCH_EXIT" -eq 124 ]; then
            status="timeout"
        fi
        codeTimes+="${BENCH_ELAPSED}"$'\n'
    done
    if [ "$status" = "ok" ]; then
        codeMedian=$(echo -n "$codeTimes" | median3)
    fi

    # Compute average of the two medians
    avgDisplay="-"
    simpleDisplay="-"
    codeDisplay="-"
    if [ -n "$simpleMedian" ] && [ -n "$codeMedian" ]; then
        avg=$(awk "BEGIN {printf \"%.1f\", ($simpleMedian + $codeMedian) / 2}")
        avgDisplay="${avg}s"
        simpleDisplay="${simpleMedian}s"
        codeDisplay="${codeMedian}s"
    fi

    printf '%-12s  %-10s  %-10s  %-10s  %s\n' "$fe" "$simpleDisplay" "$codeDisplay" "$avgDisplay" "$status"
done

echo ""
echo "================================================"
echo "  Benchmark complete"
echo "================================================"
