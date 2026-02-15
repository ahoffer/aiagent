#!/usr/bin/env bash
# Validates server-side tool execution on the native /chat path.
# The agent should invoke web_search for time-sensitive prompts and
# skip it for factual prompts answerable from training data.
# Exit codes: 0 if all checks pass, 1 if any check fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/test.env"

print_header "Native Tools Integration Tests"

# -- web_search produces cited content --
echo "--- web_search produces cited content ---"
searchResponse=$(curl -s --max-time 180 "$AGENT_URL/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "Search the web and tell me who won the Super Bowl in 2026."}' \
    2>/dev/null || echo "{}")

searchOk=$(echo "$searchResponse" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    count = int(d.get("search_count", 0) or 0)
    sources = d.get("sources", []) or []
    has_response = bool((d.get("response") or "").strip())
    if count >= 1 and len(sources) > 0 and has_response:
        print("true")
    else:
        print("false")
except Exception:
    print("false")
')
report "web_search produces cited content" "$searchOk"

if [ "$searchOk" = "false" ]; then
    echo "         Response: $(echo "$searchResponse" | head -c 200)"
fi

echo ""

# -- direct answer skips search --
echo "--- direct answer skips search ---"
directResponse=$(curl -s --max-time 120 "$AGENT_URL/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "What is 2 + 2?"}' \
    2>/dev/null || echo "{}")

directOk=$(echo "$directResponse" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    count = int(d.get("search_count", 0) or 0)
    has_response = bool((d.get("response") or "").strip())
    if count == 0 and has_response:
        print("true")
    else:
        print("false")
except Exception:
    print("false")
')
report "direct answer skips search" "$directOk"

if [ "$directOk" = "false" ]; then
    echo "         Response: $(echo "$directResponse" | head -c 200)"
fi

echo ""

# -- Summary --
print_summary
