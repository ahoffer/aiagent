#!/usr/bin/env bash
# Interactive model switcher for the agent stack.
# Sets model variables so launchers use the chosen model.
#
# Usage:
#   ./switch-model.sh                      # pick a model for all frontends
#   ./switch-model.sh aider                # pick a model for aider only
#   ./switch-model.sh goose openhands      # pick a model for goose and openhands
# Note: ollmcp has internal model switching, use /models command in the chat
#   source switch-model.sh                 # pick a model, set vars in current shell

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/defaults.sh"

# Parse frontend arguments
FRONTENDS=()
PASSTHROUGH_CMD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        aider|goose|opencode|openhands|all)
            FRONTENDS+=("$1")
            shift
            ;;
        --)
            shift
            PASSTHROUGH_CMD=("$@")
            break
            ;;
        *)
            PASSTHROUGH_CMD=("$@")
            break
            ;;
    esac
done

# Default to all frontends if none specified
if [[ ${#FRONTENDS[@]} -eq 0 ]]; then
    FRONTENDS=("all")
fi

# Fetch model list from Ollama
modelsJson=$(curl -sf --max-time 10 "$OLLAMA_URL/api/tags" 2>/dev/null)
if [[ $? -ne 0 ]] || [[ -z "$modelsJson" ]]; then
    echo "ERROR: cannot reach Ollama at $OLLAMA_URL"
    return 1 2>/dev/null || exit 1
fi

# Parse into parallel arrays of names and human-readable sizes
mapfile -t modelNames < <(python3 -c "
import json, sys
tags = json.loads(sys.stdin.read())
for m in sorted(tags.get('models', []), key=lambda x: x['name']):
    print(m['name'])
" <<< "$modelsJson")

mapfile -t modelSizes < <(python3 -c "
import json, sys
tags = json.loads(sys.stdin.read())
for m in sorted(tags.get('models', []), key=lambda x: x['name']):
    sizeBytes = m.get('size', 0)
    if sizeBytes >= 1_000_000_000:
        print(f'{sizeBytes / 1_000_000_000:.1f} GB')
    else:
        print(f'{sizeBytes / 1_000_000:.0f} MB')
" <<< "$modelsJson")

if [[ ${#modelNames[@]} -eq 0 ]]; then
    echo "No models found in Ollama."
    return 1 2>/dev/null || exit 1
fi

# Find longest model name for alignment
maxLen=0
for name in "${modelNames[@]}"; do
    (( ${#name} > maxLen )) && maxLen=${#name}
done

echo ""
echo "Available models:"
for i in "${!modelNames[@]}"; do
    num=$((i + 1))
    printf "  %2d) %-${maxLen}s  (%s)\n" "$num" "${modelNames[$i]}" "${modelSizes[$i]}"
done
echo ""

# Show which frontends will be configured
if [[ " ${FRONTENDS[*]} " == *" all "* ]]; then
    echo "Configuring: all frontends"
else
    echo "Configuring: ${FRONTENDS[*]}"
fi
echo ""

# Read selection
count=${#modelNames[@]}
while true; do
    read -rp "Select model [1-$count]: " selection
    if [[ "$selection" =~ ^[0-9]+$ ]] && (( selection >= 1 && selection <= count )); then
        break
    fi
    echo "Invalid selection, try again."
done

chosen="${modelNames[$((selection - 1))]}"

# Set the appropriate variables based on frontend selection
if [[ " ${FRONTENDS[*]} " == *" all "* ]]; then
    export AGENT_MODEL="$chosen"
    export AIDER_MODEL="$chosen"
    export GOOSE_MODEL="$chosen"
    export OPENCODE_MODEL="$chosen"
    export OPENHANDS_MODEL="$chosen"
    export MODEL="$chosen"
    echo ""
    echo "Selected: $chosen (all frontends)"
else
    for frontend in "${FRONTENDS[@]}"; do
        case "$frontend" in
            aider)
                export AIDER_MODEL="$chosen"
                ;;
            goose)
                export GOOSE_MODEL="$chosen"
                ;;
            opencode)
                export OPENCODE_MODEL="$chosen"
                ;;
            openhands)
                export OPENHANDS_MODEL="$chosen"
                ;;
        esac
    done
    echo ""
    echo "Selected: $chosen (${FRONTENDS[*]})"
fi

# If passthrough command was given, exec it
if [[ ${#PASSTHROUGH_CMD[@]} -gt 0 ]]; then
    echo "Running: ${PASSTHROUGH_CMD[*]}"
    echo ""
    exec "${PASSTHROUGH_CMD[@]}"
fi

# When sourced, vars are already in the caller's environment.
# When executed directly, remind the user.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo ""
    echo "Variables are set in this subshell only."
    echo "To set them in your current shell, run: source switch-model.sh"
fi
