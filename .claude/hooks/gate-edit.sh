#!/bin/bash
# Gate edits on required reading.
# PreToolUse/Edit hook — blocks edits to llm_client/ files if coupled docs are
# not read, and appends structured log entries for operator review.
#
# Uses relationships.yaml couplings to determine what docs must be read
# and checks the session reads file (populated by track-reads.sh).
#
# Exit codes:
#   0 - Allow (all required docs read, or file not gated)
#   2 - Block (required docs not yet read)
#
# Bypass: SKIP_READ_GATE=1 in environment, or edit non-source files.

set -euo pipefail

resolve_repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || pwd
}

normalize_repo_path() {
    local raw_path="$1"
    local rel_path="$raw_path"
    if [[ "$raw_path" == "$REPO_ROOT/"* ]]; then
        rel_path="${raw_path#$REPO_ROOT/}"
    fi
    if [[ "$rel_path" == worktrees/* ]]; then
        rel_path="$(echo "$rel_path" | sed 's|^worktrees/[^/]*/||')"
    fi
    printf '%s' "$rel_path"
}

resolve_data_path() {
    local raw_path="$1"
    if [[ "$raw_path" == /* ]]; then
        printf '%s' "$raw_path"
    else
        printf '%s' "$REPO_ROOT/$raw_path"
    fi
}

log_gate_decision() {
    local decision="$1"
    local reason="$2"
    local context_emitted="$3"
    local context_bytes="$4"
    if [[ ! -f "$HOOK_LOG_SCRIPT" ]]; then
        return 0
    fi
    local -a command=(
        python "$HOOK_LOG_SCRIPT" gate
        --file-path "$REL_PATH"
        --tool-name "${TOOL_NAME:-unknown}"
        --decision "$decision"
        --reason "$reason"
        --reads-file "$READS_FILE"
        --log-file "$LOG_FILE"
        --context-bytes "$context_bytes"
    )
    if [[ "$context_emitted" == "1" ]]; then
        command+=(--context-emitted)
    fi
    if [[ -n "${HOOK_EXPERIMENT_ID:-}" ]]; then
        command+=(--experiment-id "$HOOK_EXPERIMENT_ID")
    fi
    if [[ -n "${HOOK_VARIANT_ID:-}" ]]; then
        command+=(--variant-id "$HOOK_VARIANT_ID")
    fi
    if [[ -n "${HOOK_DOWNSTREAM_RUN_ID:-}" ]]; then
        command+=(--downstream-run-id "$HOOK_DOWNSTREAM_RUN_ID")
    fi
    "${command[@]}" \
        >/dev/null
}

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null || echo "")
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null || echo "")

REPO_ROOT="$(resolve_repo_root)"
READS_FILE="$(resolve_data_path "${CLAUDE_SESSION_READS_FILE:-${LLM_CLIENT_READS_FILE:-/tmp/.claude_session_reads}}")"
LOG_FILE="$(resolve_data_path "${CLAUDE_HOOK_LOG_FILE:-.claude/hook_log.jsonl}")"
HOOK_LOG_SCRIPT="$REPO_ROOT/scripts/meta/hook_log.py"
CHECK_SCRIPT="$REPO_ROOT/scripts/check_required_reading.py"
REL_PATH="$(normalize_repo_path "$FILE_PATH")"
HOOK_EXPERIMENT_ID="${CLAUDE_HOOK_EXPERIMENT_ID:-}"
HOOK_VARIANT_ID="${CLAUDE_HOOK_VARIANT_ID:-}"
HOOK_DOWNSTREAM_RUN_ID="${CLAUDE_HOOK_DOWNSTREAM_RUN_ID:-}"

# Only gate Edit and Write
if [[ "$TOOL_NAME" != "Edit" && "$TOOL_NAME" != "Write" ]]; then
    log_gate_decision "skip" "non-edit tool" "0" "0"
    exit 0
fi

if [[ -z "$FILE_PATH" ]]; then
    log_gate_decision "skip" "missing file path" "0" "0"
    exit 0
fi

# Only gate production source files
if [[ "$FILE_PATH" != *"/llm_client/"* ]] && [[ "$FILE_PATH" != "llm_client/"* ]]; then
    log_gate_decision "skip" "non-governed path" "0" "0"
    exit 0
fi

# Bypass check
if [[ "${SKIP_READ_GATE:-}" == "1" ]]; then
    log_gate_decision "skip" "SKIP_READ_GATE=1" "0" "0"
    exit 0
fi

if [[ ! -f "$CHECK_SCRIPT" ]]; then
    log_gate_decision "block" "missing check_required_reading.py" "0" "0"
    REASON_ESCAPED=$(printf '%s' "Required-read checker missing: scripts/check_required_reading.py" | jq -Rs .)
    cat << EOF
{
  "decision": "block",
  "reason": $REASON_ESCAPED
}
EOF
    exit 2
fi

# Run the check
set +e
RESULT=$(cd "$REPO_ROOT" && python "$CHECK_SCRIPT" "$REL_PATH" --reads-file "$READS_FILE" 2>/dev/null)
CHECK_EXIT=$?
set -e

if [[ $CHECK_EXIT -ne 0 ]]; then
    log_gate_decision "block" "required reading missing" "0" "0"
    # Escape for JSON output
    RESULT_ESCAPED=$(echo "$RESULT" | jq -Rs .)

    cat << EOF
{
  "decision": "block",
  "reason": $RESULT_ESCAPED
}
EOF
    exit 2
fi

# All required reading done — output constraints as advisory context
if [[ -n "$RESULT" ]]; then
    CONTEXT_BYTES="$(printf '%s' "$RESULT" | wc -c | tr -d '[:space:]')"
    log_gate_decision "allow" "required reading satisfied" "1" "$CONTEXT_BYTES"
    RESULT_ESCAPED=$(echo "$RESULT" | jq -Rs .)
    cat << EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "additionalContext": $RESULT_ESCAPED
  }
}
EOF
else
    log_gate_decision "allow" "required reading satisfied" "0" "0"
fi

exit 0
