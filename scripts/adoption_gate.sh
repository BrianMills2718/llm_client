#!/usr/bin/env bash
set -euo pipefail

# Local/scheduler-friendly adoption gate (cron/Jenkins/Buildkite/etc).
# No GitHub Actions required.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENTS_PATH="${LLM_CLIENT_EXPERIMENTS_PATH:-$HOME/projects/data/task_graph/experiments.jsonl}"
RUN_ID_PREFIX="${LLM_CLIENT_ADOPTION_RUN_ID_PREFIX:-nightly_}"
SINCE="${LLM_CLIENT_ADOPTION_SINCE:-}"
SINCE_DAYS="${LLM_CLIENT_ADOPTION_SINCE_DAYS:-}"
MIN_RATE="${LLM_CLIENT_ADOPTION_MIN_RATE:-0.95}"
METRIC="${LLM_CLIENT_ADOPTION_METRIC:-among_reasoning}"
MIN_SAMPLES="${LLM_CLIENT_ADOPTION_MIN_SAMPLES:-20}"
WARN_ONLY="${LLM_CLIENT_ADOPTION_WARN_ONLY:-0}"
EXIT_CODE="${LLM_CLIENT_ADOPTION_EXIT_CODE:-2}"

if [[ -z "$SINCE" && -n "$SINCE_DAYS" ]]; then
  SINCE="$(
    LLM_CLIENT_ADOPTION_SINCE_DAYS="$SINCE_DAYS" "$PYTHON_BIN" - <<'PY'
import os
from datetime import datetime, timedelta, timezone

raw = os.environ.get("LLM_CLIENT_ADOPTION_SINCE_DAYS", "").strip()
try:
    days = int(raw)
except Exception as exc:
    raise SystemExit(f"LLM_CLIENT_ADOPTION_SINCE_DAYS must be an integer, got {raw!r}") from exc
if days < 0:
    raise SystemExit(f"LLM_CLIENT_ADOPTION_SINCE_DAYS must be >= 0, got {days}")
print((datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat())
PY
  )"
fi

cmd=(
  "$PYTHON_BIN" -m llm_client adoption
  --experiments-path "$EXPERIMENTS_PATH"
  --run-id-prefix "$RUN_ID_PREFIX"
  --min-rate "$MIN_RATE"
  --metric "$METRIC"
  --min-samples "$MIN_SAMPLES"
  --gate-fail-exit-code "$EXIT_CODE"
)

if [[ -n "$SINCE" ]]; then
  cmd+=(--since "$SINCE")
fi

if [[ "$WARN_ONLY" == "1" || "$WARN_ONLY" == "true" || "$WARN_ONLY" == "yes" ]]; then
  cmd+=(--warn-only)
fi

cmd+=("$@")

exec "${cmd[@]}"
