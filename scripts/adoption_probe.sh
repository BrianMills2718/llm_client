#!/usr/bin/env bash
set -euo pipefail

# Run a single live long-thinking probe and append one task-graph experiment row.
# This is local-only (cron/Jenkins/manual), no GitHub Actions required.

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
EXPERIMENTS_PATH="${LLM_CLIENT_EXPERIMENTS_PATH:-$HOME/projects/data/task_graph/experiments.jsonl}"
MODEL="${LLM_CLIENT_ADOPTION_PROBE_MODEL:-gpt-5.2-pro}"
EFFORT="${LLM_CLIENT_ADOPTION_PROBE_EFFORT:-high}"
PROMPT="${LLM_CLIENT_ADOPTION_PROBE_PROMPT:-In 3 short bullets, explain why deterministic tests reduce production risk. Keep total response under 70 words.}"
RUN_ID_PREFIX="${LLM_CLIENT_ADOPTION_PROBE_RUN_ID_PREFIX:-adoption_probe}"
TIMEOUT_MINUTES="${LLM_CLIENT_ADOPTION_PROBE_TIMEOUT_MINUTES:-20}"

# Background-mode probes work best on direct OpenAI routing by default.
export LLM_CLIENT_OPENROUTER_ROUTING="${LLM_CLIENT_OPENROUTER_ROUTING:-off}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it before running adoption_probe.sh." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found/executable: $PYTHON_BIN" >&2
  exit 1
fi

echo "Running adoption probe..."
echo "  model=$MODEL effort=$EFFORT routing=$LLM_CLIENT_OPENROUTER_ROUTING"
echo "  experiments_path=$EXPERIMENTS_PATH"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from llm_client.task_graph import GraphMeta, TaskDef, TaskGraph, run_graph


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


run_id_prefix = os.environ.get("LLM_CLIENT_ADOPTION_PROBE_RUN_ID_PREFIX", "adoption_probe").strip() or "adoption_probe"
run_id = f"{run_id_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
model = os.environ.get("LLM_CLIENT_ADOPTION_PROBE_MODEL", "gpt-5.2-pro").strip() or "gpt-5.2-pro"
effort = os.environ.get("LLM_CLIENT_ADOPTION_PROBE_EFFORT", "high").strip() or "high"
prompt = os.environ.get(
    "LLM_CLIENT_ADOPTION_PROBE_PROMPT",
    "In 3 short bullets, explain why deterministic tests reduce production risk. Keep total response under 70 words.",
).strip()
timeout_minutes = _env_int("LLM_CLIENT_ADOPTION_PROBE_TIMEOUT_MINUTES", 20)
experiments_path = Path(
    os.environ.get(
        "LLM_CLIENT_EXPERIMENTS_PATH",
        str(Path.home() / "projects" / "data" / "task_graph" / "experiments.jsonl"),
    )
).expanduser()

graph = TaskGraph(
    meta=GraphMeta(
        id=run_id,
        description="Local long-thinking adoption probe",
        timeout_minutes=timeout_minutes,
        checkpoint="none",
    ),
    tasks={
        "probe": TaskDef(
            id="probe",
            difficulty=2,
            model=model,
            prompt=prompt,
            reasoning_effort=effort,
            timeout=max(60, timeout_minutes * 60 - 5),
        )
    },
    waves=[["probe"]],
)

report = asyncio.run(run_graph(graph, experiment_log=experiments_path))
if not report.task_results:
    raise SystemExit("Probe produced no task results.")

task_result = report.task_results[0]
payload = {
    "run_id": run_id,
    "status": report.status,
    "task_status": task_result.status.value,
    "requested_model": task_result.requested_model,
    "resolved_model": task_result.resolved_model,
    "reasoning_effort": task_result.reasoning_effort,
    "background_mode": task_result.background_mode,
    "duration_s": task_result.duration_s,
    "cost_usd": task_result.cost_usd,
    "error": task_result.error,
    "experiments_path": str(experiments_path),
}
print(json.dumps(payload, indent=2))
PY
