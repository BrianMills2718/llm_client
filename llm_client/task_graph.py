"""Task graph runner — parse YAML DAGs, dispatch to agents, validate, checkpoint.

Dumb runner, smart coordinator. The runner is mechanical — parse, sort, dispatch,
validate, log. OpenClaw (the LLM coordinator) does all reasoning about what to run.

Usage:
    from llm_client.task_graph import load_graph, run_graph

    graph = load_graph("path/to/graph.yaml")
    report = await run_graph(graph)
    for tr in report.task_results:
        print(f"{tr.task_id}: {tr.status} (${tr.cost_usd:.3f})")

See docs/TASK_GRAPH_DESIGN.md for full specification.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llm_client.client import acall_llm as _acall_llm
from llm_client.difficulty import get_effective_tier, get_model_for_difficulty
from llm_client.validators import ValidationResult, run_validators, spec_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    SPEC_LOCKED = "spec_locked"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskDef(BaseModel):
    """A single task in the graph."""

    model_config = {"populate_by_name": True}

    id: str
    difficulty: int
    agent: str = "codex"
    prompt: str
    depends_on: list[str] = []
    mcp_servers: list[str] = []
    working_directory: str | None = None
    validators: list[dict[str, Any]] = Field(default=[], alias="validate")
    outputs: dict[str, str] = {}
    model: str | None = None  # Override difficulty router
    investigate_first: list[str] = []
    timeout: int = 300  # Per-task timeout in seconds


class GraphMeta(BaseModel):
    """Top-level graph metadata."""

    id: str
    description: str = ""
    timeout_minutes: int = 120
    checkpoint: str = "git_tag"  # git_tag | git_commit | none


class TaskGraph(BaseModel):
    """Parsed task graph ready for execution."""

    meta: GraphMeta
    tasks: dict[str, TaskDef]
    waves: list[list[str]] = []  # Populated by toposort_waves()


class Uncertainty(BaseModel):
    """An unknown discovered during execution."""

    question: str
    status: str = "deferred"
    resolution: str | None = None
    raised_at: str = ""


class TaskResult(BaseModel):
    """Result of executing a single task."""

    task_id: str
    status: TaskStatus
    wave: int
    model_selected: str | None = None
    difficulty: int = 0
    duration_s: float = 0.0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    validation_results: list[ValidationResult] = []
    agent_output: str | None = None  # Truncated agent response
    spec_hash: str = ""
    uncertainties: list[Uncertainty] = []
    error: str | None = None


class ExperimentRecord(BaseModel):
    """Structured per-task experiment record written to JSONL."""

    run_id: str
    task_id: str
    wave: int
    timestamp: str
    hypothesis: str
    difficulty: int
    model_selected: str | None
    agent: str
    result: dict[str, Any]
    dimensions: dict[str, Any] = {}
    outcome: str  # "confirmed" | "hypothesis_rejected" | "error"
    prior_tier: int | None = None
    learning: str | None = None


class ExecutionReport(BaseModel):
    """Aggregate results for a full graph run."""

    graph_id: str
    started_at: str
    finished_at: str
    status: str  # "completed" | "partial" | "failed"
    task_results: list[TaskResult]
    total_cost_usd: float = 0.0
    total_duration_s: float = 0.0
    waves_completed: int = 0
    waves_total: int = 0


# ---------------------------------------------------------------------------
# Graph parsing and DAG validation
# ---------------------------------------------------------------------------


def load_graph(path: str | Path) -> TaskGraph:
    """Parse a YAML task graph file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed TaskGraph with waves populated.

    Raises:
        FileNotFoundError: YAML file not found.
        ValueError: Invalid graph structure (cycles, missing deps, etc.).
    """
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())

    meta = GraphMeta(**raw["graph"])
    tasks: dict[str, TaskDef] = {}
    for task_id, task_raw in raw.get("tasks", {}).items():
        task_raw["id"] = task_id
        tasks[task_id] = TaskDef(**task_raw)

    graph = TaskGraph(meta=meta, tasks=tasks)
    _validate_dag(graph)
    graph.waves = toposort_waves(graph)
    return graph


def _validate_dag(graph: TaskGraph) -> None:
    """Validate DAG integrity: no cycles, all deps exist."""
    task_ids = set(graph.tasks.keys())

    # Check all dependencies reference existing tasks
    for task_id, task in graph.tasks.items():
        for dep in task.depends_on:
            if dep not in task_ids:
                raise ValueError(
                    f"Task {task_id!r} depends on {dep!r}, which doesn't exist. "
                    f"Available: {sorted(task_ids)}"
                )

    # Cycle detection via DFS
    visited: set[str] = set()
    in_stack: set[str] = set()

    def _dfs(node: str) -> None:
        if node in in_stack:
            raise ValueError(f"Cycle detected in task graph involving {node!r}")
        if node in visited:
            return
        in_stack.add(node)
        for dep in graph.tasks[node].depends_on:
            _dfs(dep)
        in_stack.remove(node)
        visited.add(node)

    for task_id in graph.tasks:
        _dfs(task_id)


def toposort_waves(graph: TaskGraph) -> list[list[str]]:
    """Group tasks into parallel execution waves via topological sort.

    Tasks in the same wave have no dependencies on each other and can
    run concurrently.

    Returns:
        List of waves, each wave is a list of task IDs.
    """
    remaining = set(graph.tasks.keys())
    completed: set[str] = set()
    waves: list[list[str]] = []

    while remaining:
        # Find tasks whose dependencies are all completed
        wave = [
            tid for tid in remaining
            if all(dep in completed for dep in graph.tasks[tid].depends_on)
        ]
        if not wave:
            raise ValueError(
                f"Could not make progress — possible cycle. Remaining: {remaining}"
            )
        wave.sort()  # Deterministic ordering
        waves.append(wave)
        completed.update(wave)
        remaining -= set(wave)

    return waves


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

_TEMPLATE_RE = re.compile(r"\{(\w+)\.outputs\.(\w+)\}")


def _resolve_templates(
    text: str,
    completed_results: dict[str, TaskResult],
    completed_outputs: dict[str, dict[str, str]],
) -> str:
    """Resolve {task_id.outputs.key} references in prompt text."""

    def _replace(m: re.Match[str]) -> str:
        task_id = m.group(1)
        key = m.group(2)
        outputs = completed_outputs.get(task_id, {})
        if key in outputs:
            return outputs[key]
        return m.group(0)  # Leave unresolved if not found

    return _TEMPLATE_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------


async def run_graph(
    graph: TaskGraph,
    *,
    experiment_log: str | Path | None = None,
    model_floors: dict[str, dict[str, Any]] | None = None,
    mcp_server_configs: dict[str, dict[str, Any]] | None = None,
    working_directory: str | None = None,
    dry_run: bool = False,
) -> ExecutionReport:
    """Execute a task graph.

    Args:
        graph: Parsed TaskGraph (from load_graph).
        experiment_log: Path to experiments.jsonl. Defaults to
            ~/projects/data/task_graph/experiments.jsonl.
        model_floors: Learned model floors. None = load from disk.
        mcp_server_configs: Dict mapping MCP server names to startup configs
            (command, args, env). Required for tasks that use mcp_servers.
        working_directory: Default working directory for all tasks.
        dry_run: If True, show what would happen without dispatching agents.

    Returns:
        ExecutionReport with per-task results.
    """
    if experiment_log is None:
        experiment_log = Path.home() / "projects" / "data" / "task_graph" / "experiments.jsonl"
    else:
        experiment_log = Path(experiment_log)

    started_at = datetime.now(timezone.utc)
    task_results: dict[str, TaskResult] = {}
    task_outputs: dict[str, dict[str, str]] = {}
    waves_completed = 0

    for wave_idx, wave in enumerate(graph.waves):
        logger.info("Wave %d/%d: %s", wave_idx + 1, len(graph.waves), wave)

        if dry_run:
            for task_id in wave:
                task = graph.tasks[task_id]
                model = get_model_for_difficulty(
                    get_effective_tier(task_id, task.difficulty, model_floors),
                    override_model=task.model,
                )
                val_results = run_validators(task.validators, dry_run=True) if task.validators else []
                task_results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    wave=wave_idx,
                    model_selected=model,
                    difficulty=task.difficulty,
                    validation_results=val_results,
                    spec_hash=spec_hash(task.model_dump()),
                )
            waves_completed += 1
            continue

        # Dispatch all tasks in this wave concurrently
        wave_tasks = [
            _execute_task(
                task=graph.tasks[task_id],
                wave_idx=wave_idx,
                graph=graph,
                model_floors=model_floors,
                mcp_server_configs=mcp_server_configs or {},
                completed_results=task_results,
                completed_outputs=task_outputs,
                working_directory=working_directory,
            )
            for task_id in wave
        ]

        results = await asyncio.gather(*wave_tasks, return_exceptions=True)

        wave_failed = False
        for task_id, result in zip(wave, results):
            if isinstance(result, Exception):
                task_results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    wave=wave_idx,
                    error=str(result),
                    spec_hash=spec_hash(graph.tasks[task_id].model_dump()),
                )
                wave_failed = True
            else:
                task_results[task_id] = result
                if result.status == TaskStatus.COMPLETED:
                    task_outputs[task_id] = graph.tasks[task_id].outputs
                else:
                    wave_failed = True

        waves_completed += 1

        # Write experiment records for this wave
        for task_id in wave:
            tr = task_results[task_id]
            record = _make_experiment_record(graph, tr)
            _append_experiment(experiment_log, record)

        # Git checkpoint after successful wave
        if not wave_failed and graph.meta.checkpoint != "none":
            _git_checkpoint(graph.meta, wave_idx)

        # Stop on failure
        if wave_failed:
            logger.error(
                "Wave %d failed. Stopping graph execution.", wave_idx + 1
            )
            break

    finished_at = datetime.now(timezone.utc)
    all_results = list(task_results.values())

    # Determine overall status
    failed = [r for r in all_results if r.status == TaskStatus.FAILED]
    if not failed and waves_completed == len(graph.waves):
        status = "completed"
    elif waves_completed > 0:
        status = "partial"
    else:
        status = "failed"

    return ExecutionReport(
        graph_id=graph.meta.id,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
        status=status,
        task_results=all_results,
        total_cost_usd=sum(r.cost_usd for r in all_results),
        total_duration_s=(finished_at - started_at).total_seconds(),
        waves_completed=waves_completed,
        waves_total=len(graph.waves),
    )


async def _execute_task(
    *,
    task: TaskDef,
    wave_idx: int,
    graph: TaskGraph,
    model_floors: dict[str, dict[str, Any]] | None,
    mcp_server_configs: dict[str, dict[str, Any]],
    completed_results: dict[str, TaskResult],
    completed_outputs: dict[str, dict[str, str]],
    working_directory: str | None,
) -> TaskResult:
    """Execute a single task through its full lifecycle."""

    task_start = time.monotonic()

    # --- SPEC_LOCKED ---
    frozen_hash = spec_hash(task.model_dump())

    # --- Model selection ---
    effective_tier = get_effective_tier(task.id, task.difficulty, model_floors)
    model = get_model_for_difficulty(effective_tier, override_model=task.model)

    if model is None:
        # Tier 0: scripted task, just run validation
        val_results = run_validators(task.validators) if task.validators else []
        passed = all(v.passed for v in val_results)
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED if passed else TaskStatus.FAILED,
            wave=wave_idx,
            model_selected=None,
            difficulty=task.difficulty,
            duration_s=time.monotonic() - task_start,
            validation_results=val_results,
            spec_hash=frozen_hash,
        )

    # --- Build prompt ---
    prompt = _resolve_templates(task.prompt, completed_results, completed_outputs)

    # Prepend investigation questions
    if task.investigate_first:
        investigation = "INVESTIGATION PHASE — Answer these questions before proceeding:\n"
        for i, q in enumerate(task.investigate_first, 1):
            investigation += f"{i}. {q}\n"
        investigation += "\nAfter answering the investigation questions, proceed with the task:\n\n"
        prompt = investigation + prompt

    # Append validation preview so the agent knows what will be checked
    if task.validators:
        val_preview = run_validators(task.validators, dry_run=True)
        prompt += "\n\nVALIDATION CRITERIA (your output will be checked against these):\n"
        for vr in val_preview:
            prompt += f"- {vr.reason}\n"

    messages = [{"role": "user", "content": prompt}]

    # --- Build kwargs ---
    kwargs: dict[str, Any] = {
        "timeout": task.timeout,
        "task": f"taskgraph:{task.id}",
        "num_retries": 0,  # Agent calls have side effects; no auto-retry
    }

    # MCP servers
    if task.mcp_servers:
        mcp_configs = {}
        for server_name in task.mcp_servers:
            if server_name in mcp_server_configs:
                mcp_configs[server_name] = mcp_server_configs[server_name]
            else:
                raise ValueError(
                    f"Task {task.id!r} requests MCP server {server_name!r} "
                    f"but no config provided in mcp_server_configs. "
                    f"Available: {sorted(mcp_server_configs.keys())}"
                )
        if mcp_configs:
            kwargs["mcp_servers"] = mcp_configs

    # Working directory
    wd = task.working_directory or working_directory
    if wd:
        wd_expanded = str(Path(wd).expanduser().resolve())
        if task.agent == "codex" or task.agent.startswith("codex/"):
            kwargs["working_directory"] = wd_expanded
        elif task.agent.startswith("claude-code"):
            kwargs["cwd"] = wd_expanded

    # --- DISPATCHED → RUNNING ---
    logger.info(
        "Dispatching %s: model=%s tier=%d agent=%s",
        task.id, model, effective_tier, task.agent,
    )

    try:
        result = await asyncio.wait_for(
            _acall_llm(model, messages, **kwargs),
            timeout=task.timeout,
        )
    except asyncio.TimeoutError:
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            wave=wave_idx,
            model_selected=model,
            difficulty=task.difficulty,
            duration_s=time.monotonic() - task_start,
            spec_hash=frozen_hash,
            error=f"Task timed out after {task.timeout}s",
        )
    except Exception as e:
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.FAILED,
            wave=wave_idx,
            model_selected=model,
            difficulty=task.difficulty,
            duration_s=time.monotonic() - task_start,
            spec_hash=frozen_hash,
            error=f"{type(e).__name__}: {e}",
        )

    # --- VALIDATING ---
    agent_output = result.content
    if agent_output and len(agent_output) > 5000:
        agent_output = agent_output[:5000] + "...[truncated]"

    cost = result.cost or 0.0
    usage = result.usage or {}
    tokens_in = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
    tokens_out = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

    val_results: list[ValidationResult] = []
    if task.validators:
        val_results = run_validators(task.validators)

    passed = all(v.passed for v in val_results) if val_results else True
    status = TaskStatus.COMPLETED if passed else TaskStatus.FAILED

    duration = time.monotonic() - task_start

    return TaskResult(
        task_id=task.id,
        status=status,
        wave=wave_idx,
        model_selected=model,
        difficulty=task.difficulty,
        duration_s=round(duration, 2),
        cost_usd=round(cost, 6),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        validation_results=val_results,
        agent_output=agent_output,
        spec_hash=frozen_hash,
        error=None if passed else "; ".join(
            v.reason for v in val_results if not v.passed and v.reason
        ),
    )


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


def _make_experiment_record(graph: TaskGraph, tr: TaskResult) -> ExperimentRecord:
    """Create an experiment record from a task result."""
    task = graph.tasks[tr.task_id]
    if tr.status == TaskStatus.COMPLETED:
        outcome = "confirmed"
    elif tr.status == TaskStatus.FAILED:
        outcome = "hypothesis_rejected" if tr.error and "validation" in tr.error.lower() else "error"
    else:
        outcome = "in_progress"

    return ExperimentRecord(
        run_id=graph.meta.id,
        task_id=tr.task_id,
        wave=tr.wave,
        timestamp=datetime.now(timezone.utc).isoformat(),
        hypothesis=f"{task.agent} at tier {tr.difficulty} can handle {tr.task_id}",
        difficulty=tr.difficulty,
        model_selected=tr.model_selected,
        agent=task.agent,
        result={
            "status": tr.status.value,
            "duration_s": tr.duration_s,
            "cost_usd": tr.cost_usd,
            "tokens_in": tr.tokens_in,
            "tokens_out": tr.tokens_out,
            "validation_results": [v.model_dump() for v in tr.validation_results],
        },
        dimensions={
            "cost_per_second": round(tr.cost_usd / tr.duration_s, 6) if tr.duration_s > 0 else 0,
        },
        outcome=outcome,
    )


def _append_experiment(path: Path, record: ExperimentRecord) -> None:
    """Append an experiment record to JSONL. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(record.model_dump_json() + "\n")
    except Exception:
        logger.warning("Failed to write experiment record", exc_info=True)


# ---------------------------------------------------------------------------
# Git checkpointing
# ---------------------------------------------------------------------------


def _git_checkpoint(meta: GraphMeta, wave_idx: int) -> None:
    """Create a git tag or commit after a successful wave. Never raises."""
    import subprocess

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"taskgraph/{meta.id}/wave_{wave_idx}_{ts}"

    try:
        if meta.checkpoint == "git_tag":
            subprocess.run(
                ["git", "tag", tag],
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info("Git tag created: %s", tag)
        elif meta.checkpoint == "git_commit":
            subprocess.run(
                ["git", "add", "-A"],
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "commit", "-m", f"taskgraph checkpoint: {tag}",
                 "--allow-empty"],
                capture_output=True,
                timeout=10,
            )
            logger.info("Git commit created: %s", tag)
    except Exception:
        logger.warning("Git checkpoint failed", exc_info=True)
