#!/usr/bin/env python3
"""MCP server for llm_client observability and control.

Exposes cost tracking, trace inspection, performance analytics,
scoring, and improvement analysis to any MCP-capable agent.

Usage:
    python llm_client_mcp_server.py
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

import llm_client
from llm_client import io_log

logger = logging.getLogger(__name__)

mcp = FastMCP("llm-observability")


# ---------------------------------------------------------------------------
# Cost & Traces
# ---------------------------------------------------------------------------


@mcp.tool()
def query_cost(
    trace_id: str | None = None,
    trace_prefix: str | None = None,
    task: str | None = None,
    project: str | None = None,
    since: str | None = None,
) -> str:
    """Query cumulative LLM cost in USD.

    At least one filter required. Combines LLM call and embedding costs.

    Args:
        trace_id: Cost for a specific trace (exact match).
        trace_prefix: Cost for a trace and all its children (hierarchical).
            E.g. "openclaw.morning_brief" sums parent + all child traces.
            Mutually exclusive with trace_id.
        task: Cost for a task category (e.g. "extraction", "scoring").
        project: Cost for a project (e.g. "sam_gov", "Digimon_for_KG_application").
        since: Only include records on or after this ISO date (e.g. "2026-02-18").

    Returns:
        JSON with total_cost_usd and the filters used.
    """
    cost = io_log.get_cost(
        trace_id=trace_id,
        trace_prefix=trace_prefix,
        task=task,
        project=project,
        since=since,
    )
    return json.dumps({
        "total_cost_usd": round(cost, 6),
        "filters": {
            k: v for k, v in {
                "trace_id": trace_id, "trace_prefix": trace_prefix,
                "task": task, "project": project, "since": since,
            }.items() if v is not None
        },
    })


@mcp.tool()
def list_recent_traces(
    project: str | None = None,
    task: str | None = None,
    days: int = 7,
    limit: int = 20,
) -> str:
    """List recent traces with cost and call count rollup.

    Args:
        project: Filter to this project.
        task: Filter to this task category.
        days: Look back this many days (default 7).
        limit: Max traces to return (default 20).

    Returns:
        JSON list of traces sorted by most recent, with: trace_id,
        project, task, total_cost, call_count, first_seen, last_seen,
        models_used.
    """
    db = io_log._get_db()
    if db is None:
        return json.dumps({"error": "Observability DB not available"})

    cutoff = (datetime.now(timezone.utc).timestamp() - days * 86400)
    cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

    clauses = ["trace_id IS NOT NULL", "timestamp >= ?"]
    params: list[Any] = [cutoff_iso]

    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if task is not None:
        clauses.append("task = ?")
        params.append(task)

    where = " AND ".join(clauses)
    rows = db.execute(
        f"""SELECT
            trace_id,
            COALESCE(project, 'unknown') as project,
            COALESCE(task, 'untagged') as task,
            ROUND(SUM(CASE WHEN error IS NULL THEN cost ELSE 0 END), 6) as total_cost,
            COUNT(*) as call_count,
            SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count,
            MIN(timestamp) as first_seen,
            MAX(timestamp) as last_seen,
            GROUP_CONCAT(DISTINCT model) as models_used
        FROM llm_calls
        WHERE {where}
        GROUP BY trace_id
        ORDER BY MAX(timestamp) DESC
        LIMIT ?""",
        params + [limit],
    ).fetchall()

    traces = []
    for r in rows:
        traces.append({
            "trace_id": r[0],
            "project": r[1],
            "task": r[2],
            "total_cost_usd": r[3],
            "call_count": r[4],
            "error_count": r[5],
            "first_seen": r[6],
            "last_seen": r[7],
            "models_used": r[8].split(",") if r[8] else [],
        })

    return json.dumps(traces, indent=2)


@mcp.tool()
def get_trace_detail(trace_id: str) -> str:
    """Get detailed breakdown of a specific trace.

    Args:
        trace_id: The trace to inspect.

    Returns:
        JSON with per-call breakdown: model, task, cost, latency,
        tokens, finish_reason, error (if any). Plus summary stats.
    """
    db = io_log._get_db()
    if db is None:
        return json.dumps({"error": "Observability DB not available"})

    rows = db.execute(
        """SELECT model, task, cost, latency_s, prompt_tokens,
                  completion_tokens, finish_reason, error, timestamp
           FROM llm_calls
           WHERE trace_id = ?
           ORDER BY timestamp""",
        (trace_id,),
    ).fetchall()

    if not rows:
        return json.dumps({"error": f"No calls found for trace_id={trace_id}"})

    calls = []
    total_cost = 0.0
    total_errors = 0
    for r in rows:
        cost = r[2] or 0.0
        total_cost += cost
        if r[7]:
            total_errors += 1
        calls.append({
            "model": r[0],
            "task": r[1],
            "cost_usd": round(cost, 6),
            "latency_s": r[3],
            "prompt_tokens": r[4],
            "completion_tokens": r[5],
            "finish_reason": r[6],
            "error": r[7],
            "timestamp": r[8],
        })

    return json.dumps({
        "trace_id": trace_id,
        "summary": {
            "total_cost_usd": round(total_cost, 6),
            "call_count": len(calls),
            "error_count": total_errors,
            "duration_s": None,  # would need parsing timestamps
        },
        "calls": calls,
    }, indent=2)


# ---------------------------------------------------------------------------
# Hierarchical Traces
# ---------------------------------------------------------------------------


@mcp.tool()
def get_trace_tree(
    trace_prefix: str,
    days: int = 7,
) -> str:
    """Roll up child traces under a parent prefix.

    Hierarchical trace_ids use "/" as separator. For example, when
    OpenClaw dispatches a morning brief that spawns sam_gov research:

        trace_id = "openclaw.morning_brief/sam_gov_research_abc123"

    Querying prefix "openclaw.morning_brief" returns the parent trace
    plus all children, with per-trace cost/call/error rollup and a
    depth field (0 = parent, 1 = direct child, etc.) plus a summary.

    Args:
        trace_prefix: The parent trace_id prefix.
        days: Look-back window (default 7).

    Returns:
        JSON with: summary (total_cost, total_calls, child_count) and
        traces list with per-trace breakdown.
    """
    traces = io_log.get_trace_tree(trace_prefix, days=days)
    total_cost = sum(t["total_cost_usd"] or 0 for t in traces)
    total_calls = sum(t["call_count"] for t in traces)
    total_errors = sum(t["error_count"] for t in traces)
    return json.dumps({
        "prefix": trace_prefix,
        "summary": {
            "total_cost_usd": round(total_cost, 6),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "trace_count": len(traces),
        },
        "traces": traces,
    }, indent=2)


# ---------------------------------------------------------------------------
# Performance Analytics
# ---------------------------------------------------------------------------


@mcp.tool()
def query_performance(
    task: str | None = None,
    model: str | None = None,
    days: int = 30,
) -> str:
    """Query model performance stats from logged calls.

    Args:
        task: Filter to task category (e.g. "extraction").
        model: Filter to specific model.
        days: Look back window (default 30).

    Returns:
        JSON list with: task, model, call_count, total_cost,
        avg_latency_s, error_rate, avg_tokens.
    """
    results = llm_client.query_performance(task=task, model=model, days=days)
    return json.dumps(results, indent=2)


@mcp.tool()
def list_models(task: str | None = None) -> str:
    """List available models, optionally filtered by task suitability.

    Args:
        task: If provided, only models suitable for this task profile
              (e.g. "extraction", "synthesis", "bulk_cheap").

    Returns:
        JSON list of models with: name, litellm_id, intelligence,
        speed, cost, capabilities.
    """
    models = llm_client.list_models(task=task)
    return json.dumps(models, indent=2)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@mcp.tool()
def list_rubrics() -> str:
    """List available scoring rubrics (built-in + project-local).

    Returns:
        JSON list of rubric names (e.g. "research_quality", "code_quality").
    """
    return json.dumps(llm_client.list_rubrics())


@mcp.tool()
async def score_output(
    output: str,
    rubric: str,
    context: str = "",
    task: str | None = None,
    trace_id: str | None = None,
    judge_model: str | None = None,
) -> str:
    """Score a task output against a rubric using LLM-as-judge.

    Costs ~$0.001-0.01 per score depending on output length and judge model.

    Args:
        output: The text to score.
        rubric: Rubric name (e.g. "research_quality") or path to YAML file.
        context: Task context shown to the judge (query, config, etc.).
        task: Task tag for observability.
        trace_id: Trace ID for correlation.
        judge_model: Override judge model (default: get_model("judging")).

    Returns:
        JSON with: overall_score (0.0-1.0), dimensions (per-criterion
        scores), reasoning (per-criterion explanations), cost, latency_s.
    """
    result = await llm_client.ascore_output(
        output=output,
        rubric=rubric,
        context=context,
        task=task or "mcp_scoring",
        trace_id=trace_id or f"mcp_score_{rubric}",
        judge_model=judge_model,
    )
    return json.dumps({
        "rubric": result.rubric,
        "overall_score": result.overall_score,
        "dimensions": result.dimensions,
        "reasoning": result.reasoning,
        "judge_model": result.judge_model,
        "cost_usd": result.cost,
        "latency_s": result.latency_s,
    }, indent=2)


# ---------------------------------------------------------------------------
# Improvement Analysis
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_scores(project: str | None = None) -> str:
    """Run the self-improvement analyzer on accumulated scores.

    Reads task_scores from SQLite, classifies failure patterns
    (PROMPT_DRIFT, DATA_QUALITY, MEASUREMENT_ERROR, MODEL_OVERKILL,
    MODEL_UNDERKILL, etc.), and generates improvement proposals.

    Args:
        project: Filter scores to this project (None = all).

    Returns:
        JSON list of proposals with: category, task_id, action,
        detail, risk (low/medium/high), evidence.
    """
    proposals = llm_client.analyze_scores(project=project)
    return json.dumps(
        [
            {
                "category": p.category.value if hasattr(p.category, "value") else str(p.category),
                "task_id": p.task_id,
                "action": p.action,
                "detail": p.detail,
                "risk": p.risk,
                "evidence": p.evidence,
            }
            for p in proposals
        ],
        indent=2,
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@mcp.tool()
def list_experiment_runs(
    dataset: str | None = None,
    model: str | None = None,
    project: str | None = None,
    since: str | None = None,
    limit: int = 50,
) -> str:
    """List experiment runs with summary metrics.

    Args:
        dataset: Filter to this dataset (e.g. "HotpotQA", "MuSiQue").
        model: Filter to this model.
        project: Filter to this project.
        since: Only show runs since this ISO date.
        limit: Max runs to return (default 50).

    Returns:
        JSON list of runs with: run_id, dataset, model, n_items,
        summary_metrics, total_cost, wall_time_s, status.
    """
    runs = io_log.get_runs(
        dataset=dataset, model=model, project=project,
        since=since, limit=limit,
    )
    return json.dumps(runs, indent=2)


@mcp.tool()
def get_experiment_detail(run_id: str) -> str:
    """Get per-item detail for an experiment run.

    Args:
        run_id: The run to inspect.

    Returns:
        JSON with run summary and per-item results including
        metrics, predicted/gold, cost, latency, errors.
    """
    items = io_log.get_run_items(run_id)
    runs = io_log.get_runs(limit=1000)
    run_info = next((r for r in runs if r["run_id"] == run_id), None)
    return json.dumps({
        "run": run_info,
        "items": items,
    }, indent=2)


@mcp.tool()
def compare_experiments(run_ids: list[str]) -> str:
    """Compare 2+ experiment runs side-by-side.

    Shows summary_metrics for each run and deltas from the first
    run (baseline). Useful for A/B testing models or configurations.

    Args:
        run_ids: List of 2+ run IDs to compare.

    Returns:
        JSON with runs array and deltas_from_first array.
    """
    result = io_log.compare_runs(run_ids)
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Budget Status
# ---------------------------------------------------------------------------


@mcp.tool()
def get_budget_status(trace_id: str, max_budget: float) -> str:
    """Check remaining budget for a trace.

    Args:
        trace_id: The trace to check.
        max_budget: The budget limit in USD.

    Returns:
        JSON with: spent, remaining, budget, percent_used, exceeded.
    """
    spent = io_log.get_cost(trace_id=trace_id)
    remaining = max(0.0, max_budget - spent)
    return json.dumps({
        "trace_id": trace_id,
        "spent_usd": round(spent, 6),
        "remaining_usd": round(remaining, 6),
        "budget_usd": max_budget,
        "percent_used": round((spent / max_budget * 100) if max_budget > 0 else 0.0, 1),
        "exceeded": spent >= max_budget,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
