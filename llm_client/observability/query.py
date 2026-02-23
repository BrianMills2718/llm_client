"""Observability query APIs.

This module contains the concrete query logic that was previously in
``llm_client.io_log``. ``io_log`` remains a compatibility shim.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from llm_client import io_log as _io_log


def lookup_result(trace_id: str) -> dict[str, Any] | None:
    """Look up a successful LLM call by trace_id."""
    db = _io_log._get_db()
    if db is None:
        return None
    row = db.execute(
        "SELECT response, model, cost, latency_s, finish_reason, "
        "prompt_tokens, completion_tokens, timestamp "
        "FROM llm_calls WHERE trace_id = ? AND error IS NULL "
        "ORDER BY timestamp DESC LIMIT 1",
        (trace_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "response": row[0],
        "model": row[1],
        "cost": row[2],
        "latency_s": row[3],
        "finish_reason": row[4],
        "prompt_tokens": row[5],
        "completion_tokens": row[6],
        "timestamp": row[7],
    }


def get_completed_traces(
    *,
    project: str | None = None,
    task: str | None = None,
) -> set[str]:
    """Return trace_ids that have at least one successful call."""
    db = _io_log._get_db()
    if db is None:
        return set()
    clauses = ["error IS NULL", "trace_id IS NOT NULL"]
    params: list[str] = []
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if task is not None:
        clauses.append("task = ?")
        params.append(task)
    where = " AND ".join(clauses)
    rows = db.execute(
        f"SELECT DISTINCT trace_id FROM llm_calls WHERE {where}",  # noqa: S608
        params,
    ).fetchall()
    return {r[0] for r in rows}


def get_cost(
    *,
    trace_id: str | None = None,
    trace_prefix: str | None = None,
    task: str | None = None,
    project: str | None = None,
    since: str | date | None = None,
) -> float:
    """Query cumulative cost from observability DB."""
    if trace_id is not None and trace_prefix is not None:
        raise ValueError("trace_id and trace_prefix are mutually exclusive.")
    if not any([trace_id, trace_prefix, task, project, since]):
        raise ValueError(
            "At least one filter (trace_id, trace_prefix, task, project, since) is required."
        )
    try:
        db = _io_log._get_db()
    except Exception:
        return 0.0

    total = 0.0
    for table in ("llm_calls", "embeddings"):
        clauses: list[str] = ["error IS NULL"]
        params: list[str] = []
        if trace_id is not None:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        if trace_prefix is not None:
            clauses.append("(trace_id = ? OR trace_id LIKE ?)")
            params.extend([trace_prefix, trace_prefix + "/%"])
        if task is not None:
            clauses.append("task = ?")
            params.append(task)
        if project is not None:
            clauses.append("project = ?")
            params.append(project)
        if since is not None:
            since_str = since.isoformat() if isinstance(since, date) else since
            clauses.append("timestamp >= ?")
            params.append(since_str)
        where = " AND ".join(clauses)
        sum_expr = (
            "COALESCE(SUM(COALESCE(marginal_cost, cost)), 0)"
            if table == "llm_calls"
            else "COALESCE(SUM(cost), 0)"
        )
        row = db.execute(
            f"SELECT {sum_expr} FROM {table} WHERE {where}",  # noqa: S608
            params,
        ).fetchone()
        total += row[0] if row else 0.0
    return total


def get_trace_tree(
    trace_prefix: str,
    *,
    days: int = 7,
) -> list[dict[str, Any]]:
    """Roll up child traces under a parent prefix."""
    db = _io_log._get_db()
    if db is None:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()
    like_pattern = trace_prefix + "/%"

    rows = db.execute(
        """SELECT
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
        WHERE trace_id IS NOT NULL
          AND (trace_id = ? OR trace_id LIKE ?)
          AND timestamp >= ?
        GROUP BY trace_id
        ORDER BY MAX(timestamp) DESC""",
        (trace_prefix, like_pattern, cutoff_iso),
    ).fetchall()

    prefix_len = len(trace_prefix)
    result: list[dict[str, Any]] = []
    for r in rows:
        tid = r[0]
        if tid == trace_prefix:
            depth = 0
        else:
            suffix = tid[prefix_len + 1 :]
            depth = suffix.count("/") + 1

        result.append(
            {
                "trace_id": tid,
                "depth": depth,
                "project": r[1],
                "task": r[2],
                "total_cost_usd": r[3],
                "call_count": r[4],
                "error_count": r[5],
                "first_seen": r[6],
                "last_seen": r[7],
                "models_used": r[8].split(",") if r[8] else [],
            }
        )

    return result


def import_jsonl(path: str | Path, *, table: str = "llm_calls") -> int:
    return _io_log.import_jsonl(path, table=table)


def get_runs(**kwargs: Any) -> list[dict[str, Any]]:
    from llm_client.observability.experiments import get_runs as _get_runs

    return _get_runs(**kwargs)


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    from llm_client.observability.experiments import compare_runs as _compare_runs

    return _compare_runs(run_ids)


def compare_cohorts(**kwargs: Any) -> dict[str, Any]:
    from llm_client.observability.experiments import compare_cohorts as _compare_cohorts

    return _compare_cohorts(**kwargs)
