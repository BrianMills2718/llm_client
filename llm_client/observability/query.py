"""Observability query APIs.

This module contains the concrete query logic that was previously in
``llm_client.io_log``. ``io_log`` remains a compatibility shim.
"""

from __future__ import annotations

import json
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
        "prompt_tokens, completion_tokens, timestamp, prompt_ref "
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
        "prompt_ref": row[8],
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


def get_active_llm_calls(
    *,
    project: str | None = None,
    task: str | None = None,
    trace_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return the latest known non-terminal lifecycle state for active public calls.

    A call is considered active when its most recent Foundation
    ``LLMCallLifecycle`` event has phase ``started``, ``heartbeat``,
    ``progress``, or ``stalled``. Terminal ``completed`` and ``failed`` phases
    are excluded.
    """
    db = _io_log._get_db()
    if db is None:
        return []

    clauses = ["event_type = 'LLMCallLifecycle'"]
    params: list[Any] = []
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if task is not None:
        clauses.append("task = ?")
        params.append(task)
    if trace_id is not None:
        clauses.append("trace_id = ?")
        params.append(trace_id)

    where = " AND ".join(clauses)
    rows = db.execute(
        f"""SELECT timestamp, task, trace_id, payload
            FROM foundation_events
            WHERE {where}
            ORDER BY timestamp ASC""",  # noqa: S608
        params,
    ).fetchall()

    by_call_id: dict[str, dict[str, Any]] = {}
    for timestamp, task_value, trace_value, payload_text in rows:
        try:
            payload = json.loads(payload_text)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        lifecycle = payload.get("llm_call_lifecycle")
        if not isinstance(lifecycle, dict):
            continue
        call_id = lifecycle.get("call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            continue
        record = by_call_id.setdefault(
            call_id,
            {
                "call_id": call_id,
                "requested_model_id": lifecycle.get("requested_model_id"),
                "resolved_model_id": lifecycle.get("resolved_model_id"),
                "call_kind": lifecycle.get("call_kind"),
                "prompt_ref": lifecycle.get("prompt_ref"),
                "timeout_policy": lifecycle.get("timeout_policy"),
                "provider_timeout_s": lifecycle.get("provider_timeout_s"),
                "heartbeat_interval_s": lifecycle.get("heartbeat_interval_s"),
                "stall_after_s": lifecycle.get("stall_after_s"),
                "task": task_value,
                "trace_id": trace_value,
                "started_at": timestamp,
                "last_event_at": timestamp,
                "phase": lifecycle.get("phase"),
                "elapsed_s": lifecycle.get("elapsed_s"),
                "latency_s": lifecycle.get("latency_s"),
                "progress_observable": lifecycle.get("progress_observable"),
                "progress_source": lifecycle.get("progress_source"),
                "progress_event_count": lifecycle.get("progress_event_count"),
                "last_progress_at": None,
            },
        )
        record["requested_model_id"] = lifecycle.get("requested_model_id", record["requested_model_id"])
        record["resolved_model_id"] = lifecycle.get("resolved_model_id", record["resolved_model_id"])
        record["call_kind"] = lifecycle.get("call_kind", record["call_kind"])
        record["prompt_ref"] = lifecycle.get("prompt_ref", record["prompt_ref"])
        record["timeout_policy"] = lifecycle.get("timeout_policy", record["timeout_policy"])
        record["provider_timeout_s"] = lifecycle.get("provider_timeout_s", record["provider_timeout_s"])
        record["heartbeat_interval_s"] = lifecycle.get("heartbeat_interval_s", record["heartbeat_interval_s"])
        record["stall_after_s"] = lifecycle.get("stall_after_s", record["stall_after_s"])
        record["task"] = task_value or record["task"]
        record["trace_id"] = trace_value or record["trace_id"]
        record["last_event_at"] = timestamp
        record["phase"] = lifecycle.get("phase", record["phase"])
        record["elapsed_s"] = lifecycle.get("elapsed_s", record["elapsed_s"])
        record["latency_s"] = lifecycle.get("latency_s", record["latency_s"])
        progress_observable = lifecycle.get("progress_observable")
        if isinstance(progress_observable, bool):
            record["progress_observable"] = progress_observable
        progress_source = lifecycle.get("progress_source")
        if isinstance(progress_source, str) and progress_source.strip():
            record["progress_source"] = progress_source
        progress_event_count = lifecycle.get("progress_event_count")
        if isinstance(progress_event_count, int) and progress_event_count >= 0:
            record["progress_event_count"] = progress_event_count
        if lifecycle.get("phase") == "progress":
            record["last_progress_at"] = timestamp

    active = [
        record for record in by_call_id.values()
        if record.get("phase") in {"started", "heartbeat", "progress", "stalled"}
    ]
    now = datetime.now(timezone.utc)
    for record in active:
        progress_observable = bool(record.get("progress_observable"))
        last_progress_at = record.get("last_progress_at")
        last_progress_dt = _parse_iso_datetime(last_progress_at)
        idle_for_s: float | None = None
        if progress_observable and last_progress_dt is not None:
            idle_for_s = max((now - last_progress_dt).total_seconds(), 0.0)
        record["idle_for_s"] = idle_for_s
        if progress_observable:
            if record.get("phase") == "stalled" and last_progress_dt is not None:
                record["activity_state"] = "idle"
            elif last_progress_dt is not None:
                record["activity_state"] = "progressing"
            else:
                record["activity_state"] = "waiting"
        else:
            record["activity_state"] = "waiting"
    active.sort(key=lambda record: str(record.get("last_event_at") or ""), reverse=True)
    if limit > 0:
        return active[:limit]
    return active


def import_governed_repo_hook_log(
    log_path: str | Path,
    *,
    repo_name: str | None = None,
    experiment_id: str | None = None,
    variant_id: str | None = None,
    downstream_run_id: str | None = None,
) -> int:
    """Import canonical governed-repo hook logs into shared observability."""
    from llm_client.observability.governed_repo import (
        import_governed_repo_hook_log as _import_governed_repo_hook_log,
    )

    return _import_governed_repo_hook_log(
        log_path,
        repo_name=repo_name,
        experiment_id=experiment_id,
        variant_id=variant_id,
        downstream_run_id=downstream_run_id,
    )


def get_governed_repo_friction_summary(
    *,
    repo_name: str | None = None,
    days: int = 7,
    limit: int = 10,
) -> dict[str, Any]:
    """Return a cross-repo summary of governed hook friction."""
    from llm_client.observability.governed_repo import (
        get_governed_repo_friction_summary as _get_governed_repo_friction_summary,
    )

    return _get_governed_repo_friction_summary(repo_name=repo_name, days=days, limit=limit)


def get_governed_repo_top_missing_reads(
    *,
    repo_name: str | None = None,
    days: int = 7,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Rank the most frequently missed governed-doc reads."""
    from llm_client.observability.governed_repo import (
        get_governed_repo_top_missing_reads as _get_governed_repo_top_missing_reads,
    )

    return _get_governed_repo_top_missing_reads(repo_name=repo_name, days=days, limit=limit)


def get_governed_repo_variant_comparison(
    *,
    experiment_id: str,
    repo_name: str | None = None,
    days: int = 7,
) -> dict[str, Any]:
    """Compare governed-repo friction metrics across experiment variants."""
    from llm_client.observability.governed_repo import (
        get_governed_repo_variant_comparison as _get_governed_repo_variant_comparison,
    )

    return _get_governed_repo_variant_comparison(
        experiment_id=experiment_id,
        repo_name=repo_name,
        days=days,
    )


def import_jsonl(path: str | Path, *, table: str = "llm_calls") -> int:
    return _io_log.import_jsonl(path, table=table)


def get_runs(**kwargs: Any) -> list[dict[str, Any]]:
    from llm_client.observability.experiments import get_runs as _get_runs

    return _get_runs(**kwargs)


def get_experiment_aggregates(**kwargs: Any) -> list[dict[str, Any]]:
    from llm_client.observability.experiments import (
        get_experiment_aggregates as _get_experiment_aggregates,
    )

    return _get_experiment_aggregates(**kwargs)


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    from llm_client.observability.experiments import compare_runs as _compare_runs

    return _compare_runs(run_ids)


def compare_cohorts(**kwargs: Any) -> dict[str, Any]:
    from llm_client.observability.experiments import compare_cohorts as _compare_cohorts

    return _compare_cohorts(**kwargs)


def _parse_iso_datetime(value: Any) -> datetime | None:
    """Parse common ISO-8601 timestamp strings into timezone-aware datetimes."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def get_background_mode_adoption(
    *,
    experiments_path: str | Path | None = None,
    since: str | date | datetime | None = None,
    run_id_prefix: str | None = None,
) -> dict[str, Any]:
    """Compute lightweight long-thinking adoption metrics from task-graph JSONL logs.

    The log format is the task-graph `ExperimentRecord` JSONL, where
    `dimensions.reasoning_effort` and `dimensions.background_mode` are recorded.
    """
    if experiments_path is None:
        path = Path.home() / "projects" / "data" / "task_graph" / "experiments.jsonl"
    else:
        path = Path(experiments_path)

    if since is None:
        since_dt: datetime | None = None
    elif isinstance(since, datetime):
        since_dt = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
        since_dt = since_dt.astimezone(timezone.utc)
    elif isinstance(since, date):
        since_dt = datetime.combine(since, datetime.min.time(), tzinfo=timezone.utc)
    else:
        parsed = _parse_iso_datetime(since)
        if parsed is None:
            raise ValueError(f"Invalid since timestamp: {since!r}")
        since_dt = parsed

    summary: dict[str, Any] = {
        "experiments_path": str(path),
        "exists": path.exists(),
        "total_records": 0,
        "records_considered": 0,
        "invalid_lines": 0,
        "records_with_reasoning_effort_key": 0,
        "records_with_background_mode_key": 0,
        "records_with_routing_trace": 0,
        "model_switches": 0,
        "fallback_records": 0,
        "with_reasoning_effort": 0,
        "background_mode_true": 0,
        "background_mode_false": 0,
        "background_mode_unknown": 0,
        "background_mode_rate_among_reasoning": 0.0,
        "background_mode_rate_overall": 0.0,
        "reasoning_effort_counts": {},
        "run_id_prefix": run_id_prefix,
        "since": since_dt.isoformat() if since_dt is not None else None,
    }
    if not path.exists():
        return summary

    reasoning_effort_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            summary["total_records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                summary["invalid_lines"] += 1
                continue
            if not isinstance(record, dict):
                summary["invalid_lines"] += 1
                continue

            run_id = str(record.get("run_id", "")).strip()
            if run_id_prefix and not run_id.startswith(run_id_prefix):
                continue

            if since_dt is not None:
                ts = _parse_iso_datetime(record.get("timestamp"))
                if ts is None or ts < since_dt:
                    continue

            summary["records_considered"] += 1
            dims = record.get("dimensions")
            if not isinstance(dims, dict):
                summary["background_mode_unknown"] += 1
                continue

            result_payload = record.get("result")
            if isinstance(result_payload, dict):
                switched = False
                requested = result_payload.get("requested_model")
                resolved = result_payload.get("resolved_model")
                if (
                    isinstance(requested, str)
                    and requested.strip()
                    and isinstance(resolved, str)
                    and resolved.strip()
                    and requested.strip() != resolved.strip()
                ):
                    switched = True

                routing_trace = result_payload.get("routing_trace")
                if isinstance(routing_trace, dict):
                    summary["records_with_routing_trace"] += 1
                    if routing_trace.get("normalized_from") != routing_trace.get("normalized_to"):
                        if routing_trace.get("normalized_from") and routing_trace.get("normalized_to"):
                            switched = True

                    attempted_models = routing_trace.get("attempted_models")
                    if isinstance(attempted_models, list):
                        valid_attempts = [
                            m for m in attempted_models
                            if isinstance(m, str) and m.strip()
                        ]
                        if len(valid_attempts) > 1:
                            summary["fallback_records"] += 1

                if switched:
                    summary["model_switches"] += 1

            if "reasoning_effort" in dims:
                summary["records_with_reasoning_effort_key"] += 1
            if "background_mode" in dims:
                summary["records_with_background_mode_key"] += 1

            raw_effort = dims.get("reasoning_effort")
            if isinstance(raw_effort, str) and raw_effort.strip():
                effort = raw_effort.strip().lower()
                if effort not in {"none", "null"}:
                    summary["with_reasoning_effort"] += 1
                    reasoning_effort_counts[effort] = reasoning_effort_counts.get(effort, 0) + 1

            raw_bg = dims.get("background_mode")
            bg = _normalize_bool(raw_bg)
            if bg is True:
                summary["background_mode_true"] += 1
            elif bg is False:
                summary["background_mode_false"] += 1
            else:
                summary["background_mode_unknown"] += 1

    summary["reasoning_effort_counts"] = reasoning_effort_counts

    denom_reasoning = int(summary["with_reasoning_effort"])
    if denom_reasoning > 0:
        summary["background_mode_rate_among_reasoning"] = (
            float(summary["background_mode_true"]) / float(denom_reasoning)
        )

    denom_overall = int(summary["records_considered"])
    if denom_overall > 0:
        summary["background_mode_rate_overall"] = (
            float(summary["background_mode_true"]) / float(denom_overall)
        )

    return summary
