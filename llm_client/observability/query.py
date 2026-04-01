"""Observability query APIs.

This module contains the concrete query logic that was previously in
``llm_client.io_log``. ``io_log`` remains a compatibility shim.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import llm_client.io_log as _io_log


def _trace_family_match(candidate: Any, trace_id: str) -> bool:
    """Return True when one stored trace belongs to the requested trace family."""

    if not isinstance(candidate, str) or not candidate:
        return False
    if candidate == trace_id:
        return True
    if candidate.startswith(trace_id + "/"):
        return True
    if f"/{trace_id}/" in candidate:
        return True
    return candidate.endswith(f"/{trace_id}")


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


def _decode_optional_json_object(payload: Any) -> dict[str, Any]:
    """Decode one optional JSON object payload conservatively."""

    if payload is None:
        return {}
    try:
        decoded = json.loads(payload) if isinstance(payload, str) else payload
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def get_active_run_progress(project: str | None = None) -> list[dict[str, Any]]:
    """Return the latest known progress state for active experiment runs."""

    db = _io_log._get_db()
    if db is None:
        return []

    clauses = ["status = 'running'"]
    params: list[Any] = []
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    where = " AND ".join(clauses)

    runs = db.execute(
        f"""SELECT run_id, timestamp, project, dataset, model, provenance, status
            FROM experiment_runs
            WHERE {where}
            ORDER BY timestamp DESC""",  # noqa: S608
        params,
    ).fetchall()

    if not runs:
        return []

    run_ids = [row[0] for row in runs]
    placeholders = ",".join("?" for _ in run_ids)
    event_rows = db.execute(
        f"""SELECT run_id, timestamp, event_type, stage, message, total, completed,
                   failed, progress_unit, avg_latency_s, checkpoint_ref, metadata, reason
            FROM experiment_run_progress_events
            WHERE run_id IN ({placeholders})
            ORDER BY id ASC""",  # noqa: S608
        run_ids,
    ).fetchall()

    latest_event_by_run: dict[str, dict[str, Any]] = {}
    latest_progress_by_run: dict[str, dict[str, Any]] = {}
    latest_progress_at_by_run: dict[str, str] = {}
    for row in event_rows:
        run_id = row[0]
        record = {
            "timestamp": row[1],
            "event_type": row[2],
            "stage": row[3],
            "message": row[4],
            "total": row[5],
            "completed": row[6],
            "failed": row[7],
            "progress_unit": row[8],
            "avg_latency_s": row[9],
            "checkpoint_ref": row[10],
            "metadata": _decode_optional_json_object(row[11]),
            "reason": row[12],
        }
        latest_event_by_run[run_id] = record
        if row[2] == "run_progress":
            latest_progress_by_run[run_id] = record
            latest_progress_at_by_run[run_id] = row[1]

    summaries: list[dict[str, Any]] = []
    for run_id, started_at, project_name, dataset, model, provenance_text, status in runs:
        provenance = _decode_optional_json_object(provenance_text)
        latest = latest_event_by_run.get(run_id, {})
        latest_progress = latest_progress_by_run.get(run_id, {})
        summaries.append(
            {
                "run_id": run_id,
                "project": project_name,
                "dataset": dataset,
                "model": model,
                "task": provenance.get("task"),
                "status": status,
                "started_at": started_at,
                "last_event_at": latest.get("timestamp", started_at),
                "stage": latest.get("stage"),
                "message": latest.get("message"),
                "total": latest_progress.get("total"),
                "completed": latest_progress.get("completed"),
                "failed": latest_progress.get("failed"),
                "progress_unit": latest_progress.get("progress_unit"),
                "avg_latency_s": latest_progress.get("avg_latency_s"),
                "checkpoint_ref": (
                    latest.get("checkpoint_ref") or latest_progress.get("checkpoint_ref")
                ),
                "last_progress_at": latest_progress_at_by_run.get(run_id),
                "stagnated": latest.get("event_type") == "run_stagnated",
                "stagnation_reason": latest.get("reason"),
                "metadata": latest.get("metadata") or None,
            }
        )

    return summaries


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


def summarize_trace(trace_id: str) -> dict[str, Any] | None:
    """Return one compact cross-table summary for a trace.

    The goal is diagnosis, not replay fidelity. The summary answers the common
    questions that surfaced during grounded-research benchmark debugging:
    which LLM calls ran, which tool calls ran, what failed, and what completed
    last without requiring bespoke SQL each time.
    """

    db = _io_log._get_db()
    if db is None:
        return None

    like_family = f"%/{trace_id}/%"
    like_suffix = f"%/{trace_id}"
    prefix = f"{trace_id}/%"

    llm_rows = db.execute(
        """
        SELECT trace_id, timestamp, model, caller, task, error, latency_s, finish_reason,
               COALESCE(marginal_cost, cost), total_tokens
        FROM llm_calls
        WHERE trace_id = ?
           OR trace_id LIKE ?
           OR trace_id LIKE ?
           OR trace_id LIKE ?
        ORDER BY timestamp ASC, id ASC
        """,
        (trace_id, prefix, like_family, like_suffix),
    ).fetchall()
    tool_rows = db.execute(
        """
        SELECT trace_id, timestamp, tool_name, operation, provider, status, task, target,
               duration_ms, error_type, error_message, cost, result_count
        FROM tool_calls
        WHERE trace_id = ?
           OR trace_id LIKE ?
           OR trace_id LIKE ?
           OR trace_id LIKE ?
        ORDER BY timestamp ASC, id ASC
        """,
        (trace_id, prefix, like_family, like_suffix),
    ).fetchall()

    llm_rows = [row for row in llm_rows if _trace_family_match(row[0], trace_id)]
    tool_rows = [row for row in tool_rows if _trace_family_match(row[0], trace_id)]

    if not llm_rows and not tool_rows:
        return None

    llm_events = [
        {
            "kind": "llm_call",
            "trace_id": row[0],
            "timestamp": row[1],
            "model": row[2],
            "caller": row[3],
            "task": row[4],
            "error": row[5],
            "latency_s": row[6],
            "finish_reason": row[7],
            "cost": row[8],
            "total_tokens": row[9],
        }
        for row in llm_rows
    ]
    tool_events = [
        {
            "kind": "tool_call",
            "trace_id": row[0],
            "timestamp": row[1],
            "tool_name": row[2],
            "operation": row[3],
            "provider": row[4],
            "status": row[5],
            "task": row[6],
            "target": row[7],
            "duration_ms": row[8],
            "error_type": row[9],
            "error_message": row[10],
            "cost": row[11],
            "result_count": row[12],
        }
        for row in tool_rows
    ]
    timeline = sorted(
        llm_events + tool_events,
        key=lambda item: (str(item.get("timestamp") or ""), item["kind"]),
    )

    last_completed_llm = next(
        (
            event for event in reversed(llm_events)
            if not event["error"]
        ),
        None,
    )
    last_tool_call = tool_events[-1] if tool_events else None

    total_llm_cost = sum(float(event["cost"] or 0.0) for event in llm_events)
    total_tool_cost = sum(float(event["cost"] or 0.0) for event in tool_events)
    total_tokens = sum(int(event["total_tokens"] or 0) for event in llm_events)

    return {
        "trace_id": trace_id,
        "llm_calls": len(llm_events),
        "tool_calls": len(tool_events),
        "llm_errors": sum(1 for event in llm_events if event["error"]),
        "tool_failures": sum(1 for event in tool_events if event["status"] == "failed"),
        "total_cost": total_llm_cost + total_tool_cost,
        "total_tokens": total_tokens,
        "first_timestamp": timeline[0]["timestamp"],
        "last_timestamp": timeline[-1]["timestamp"],
        "matched_trace_ids": sorted({event["trace_id"] for event in timeline}),
        "last_completed_llm_call": last_completed_llm,
        "last_tool_call": last_tool_call,
        "timeline": timeline,
    }


def _current_host_name() -> str | None:
    """Return the current host name for same-host lifecycle filtering."""

    try:
        hostname = socket.gethostname().strip()
    except Exception:
        return None
    return hostname or None


def _linux_process_start_token(pid: int) -> str | None:
    """Return the Linux procfs start token for one process when available."""

    if pid <= 0:
        return None
    try:
        stat_text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    _, _, remainder = stat_text.partition(") ")
    if not remainder:
        return None
    fields = remainder.split()
    if len(fields) <= 19:
        return None
    start_ticks = fields[19].strip()
    return f"linux-proc-start:{start_ticks}" if start_ticks else None


def _same_host_process_status(
    *,
    host_name: Any,
    process_id: Any,
    process_start_token: Any,
) -> bool | None:
    """Return same-host process liveness for one lifecycle record.

    Returns `True` when the originating process is definitely still alive,
    `False` when it is definitely gone or no longer matches the original
    process identity, and `None` when liveness cannot be determined honestly.
    """

    if not isinstance(host_name, str) or not host_name.strip():
        return None
    current_host = _current_host_name()
    if current_host is None or host_name.strip() != current_host:
        return None
    if not isinstance(process_id, int) or process_id <= 0:
        return None

    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return None

    if isinstance(process_start_token, str) and process_start_token.strip():
        current_token = _linux_process_start_token(process_id)
        if current_token is not None and current_token != process_start_token.strip():
            return False
    return True


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
    are excluded. Same-host calls are also excluded when the recorded process
    identity proves the originating process is gone.
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
                "host_name": lifecycle.get("host_name"),
                "process_id": lifecycle.get("process_id"),
                "process_start_token": lifecycle.get("process_start_token"),
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
        record["host_name"] = lifecycle.get("host_name", record["host_name"])
        record["process_id"] = lifecycle.get("process_id", record["process_id"])
        record["process_start_token"] = lifecycle.get(
            "process_start_token",
            record["process_start_token"],
        )
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
    filtered_active: list[dict[str, Any]] = []
    for record in active:
        process_alive = _same_host_process_status(
            host_name=record.get("host_name"),
            process_id=record.get("process_id"),
            process_start_token=record.get("process_start_token"),
        )
        record["process_alive"] = process_alive
        if process_alive is False:
            continue
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
        filtered_active.append(record)
    filtered_active.sort(key=lambda record: str(record.get("last_event_at") or ""), reverse=True)
    if limit > 0:
        return filtered_active[:limit]
    return filtered_active


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
