"""Intervention storage helpers for observability analytics.

This module owns the structured intervention log: durable records that connect
one diagnosed problem to the change made, the expected impact, and the measured
verification result. ``llm_client.io_log`` keeps the compatibility facade, but
the storage/query/update behavior lives here so intervention analytics do not
inflate the core call-logging module further.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from llm_client import io_log as _io_log

logger = logging.getLogger(__name__)


def log_intervention(
    *,
    description: str,
    problem: str,
    fix: str,
    category: str = "infra",
    project: str | None = None,
    dataset: str | None = None,
    git_commit: str | None = None,
    baseline_run_id: str | None = None,
    verification_run_id: str | None = None,
    affected_items: list[str] | None = None,
    expected_impact: str | None = None,
    measured_impact: str | None = None,
    status: str = "verified",
) -> str:
    """Persist one intervention record and return its stable intervention id."""

    conn = _io_log._get_db()
    if conn is None:
        return ""

    intervention_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()

    if git_commit is None:
        try:
            from llm_client.git_utils import get_git_head

            git_commit = get_git_head()
        except Exception:
            git_commit = None

    resolved_project = project if project is not None else _io_log._get_project()
    affected_json = json.dumps(affected_items) if affected_items else None

    try:
        conn.execute(
            """INSERT INTO interventions
            (intervention_id, timestamp, project, dataset, git_commit, category,
             description, problem, fix, baseline_run_id, verification_run_id,
             affected_items, expected_impact, measured_impact, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                intervention_id,
                timestamp,
                resolved_project,
                dataset,
                git_commit,
                category,
                description,
                problem,
                fix,
                baseline_run_id,
                verification_run_id,
                affected_json,
                expected_impact,
                measured_impact,
                status,
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Failed to log intervention: %s", exc)
        return ""

    return intervention_id


def get_interventions(
    *,
    project: str | None = None,
    dataset: str | None = None,
    category: str | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return intervention records, most recent first, with decoded item lists."""

    conn = _io_log._get_db()
    if conn is None:
        return []

    clauses: list[str] = []
    params: list[Any] = []
    if project:
        clauses.append("project = ?")
        params.append(project)
    if dataset:
        clauses.append("dataset = ?")
        params.append(dataset)
    if category:
        clauses.append("category = ?")
        params.append(category)
    if status:
        clauses.append("status = ?")
        params.append(status)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)

    rows = conn.execute(
        f"SELECT * FROM interventions {where} ORDER BY timestamp DESC LIMIT ?",
        params,
    ).fetchall()

    columns = [desc[0] for desc in conn.execute("SELECT * FROM interventions LIMIT 0").description]
    results: list[dict[str, Any]] = []
    for row in rows:
        record = dict(zip(columns, row))
        if record.get("affected_items"):
            try:
                record["affected_items"] = json.loads(record["affected_items"])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append(record)
    return results


def update_intervention(
    intervention_id: str,
    *,
    verification_run_id: str | None = None,
    measured_impact: str | None = None,
    status: str | None = None,
) -> bool:
    """Update one existing intervention with verification results."""

    conn = _io_log._get_db()
    if conn is None:
        return False

    sets: list[str] = []
    params: list[Any] = []
    if verification_run_id is not None:
        sets.append("verification_run_id = ?")
        params.append(verification_run_id)
    if measured_impact is not None:
        sets.append("measured_impact = ?")
        params.append(measured_impact)
    if status is not None:
        sets.append("status = ?")
        params.append(status)

    if not sets:
        return False

    params.append(intervention_id)
    try:
        conn.execute(
            f"UPDATE interventions SET {', '.join(sets)} WHERE intervention_id = ?",
            params,
        )
        conn.commit()
        return True
    except Exception as exc:
        logger.warning("Failed to update intervention %s: %s", intervention_id, exc)
        return False
