"""Governed-repo friction telemetry import and query helpers.

Repo-local read-gating hooks write canonical JSONL logs under
``.claude/hook_log.jsonl``. This module imports those edge-buffer logs into the
shared ``llm_client`` observability substrate and exposes the first operator
queries needed to evaluate governed-repo rollout quality.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from llm_client import io_log as _io_log
from llm_client.foundation import sha256_text, validate_foundation_event


class CanonicalHookLogRow(BaseModel):
    """Normalized repo-local hook log row emitted by governed-repo hooks."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(ge=1)
    timestamp: str
    hook: str = Field(min_length=1)
    tool_name: str | None = None
    file_path: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    decision_reason: str | None = None
    reads_file: str | None = None
    required_reads: list[str] = Field(default_factory=list)
    reads_completed: list[str] = Field(default_factory=list)
    missing_reads: list[str] = Field(default_factory=list)
    coupled_docs: list[str] = Field(default_factory=list)
    context_bytes: int | None = Field(default=None, ge=0)
    experiment_id: str | None = None
    variant_id: str | None = None

    @field_validator(
        "timestamp",
        "hook",
        "tool_name",
        "file_path",
        "decision",
        "decision_reason",
        "reads_file",
        "experiment_id",
        "variant_id",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator(
        "required_reads",
        "reads_completed",
        "missing_reads",
        "coupled_docs",
        mode="before",
    )
    @classmethod
    def _normalize_text_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("expected a list of strings")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("expected a list of strings")
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
        return normalized

    @model_validator(mode="after")
    def _validate_hook_specific_fields(self) -> "CanonicalHookLogRow":
        _normalize_timestamp(self.timestamp)
        if self.hook == "gate-edit" and not self.tool_name:
            raise ValueError("gate-edit rows require tool_name")
        return self


def _normalize_timestamp(raw_timestamp: str) -> str:
    """Return a stable UTC ISO-8601 timestamp for imported hook events."""
    text = raw_timestamp.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {raw_timestamp!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


def _hash_suffix(text: str, *, length: int = 24) -> str:
    """Return a deterministic lowercase hash suffix safe for Foundation ids."""
    return sha256_text(text).split(":", 1)[1][:length]


def _infer_repo_name(log_path: Path) -> str:
    """Infer the repo name from ``<repo>/.claude/hook_log.jsonl`` paths."""
    resolved = log_path.resolve()
    if resolved.parent.name != ".claude":
        raise ValueError(
            "repo_name is required when hook logs are not stored under '<repo>/.claude/'"
        )
    repo_name = resolved.parent.parent.name.strip()
    if not repo_name:
        raise ValueError(f"unable to infer repo name from hook log path: {log_path}")
    return repo_name


def _session_source_for_row(row: CanonicalHookLogRow) -> str:
    """Return the best available stable session source for one hook row."""
    if row.reads_file:
        return f"reads_file:{row.reads_file}"
    return f"synthetic:{row.hook}:{row.file_path}"


def _build_event(
    *,
    repo_name: str,
    row: CanonicalHookLogRow,
    raw_line: str,
) -> dict[str, Any]:
    """Translate one canonical hook-log row into a Foundation event payload."""
    event_timestamp = _normalize_timestamp(row.timestamp)
    session_source = _session_source_for_row(row)
    session_id = f"sess_govhook_{_hash_suffix(repo_name + '|' + session_source)}"
    run_id = f"run_governed_repo_{_hash_suffix(repo_name, length=16)}"
    event_id = f"evt_govhook_{_hash_suffix(repo_name + '|' + raw_line)}"
    event_payload = {
        "event_id": event_id,
        "event_type": "GovernedRepoHook",
        "timestamp": event_timestamp,
        "run_id": run_id,
        "session_id": session_id,
        "actor_id": "service:llm_client:governed_repo_importer:1",
        "operation": {
            "name": "import_governed_repo_hook_log",
            "version": "1.0.0",
        },
        "inputs": {
            "artifact_ids": [],
            "params": {
                "repo_name": repo_name,
                "hook": row.hook,
                "schema_version": row.schema_version,
                "file_path": row.file_path,
            },
            "bindings": {},
        },
        "outputs": {"artifact_ids": [], "payload_hashes": []},
        "governed_repo_hook": {
            "repo_name": repo_name,
            "hook_name": row.hook,
            "decision": row.decision,
            "file_path": row.file_path,
            "tool_name": row.tool_name,
            "decision_reason": row.decision_reason,
            "reads_file": row.reads_file,
            "required_reads": row.required_reads,
            "reads_completed": row.reads_completed,
            "missing_reads": row.missing_reads,
            "coupled_docs": row.coupled_docs,
            "context_bytes": row.context_bytes,
            "session_source": session_source,
            "experiment_id": row.experiment_id,
            "variant_id": row.variant_id,
        },
    }
    return validate_foundation_event(event_payload)


def _iter_import_rows(log_path: Path) -> list[tuple[int, str, CanonicalHookLogRow]]:
    """Load and validate non-empty hook-log rows, failing loudly on malformed lines."""
    rows: list[tuple[int, str, CanonicalHookLogRow]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{log_path}:{line_number}: invalid JSON hook row") from exc
            try:
                row = CanonicalHookLogRow.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{log_path}:{line_number}: invalid hook row: {exc}") from exc
            rows.append((line_number, stripped, row))
    return rows


def import_governed_repo_hook_log(
    log_path: str | Path,
    *,
    repo_name: str | None = None,
) -> int:
    """Import canonical governed-repo hook logs into shared Foundation observability.

    Returns the number of new events inserted. Re-importing the same hook log is
    idempotent because imported event ids are deterministic and the shared
    ``foundation_events`` table enforces uniqueness on ``event_id``.
    """
    resolved_path = Path(log_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"hook log not found: {resolved_path}")
    resolved_repo_name = repo_name or _infer_repo_name(resolved_path)
    rows = _iter_import_rows(resolved_path)
    imported = 0
    for _, raw_line, row in rows:
        event = _build_event(repo_name=resolved_repo_name, row=row, raw_line=raw_line)
        try:
            _io_log._write_foundation_event_to_db(
                timestamp=str(event["timestamp"]),
                project=resolved_repo_name,
                run_id=str(event["run_id"]),
                trace_id=f"governed_repo/{resolved_repo_name}",
                event_id=str(event["event_id"]),
                event_type=str(event["event_type"]),
                payload=event,
                caller="llm_client.observability.governed_repo",
                task="governed_repo_friction",
                raise_on_error=True,
            )
        except sqlite3.IntegrityError:
            continue
        imported += 1
    return imported


def _load_governed_repo_rows(
    *,
    repo_name: str | None,
    days: int,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Return normalized governed-repo Foundation payloads from shared observability."""
    db = _io_log._get_db()
    if db is None:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    clauses = ["event_type = 'GovernedRepoHook'", "timestamp >= ?"]
    params: list[Any] = [cutoff.isoformat()]
    if repo_name is not None:
        clauses.append("project = ?")
        params.append(repo_name)
    where = " AND ".join(clauses)
    results: list[tuple[str, str, dict[str, Any]]] = []
    for timestamp, payload_text in db.execute(
        f"""SELECT timestamp, payload
            FROM foundation_events
            WHERE {where}
            ORDER BY timestamp DESC""",  # noqa: S608
        params,
    ).fetchall():
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        hook_payload = payload.get("governed_repo_hook")
        if not isinstance(hook_payload, dict):
            continue
        results.append((str(timestamp), str(payload.get("session_id") or ""), hook_payload))
    return results


def _rank_counter(counter: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    """Return a stable ranked list of ``path``/``count`` dictionaries."""
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit > 0:
        ranked = ranked[:limit]
    return [{"path": path, "count": count} for path, count in ranked]


def _rank_session_counter(counter: Counter[str], *, limit: int) -> list[dict[str, Any]]:
    """Return a stable ranked list of session summaries for friction-heavy sessions."""
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit > 0:
        ranked = ranked[:limit]
    return [{"session_id": session_id, "count": count} for session_id, count in ranked]


def get_governed_repo_top_missing_reads(
    *,
    repo_name: str | None = None,
    days: int = 7,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Rank repeated missing governed-doc reads across imported hook telemetry."""
    missing_read_counts: Counter[str] = Counter()
    for _, _, payload in _load_governed_repo_rows(repo_name=repo_name, days=days):
        for path in payload.get("missing_reads", []):
            if isinstance(path, str) and path.strip():
                missing_read_counts[path.strip()] += 1
    return _rank_counter(missing_read_counts, limit=limit)


def get_governed_repo_friction_summary(
    *,
    repo_name: str | None = None,
    days: int = 7,
    limit: int = 10,
) -> dict[str, Any]:
    """Return the first operator summary view for governed-repo hook friction."""
    rows = _load_governed_repo_rows(repo_name=repo_name, days=days)
    decision_counts: Counter[str] = Counter()
    hook_counts: Counter[str] = Counter()
    friction_file_counts: Counter[str] = Counter()
    session_friction_counts: Counter[str] = Counter()
    missing_read_counts: Counter[str] = Counter()

    for _, session_id, payload in rows:
        decision = str(payload.get("decision") or "").strip() or "unknown"
        hook_name = str(payload.get("hook_name") or "").strip() or "unknown"
        file_path = str(payload.get("file_path") or "").strip()
        missing_reads = [
            path.strip()
            for path in payload.get("missing_reads", [])
            if isinstance(path, str) and path.strip()
        ]
        decision_counts[decision] += 1
        hook_counts[hook_name] += 1
        for path in missing_reads:
            missing_read_counts[path] += 1
        if decision in {"block", "error"} or missing_reads:
            if file_path:
                friction_file_counts[file_path] += 1
            if session_id:
                session_friction_counts[session_id] += 1

    return {
        "repo_name": repo_name,
        "days": days,
        "total_events": len(rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "hook_counts": dict(sorted(hook_counts.items())),
        "top_missing_reads": _rank_counter(missing_read_counts, limit=limit),
        "top_friction_files": _rank_counter(friction_file_counts, limit=limit),
        "top_friction_sessions": _rank_session_counter(session_friction_counts, limit=limit),
    }


__all__ = [
    "get_governed_repo_friction_summary",
    "get_governed_repo_top_missing_reads",
    "import_governed_repo_hook_log",
]
