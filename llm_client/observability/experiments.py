"""Observability experiment-run APIs.

This module contains concrete experiment lifecycle logic previously housed in
``llm_client.io_log``. ``io_log`` remains a compatibility shim.
"""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from llm_client.experiment_summary import summarize_adoption_profiles, summarize_agent_outcomes
import llm_client.io_log as _io_log

logger = logging.getLogger(__name__)

ActiveExperimentRun = _io_log.ActiveExperimentRun


def activate_experiment_run(run_id: str) -> ActiveExperimentRun:
    return _io_log.activate_experiment_run(run_id)


def _build_auto_run_provenance(*, git_commit: str | None) -> dict[str, Any]:
    """Build automatic provenance metadata for an experiment run."""
    provenance: dict[str, Any] = {
        "git_dirty": False,
        "changed_files": [],
        "diff_categories": [],
    }
    try:
        from llm_client.utils.git_utils import classify_diff_files, get_working_tree_files, is_git_dirty

        changed_files = get_working_tree_files()
        provenance["git_dirty"] = is_git_dirty()
        provenance["changed_files"] = changed_files
        provenance["diff_categories"] = sorted(classify_diff_files(changed_files))
    except Exception:
        logger.debug("observability.experiments._build_auto_run_provenance failed", exc_info=True)

    if git_commit:
        provenance["git_commit"] = git_commit
    return provenance


def start_run(
    *,
    dataset: str,
    model: str,
    task: str | None = None,
    config: dict[str, Any] | None = None,
    condition_id: str | None = None,
    seed: int | None = None,
    replicate: int | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    metrics_schema: list[str] | None = None,
    run_id: str | None = None,
    git_commit: str | None = None,
    provenance: dict[str, Any] | None = None,
    feature_profile: str | dict[str, Any] | None = None,
    agent_spec: str | Path | dict[str, Any] | None = None,
    allow_missing_agent_spec: bool = False,
    missing_agent_spec_reason: str | None = None,
    project: str | None = None,
) -> str:
    """Register an experiment run. Returns run_id."""
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]
    _io_log._start_run_timer(run_id)

    if git_commit is None:
        from llm_client.utils.git_utils import get_git_head

        git_commit = get_git_head()

    timestamp = datetime.now(timezone.utc).isoformat()
    proj = project or _io_log._get_project()
    normalized_condition_id = str(condition_id).strip() if condition_id is not None else None
    if normalized_condition_id == "":
        normalized_condition_id = None
    normalized_scenario_id = str(scenario_id).strip() if scenario_id is not None else None
    if normalized_scenario_id == "":
        normalized_scenario_id = None
    normalized_phase = str(phase).strip() if phase is not None else None
    if normalized_phase == "":
        normalized_phase = None
    normalized_seed = int(seed) if seed is not None else None
    normalized_replicate = int(replicate) if replicate is not None else None
    auto_provenance = _build_auto_run_provenance(git_commit=git_commit)
    merged_provenance = dict(auto_provenance)
    if provenance:
        merged_provenance.update(provenance)
    resolved_profile: dict[str, Any] | None = None
    try:
        if feature_profile is not None:
            resolved_profile = _io_log._normalize_feature_profile(feature_profile)
        else:
            active_profile = _io_log.get_active_feature_profile()
            if active_profile is not None:
                resolved_profile = dict(active_profile)
    except Exception:
        logger.warning("Failed to normalize feature profile for run provenance.", exc_info=True)
    if resolved_profile is not None:
        merged_provenance["feature_profile"] = resolved_profile

    agent_spec_payload: dict[str, Any] | None = None
    if agent_spec is not None:
        try:
            from llm_client.agent_spec import load_agent_spec
        except Exception as exc:
            raise ValueError("Failed to import AgentSpec loader from llm_client.agent_spec") from exc
        _, agent_spec_summary = load_agent_spec(agent_spec)
        agent_spec_payload = {
            "summary": agent_spec_summary,
        }
        merged_provenance["agent_spec"] = agent_spec_payload

    _io_log.enforce_agent_spec(
        task,
        has_agent_spec=agent_spec_payload is not None,
        allow_missing=allow_missing_agent_spec,
        missing_reason=missing_agent_spec_reason,
        caller="llm_client.io_log.start_run",
    )

    if allow_missing_agent_spec:
        reason = (missing_agent_spec_reason or "").strip()
        merged_provenance["agent_spec_opt_out"] = {
            "enabled": True,
            "reason": reason or None,
        }
    if task:
        merged_provenance["task"] = task
    provenance_payload = merged_provenance or None

    try:
        d = _io_log._log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "run_start",
            "run_id": run_id,
            "timestamp": timestamp,
            "project": proj,
            "dataset": dataset,
            "model": model,
            "config": config,
            "condition_id": normalized_condition_id,
            "seed": normalized_seed,
            "replicate": normalized_replicate,
            "scenario_id": normalized_scenario_id,
            "phase": normalized_phase,
            "metrics_schema": metrics_schema,
            "git_commit": git_commit,
            "provenance": provenance_payload,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("observability.experiments.start_run JSONL write failed", exc_info=True)

    try:
        db = _io_log._get_db()
        db.execute(
            """INSERT INTO experiment_runs
               (run_id, timestamp, project, dataset, model, config,
                provenance, condition_id, seed, replicate, scenario_id, phase,
                metrics_schema, git_commit, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')""",
            (
                run_id,
                timestamp,
                proj,
                dataset,
                model,
                json.dumps(config, default=str) if config else None,
                json.dumps(provenance_payload, default=str) if provenance_payload else None,
                normalized_condition_id,
                normalized_seed,
                normalized_replicate,
                normalized_scenario_id,
                normalized_phase,
                json.dumps(metrics_schema) if metrics_schema else None,
                git_commit,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("observability.experiments.start_run DB write failed", exc_info=True)

    return run_id


def _validate_run_id(run_id: str) -> str:
    """Return a non-empty run id or raise a truthful error."""

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must be a non-empty string")
    return normalized


def _normalize_optional_stage(stage: str | None) -> str | None:
    """Normalize an optional stage label."""

    if stage is None:
        return None
    normalized = stage.strip()
    if not normalized:
        raise ValueError("stage must be non-empty when provided")
    return normalized


def _normalize_optional_message(message: str | None) -> str | None:
    """Normalize an optional message payload."""

    if message is None:
        return None
    normalized = message.strip()
    return normalized or None


def _validate_optional_nonnegative_int(name: str, value: int | None) -> int | None:
    """Validate that an optional integer counter is non-negative."""

    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _validate_optional_nonnegative_float(name: str, value: float | None) -> float | None:
    """Validate that an optional float value is non-negative."""

    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _normalize_optional_text(name: str, value: str | None) -> str | None:
    """Normalize an optional text field and reject empty strings."""

    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty when provided")
    return normalized


def _serialize_progress_event(
    *,
    event_type: Literal["run_stage", "run_progress", "run_stagnated"],
    run_id: str,
    stage: str | None,
    message: str | None,
    total: int | None,
    completed: int | None,
    failed: int | None,
    progress_unit: str | None,
    avg_latency_s: float | None,
    checkpoint_ref: str | None,
    metadata: dict[str, Any] | None,
    reason: str | None,
) -> dict[str, Any]:
    """Build one normalized run-progress event payload."""

    normalized_run_id = _validate_run_id(run_id)
    normalized_stage = _normalize_optional_stage(stage)
    normalized_message = _normalize_optional_message(message)
    normalized_total = _validate_optional_nonnegative_int("total", total)
    normalized_completed = _validate_optional_nonnegative_int("completed", completed)
    normalized_failed = _validate_optional_nonnegative_int("failed", failed)
    if (
        normalized_total is not None
        and normalized_completed is not None
        and normalized_completed > normalized_total
    ):
        raise ValueError("completed cannot exceed total")
    if (
        normalized_total is not None
        and normalized_failed is not None
        and normalized_failed > normalized_total
    ):
        raise ValueError("failed cannot exceed total")
    if (
        normalized_total is not None
        and normalized_completed is not None
        and normalized_failed is not None
        and normalized_completed + normalized_failed > normalized_total
    ):
        raise ValueError("completed + failed cannot exceed total")
    normalized_progress_unit = _normalize_optional_text("progress_unit", progress_unit)
    normalized_checkpoint_ref = _normalize_optional_text("checkpoint_ref", checkpoint_ref)
    normalized_reason = _normalize_optional_text("reason", reason)
    normalized_avg_latency = _validate_optional_nonnegative_float(
        "avg_latency_s",
        avg_latency_s,
    )
    normalized_metadata = dict(metadata) if metadata is not None else None
    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "type": event_type,
        "run_id": normalized_run_id,
        "timestamp": timestamp,
        "stage": normalized_stage,
        "message": normalized_message,
        "total": normalized_total,
        "completed": normalized_completed,
        "failed": normalized_failed,
        "progress_unit": normalized_progress_unit,
        "avg_latency_s": (
            round(normalized_avg_latency, 3)
            if normalized_avg_latency is not None
            else None
        ),
        "checkpoint_ref": normalized_checkpoint_ref,
        "metadata": normalized_metadata,
        "reason": normalized_reason,
    }


def _persist_run_progress_event(event: dict[str, Any]) -> None:
    """Persist one run-progress event to both SQLite and JSONL."""

    try:
        d = _io_log._log_dir()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except Exception:
        logger.debug(
            "observability.experiments._persist_run_progress_event JSONL write failed",
            exc_info=True,
        )

    try:
        db = _io_log._get_db()
        db.execute(
            """INSERT INTO experiment_run_progress_events
               (run_id, timestamp, event_type, stage, message, total, completed,
                failed, progress_unit, avg_latency_s, checkpoint_ref, metadata, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event["run_id"],
                event["timestamp"],
                event["type"],
                event["stage"],
                event["message"],
                event["total"],
                event["completed"],
                event["failed"],
                event["progress_unit"],
                event["avg_latency_s"],
                event["checkpoint_ref"],
                json.dumps(event["metadata"], default=str) if event["metadata"] else None,
                event["reason"],
            ),
        )
        db.commit()
    except Exception:
        logger.debug(
            "observability.experiments._persist_run_progress_event DB write failed",
            exc_info=True,
        )


def log_run_stage(
    run_id: str,
    *,
    stage: str,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a named stage transition for an existing run."""

    event = _serialize_progress_event(
        event_type="run_stage",
        run_id=run_id,
        stage=stage,
        message=message,
        total=None,
        completed=None,
        failed=None,
        progress_unit=None,
        avg_latency_s=None,
        checkpoint_ref=None,
        metadata=metadata,
        reason=None,
    )
    _persist_run_progress_event(event)


def log_run_progress(
    run_id: str,
    *,
    stage: str | None = None,
    total: int | None = None,
    completed: int | None = None,
    failed: int | None = None,
    progress_unit: str | None = None,
    avg_latency_s: float | None = None,
    checkpoint_ref: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a numeric progress snapshot for an existing run."""

    event = _serialize_progress_event(
        event_type="run_progress",
        run_id=run_id,
        stage=stage,
        message=None,
        total=total,
        completed=completed,
        failed=failed,
        progress_unit=progress_unit,
        avg_latency_s=avg_latency_s,
        checkpoint_ref=checkpoint_ref,
        metadata=metadata,
        reason=None,
    )
    _persist_run_progress_event(event)


def mark_run_stagnated(
    run_id: str,
    *,
    stage: str | None = None,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist an explicit stagnation event for an existing run."""

    event = _serialize_progress_event(
        event_type="run_stagnated",
        run_id=run_id,
        stage=stage,
        message=None,
        total=None,
        completed=None,
        failed=None,
        progress_unit=None,
        avg_latency_s=None,
        checkpoint_ref=None,
        metadata=metadata,
        reason=reason,
    )
    _persist_run_progress_event(event)


def log_item(
    *,
    run_id: str,
    item_id: str,
    metrics: dict[str, Any],
    predicted: str | None = None,
    gold: str | None = None,
    latency_s: float | None = None,
    cost: float | None = None,
    n_tool_calls: int | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> None:
    """Log one item result. Dual-write SQLite + JSONL. Never raises."""
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        d = _io_log._log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "item",
            "run_id": run_id,
            "item_id": item_id,
            "timestamp": timestamp,
            "metrics": metrics,
            "predicted": predicted,
            "gold": gold,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "cost": cost,
            "n_tool_calls": n_tool_calls,
            "error": error,
            "extra": extra,
            "trace_id": trace_id,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("observability.experiments.log_item JSONL write failed", exc_info=True)

    try:
        db = _io_log._get_db()
        db.execute(
            """INSERT OR REPLACE INTO experiment_items
               (run_id, item_id, timestamp, metrics, predicted, gold,
                latency_s, cost, n_tool_calls, error, extra, trace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                item_id,
                timestamp,
                json.dumps(metrics, default=str),
                predicted,
                gold,
                round(latency_s, 3) if latency_s is not None else None,
                cost,
                n_tool_calls,
                error,
                json.dumps(extra, default=str) if extra else None,
                trace_id,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("observability.experiments.log_item DB write failed", exc_info=True)


def finish_run(
    *,
    run_id: str,
    summary_metrics: dict[str, Any] | None = None,
    status: str = "completed",
    wall_time_s: float | None = None,
    cpu_time_s: float | None = None,
    cpu_user_s: float | None = None,
    cpu_system_s: float | None = None,
) -> dict[str, Any]:
    """Finalize run and return run record."""
    db = _io_log._get_db()

    item_rows = db.execute(
        "SELECT metrics, cost, error, predicted, extra, item_id FROM experiment_items WHERE run_id = ?",
        (run_id,),
    ).fetchall()

    n_items = len(item_rows)
    n_errors = sum(1 for r in item_rows if r[2] is not None)
    n_completed = n_items - n_errors
    total_cost = sum(r[1] or 0.0 for r in item_rows)

    wall_time_s, cpu_time_s, cpu_user_s, cpu_system_s = _io_log._auto_capture_run_timing(
        run_id=run_id,
        wall_time_s=wall_time_s,
        cpu_time_s=cpu_time_s,
        cpu_user_s=cpu_user_s,
        cpu_system_s=cpu_system_s,
    )

    if summary_metrics is None:
        schema_row = db.execute(
            "SELECT metrics_schema FROM experiment_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        schema: list[str] = []
        if schema_row and schema_row[0]:
            schema = json.loads(schema_row[0])

        summary_metrics = {}
        for metric_name in schema:
            values = []
            for r in item_rows:
                m = json.loads(r[0])
                v = m.get(metric_name)
                if v is not None:
                    values.append(float(v))
            if values:
                summary_metrics[f"avg_{metric_name}"] = round(
                    100.0 * sum(values) / len(values), 2
                )
    else:
        summary_metrics = dict(summary_metrics)

    try:
        outcome_items: list[dict[str, Any]] = []
        for metrics_json, _cost, error, predicted, extra_json, item_id in item_rows:
            extra = json.loads(extra_json) if extra_json else None
            outcome_items.append(
                {
                    "item_id": item_id,
                    "predicted": predicted,
                    "error": error,
                    "metrics": json.loads(metrics_json) if metrics_json else {},
                    "extra": extra,
                }
            )
        outcome_summary = summarize_agent_outcomes(outcome_items)
        adoption_summary = summarize_adoption_profiles(outcome_items)
        for key in (
            "answer_present_count",
            "answer_present_rate",
            "grounded_completed_count",
            "grounded_completed_rate",
            "forced_terminal_accepted_count",
            "forced_terminal_accepted_rate",
            "reliability_completed_count",
            "reliability_completed_rate",
            "required_submit_missing_count",
            "required_submit_missing_rate",
            "submit_validator_accepted_count",
            "submit_validator_accepted_rate",
        ):
            summary_metrics.setdefault(key, outcome_summary.get(key))
        for key, value in (outcome_summary.get("submit_completion_mode_counts") or {}).items():
            summary_metrics.setdefault(f"submit_mode_{key}_count", value)
        for key, value in (outcome_summary.get("primary_failure_class_counts") or {}).items():
            summary_metrics.setdefault(f"primary_failure_{key}_count", value)
        for key in (
            "n_items_with_metadata",
            "metadata_coverage_rate",
            "satisfied_count",
            "satisfied_rate",
        ):
            summary_metrics.setdefault(f"adoption_{key}", adoption_summary.get(key))
        for key, value in (adoption_summary.get("effective_profile_counts") or {}).items():
            summary_metrics.setdefault(f"adoption_profile_{key}_count", value)
    except Exception:
        logger.debug("observability.experiments.finish_run outcome summary failed", exc_info=True)

    try:
        db.execute(
            """UPDATE experiment_runs
               SET n_items = ?, n_completed = ?, n_errors = ?,
                   summary_metrics = ?, total_cost = ?,
                   wall_time_s = ?, cpu_time_s = ?, cpu_user_s = ?, cpu_system_s = ?,
                   status = ?
               WHERE run_id = ?""",
            (
                n_items,
                n_completed,
                n_errors,
                json.dumps(summary_metrics, default=str) if summary_metrics else None,
                round(total_cost, 6),
                round(wall_time_s, 1) if wall_time_s is not None else None,
                round(cpu_time_s, 3) if cpu_time_s is not None else None,
                round(cpu_user_s, 3) if cpu_user_s is not None else None,
                round(cpu_system_s, 3) if cpu_system_s is not None else None,
                status,
                run_id,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("observability.experiments.finish_run DB update failed", exc_info=True)

    try:
        d = _io_log._log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "run_finish",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_items": n_items,
            "n_completed": n_completed,
            "n_errors": n_errors,
            "summary_metrics": summary_metrics,
            "total_cost": round(total_cost, 6),
            "wall_time_s": round(wall_time_s, 1) if wall_time_s is not None else None,
            "cpu_time_s": round(cpu_time_s, 3) if cpu_time_s is not None else None,
            "cpu_user_s": round(cpu_user_s, 3) if cpu_user_s is not None else None,
            "cpu_system_s": round(cpu_system_s, 3) if cpu_system_s is not None else None,
            "status": status,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("observability.experiments.finish_run JSONL write failed", exc_info=True)

    row = db.execute(
        """SELECT run_id, timestamp, project, dataset, model, config, provenance,
                  condition_id, seed, replicate, scenario_id, phase,
                  metrics_schema, n_items, n_completed, n_errors,
                  summary_metrics, total_cost, wall_time_s,
                  cpu_time_s, cpu_user_s, cpu_system_s,
                  git_commit, status
           FROM experiment_runs WHERE run_id = ?""",
        (run_id,),
    ).fetchone()

    _io_log._pop_run_timer(run_id)

    if row is None:
        return {"run_id": run_id, "status": status}

    return {
        "run_id": row[0],
        "timestamp": row[1],
        "project": row[2],
        "dataset": row[3],
        "model": row[4],
        "config": json.loads(row[5]) if row[5] else None,
        "provenance": json.loads(row[6]) if row[6] else None,
        "condition_id": row[7],
        "seed": row[8],
        "replicate": row[9],
        "scenario_id": row[10],
        "phase": row[11],
        "metrics_schema": json.loads(row[12]) if row[12] else None,
        "n_items": row[13],
        "n_completed": row[14],
        "n_errors": row[15],
        "summary_metrics": json.loads(row[16]) if row[16] else None,
        "total_cost": row[17],
        "wall_time_s": row[18],
        "cpu_time_s": row[19],
        "cpu_user_s": row[20],
        "cpu_system_s": row[21],
        "git_commit": row[22],
        "status": row[23],
    }


class ExperimentRun:
    """Managed experiment run with auto timing, context, and convenience APIs."""

    def __init__(
        self,
        *,
        dataset: str,
        model: str,
        config: dict[str, Any] | None = None,
        condition_id: str | None = None,
        seed: int | None = None,
        replicate: int | None = None,
        scenario_id: str | None = None,
        phase: str | None = None,
        metrics_schema: list[str] | None = None,
        run_id: str | None = None,
        git_commit: str | None = None,
        provenance: dict[str, Any] | None = None,
        feature_profile: str | dict[str, Any] | None = None,
        project: str | None = None,
        status_on_exception: str = "interrupted",
    ) -> None:
        self._feature_profile = (
            _io_log._normalize_feature_profile(feature_profile)
            if feature_profile is not None
            else None
        )
        self.run_id = start_run(
            dataset=dataset,
            model=model,
            config=config,
            condition_id=condition_id,
            seed=seed,
            replicate=replicate,
            scenario_id=scenario_id,
            phase=phase,
            metrics_schema=metrics_schema,
            run_id=run_id,
            git_commit=git_commit,
            provenance=provenance,
            feature_profile=self._feature_profile,
            project=project,
        )
        self._status_on_exception = status_on_exception
        self._finished = False
        self._active_token: contextvars.Token[str | None] | None = None
        self._feature_profile_token: contextvars.Token[dict[str, Any] | None] | None = None

    def __enter__(self) -> "ExperimentRun":
        if self._active_token is None and _io_log.get_active_experiment_run_id() != self.run_id:
            self._active_token = _io_log._active_experiment_run_id.set(self.run_id)
        if self._feature_profile is not None and self._feature_profile_token is None:
            self._feature_profile_token = _io_log._active_feature_profile.set(self._feature_profile)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        try:
            if not self._finished:
                final_status = "completed" if exc_type is None else self._status_on_exception
                self.finish(status=final_status)
        finally:
            if self._active_token is not None:
                _io_log._active_experiment_run_id.reset(self._active_token)
                self._active_token = None
            if self._feature_profile_token is not None:
                _io_log._active_feature_profile.reset(self._feature_profile_token)
                self._feature_profile_token = None
        return False

    async def __aenter__(self) -> "ExperimentRun":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        return self.__exit__(exc_type, exc, tb)

    def log_item(
        self,
        *,
        item_id: str,
        metrics: dict[str, Any],
        predicted: str | None = None,
        gold: str | None = None,
        latency_s: float | None = None,
        cost: float | None = None,
        n_tool_calls: int | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        log_item(
            run_id=self.run_id,
            item_id=item_id,
            metrics=metrics,
            predicted=predicted,
            gold=gold,
            latency_s=latency_s,
            cost=cost,
            n_tool_calls=n_tool_calls,
            error=error,
            extra=extra,
            trace_id=trace_id,
        )

    def finish(
        self,
        *,
        summary_metrics: dict[str, Any] | None = None,
        status: str = "completed",
        wall_time_s: float | None = None,
        cpu_time_s: float | None = None,
        cpu_user_s: float | None = None,
        cpu_system_s: float | None = None,
    ) -> dict[str, Any]:
        if self._finished:
            existing = get_run(self.run_id)
            if existing is not None:
                return existing
            return {"run_id": self.run_id, "status": status}

        result = finish_run(
            run_id=self.run_id,
            summary_metrics=summary_metrics,
            status=status,
            wall_time_s=wall_time_s,
            cpu_time_s=cpu_time_s,
            cpu_user_s=cpu_user_s,
            cpu_system_s=cpu_system_s,
        )
        self._finished = True
        return result


def experiment_run(
    *,
    dataset: str,
    model: str,
    config: dict[str, Any] | None = None,
    condition_id: str | None = None,
    seed: int | None = None,
    replicate: int | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    metrics_schema: list[str] | None = None,
    run_id: str | None = None,
    git_commit: str | None = None,
    provenance: dict[str, Any] | None = None,
    feature_profile: str | dict[str, Any] | None = None,
    project: str | None = None,
    status_on_exception: str = "interrupted",
) -> ExperimentRun:
    return ExperimentRun(
        dataset=dataset,
        model=model,
        config=config,
        condition_id=condition_id,
        seed=seed,
        replicate=replicate,
        scenario_id=scenario_id,
        phase=phase,
        metrics_schema=metrics_schema,
        run_id=run_id,
        git_commit=git_commit,
        provenance=provenance,
        feature_profile=feature_profile,
        project=project,
        status_on_exception=status_on_exception,
    )


def get_runs(
    *,
    dataset: str | None = None,
    model: str | None = None,
    project: str | None = None,
    condition_id: str | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    seed: int | None = None,
    since: str | date | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Query experiment runs, newest first."""
    db = _io_log._get_db()

    clauses: list[str] = []
    params: list[Any] = []

    if dataset is not None:
        clauses.append("dataset = ?")
        params.append(dataset)
    if model is not None:
        clauses.append("model = ?")
        params.append(model)
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if condition_id is not None:
        clauses.append("condition_id = ?")
        params.append(condition_id)
    if scenario_id is not None:
        clauses.append("scenario_id = ?")
        params.append(scenario_id)
    if phase is not None:
        clauses.append("phase = ?")
        params.append(phase)
    if seed is not None:
        clauses.append("seed = ?")
        params.append(int(seed))
    if since is not None:
        since_str = since.isoformat() if isinstance(since, date) else since
        clauses.append("timestamp >= ?")
        params.append(since_str)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    rows = db.execute(
        f"""SELECT run_id, timestamp, project, dataset, model,
                   config, provenance, condition_id, seed, replicate, scenario_id, phase,
                   n_items, n_completed, n_errors,
                   summary_metrics, total_cost, wall_time_s,
                   cpu_time_s, cpu_user_s, cpu_system_s,
                   git_commit, status
            FROM experiment_runs
            {where}
            ORDER BY timestamp DESC
            LIMIT ?""",  # noqa: S608
        params,
    ).fetchall()

    results = []
    for r in rows:
        results.append(
            {
                "run_id": r[0],
                "timestamp": r[1],
                "project": r[2],
                "dataset": r[3],
                "model": r[4],
                "config": json.loads(r[5]) if r[5] else None,
                "provenance": json.loads(r[6]) if r[6] else None,
                "condition_id": r[7],
                "seed": r[8],
                "replicate": r[9],
                "scenario_id": r[10],
                "phase": r[11],
                "n_items": r[12],
                "n_completed": r[13],
                "n_errors": r[14],
                "summary_metrics": json.loads(r[15]) if r[15] else None,
                "total_cost": r[16],
                "wall_time_s": r[17],
                "cpu_time_s": r[18],
                "cpu_user_s": r[19],
                "cpu_system_s": r[20],
                "git_commit": r[21],
                "status": r[22],
            }
        )
    return results


def get_run(run_id: str) -> dict[str, Any] | None:
    """Fetch one experiment run by run_id."""
    db = _io_log._get_db()
    row = db.execute(
        """SELECT run_id, timestamp, project, dataset, model,
                  config, provenance, condition_id, seed, replicate, scenario_id, phase, metrics_schema,
                  n_items, n_completed, n_errors, summary_metrics,
                  total_cost, wall_time_s, cpu_time_s, cpu_user_s, cpu_system_s,
                  git_commit, status
           FROM experiment_runs
           WHERE run_id = ?""",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": row[0],
        "timestamp": row[1],
        "project": row[2],
        "dataset": row[3],
        "model": row[4],
        "config": json.loads(row[5]) if row[5] else None,
        "provenance": json.loads(row[6]) if row[6] else None,
        "condition_id": row[7],
        "seed": row[8],
        "replicate": row[9],
        "scenario_id": row[10],
        "phase": row[11],
        "metrics_schema": json.loads(row[12]) if row[12] else None,
        "n_items": row[13],
        "n_completed": row[14],
        "n_errors": row[15],
        "summary_metrics": json.loads(row[16]) if row[16] else None,
        "total_cost": row[17],
        "wall_time_s": row[18],
        "cpu_time_s": row[19],
        "cpu_user_s": row[20],
        "cpu_system_s": row[21],
        "git_commit": row[22],
        "status": row[23],
    }


def get_run_items(run_id: str) -> list[dict[str, Any]]:
    """All items for a run, ordered by timestamp."""
    db = _io_log._get_db()

    rows = db.execute(
        """SELECT item_id, timestamp, metrics, predicted, gold,
                  latency_s, cost, n_tool_calls, error, extra, trace_id
           FROM experiment_items
           WHERE run_id = ?
           ORDER BY timestamp""",
        (run_id,),
    ).fetchall()

    results = []
    for r in rows:
        results.append(
            {
                "item_id": r[0],
                "timestamp": r[1],
                "metrics": json.loads(r[2]) if r[2] else {},
                "predicted": r[3],
                "gold": r[4],
                "latency_s": r[5],
                "cost": r[6],
                "n_tool_calls": r[7],
                "error": r[8],
                "extra": json.loads(r[9]) if r[9] else None,
                "trace_id": r[10],
            }
        )
    return results


def log_experiment_aggregate(
    *,
    dataset: str,
    family_id: str,
    aggregate_type: str,
    metrics: dict[str, Any],
    aggregate_id: str | None = None,
    project: str | None = None,
    condition_id: str | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    provenance: dict[str, Any] | None = None,
    source_run_ids: list[str] | None = None,
) -> str:
    """Persist one family-level experiment aggregate and return its ID.

    This is the shared observability surface for metrics derived across multiple
    runs rather than within a single run. Prompt-eval corpus metrics are the
    first consumer, but the record shape is generic.
    """
    normalized_dataset = str(dataset).strip()
    normalized_family_id = str(family_id).strip()
    normalized_type = str(aggregate_type).strip()
    if not normalized_dataset:
        raise ValueError("log_experiment_aggregate requires a non-empty dataset.")
    if not normalized_family_id:
        raise ValueError("log_experiment_aggregate requires a non-empty family_id.")
    if not normalized_type:
        raise ValueError("log_experiment_aggregate requires a non-empty aggregate_type.")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("log_experiment_aggregate requires a non-empty metrics dict.")

    normalized_condition_id = str(condition_id).strip() if condition_id is not None else None
    if normalized_condition_id == "":
        normalized_condition_id = None
    normalized_scenario_id = str(scenario_id).strip() if scenario_id is not None else None
    if normalized_scenario_id == "":
        normalized_scenario_id = None
    normalized_phase = str(phase).strip() if phase is not None else None
    if normalized_phase == "":
        normalized_phase = None
    normalized_source_run_ids = None
    if source_run_ids is not None:
        normalized_source_run_ids = [str(run_id).strip() for run_id in source_run_ids if str(run_id).strip()]

    if aggregate_id is None:
        aggregate_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()
    proj = project or _io_log._get_project()

    try:
        d = _io_log._log_dir()
        d.mkdir(parents=True, exist_ok=True)
        record = {
            "type": "aggregate",
            "aggregate_id": aggregate_id,
            "timestamp": timestamp,
            "project": proj,
            "dataset": normalized_dataset,
            "family_id": normalized_family_id,
            "aggregate_type": normalized_type,
            "condition_id": normalized_condition_id,
            "scenario_id": normalized_scenario_id,
            "phase": normalized_phase,
            "metrics": metrics,
            "provenance": provenance,
            "source_run_ids": normalized_source_run_ids,
        }
        with open(d / "experiments.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("observability.experiments.log_experiment_aggregate JSONL write failed", exc_info=True)

    try:
        db = _io_log._get_db()
        db.execute(
            """INSERT INTO experiment_aggregates
               (aggregate_id, timestamp, project, dataset, family_id, aggregate_type,
                condition_id, scenario_id, phase, metrics, provenance, source_run_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                aggregate_id,
                timestamp,
                proj,
                normalized_dataset,
                normalized_family_id,
                normalized_type,
                normalized_condition_id,
                normalized_scenario_id,
                normalized_phase,
                json.dumps(metrics, default=str),
                json.dumps(provenance, default=str) if provenance else None,
                json.dumps(normalized_source_run_ids) if normalized_source_run_ids else None,
            ),
        )
        db.commit()
    except Exception:
        logger.debug("observability.experiments.log_experiment_aggregate DB write failed", exc_info=True)

    return aggregate_id


def get_experiment_aggregates(
    *,
    dataset: str | None = None,
    family_id: str | None = None,
    aggregate_type: str | None = None,
    project: str | None = None,
    condition_id: str | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Query family-level experiment aggregates, newest first."""
    db = _io_log._get_db()

    clauses: list[str] = []
    params: list[Any] = []

    if dataset is not None:
        clauses.append("dataset = ?")
        params.append(dataset)
    if family_id is not None:
        clauses.append("family_id = ?")
        params.append(family_id)
    if aggregate_type is not None:
        clauses.append("aggregate_type = ?")
        params.append(aggregate_type)
    if project is not None:
        clauses.append("project = ?")
        params.append(project)
    if condition_id is not None:
        clauses.append("condition_id = ?")
        params.append(condition_id)
    if scenario_id is not None:
        clauses.append("scenario_id = ?")
        params.append(scenario_id)
    if phase is not None:
        clauses.append("phase = ?")
        params.append(phase)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    rows = db.execute(
        f"""SELECT aggregate_id, timestamp, project, dataset, family_id, aggregate_type,
                   condition_id, scenario_id, phase, metrics, provenance, source_run_ids
            FROM experiment_aggregates
            {where}
            ORDER BY timestamp DESC
            LIMIT ?""",  # noqa: S608
        params,
    ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "aggregate_id": row[0],
                "timestamp": row[1],
                "project": row[2],
                "dataset": row[3],
                "family_id": row[4],
                "aggregate_type": row[5],
                "condition_id": row[6],
                "scenario_id": row[7],
                "phase": row[8],
                "metrics": json.loads(row[9]) if row[9] else {},
                "provenance": json.loads(row[10]) if row[10] else None,
                "source_run_ids": json.loads(row[11]) if row[11] else [],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Comparison / analysis (extracted to observability/comparison.py)
# ---------------------------------------------------------------------------
from llm_client.observability.comparison import (  # noqa: E402, F401
    _metric_distribution_stats,
    _numeric_summary_metrics,
    compare_cohorts,
    compare_runs,
)
