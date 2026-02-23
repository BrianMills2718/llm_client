"""Observability experiment-run APIs.

This module contains concrete experiment lifecycle logic previously housed in
``llm_client.io_log``. ``io_log`` remains a compatibility shim.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import statistics
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from llm_client import io_log as _io_log

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
        from llm_client.git_utils import classify_diff_files, get_working_tree_files, is_git_dirty

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
        from llm_client.git_utils import get_git_head

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
        "SELECT metrics, cost, error FROM experiment_items WHERE run_id = ?",
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


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    """Side-by-side summary_metrics for 2+ runs, with deltas from first run."""
    if len(run_ids) < 2:
        raise ValueError("compare_runs requires at least 2 run_ids.")

    def _to_float(v: Any) -> float | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if isinstance(v, str) and v.strip():
                return float(v)
        except ValueError:
            return None
        return None

    def _item_delta(base_items: list[dict[str, Any]], cand_items: list[dict[str, Any]]) -> dict[str, Any]:
        base_map = {str(it["item_id"]): it for it in base_items}
        cand_map = {str(it["item_id"]): it for it in cand_items}
        shared_ids = sorted(set(base_map) & set(cand_map))
        new_ids = sorted(set(cand_map) - set(base_map))
        missing_ids = sorted(set(base_map) - set(cand_map))

        improved: dict[str, list[str]] = {}
        regressed: dict[str, list[str]] = {}
        unchanged_items = 0

        for item_id in shared_ids:
            base_metrics = base_map[item_id].get("metrics", {}) or {}
            cand_metrics = cand_map[item_id].get("metrics", {}) or {}
            changed = False
            for metric_name in sorted(set(base_metrics) | set(cand_metrics)):
                base_v = _to_float(base_metrics.get(metric_name))
                cand_v = _to_float(cand_metrics.get(metric_name))
                if base_v is None or cand_v is None or cand_v == base_v:
                    continue
                changed = True
                if cand_v > base_v:
                    improved.setdefault(metric_name, []).append(item_id)
                else:
                    regressed.setdefault(metric_name, []).append(item_id)
            if not changed:
                unchanged_items += 1

        return {
            "shared_items": len(shared_ids),
            "unchanged_items": unchanged_items,
            "improved": improved,
            "regressed": regressed,
            "new_in_candidate": new_ids,
            "missing_in_candidate": missing_ids,
        }

    db = _io_log._get_db()
    runs = []
    for rid in run_ids:
        row = db.execute(
            """SELECT run_id, dataset, model, condition_id, seed, replicate, scenario_id, phase,
                      n_items, n_completed, n_errors,
                      summary_metrics, total_cost, wall_time_s,
                      cpu_time_s, cpu_user_s, cpu_system_s,
                      status, timestamp, git_commit, provenance
               FROM experiment_runs WHERE run_id = ?""",
            (rid,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Run not found: {rid}")
        runs.append(
            {
                "run_id": row[0],
                "dataset": row[1],
                "model": row[2],
                "condition_id": row[3],
                "seed": row[4],
                "replicate": row[5],
                "scenario_id": row[6],
                "phase": row[7],
                "n_items": row[8],
                "n_completed": row[9],
                "n_errors": row[10],
                "summary_metrics": json.loads(row[11]) if row[11] else {},
                "total_cost": row[12],
                "wall_time_s": row[13],
                "cpu_time_s": row[14],
                "cpu_user_s": row[15],
                "cpu_system_s": row[16],
                "status": row[17],
                "timestamp": row[18],
                "git_commit": row[19],
                "provenance": json.loads(row[20]) if row[20] else None,
            }
        )

    baseline = runs[0]["summary_metrics"]
    deltas = []
    baseline_items = get_run_items(run_ids[0])
    item_deltas = []
    for run in runs[1:]:
        d: dict[str, float] = {}
        for k, v in run["summary_metrics"].items():
            if k in baseline and isinstance(v, (int, float)) and isinstance(baseline[k], (int, float)):
                d[k] = round(v - baseline[k], 2)
        deltas.append(d)
        per_item = _item_delta(baseline_items, get_run_items(run["run_id"]))
        per_item["run_id"] = run["run_id"]
        item_deltas.append(per_item)

    return {
        "runs": runs,
        "deltas_from_first": deltas,
        "item_deltas_from_first": item_deltas,
    }


def _metric_distribution_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    n = len(values)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if n >= 2 else 0.0
    sem = stdev / math.sqrt(n) if n >= 2 else 0.0
    ci95_half_width = 1.96 * sem
    return {
        "n": float(n),
        "mean": round(mean, 6),
        "stdev": round(stdev, 6),
        "sem": round(sem, 6),
        "ci95_low": round(mean - ci95_half_width, 6),
        "ci95_high": round(mean + ci95_half_width, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _numeric_summary_metrics(run: dict[str, Any]) -> dict[str, float]:
    summary = run.get("summary_metrics")
    if not isinstance(summary, dict):
        return {}
    numeric: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric[str(key)] = float(value)
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            try:
                numeric[str(key)] = float(text)
            except ValueError:
                continue
    return numeric


def compare_cohorts(
    *,
    condition_ids: list[str] | None = None,
    baseline_condition_id: str | None = None,
    dataset: str | None = None,
    model: str | None = None,
    project: str | None = None,
    scenario_id: str | None = None,
    phase: str | None = None,
    since: str | date | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """Compare experiment cohorts by condition with aggregate stats and matched-seed deltas."""
    if limit <= 0:
        raise ValueError("limit must be > 0")

    normalized_condition_ids = None
    if condition_ids is not None:
        normalized_condition_ids = [str(cid).strip() for cid in condition_ids if str(cid).strip()]
        if not normalized_condition_ids:
            raise ValueError("condition_ids must contain at least one non-empty value")

    runs = get_runs(
        dataset=dataset,
        model=model,
        project=project,
        scenario_id=scenario_id,
        phase=phase,
        since=since,
        limit=limit,
    )

    cohorts: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        condition = str(run.get("condition_id") or "__unlabeled__")
        if normalized_condition_ids is not None and condition not in normalized_condition_ids:
            continue
        cohorts.setdefault(condition, []).append(run)

    if not cohorts:
        return {
            "cohorts": [],
            "matched_seed_deltas_from_baseline": [],
            "baseline_condition_id": baseline_condition_id,
            "settings": {
                "dataset": dataset,
                "model": model,
                "project": project,
                "scenario_id": scenario_id,
                "phase": phase,
                "since": since.isoformat() if isinstance(since, date) else since,
                "limit": int(limit),
            },
        }

    ordered_conditions = sorted(cohorts.keys())
    resolved_baseline = baseline_condition_id.strip() if isinstance(baseline_condition_id, str) else ""
    if not resolved_baseline:
        resolved_baseline = ordered_conditions[0]
    if resolved_baseline not in cohorts:
        raise ValueError(f"baseline_condition_id not found in selected runs: {resolved_baseline}")

    cohort_rows: list[dict[str, Any]] = []
    for condition in ordered_conditions:
        cohort_runs = cohorts[condition]
        metric_values: dict[str, list[float]] = {}
        for run in cohort_runs:
            for key, value in _numeric_summary_metrics(run).items():
                metric_values.setdefault(key, []).append(value)
        aggregated = {
            metric: _metric_distribution_stats(values)
            for metric, values in sorted(metric_values.items())
            if values
        }
        cohort_rows.append(
            {
                "condition_id": condition,
                "n_runs": len(cohort_runs),
                "run_ids": [str(r.get("run_id")) for r in cohort_runs],
                "seeds": sorted({int(r["seed"]) for r in cohort_runs if isinstance(r.get("seed"), int)}),
                "metrics": aggregated,
            }
        )

    def _matched_key(run: dict[str, Any]) -> tuple[int, int | None] | None:
        seed_value = run.get("seed")
        if not isinstance(seed_value, int):
            return None
        replicate_value = run.get("replicate")
        if isinstance(replicate_value, int):
            return (seed_value, replicate_value)
        return (seed_value, None)

    baseline_runs = cohorts[resolved_baseline]
    baseline_by_key: dict[tuple[int, int | None], dict[str, Any]] = {}
    for run in baseline_runs:
        baseline_key = _matched_key(run)
        if baseline_key is None:
            continue
        baseline_by_key.setdefault(baseline_key, run)

    matched_deltas: list[dict[str, Any]] = []
    baseline_metrics_by_key = {key: _numeric_summary_metrics(run) for key, run in baseline_by_key.items()}
    for condition in ordered_conditions:
        if condition == resolved_baseline:
            continue
        candidate_by_key: dict[tuple[int, int | None], dict[str, Any]] = {}
        for run in cohorts[condition]:
            candidate_key = _matched_key(run)
            if candidate_key is None:
                continue
            candidate_by_key.setdefault(candidate_key, run)
        shared_keys = sorted(set(baseline_by_key).intersection(candidate_by_key))
        delta_values: dict[str, list[float]] = {}
        for match_key in shared_keys:
            base_metrics = baseline_metrics_by_key.get(match_key, {})
            cand_metrics = _numeric_summary_metrics(candidate_by_key[match_key])
            for metric_name in sorted(set(base_metrics).intersection(cand_metrics)):
                delta_values.setdefault(metric_name, []).append(cand_metrics[metric_name] - base_metrics[metric_name])
        matched_deltas.append(
            {
                "condition_id": condition,
                "baseline_condition_id": resolved_baseline,
                "n_matched_pairs": len(shared_keys),
                "matched_keys": [[k[0], k[1]] for k in shared_keys],
                "metric_deltas": {
                    metric: _metric_distribution_stats(values)
                    for metric, values in sorted(delta_values.items())
                    if values
                },
            }
        )

    return {
        "settings": {
            "dataset": dataset,
            "model": model,
            "project": project,
            "scenario_id": scenario_id,
            "phase": phase,
            "since": since.isoformat() if isinstance(since, date) else since,
            "limit": int(limit),
            "condition_ids": normalized_condition_ids,
        },
        "baseline_condition_id": resolved_baseline,
        "cohorts": cohort_rows,
        "matched_seed_deltas_from_baseline": matched_deltas,
    }
