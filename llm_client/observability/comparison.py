"""Run comparison and cohort analysis for experiment observability.

This module owns the comparative analysis surface: side-by-side run
comparison with per-item deltas, and cohort analysis by condition with
matched-seed statistical summaries. These are read-only operations that
query the experiment database but never mutate it.

Extracted from ``experiments.py`` to separate the analysis concern from
the run lifecycle write path.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import date
from typing import Any

import llm_client.io_log as _io_log
from llm_client.experiment_summary import summarize_adoption_profiles, summarize_agent_outcomes


def _metric_distribution_stats(values: list[float]) -> dict[str, float]:
    """Compute descriptive statistics for a list of metric values."""
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
    """Extract numeric fields from a run's summary_metrics."""
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


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    """Side-by-side summary_metrics for 2+ runs, with deltas from first run."""
    from llm_client.observability.experiments import get_run_items

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
    all_items_by_run = {rid: get_run_items(rid) for rid in run_ids}
    for run in runs:
        run["outcome_summary"] = summarize_agent_outcomes(all_items_by_run[run["run_id"]])
        run["adoption_summary"] = summarize_adoption_profiles(all_items_by_run[run["run_id"]])
    baseline_items = all_items_by_run[run_ids[0]]
    item_deltas = []
    outcome_deltas = []
    adoption_deltas = []
    for run in runs[1:]:
        d: dict[str, float] = {}
        for k, v in run["summary_metrics"].items():
            if k in baseline and isinstance(v, (int, float)) and isinstance(baseline[k], (int, float)):
                d[k] = round(v - baseline[k], 2)
        deltas.append(d)
        per_item = _item_delta(baseline_items, all_items_by_run[run["run_id"]])
        per_item["run_id"] = run["run_id"]
        item_deltas.append(per_item)
        outcome_delta: dict[str, float] = {}
        base_outcome = runs[0]["outcome_summary"]
        cand_outcome = run["outcome_summary"]
        for key in (
            "answer_present_rate",
            "grounded_completed_rate",
            "forced_terminal_accepted_rate",
            "reliability_completed_rate",
            "required_submit_missing_rate",
            "submit_validator_accepted_rate",
        ):
            base_v = base_outcome.get(key)
            cand_v = cand_outcome.get(key)
            if isinstance(base_v, (int, float)) and isinstance(cand_v, (int, float)):
                outcome_delta[key] = round(float(cand_v) - float(base_v), 4)
        outcome_deltas.append(outcome_delta)
        adoption_delta: dict[str, float] = {}
        base_adoption = runs[0]["adoption_summary"]
        cand_adoption = run["adoption_summary"]
        for key in ("metadata_coverage_rate", "satisfied_rate"):
            base_v = base_adoption.get(key)
            cand_v = cand_adoption.get(key)
            if isinstance(base_v, (int, float)) and isinstance(cand_v, (int, float)):
                adoption_delta[key] = round(float(cand_v) - float(base_v), 4)
        adoption_deltas.append(adoption_delta)

    return {
        "runs": runs,
        "deltas_from_first": deltas,
        "item_deltas_from_first": item_deltas,
        "outcome_deltas_from_first": outcome_deltas,
        "adoption_deltas_from_first": adoption_deltas,
    }


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
    from llm_client.observability.experiments import get_runs

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
