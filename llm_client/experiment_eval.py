"""Experiment review, gating, and triage helpers.

This module keeps evaluation logic reusable across projects and CLIs:
- optional rubric-based LLM review
- deterministic structural checks
- gate-policy evaluation over run/item signals
- automatic item triage summaries
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from llm_client.scoring import score_output

DEFAULT_DETERMINISTIC_CHECKS: tuple[str, ...] = (
    "prediction_present",
    "no_item_error",
    "trace_id_present",
    "metrics_in_unit_interval",
)

_GATE_SUFFIX_OPS: tuple[tuple[str, str], ...] = (
    ("_gte", ">="),
    ("_lte", "<="),
    ("_neq", "!="),
    ("_gt", ">"),
    ("_lt", "<"),
    ("_eq", "=="),
)

_CMP_RE = re.compile(r"^\s*(>=|<=|!=|==|>|<)\s*(.+?)\s*$")


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    return 0


def _coerce_check_names(checks: str | list[str] | None) -> list[str]:
    if checks is None:
        return list(DEFAULT_DETERMINISTIC_CHECKS)
    if isinstance(checks, str):
        text = checks.strip().lower()
        if text in {"", "default"}:
            return list(DEFAULT_DETERMINISTIC_CHECKS)
        if text in {"none", "off", "false", "0"}:
            return []
        return [part.strip() for part in checks.split(",") if part.strip()]
    result: list[str] = []
    for item in checks:
        name = str(item).strip()
        if name:
            result.append(name)
    return result


def run_deterministic_checks_for_item(
    item: dict[str, Any],
    *,
    checks: str | list[str] | None = None,
) -> list[dict[str, Any]]:
    names = _coerce_check_names(checks)
    metrics = item.get("metrics") or {}
    prediction = str(item.get("predicted") or "").strip()
    trace_id = str(item.get("trace_id") or "").strip()
    item_error = str(item.get("error") or "").strip()

    results: list[dict[str, Any]] = []
    for name in names:
        if name == "prediction_present":
            passed = bool(prediction)
            results.append({
                "name": name,
                "passed": passed,
                "value": prediction[:120],
                "reason": None if passed else "predicted answer is empty",
            })
            continue
        if name == "no_item_error":
            passed = not item_error
            results.append({
                "name": name,
                "passed": passed,
                "value": item_error[:160],
                "reason": None if passed else "item contains a runtime error",
            })
            continue
        if name == "trace_id_present":
            passed = bool(trace_id)
            results.append({
                "name": name,
                "passed": passed,
                "value": trace_id,
                "reason": None if passed else "trace_id missing",
            })
            continue
        if name == "metrics_in_unit_interval":
            bad: list[str] = []
            for k, v in metrics.items():
                fv = _to_float(v)
                if fv is None:
                    continue
                if fv < 0.0 or fv > 1.0:
                    bad.append(f"{k}={fv}")
            passed = not bad
            results.append({
                "name": name,
                "passed": passed,
                "value": list(metrics.keys()),
                "reason": None if passed else f"metric(s) outside [0,1]: {', '.join(bad)}",
            })
            continue

        results.append({
            "name": name,
            "passed": False,
            "value": None,
            "reason": f"unknown deterministic check: {name}",
        })
    return results


def run_deterministic_checks_for_items(
    items: list[dict[str, Any]],
    *,
    checks: str | list[str] | None = None,
) -> dict[str, Any]:
    names = _coerce_check_names(checks)
    per_item: list[dict[str, Any]] = []
    total_checks = 0
    total_passed = 0
    failed_items = 0

    for item in items:
        item_id = str(item.get("item_id") or "")
        results = run_deterministic_checks_for_item(item, checks=names)
        n_passed = sum(1 for r in results if r.get("passed"))
        n_total = len(results)
        if n_passed < n_total:
            failed_items += 1
        total_checks += n_total
        total_passed += n_passed
        per_item.append({
            "item_id": item_id,
            "checks": results,
            "passed_checks": n_passed,
            "failed_checks": n_total - n_passed,
        })

    pass_rate = (total_passed / total_checks) if total_checks else None
    return {
        "checks": names,
        "n_items": len(items),
        "n_failed_items": failed_items,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
        "items": per_item,
    }


def review_items_with_rubric(
    items: list[dict[str, Any]],
    *,
    rubric: str,
    judge_model: str | None = None,
    task_prefix: str = "experiment_review",
    max_items: int | None = None,
) -> dict[str, Any]:
    reviewed_items: list[dict[str, Any]] = []
    failures = 0
    scored = 0
    total = 0.0
    min_score: float | None = None
    max_score: float | None = None

    cap = len(items) if not max_items or max_items <= 0 else min(max_items, len(items))
    for item in items[:cap]:
        item_id = str(item.get("item_id") or "")
        output = str(item.get("predicted") or "")
        context = {
            "item_id": item_id,
            "gold": item.get("gold"),
            "metrics": item.get("metrics") or {},
            "error": item.get("error"),
            "extra": item.get("extra") or {},
        }
        try:
            result = score_output(
                output=output,
                rubric=rubric,
                context=context,
                task=f"{task_prefix}.{item_id or 'item'}",
                trace_id=item.get("trace_id"),
                judge_model=judge_model,
            )
            score = float(result.overall_score)
            scored += 1
            total += score
            min_score = score if min_score is None else min(min_score, score)
            max_score = score if max_score is None else max(max_score, score)
            reviewed_items.append({
                "item_id": item_id,
                "overall_score": score,
                "dimensions": dict(result.dimensions),
                "reasoning": dict(result.reasoning),
                "cost": float(result.cost or 0.0),
                "latency_s": float(result.latency_s or 0.0),
                "error": None,
            })
        except Exception as exc:
            failures += 1
            reviewed_items.append({
                "item_id": item_id,
                "overall_score": None,
                "dimensions": {},
                "reasoning": {},
                "cost": 0.0,
                "latency_s": 0.0,
                "error": str(exc),
            })

    avg = (total / scored) if scored else None
    return {
        "rubric": rubric,
        "judge_model": judge_model or "",
        "n_items_considered": cap,
        "n_scored": scored,
        "n_failed": failures,
        "avg_overall_score": round(avg, 4) if avg is not None else None,
        "min_overall_score": round(min_score, 4) if min_score is not None else None,
        "max_overall_score": round(max_score, 4) if max_score is not None else None,
        "items": reviewed_items,
    }


def load_gate_policy(policy: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(policy, dict):
        return dict(policy)
    text = str(policy).strip()
    if not text:
        raise ValueError("gate policy cannot be empty")
    if text.startswith("@"):
        text = text[1:].strip()
    path = Path(text).expanduser()
    if path.is_file():
        return json.loads(path.read_text())
    return json.loads(text)


def build_gate_signals(
    *,
    run_info: dict[str, Any] | None,
    items: list[dict[str, Any]],
    deterministic_report: dict[str, Any] | None = None,
    review_report: dict[str, Any] | None = None,
) -> dict[str, float]:
    signals: dict[str, float] = {}
    run_info = run_info or {}
    for key in ("n_items", "n_completed", "n_errors", "total_cost", "wall_time_s", "cpu_time_s"):
        value = _to_float(run_info.get(key))
        if value is not None:
            signals[key] = value

    summary = run_info.get("summary_metrics") or {}
    if isinstance(summary, dict):
        for k, v in summary.items():
            fv = _to_float(v)
            if fv is not None:
                signals[k] = fv

    item_error_count = 0
    tool_error_total = 0
    tool_interface_total = 0
    tool_unavailable_total = 0
    tool_prereq_total = 0
    control_loop_total = 0
    arg_validation_total = 0

    for item in items:
        if str(item.get("error") or "").strip():
            item_error_count += 1
        comp = (item.get("extra") or {}).get("composability") or {}
        if not isinstance(comp, dict):
            comp = {}
        categories = comp.get("error_categories") or {}
        if not isinstance(categories, dict):
            categories = {}
        tool_error_total += _to_int(comp.get("n_errors"))
        tool_interface_total += _to_int(categories.get("tool_interface_mismatch"))
        tool_unavailable_total += _to_int(categories.get("tool_unavailable"))
        tool_prereq_total += _to_int(categories.get("missing_prerequisite"))
        control_loop_total += _to_int(comp.get("n_control_loop_suppressed"))
        arg_validation_total += _to_int(comp.get("n_arg_validation_rejections"))

    signals["item_error_count"] = float(item_error_count)
    signals["tool_error_total"] = float(tool_error_total)
    signals["tool_interface_error_total"] = float(tool_interface_total)
    signals["tool_unavailable_error_total"] = float(tool_unavailable_total)
    signals["tool_prerequisite_error_total"] = float(tool_prereq_total)
    signals["control_loop_suppressed_total"] = float(control_loop_total)
    signals["tool_arg_validation_rejection_total"] = float(arg_validation_total)

    if deterministic_report:
        pass_rate = _to_float(deterministic_report.get("pass_rate"))
        failed_items = _to_float(deterministic_report.get("n_failed_items"))
        if pass_rate is not None:
            signals["deterministic_pass_rate"] = pass_rate
        if failed_items is not None:
            signals["deterministic_failed_items"] = failed_items

    if review_report:
        avg = _to_float(review_report.get("avg_overall_score"))
        if avg is not None:
            signals["avg_review_score"] = avg
        min_score = _to_float(review_report.get("min_overall_score"))
        if min_score is not None:
            signals["min_review_score"] = min_score
        max_score = _to_float(review_report.get("max_overall_score"))
        if max_score is not None:
            signals["max_review_score"] = max_score
        n_failed = _to_float(review_report.get("n_failed"))
        if n_failed is not None:
            signals["review_failed_items"] = n_failed

    return signals


def _parse_rule(key: str, value: Any) -> tuple[str, str, float]:
    metric = key
    op = "=="
    threshold_val = _to_float(value)

    for suffix, candidate_op in _GATE_SUFFIX_OPS:
        if key.endswith(suffix):
            metric = key[:-len(suffix)]
            op = candidate_op
            break

    if threshold_val is None and isinstance(value, str):
        m = _CMP_RE.match(value)
        if m:
            op = m.group(1)
            threshold_val = _to_float(m.group(2))

    if threshold_val is None:
        raise ValueError(f"invalid gate threshold for {key!r}: {value!r}")
    metric = metric.strip()
    if not metric:
        raise ValueError(f"invalid gate metric name in {key!r}")
    return metric, op, threshold_val


def _apply_cmp(lhs: float | None, op: str, rhs: float) -> bool:
    if lhs is None:
        return False
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    if op == "!=":
        return lhs != rhs
    if op == "==":
        return lhs == rhs
    raise ValueError(f"unsupported operator: {op}")


def evaluate_gate_policy(
    *,
    policy: dict[str, Any],
    signals: dict[str, float],
) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise ValueError("gate policy must be a JSON object")

    fail_if = policy.get("fail_if")
    pass_if = policy.get("pass_if")
    if fail_if is None and pass_if is None:
        pass_if = policy

    if fail_if is not None and not isinstance(fail_if, dict):
        raise ValueError("gate policy fail_if must be an object")
    if pass_if is not None and not isinstance(pass_if, dict):
        raise ValueError("gate policy pass_if must be an object")

    fail_rules = fail_if or {}
    pass_rules = pass_if or {}

    triggered_fail_if: list[dict[str, Any]] = []
    unsatisfied_pass_if: list[dict[str, Any]] = []

    for key, value in fail_rules.items():
        metric, op, threshold = _parse_rule(str(key), value)
        actual = _to_float(signals.get(metric))
        if _apply_cmp(actual, op, threshold):
            triggered_fail_if.append({
                "rule": key,
                "metric": metric,
                "operator": op,
                "threshold": threshold,
                "actual": actual,
            })

    for key, value in pass_rules.items():
        metric, op, threshold = _parse_rule(str(key), value)
        actual = _to_float(signals.get(metric))
        if not _apply_cmp(actual, op, threshold):
            unsatisfied_pass_if.append({
                "rule": key,
                "metric": metric,
                "operator": op,
                "threshold": threshold,
                "actual": actual,
            })

    passed = not triggered_fail_if and not unsatisfied_pass_if
    return {
        "passed": passed,
        "signals": signals,
        "triggered_fail_if": triggered_fail_if,
        "unsatisfied_pass_if": unsatisfied_pass_if,
    }


def _triage_categories_for_item(item: dict[str, Any]) -> list[str]:
    categories: list[str] = []

    if str(item.get("error") or "").strip():
        categories.append("runtime_error")

    comp = (item.get("extra") or {}).get("composability") or {}
    if not isinstance(comp, dict):
        comp = {}
    comp_categories = comp.get("error_categories") or {}
    if not isinstance(comp_categories, dict):
        comp_categories = {}

    interface_err = _to_int(comp_categories.get("tool_interface_mismatch"))
    unavailable_err = _to_int(comp_categories.get("tool_unavailable"))
    prereq_err = _to_int(comp_categories.get("missing_prerequisite"))
    n_tool_err = _to_int(comp.get("n_errors"))
    control_suppressed = _to_int(comp.get("n_control_loop_suppressed"))

    if interface_err > 0:
        categories.append("tool_interface")
    if unavailable_err > 0:
        categories.append("tool_unavailable")
    if prereq_err > 0:
        categories.append("missing_prerequisite")
    if n_tool_err > 0 and interface_err == 0 and unavailable_err == 0 and prereq_err == 0:
        categories.append("tool_runtime")
    if control_suppressed > 0:
        categories.append("control_loop")

    metrics = item.get("metrics") or {}
    em = _to_float(metrics.get("em"))
    llm_em = _to_float(metrics.get("llm_em"))
    if em is not None and llm_em is not None and em != llm_em:
        categories.append("judge_mismatch")

    failed_metric = False
    if em is not None and em < 0.5:
        failed_metric = True
    if llm_em is not None and llm_em < 0.5:
        failed_metric = True
    if failed_metric and not {"runtime_error", "tool_interface", "tool_unavailable"} & set(categories):
        categories.append("reasoning_or_retrieval")

    if not categories:
        categories.append("clean")
    return categories


def triage_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    per_item: list[dict[str, Any]] = []

    for item in items:
        item_id = str(item.get("item_id") or "")
        categories = _triage_categories_for_item(item)
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        per_item.append({
            "item_id": item_id,
            "categories": categories,
            "error": item.get("error"),
            "metrics": item.get("metrics") or {},
        })

    sorted_counts = dict(sorted(category_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "n_items": len(items),
        "category_counts": sorted_counts,
        "items": per_item,
    }
