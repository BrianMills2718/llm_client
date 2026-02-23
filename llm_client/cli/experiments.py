"""Experiments CLI command."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def cmd_experiments(args: argparse.Namespace) -> None:
    from llm_client import io_log

    if args.compare_diff:
        _cmd_experiments_compare_diff(args)
        return

    if args.compare:
        _cmd_experiments_compare(args)
        return

    if args.detail:
        _cmd_experiments_detail(args)
        return

    runs = io_log.get_runs(
        dataset=args.dataset,
        model=args.model,
        project=args.project,
        since=args.since,
        limit=args.limit,
    )

    if args.format == "json":
        print(json.dumps(runs, indent=2))
        return

    if not runs:
        print("No experiment runs found.")
        return

    headers = ["Run ID", "Dataset", "Model", "Items", "Status", "Metrics", "Cost", "Wall", "Time"]
    col_widths = [max(len(h), 6) for h in headers]

    display_rows = []
    for r in runs:
        rid = r["run_id"][:12]
        ds = (r["dataset"] or "-")[:20]
        mdl = (r["model"] or "-")[:25]
        items = f"{r['n_completed'] or 0}/{r['n_items'] or 0}"
        status = r["status"] or "-"

        sm = r.get("summary_metrics") or {}
        metrics_str = "  ".join(f"{k}={v}" for k, v in sm.items()) if sm else "-"
        if len(metrics_str) > 40:
            metrics_str = metrics_str[:37] + "..."

        cost = f"${r.get('total_cost', 0.0):.4f}"
        wall = f"{r['wall_time_s']:.0f}s" if r.get("wall_time_s") else "-"
        ts = r["timestamp"][:16] if r.get("timestamp") else "-"

        row = (rid, ds, mdl, items, status, metrics_str, cost, wall, ts)
        display_rows.append(row)
        for i, v in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(v)))

    print("\nExperiment Runs:")
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in display_rows:
        print(fmt.format(*row))


def _cmd_experiments_compare(args: argparse.Namespace) -> None:
    from llm_client import io_log

    try:
        result = io_log.compare_runs(args.compare)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(result, indent=2))
        return

    runs = result["runs"]
    deltas = result["deltas_from_first"]
    item_deltas = result.get("item_deltas_from_first", [])

    print("\nExperiment Comparison:")
    print("─" * 70)
    for i, r in enumerate(runs):
        label = "BASELINE" if i == 0 else "vs base"
        sm = r.get("summary_metrics") or {}
        metrics_str = "  ".join(f"{k}={v}" for k, v in sm.items())
        print(f"  [{r['run_id'][:12]}] {r['dataset']} / {r['model']}")
        print(f"    {r['n_completed']}/{r['n_items']} items  ${r['total_cost']:.4f}  {label}")
        print(f"    {metrics_str}")
        if r.get("cpu_time_s") is not None:
            print(
                f"    CPU: {r['cpu_time_s']:.2f}s"
                f" (user={r.get('cpu_user_s') or 0:.2f}s sys={r.get('cpu_system_s') or 0:.2f}s)"
            )
        if i > 0 and deltas[i - 1]:
            delta_str = "  ".join(f"{k}={v:+.2f}" for k, v in deltas[i - 1].items())
            print(f"    Deltas: {delta_str}")
        if i > 0 and i - 1 < len(item_deltas):
            item_delta = item_deltas[i - 1]
            improved = item_delta.get("improved", {})
            regressed = item_delta.get("regressed", {})
            em_up = len(improved.get("em", []))
            em_down = len(regressed.get("em", []))
            llm_up = len(improved.get("llm_em", []))
            llm_down = len(regressed.get("llm_em", []))
            f1_up = len(improved.get("f1", []))
            f1_down = len(regressed.get("f1", []))
            shared = item_delta.get("shared_items", 0)
            unchanged = item_delta.get("unchanged_items", 0)
            print(
                "    Item deltas:"
                f" shared={shared} unchanged={unchanged}"
                f" | EM +{em_up}/-{em_down}"
                f" | LLM_EM +{llm_up}/-{llm_down}"
                f" | F1 +{f1_up}/-{f1_down}"
            )
            if em_up or em_down or llm_up or llm_down:
                em_up_ids = ", ".join(improved.get("em", [])[:8]) or "-"
                em_down_ids = ", ".join(regressed.get("em", [])[:8]) or "-"
                llm_up_ids = ", ".join(improved.get("llm_em", [])[:8]) or "-"
                llm_down_ids = ", ".join(regressed.get("llm_em", [])[:8]) or "-"
                print(f"      EM up: {em_up_ids}")
                print(f"      EM down: {em_down_ids}")
                print(f"      LLM_EM up: {llm_up_ids}")
                print(f"      LLM_EM down: {llm_down_ids}")
        print()


def _cmd_experiments_compare_diff(args: argparse.Namespace) -> None:
    from llm_client import io_log
    from llm_client.git_utils import classify_diff_files, get_diff_files

    base_run_id, cand_run_id = args.compare_diff
    base_run = io_log.get_run(base_run_id)
    cand_run = io_log.get_run(cand_run_id)
    if base_run is None:
        print(f"Error: Run not found: {base_run_id}", file=sys.stderr)
        sys.exit(1)
    if cand_run is None:
        print(f"Error: Run not found: {cand_run_id}", file=sys.stderr)
        sys.exit(1)

    base_commit = base_run.get("git_commit")
    cand_commit = cand_run.get("git_commit")
    if not base_commit or not cand_commit:
        print(
            "Error: compare-diff requires both runs to have git_commit metadata.",
            file=sys.stderr,
        )
        sys.exit(1)

    files = get_diff_files(base_commit, cand_commit)
    categories = sorted(classify_diff_files(files))

    payload = {
        "base_run_id": base_run_id,
        "candidate_run_id": cand_run_id,
        "base_commit": base_commit,
        "candidate_commit": cand_commit,
        "changed_file_count": len(files),
        "categories": categories,
        "changed_files": files,
    }

    if args.format == "json":
        print(json.dumps(payload, indent=2))
        return

    print("\nExperiment Git Diff:")
    print("─" * 70)
    print(f"  Base run:      {base_run_id[:12]} ({base_run.get('dataset')} / {base_run.get('model')})")
    print(f"  Candidate run: {cand_run_id[:12]} ({cand_run.get('dataset')} / {cand_run.get('model')})")
    print(f"  Commits:       {base_commit} -> {cand_commit}")
    print(f"  Files changed: {len(files)}")
    print(f"  Categories:    {', '.join(categories) if categories else '-'}")
    if files:
        print("  Changed files:")
        for path in files[:200]:
            print(f"    - {path}")
        if len(files) > 200:
            print(f"    ... ({len(files) - 200} more)")


def _cmd_experiments_detail(args: argparse.Namespace) -> None:
    from llm_client import io_log
    from llm_client.experiment_eval import (
        build_gate_signals,
        evaluate_gate_policy,
        load_gate_policy,
        review_items_with_rubric,
        run_deterministic_checks_for_items,
        triage_items,
    )

    run_id = args.detail
    items = io_log.get_run_items(run_id)

    if not items:
        print(f"No items found for run {run_id}")
        return

    run_info = io_log.get_run(run_id)
    triage_report = triage_items(items)

    deterministic_report = None
    if args.det_checks and args.det_checks.strip().lower() not in {"none", "off", "0", "false"}:
        deterministic_report = run_deterministic_checks_for_items(items, checks=args.det_checks)

    review_report = None
    if args.review_rubric:
        review_report = review_items_with_rubric(
            items,
            rubric=args.review_rubric,
            judge_model=args.review_model or None,
            task_prefix=f"experiments.detail.{run_id}",
            max_items=args.review_max_items if args.review_max_items and args.review_max_items > 0 else None,
        )

    gate_policy = None
    gate_report = None
    if args.gate_policy:
        gate_policy = load_gate_policy(args.gate_policy)
        signals = build_gate_signals(
            run_info=run_info,
            items=items,
            deterministic_report=deterministic_report,
            review_report=review_report,
        )
        gate_report = evaluate_gate_policy(policy=gate_policy, signals=signals)

    if args.format == "json":
        include_bundle = bool(
            args.include_triage
            or deterministic_report is not None
            or review_report is not None
            or gate_report is not None
        )
        if include_bundle:
            payload = {
                "run_id": run_id,
                "run": run_info,
                "items": items,
                "triage": triage_report,
            }
            if deterministic_report is not None:
                payload["deterministic_checks"] = deterministic_report
            if review_report is not None:
                payload["review"] = review_report
            if gate_policy is not None:
                payload["gate_policy"] = gate_policy
            if gate_report is not None:
                payload["gate_result"] = gate_report
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(items, indent=2))
        if gate_report is not None and not gate_report.get("passed", False) and args.gate_fail_exit_code:
            sys.exit(2)
        return

    if run_info:
        print(f"\nRun: {run_id}")
        print(f"  Dataset: {run_info['dataset']}  Model: {run_info['model']}  Status: {run_info['status']}")
        sm = run_info.get("summary_metrics") or {}
        if sm:
            print(f"  Summary: {'  '.join(f'{k}={v}' for k, v in sm.items())}")
        print()

    headers = ["Item", "Metrics", "Predicted", "Gold", "Cost", "Latency", "Error"]
    col_widths = [max(len(h), 6) for h in headers]

    display_rows = []
    for it in items:
        iid = str(it["item_id"])[:10]
        m = it.get("metrics") or {}
        metrics_str = " ".join(f"{k}={v}" for k, v in m.items())
        if len(metrics_str) > 30:
            metrics_str = metrics_str[:27] + "..."
        pred = (it.get("predicted") or "-")[:30]
        gold = (it.get("gold") or "-")[:30]
        cost = f"${(it.get('cost') or 0.0):.4f}"
        lat = f"{it['latency_s']:.1f}s" if it.get("latency_s") else "-"
        err = (it.get("error") or "")[:20]

        row = (iid, metrics_str, pred, gold, cost, lat, err)
        display_rows.append(row)
        for i, v in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(v)))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in display_rows:
        print(fmt.format(*row))

    cat_counts = triage_report.get("category_counts") or {}
    if cat_counts:
        print("\nTriage:")
        print("  " + "  ".join(f"{k}={v}" for k, v in cat_counts.items()))
        flagged = [
            it for it in (triage_report.get("items") or []) if "clean" not in set(it.get("categories") or [])
        ]
        if flagged:
            print("  Flagged items:")
            for item in flagged[:12]:
                cats = ",".join(item.get("categories") or [])
                print(f"    {item.get('item_id')}: {cats}")

    if deterministic_report is not None:
        print("\nDeterministic Checks:")
        print(
            "  "
            f"checks={','.join(deterministic_report.get('checks') or [])} "
            f"pass_rate={deterministic_report.get('pass_rate')} "
            f"failed_items={deterministic_report.get('n_failed_items')}/{deterministic_report.get('n_items')}"
        )
        failed = [it for it in deterministic_report.get("items", []) if it.get("failed_checks", 0) > 0]
        for item in failed[:12]:
            reasons = [
                r.get("reason")
                for r in item.get("checks", [])
                if not r.get("passed") and r.get("reason")
            ]
            print(f"    {item.get('item_id')}: {' | '.join(reasons[:3])}")

    if review_report is not None:
        print("\nReview:")
        print(
            "  "
            f"rubric={review_report.get('rubric')} "
            f"judge={review_report.get('judge_model') or '-'} "
            f"scored={review_report.get('n_scored')}/{review_report.get('n_items_considered')} "
            f"avg={review_report.get('avg_overall_score')} "
            f"failed={review_report.get('n_failed')}"
        )
        low_scores = [
            it for it in review_report.get("items", []) if isinstance(it.get("overall_score"), (int, float))
        ]
        low_scores.sort(key=lambda x: float(x.get("overall_score")))
        for item in low_scores[:10]:
            print(f"    {item.get('item_id')}: score={item.get('overall_score')}")
        review_errors = [it for it in review_report.get("items", []) if it.get("error")]
        for item in review_errors[:10]:
            print(f"    {item.get('item_id')}: review_error={item.get('error')}")

    if gate_report is not None:
        verdict = "PASS" if gate_report.get("passed") else "FAIL"
        print(f"\nGate: {verdict}")
        for rule in gate_report.get("triggered_fail_if", []):
            print(
                "  triggered fail_if:"
                f" {rule.get('rule')} (actual={rule.get('actual')} {rule.get('operator')} {rule.get('threshold')})"
            )
        for rule in gate_report.get("unsatisfied_pass_if", []):
            print(
                "  unsatisfied pass_if:"
                f" {rule.get('rule')} (actual={rule.get('actual')} {rule.get('operator')} {rule.get('threshold')})"
            )
        if not gate_report.get("passed", False) and args.gate_fail_exit_code:
            sys.exit(2)


def register_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("experiments", help="List and compare experiment runs")
    parser.add_argument("--dataset", help="Filter to a dataset")
    parser.add_argument("--model", help="Filter to a model")
    parser.add_argument("--project", help="Filter to a project")
    parser.add_argument("--since", help="Only show runs since this ISO date")
    parser.add_argument("--limit", type=int, default=50, help="Max runs to show")
    parser.add_argument("--compare", nargs="+", metavar="RUN_ID", help="Compare 2+ run IDs side-by-side")
    parser.add_argument(
        "--compare-diff",
        nargs=2,
        metavar=("BASE_RUN", "CAND_RUN"),
        help="Show git diff (changed files/categories) between two run commits",
    )
    parser.add_argument("--detail", metavar="RUN_ID", help="Show per-item detail for a run")
    parser.add_argument(
        "--include-triage",
        action="store_true",
        help="When using --detail --format json, include triage/review bundle payload",
    )
    parser.add_argument(
        "--det-checks",
        default="none",
        help="Deterministic checks for --detail (none|default|comma-separated names)",
    )
    parser.add_argument("--review-rubric", help="Rubric name/path to run LLM review over item outputs for --detail")
    parser.add_argument("--review-model", help="Judge model override used with --review-rubric")
    parser.add_argument(
        "--review-max-items",
        type=int,
        default=0,
        help="Max items to review with --review-rubric (0=all)",
    )
    parser.add_argument(
        "--gate-policy",
        help="Gate policy JSON string/path for --detail "
        "(supports pass_if/fail_if with _gt/_gte/_lt/_lte/_eq/_neq)",
    )
    parser.add_argument(
        "--gate-fail-exit-code",
        action="store_true",
        help="Exit with code 2 when --gate-policy evaluates to FAIL",
    )
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.set_defaults(handler=cmd_experiments)
