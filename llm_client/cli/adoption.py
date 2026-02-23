"""Long-thinking adoption telemetry CLI command."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _gate_metric(summary: dict[str, Any], metric: str) -> tuple[float, int]:
    if metric == "overall":
        rate = float(summary.get("background_mode_rate_overall", 0.0))
        samples = int(summary.get("records_considered", 0))
        return rate, samples
    rate = float(summary.get("background_mode_rate_among_reasoning", 0.0))
    samples = int(summary.get("with_reasoning_effort", 0))
    return rate, samples


def _apply_gate(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    min_rate = args.min_rate
    if min_rate is None:
        return None
    if min_rate < 0.0 or min_rate > 1.0:
        raise ValueError(f"--min-rate must be between 0 and 1, got {min_rate!r}")
    if args.min_samples < 0:
        raise ValueError(f"--min-samples must be >= 0, got {args.min_samples!r}")

    actual_rate, samples = _gate_metric(summary, args.metric)
    considered = int(summary.get("records_considered", 0))
    effort_dim_key_count = int(summary.get("records_with_reasoning_effort_key", 0))
    if samples < args.min_samples:
        if args.metric == "among_reasoning" and considered > 0 and effort_dim_key_count == 0:
            passed = False
            reason = "missing_reasoning_effort_dimension"
        elif args.metric == "among_reasoning" and considered > 0 and samples == 0:
            passed = False
            reason = "no_reasoning_effort_records"
        else:
            passed = False
            reason = "insufficient_samples"
    elif actual_rate < min_rate:
        passed = False
        reason = "rate_below_threshold"
    else:
        passed = True
        reason = "ok"

    gate = {
        "enabled": True,
        "metric": args.metric,
        "min_rate": min_rate,
        "actual_rate": actual_rate,
        "min_samples": args.min_samples,
        "samples": samples,
        "passed": passed,
        "reason": reason,
        "warn_only": bool(args.warn_only),
    }
    summary["gate"] = gate
    return gate


def cmd_adoption(args: argparse.Namespace) -> None:
    from llm_client import get_background_mode_adoption

    try:
        summary = get_background_mode_adoption(
            experiments_path=args.experiments_path,
            since=args.since,
            run_id_prefix=args.run_id_prefix,
        )
        gate = _apply_gate(summary, args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
        if gate is not None and not gate.get("passed", False) and not args.warn_only:
            sys.exit(args.gate_fail_exit_code)
        return

    print("\nLong-Thinking Adoption:")
    print("─" * 72)
    print(f"Experiments file:   {summary.get('experiments_path')}")
    print(f"File exists:        {summary.get('exists')}")
    print(f"Run prefix filter:  {summary.get('run_id_prefix') or '-'}")
    print(f"Since filter:       {summary.get('since') or '-'}")
    print(f"Total lines:        {summary.get('total_records')}")
    print(f"Invalid lines:      {summary.get('invalid_lines')}")
    print(f"Records considered: {summary.get('records_considered')}")
    print(f"Records w/ effort key: {summary.get('records_with_reasoning_effort_key')}")
    print(f"Records w/ bg key:     {summary.get('records_with_background_mode_key')}")
    print(f"Records w/ routing:    {summary.get('records_with_routing_trace')}")
    print(f"With effort set:    {summary.get('with_reasoning_effort')}")
    print(f"background=true:    {summary.get('background_mode_true')}")
    print(f"background=false:   {summary.get('background_mode_false')}")
    print(f"background=unknown: {summary.get('background_mode_unknown')}")
    print(f"Model switches:     {summary.get('model_switches')}")
    print(f"Fallback records:   {summary.get('fallback_records')}")
    print(
        "Background rate "
        f"(among reasoning): {_pct(float(summary.get('background_mode_rate_among_reasoning', 0.0)))}"
    )
    print(
        "Background rate "
        f"(overall): {_pct(float(summary.get('background_mode_rate_overall', 0.0)))}"
    )

    counts = summary.get("reasoning_effort_counts") or {}
    if isinstance(counts, dict) and counts:
        print("\nReasoning Effort Counts:")
        for effort, count in sorted(counts.items()):
            print(f"  {effort}: {count}")

    if gate is not None:
        verdict = "PASS" if gate.get("passed") else "FAIL"
        print("\nGate:")
        print(f"  Verdict:          {verdict}")
        print(f"  Metric:           {gate.get('metric')}")
        print(f"  Actual rate:      {_pct(float(gate.get('actual_rate', 0.0)))}")
        print(f"  Required rate:    {_pct(float(gate.get('min_rate', 0.0)))}")
        print(f"  Samples:          {gate.get('samples')} (min {gate.get('min_samples')})")
        print(f"  Reason:           {gate.get('reason')}")
        if not gate.get("passed", False) and not args.warn_only:
            sys.exit(args.gate_fail_exit_code)


def register_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "adoption",
        help="Summarize long-thinking background_mode adoption from task-graph JSONL",
    )
    parser.add_argument(
        "--experiments-path",
        help="Path to task-graph experiments JSONL (default: ~/projects/data/task_graph/experiments.jsonl)",
    )
    parser.add_argument(
        "--since",
        help="Only include records since this ISO timestamp/date (UTC if date-only)",
    )
    parser.add_argument(
        "--run-id-prefix",
        help="Only include runs whose run_id starts with this prefix",
    )
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.add_argument(
        "--min-rate",
        type=float,
        help="Gate: minimum required background-mode rate (0-1).",
    )
    parser.add_argument(
        "--metric",
        choices=["among_reasoning", "overall"],
        default="among_reasoning",
        help="Gate metric basis (among_reasoning or overall).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Gate: minimum sample count required before evaluating the rate.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Do not exit non-zero when the gate fails; print/report only.",
    )
    parser.add_argument(
        "--gate-fail-exit-code",
        type=int,
        default=2,
        help="Exit code used when a configured gate fails (unless --warn-only).",
    )
    parser.set_defaults(handler=cmd_adoption)
