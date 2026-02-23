"""Long-thinking adoption telemetry CLI command."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def cmd_adoption(args: argparse.Namespace) -> None:
    from llm_client import get_background_mode_adoption

    try:
        summary = get_background_mode_adoption(
            experiments_path=args.experiments_path,
            since=args.since,
            run_id_prefix=args.run_id_prefix,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(summary, indent=2))
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
    print(f"With effort set:    {summary.get('with_reasoning_effort')}")
    print(f"background=true:    {summary.get('background_mode_true')}")
    print(f"background=false:   {summary.get('background_mode_false')}")
    print(f"background=unknown: {summary.get('background_mode_unknown')}")
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
    parser.set_defaults(handler=cmd_adoption)
