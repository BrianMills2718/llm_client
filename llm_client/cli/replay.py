"""CLI commands for call snapshot comparison and replay."""

from __future__ import annotations

import argparse
import json
from typing import Any

import llm_client


def _cmd_replay_compare(args: argparse.Namespace) -> None:
    """Compare two captured call snapshots and print a compact diff."""

    report = llm_client.compare_call_snapshots(args.left_call_id, args.right_call_id)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(llm_client.format_call_diff(report))


def _cmd_replay_rerun(args: argparse.Namespace) -> None:
    """Replay one captured call snapshot under a fresh trace id."""

    result = llm_client.replay_call_snapshot(
        args.call_id,
        trace_id=args.trace_id,
        task=args.task,
        max_budget=args.max_budget,
        project=args.project,
    )
    if args.format == "json":
        print(
            json.dumps(
                {
                    "source_call_id": result["source_call_id"],
                    "replay_trace_id": result["replay_trace_id"],
                    "task": result["task"],
                    "project": result["project"],
                    "public_api": result["public_api"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(f"replayed call {result['source_call_id']} -> trace {result['replay_trace_id']}")
    print(f"task={result['task']} project={result['project']} api={result['public_api']}")


def register_parser(subparsers: Any) -> None:
    """Register the replay CLI subtree."""

    parser = subparsers.add_parser(
        "replay",
        help="Compare captured calls and replay one call snapshot",
    )
    replay_subparsers = parser.add_subparsers(dest="replay_command")

    compare_parser = replay_subparsers.add_parser(
        "compare",
        help="Compare two captured call snapshots",
    )
    compare_parser.add_argument("--left-call-id", type=int, required=True)
    compare_parser.add_argument("--right-call-id", type=int, required=True)
    compare_parser.add_argument("--format", choices=["text", "json"], default="text")
    compare_parser.set_defaults(handler=_cmd_replay_compare)

    rerun_parser = replay_subparsers.add_parser(
        "rerun",
        help="Replay one captured call snapshot under a fresh trace id",
    )
    rerun_parser.add_argument("--call-id", type=int, required=True)
    rerun_parser.add_argument("--trace-id", required=True)
    rerun_parser.add_argument("--task")
    rerun_parser.add_argument("--project")
    rerun_parser.add_argument("--max-budget", type=float, default=0.0)
    rerun_parser.add_argument("--format", choices=["text", "json"], default="text")
    rerun_parser.set_defaults(handler=_cmd_replay_rerun)
