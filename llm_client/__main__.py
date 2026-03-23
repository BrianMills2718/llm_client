"""Cost dashboard CLI for llm_client observability.

Usage:
    python -m llm_client cost
    python -m llm_client models {list,show,tasks}
    python -m llm_client traces
    python -m llm_client scores
    python -m llm_client experiments
    python -m llm_client adoption
    python -m llm_client backfill
    python -m llm_client tool-lint
"""

from __future__ import annotations

import argparse
import sys

from llm_client.cli.adoption import register_parser as register_adoption_parser
from llm_client.cli.backfill import register_parser as register_backfill_parser
from llm_client.cli.cost import register_parser as register_cost_parser
from llm_client.cli.experiments import register_parser as register_experiments_parser
from llm_client.cli.models import register_parser as register_models_parser
from llm_client.cli.replay import register_parser as register_replay_parser
from llm_client.cli.scores import register_parser as register_scores_parser
from llm_client.cli.tool_lint import register_parser as register_tool_lint_parser
from llm_client.cli.traces import register_parser as register_traces_parser


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm_client",
        description="LLM cost dashboard and observability tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    register_cost_parser(subparsers)
    register_models_parser(subparsers)
    register_replay_parser(subparsers)
    register_traces_parser(subparsers)
    register_scores_parser(subparsers)
    register_experiments_parser(subparsers)
    register_adoption_parser(subparsers)
    register_backfill_parser(subparsers)
    register_tool_lint_parser(subparsers)

    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if args.command is None or handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
