"""CLI command modules for ``python -m llm_client``."""

from llm_client.cli.backfill import cmd_backfill, register_parser as register_backfill_parser
from llm_client.cli.cost import cmd_cost, register_parser as register_cost_parser
from llm_client.cli.experiments import cmd_experiments, register_parser as register_experiments_parser
from llm_client.cli.scores import cmd_scores, register_parser as register_scores_parser
from llm_client.cli.traces import cmd_traces, register_parser as register_traces_parser

__all__ = [
    "cmd_backfill",
    "cmd_cost",
    "cmd_experiments",
    "cmd_scores",
    "cmd_traces",
    "register_backfill_parser",
    "register_cost_parser",
    "register_experiments_parser",
    "register_scores_parser",
    "register_traces_parser",
]
