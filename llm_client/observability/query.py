"""Observability query adapters.

This module is a boundary layer around ``llm_client.io_log`` while we
incrementally migrate internals out of that monolith.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_client import io_log as _io_log


def get_completed_traces(*, limit: int = 1000, max_chars: int = 2000) -> list[dict[str, Any]]:
    return _io_log.get_completed_traces(limit=limit, max_chars=max_chars)


def get_cost(
    *,
    trace_id: str | None = None,
    trace_prefix: str | None = None,
    project: str | None = None,
    caller: str | None = None,
    model: str | None = None,
) -> float:
    return _io_log.get_cost(
        trace_id=trace_id,
        trace_prefix=trace_prefix,
        project=project,
        caller=caller,
        model=model,
    )


def get_runs(**kwargs: Any) -> list[dict[str, Any]]:
    return _io_log.get_runs(**kwargs)


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    return _io_log.compare_runs(run_ids)


def get_trace_tree(trace_prefix: str) -> dict[str, Any]:
    return _io_log.get_trace_tree(trace_prefix)


def import_jsonl(path: str | Path, *, table: str = "llm_calls") -> int:
    return _io_log.import_jsonl(path, table=table)


def lookup_result(
    *,
    task: str,
    item_id: str,
    metric: str = "llm_em",
    prefer_best: bool = True,
    min_metric: float | None = None,
) -> dict[str, Any] | None:
    return _io_log.lookup_result(
        task=task,
        item_id=item_id,
        metric=metric,
        prefer_best=prefer_best,
        min_metric=min_metric,
    )
