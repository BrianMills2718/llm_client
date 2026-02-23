"""Observability experiment-run adapters.

This module is a boundary layer around ``llm_client.io_log`` while we
incrementally migrate internals out of that monolith.
"""

from __future__ import annotations

from typing import Any

from llm_client import io_log as _io_log

ActiveExperimentRun = _io_log.ActiveExperimentRun
ExperimentRun = _io_log.ExperimentRun


def activate_experiment_run(run_id: str) -> ActiveExperimentRun:
    return _io_log.activate_experiment_run(run_id)


def experiment_run(**kwargs: Any) -> ExperimentRun:
    return _io_log.experiment_run(**kwargs)


def start_run(**kwargs: Any) -> str:
    return _io_log.start_run(**kwargs)


def log_item(**kwargs: Any) -> None:
    _io_log.log_item(**kwargs)


def finish_run(**kwargs: Any) -> dict[str, Any]:
    return _io_log.finish_run(**kwargs)


def get_run(run_id: str) -> dict[str, Any] | None:
    return _io_log.get_run(run_id)


def get_run_items(run_id: str) -> list[dict[str, Any]]:
    return _io_log.get_run_items(run_id)
