"""Compatibility shim — experiment_eval was extracted to prompt_eval (Plan #17).

This module re-exports the public API so that in-package imports
(e.g. ``llm_client.cli.experiments``) continue to work without
requiring prompt_eval as a hard dependency at import time.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prompt_eval.experiment_eval import (  # type: ignore[import-untyped]
        build_gate_signals,
        evaluate_gate_policy,
        load_gate_policy,
        review_items_with_rubric,
        run_deterministic_checks_for_items,
        triage_items,
    )
except ImportError as exc:
    _import_err = exc

    def _raise(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ImportError(
            "experiment_eval was extracted from llm_client to prompt_eval (Plan #17). "
            "Install prompt_eval: pip install -e ~/projects/prompt_eval"
        ) from _import_err

    build_gate_signals = _raise
    evaluate_gate_policy = _raise
    load_gate_policy = _raise
    review_items_with_rubric = _raise
    run_deterministic_checks_for_items = _raise
    triage_items = _raise

__all__ = [
    "build_gate_signals",
    "evaluate_gate_policy",
    "load_gate_policy",
    "review_items_with_rubric",
    "run_deterministic_checks_for_items",
    "triage_items",
]
