"""Compatibility shim — agent_spec was extracted to prompt_eval (Plan #90).

This module re-exports ``load_agent_spec`` so that in-package imports
(e.g. ``llm_client.observability.experiments``) continue to work.
"""

from __future__ import annotations

try:
    from prompt_eval.agent_spec import load_agent_spec
except ImportError:
    def load_agent_spec(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise ImportError(
            "agent_spec was extracted from llm_client to prompt_eval.agent_spec "
            "(Plan #90). Ensure prompt_eval is installed in this environment."
        )

__all__ = ["load_agent_spec"]
