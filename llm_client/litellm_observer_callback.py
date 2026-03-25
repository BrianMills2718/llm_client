"""LiteLLM callback that writes to llm_client's JSONL+SQLite observability.

Captures cost, tokens, latency, and model for ALL litellm calls — including
calls from projects that haven't migrated to llm_client yet (Digimon's
LiteLLMProvider, sam_gov, etc.).

Activate by adding this callback to litellm in your project's entry point::

    import litellm
    from llm_client.litellm_observer_callback import LLMClientObserverCallback

    litellm.callbacks = [LLMClientObserverCallback()]

Or via environment (requires registration in litellm's callback registry)::

    export LITELLM_CALLBACKS=llm_client_observer

Calls already routed through llm_client's call_llm/acall_llm are logged by
the normal path and will NOT be double-logged — the callback checks for the
``_llm_client_logged`` marker in metadata.
"""

from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


class LLMClientObserverCallback(CustomLogger):
    """LiteLLM callback that logs to llm_client's observability store."""

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Log successful litellm calls to JSONL+SQLite."""
        try:
            self._log_event(kwargs, response_obj, start_time, end_time, error=None)
        except Exception:
            logger.debug("LLMClientObserverCallback.log_success_event failed", exc_info=True)

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Log failed litellm calls to JSONL+SQLite."""
        try:
            error = kwargs.get("exception") or kwargs.get("original_exception")
            self._log_event(kwargs, response_obj, start_time, end_time, error=error)
        except Exception:
            logger.debug("LLMClientObserverCallback.log_failure_event failed", exc_info=True)

    def _log_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        error: Any,
    ) -> None:
        """Extract fields from litellm callback args and write to io_log."""
        # Skip if already logged by llm_client's normal path
        metadata = kwargs.get("litellm_params", {}).get("metadata", {}) or {}
        if metadata.get("_llm_client_logged"):
            return

        # Late import to avoid circular dependencies at module load time
        from llm_client.io_log import log_call

        # Extract standard logging payload if available
        slp = kwargs.get("standard_logging_object")

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages")

        # Compute latency
        latency_s = None
        if start_time and end_time:
            try:
                latency_s = (end_time - start_time).total_seconds()
            except (TypeError, AttributeError):
                pass

        # Build a lightweight result-like object for log_call
        if slp:
            result = _SLPResult(
                content=slp.get("response") if isinstance(slp.get("response"), str) else None,
                usage={
                    "prompt_tokens": slp.get("prompt_tokens", 0),
                    "completion_tokens": slp.get("completion_tokens", 0),
                    "total_tokens": slp.get("total_tokens", 0),
                },
                cost=slp.get("response_cost", 0.0),
                cost_source="provider_reported",
                cache_hit=bool(slp.get("cache_hit")),
                finish_reason=None,
            )
            task = metadata.get("task", f"litellm_callback/{slp.get('call_type', 'completion')}")
        elif response_obj is not None:
            usage_raw = getattr(response_obj, "usage", None)
            usage = {}
            if usage_raw:
                usage = {
                    "prompt_tokens": getattr(usage_raw, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage_raw, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage_raw, "total_tokens", 0) or 0,
                }
            content = None
            choices = getattr(response_obj, "choices", None)
            if choices and len(choices) > 0:
                msg = getattr(choices[0], "message", None)
                if msg:
                    content = getattr(msg, "content", None)

            from litellm import completion_cost
            try:
                cost = completion_cost(completion_response=response_obj)
            except Exception:
                cost = 0.0

            result = _SLPResult(
                content=content,
                usage=usage,
                cost=cost,
                cost_source="computed",
                cache_hit=False,
                finish_reason=getattr(choices[0], "finish_reason", None) if choices else None,
            )
            task = metadata.get("task", "litellm_callback/completion")
        else:
            result = None
            task = metadata.get("task", "litellm_callback/unknown")

        trace_id = metadata.get("trace_id", f"litellm_callback/{model}")

        log_call(
            model=model,
            messages=messages,
            result=result,
            error=Exception(str(error)) if error else None,
            latency_s=latency_s,
            caller="litellm_callback",
            task=task,
            trace_id=trace_id,
        )


class _SLPResult:
    """Minimal result object compatible with io_log.log_call field extraction."""

    __slots__ = ("content", "usage", "cost", "cost_source", "cache_hit",
                 "finish_reason", "billing_mode", "marginal_cost", "warnings",
                 "tool_calls")

    def __init__(
        self,
        *,
        content: str | None,
        usage: dict[str, int],
        cost: float,
        cost_source: str,
        cache_hit: bool,
        finish_reason: str | None,
    ) -> None:
        self.content = content
        self.usage = usage
        self.cost = cost
        self.cost_source = cost_source
        self.cache_hit = cache_hit
        self.finish_reason = finish_reason
        self.billing_mode = "api_metered"
        self.marginal_cost = 0.0 if cache_hit else cost
        self.warnings = None
        self.tool_calls = None
