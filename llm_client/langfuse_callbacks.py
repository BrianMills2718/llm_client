"""Langfuse observability integration via LiteLLM's built-in callback mechanism.

Activates Langfuse as a complementary observability backend alongside the default
JSONL+SQLite logging. Langfuse is never required -- it activates only when both
conditions are met:

    1. ``LITELLM_CALLBACKS`` env var includes ``langfuse``
    2. The ``langfuse`` package is importable

LiteLLM reads Langfuse credentials from standard env vars automatically:
    ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, ``LANGFUSE_HOST``

Usage (shell)::

    export LITELLM_CALLBACKS=langfuse
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=https://cloud.langfuse.com   # or self-hosted

Once configured, every ``litellm.completion()`` / ``litellm.acompletion()`` call
is automatically traced in Langfuse. The ``task=`` and ``trace_id=`` metadata
from llm_client calls flows through as Langfuse trace metadata via litellm's
``metadata`` kwarg.

Install the optional dependency::

    pip install llm-client[langfuse]
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_initialized: bool = False


def configure_langfuse_callbacks() -> bool:
    """Register Langfuse in LiteLLM's callback lists if env-configured.

    Reads ``LITELLM_CALLBACKS`` for a comma-separated list of callback names.
    If ``langfuse`` is present, verifies the package is importable, then appends
    ``"langfuse"`` to ``litellm.success_callback`` and ``litellm.failure_callback``.

    Returns True if Langfuse callbacks were activated, False otherwise.
    Idempotent -- safe to call multiple times.
    """
    global _initialized  # noqa: PLW0603

    if _initialized:
        return _is_active()

    _initialized = True

    callbacks_env = os.environ.get("LITELLM_CALLBACKS", "")
    requested = [cb.strip().lower() for cb in callbacks_env.split(",") if cb.strip()]

    if "langfuse" not in requested:
        return False

    try:
        import langfuse  # noqa: F401
    except ImportError:
        logger.warning(
            "LITELLM_CALLBACKS includes 'langfuse' but the langfuse package is not "
            "installed. Install it with: pip install llm-client[langfuse]"
        )
        return False

    import litellm

    if "langfuse" not in litellm.success_callback:
        litellm.success_callback.append("langfuse")  # type: ignore[attr-defined]
    if "langfuse" not in litellm.failure_callback:
        litellm.failure_callback.append("langfuse")  # type: ignore[attr-defined]

    logger.info("Langfuse callbacks activated via LITELLM_CALLBACKS")
    return True


def _is_active() -> bool:
    """Check whether Langfuse callbacks are currently registered in LiteLLM."""
    try:
        import litellm
        return "langfuse" in litellm.success_callback  # type: ignore[attr-defined]
    except ImportError:
        return False


def inject_metadata(
    kwargs: dict[str, object],
    *,
    task: str | None = None,
    trace_id: str | None = None,
) -> None:
    """Inject task/trace_id into litellm's metadata kwarg for callback propagation.

    Merges llm_client's ``task`` and ``trace_id`` into the ``metadata`` dict that
    litellm passes to all registered callbacks (including Langfuse). Preserves any
    existing metadata the caller already set.

    This is a no-op when both task and trace_id are None.
    """
    meta: dict[str, object] = {"_llm_client_logged": True}
    if task is not None:
        meta["task"] = task
    if trace_id is not None:
        meta["trace_id"] = trace_id

    existing = kwargs.get("metadata")
    if isinstance(existing, dict):
        existing.update(meta)
    else:
        kwargs["metadata"] = meta
