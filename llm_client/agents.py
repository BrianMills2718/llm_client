"""Agent SDK routing for llm_client.

Routes agent model strings to the appropriate agent SDK instead of litellm.
Provides the same LLMCallResult interface regardless of SDK.

Supports:
- Basic calls (_route_call / _route_acall)
- Structured output (_route_call_structured / _route_acall_structured)
- Streaming (_route_stream / _route_astream)

Agent models are detected by prefix:
- "claude-code" or "claude-code/<model>" → Claude Agent SDK
- "codex" or "codex/<model>" → Codex SDK
- "openai-agents/<model>" → Reserved (NotImplementedError)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json as _json
import logging
import os
from typing import Any, Awaitable, Callable, TypeVar, cast

from pydantic import BaseModel

from llm_client.client import Hooks, LLMCallResult
from llm_client.timeout_policy import normalize_timeout as _normalize_timeout

_T = TypeVar("_T")

logger = logging.getLogger(__name__)
_CODEX_AGENT_ALIASES: frozenset[str] = frozenset({"codex-mini-latest"})
_CODEX_TRANSPORT_FALLBACK_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


def _agent_billing_mode() -> str:
    """Resolve billing mode for agent SDK models.

    Modes:
    - subscription (default): no API USD attribution for claude-code/codex
    - api: attribute per-call USD when available/estimable
    """
    mode = os.environ.get("LLM_CLIENT_AGENT_BILLING_MODE", "subscription").strip().lower()
    if mode in {"subscription", "api"}:
        return mode
    return "subscription"


def _parse_agent_model(model: str) -> tuple[str, str | None]:
    """Parse an agent model string into (sdk_name, underlying_model).

    Examples:
        "claude-code"         → ("claude-code", None)
        "claude-code/opus"    → ("claude-code", "opus")
        "codex"               → ("codex", None)
        "codex/gpt-5"         → ("codex", "gpt-5")
        "openai-agents/gpt-5" → ("openai-agents", "gpt-5")
    """
    if "/" in model:
        sdk, _, underlying = model.partition("/")
        return (sdk.lower(), underlying)
    lower = model.lower()
    # Support Codex aliases like "codex-mini-latest" as shorthand for
    # sdk=codex, underlying_model=codex-mini-latest.
    if lower in _CODEX_AGENT_ALIASES:
        return ("codex", model)
    return (lower, None)


def _safe_error_text(exc: BaseException) -> str:
    """Extract error text, falling back to type name if empty."""
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__


def _safe_line_preview(value: Any, *, max_chars: int = 240) -> str:
    """Best-effort compact single-line preview for Codex exec stream items."""
    try:
        if isinstance(value, str):
            text = value
        else:
            text = _json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        text = repr(value)
    text = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _compact_json(payload: dict[str, Any], *, max_chars: int = 1800) -> str:
    """Compact JSON render with truncation guard."""
    try:
        rendered = _json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        rendered = str(payload)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars] + "...(truncated)"


def _is_codex_transport_fallback_error(exc: BaseException) -> bool:
    """Return whether a Codex SDK failure should fall back to CLI transport."""

    if isinstance(exc, _CODEX_TRANSPORT_FALLBACK_EXCEPTIONS):
        return True
    if isinstance(exc, RuntimeError):
        message = _safe_error_text(exc)
        return message.startswith(
            (
                "CODEX_WORKER_ERROR",
                "CODEX_STRUCTURED_WORKER_ERROR",
                "CODEX_WORKER_EOF",
                "CODEX_STRUCTURED_WORKER_EOF",
            )
        )
    return False


def _codex_transport_fallback_warning(exc: BaseException) -> str:
    """Build the standard SDK-to-CLI fallback warning string."""

    return (
        "CODEX_TRANSPORT_FALLBACK[sdk->cli]: "
        f"{type(exc).__name__}: {_safe_error_text(exc)}"
    )


async def _acall_codex_auto_fallback(
    sdk_call: Callable[[], Awaitable[_T]],
    cli_call: Callable[[str], Awaitable[_T]],
) -> _T:
    """Run one async Codex SDK call with bounded auto fallback to CLI transport."""

    try:
        return await sdk_call()
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            raise
        if not _is_codex_transport_fallback_error(exc):
            raise
        warning = _codex_transport_fallback_warning(exc)
        logger.warning(warning)
        return await cli_call(warning)


def _call_codex_auto_fallback(
    sdk_call: Callable[[], _T],
    cli_call: Callable[[str], _T],
) -> _T:
    """Run one sync Codex SDK call with bounded auto fallback to CLI transport."""

    try:
        return sdk_call()
    except BaseException as exc:
        if not _is_codex_transport_fallback_error(exc):
            raise
        warning = _codex_transport_fallback_warning(exc)
        logger.warning(warning)
        return cli_call(warning)


# kwargs consumed by agent SDKs (not passed through)
_AGENT_KWARGS = frozenset({
    # Claude Agent SDK
    "allowed_tools", "cwd", "max_turns", "permission_mode", "max_budget_usd",
    # Codex SDK
    "sandbox_mode", "working_directory", "approval_policy",
    "model_reasoning_effort", "network_access_enabled", "web_search_enabled",
    "additional_directories", "skip_git_repo_check", "yolo_mode",
    "api_key", "base_url",
    # Codex MCP server control
    "codex_home", "mcp_servers",
    # Codex transport controls
    "codex_transport", "codex_cli_path", "agent_hard_timeout",
    # Codex runtime isolation controls
    "codex_process_isolation", "codex_process_start_method", "codex_process_grace_s",
})

_CODEX_PROCESS_ISOLATION_ENV = "LLM_CLIENT_CODEX_PROCESS_ISOLATION"
_CODEX_PROCESS_START_METHOD_ENV = "LLM_CLIENT_CODEX_PROCESS_START_METHOD"
_CODEX_PROCESS_GRACE_ENV = "LLM_CLIENT_CODEX_PROCESS_GRACE_S"
_CODEX_ALLOW_MINIMAL_EFFORT_ENV = "LLM_CLIENT_CODEX_ALLOW_MINIMAL_EFFORT"
_CODEX_TRANSPORT_ENV = "LLM_CLIENT_CODEX_TRANSPORT"


def _normalize_codex_reasoning_effort(value: Any) -> str:
    """Normalize Codex reasoning effort to SDK-accepted values.

    Some Codex-backed models reject ``xhigh``; normalize aliases to
    minimal/low/medium/high and default to ``high`` when omitted.
    """
    raw = str(value or "").strip().lower()
    if not raw:
        return "high"
    if raw in {"low", "medium", "high"}:
        return raw
    if raw == "minimal":
        # Codex ChatGPT-account lanes can reject minimal effort because the
        # platform auto-enables web_search tooling. Keep a controlled override
        # for environments where minimal has been validated.
        if _as_bool(os.environ.get(_CODEX_ALLOW_MINIMAL_EFFORT_ENV), default=False):
            return "minimal"
        logger.warning(
            "Codex model_reasoning_effort=minimal is often rejected by the platform; "
            "coercing to low (set %s=1 to force minimal).",
            _CODEX_ALLOW_MINIMAL_EFFORT_ENV,
        )
        return "low"
    if raw in {"xhigh", "very_high", "highest", "max"}:
        return "high"
    if raw in {"none", "off", "disabled"}:
        return "minimal"
    return "high"


def _apply_agent_yolo_defaults(agent_name: str, agent_kw: dict[str, Any]) -> dict[str, Any]:
    """Apply convenience defaults for an explicitly autonomous agent run.

    `yolo_mode=True` is syntactic sugar for callers who want the agent to run
    headlessly without having to remember the exact per-SDK knobs. Explicit
    kwargs still win over the convenience defaults.
    """

    if not agent_kw.get("yolo_mode"):
        return agent_kw

    if agent_name == "codex":
        agent_kw.setdefault("sandbox_mode", "workspace-write")
        agent_kw.setdefault("approval_policy", "never")
        agent_kw.setdefault("skip_git_repo_check", True)
    elif agent_name == "claude-code":
        agent_kw.setdefault("permission_mode", "bypassPermissions")
    return agent_kw


def _messages_to_agent_prompt(
    messages: list[dict[str, Any]],
) -> tuple[str, str | None]:
    """Convert OpenAI-format messages to (prompt, system_prompt).

    - role="system" → system_prompt (first system message only)
    - Single user message → prompt directly
    - Multi-turn → "User: ...\\nAssistant: ...\\nUser: ..."
    """
    system_prompt: str | None = None
    conversation: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system" and system_prompt is None:
            system_prompt = content
        else:
            conversation.append(msg)

    if not conversation:
        raise ValueError("No user/assistant messages found in messages list")

    # Single user message → prompt directly
    if len(conversation) == 1 and conversation[0].get("role") == "user":
        return (conversation[0]["content"], system_prompt)

    # Multi-turn → concatenate
    parts: list[str] = []
    for msg in conversation:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        label = role.capitalize()
        parts.append(f"{label}: {content}")

    return ("\n".join(parts), system_prompt)


def _run_sync(coro: Any) -> Any:
    """Run a coroutine synchronously, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default



# ---------------------------------------------------------------------------
# Re-exports from agents_claude.py (backward compatibility)
# ---------------------------------------------------------------------------
from llm_client.agents_claude import (  # noqa: F401
    AsyncAgentStream,
    AgentStream,
    _acall_agent,
    _acall_agent_structured,
    _astream_agent,
    _build_agent_options,
    _call_agent,
    _call_agent_structured,
    _capture_tool_result,
    _import_sdk,
    _result_from_agent,
    _stream_agent,
)

# ---------------------------------------------------------------------------
# Re-exports from agents_codex.py (backward compatibility)
# ---------------------------------------------------------------------------
from llm_client.agents_codex import (  # noqa: F401
    AsyncCodexStream,
    CodexStream,
    _acall_codex,
    _acall_codex_inproc,
    _acall_codex_structured,
    _acall_codex_structured_inproc,
    _acall_codex_via_cli,
    _agent_hard_timeout,
    _astream_codex,
    _await_codex_turn_with_hard_timeout,
    _build_codex_cli_command,
    _build_codex_options,
    _call_codex,
    _call_codex_in_isolated_process,
    _call_codex_structured,
    _call_codex_structured_in_isolated_process,
    _call_codex_via_cli,
    _cleanup_tmp,
    _codex_cli_path,
    _codex_exec_diagnostics,
    _codex_process_grace_s,
    _codex_process_isolation_enabled,
    _codex_process_start_method,
    _codex_structured_worker_entry,
    _codex_text_worker_entry,
    _codex_timeout_message,
    _codex_transport,
    _collect_process_tree_snapshot,
    _create_codex_home,
    _deserialize_llm_result,
    _estimate_codex_cost,
    _extract_codex_tool_calls,
    _import_codex_sdk,
    _patch_codex_buffer_limit,
    _prepare_codex_mcp,
    _process_exists,
    _result_from_codex,
    _result_from_codex_cli,
    _serialize_llm_result,
    _stream_codex,
    _strip_fences,
    _terminate_pid_tree,
)

# ===========================================================================
# Routing dispatchers — dispatch by SDK name
# ===========================================================================


def _route_call(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> LLMCallResult:
    """Route a sync agent call to the appropriate SDK."""
    timeout = _normalize_timeout(timeout, caller="_route_call", logger=logger)
    on_turn = kwargs.pop("on_turn", None)
    sdk_name, _ = _parse_agent_model(model)
    if sdk_name == "codex":
        return _call_codex(model, messages, timeout=timeout, on_turn=on_turn, **kwargs)
    if sdk_name == "claude-code":
        return _call_agent(model, messages, timeout=timeout, on_turn=on_turn, **kwargs)
    if sdk_name == "openai-agents":
        raise NotImplementedError(
            f"Agent SDK '{sdk_name}' is not yet supported. "
            "Only 'claude-code' and 'codex' are implemented."
        )
    raise ValueError(f"Unknown agent SDK: {sdk_name}")


async def _route_acall(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> LLMCallResult:
    """Route an async agent call to the appropriate SDK."""
    timeout = _normalize_timeout(timeout, caller="_route_acall", logger=logger)
    on_turn = kwargs.pop("on_turn", None)
    sdk_name, _ = _parse_agent_model(model)
    if sdk_name == "codex":
        return await _acall_codex(model, messages, timeout=timeout, on_turn=on_turn, **kwargs)
    if sdk_name == "claude-code":
        return await _acall_agent(model, messages, timeout=timeout, on_turn=on_turn, **kwargs)
    if sdk_name == "openai-agents":
        raise NotImplementedError(
            f"Agent SDK '{sdk_name}' is not yet supported. "
            "Only 'claude-code' and 'codex' are implemented."
        )
    raise ValueError(f"Unknown agent SDK: {sdk_name}")


def _route_call_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Route a sync structured agent call to the appropriate SDK."""
    timeout = _normalize_timeout(timeout, caller="_route_call_structured", logger=logger)
    sdk_name, _ = _parse_agent_model(model)
    if sdk_name == "codex":
        return _call_codex_structured(model, messages, response_model, timeout=timeout, **kwargs)
    if sdk_name == "claude-code":
        return _call_agent_structured(model, messages, response_model, timeout=timeout, **kwargs)
    if sdk_name == "openai-agents":
        raise NotImplementedError(
            f"Agent SDK '{sdk_name}' is not yet supported. "
            "Only 'claude-code' and 'codex' are implemented."
        )
    raise ValueError(f"Unknown agent SDK: {sdk_name}")


async def _route_acall_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Route an async structured agent call to the appropriate SDK."""
    timeout = _normalize_timeout(timeout, caller="_route_acall_structured", logger=logger)
    sdk_name, _ = _parse_agent_model(model)
    if sdk_name == "codex":
        return await _acall_codex_structured(model, messages, response_model, timeout=timeout, **kwargs)
    if sdk_name == "claude-code":
        return await _acall_agent_structured(model, messages, response_model, timeout=timeout, **kwargs)
    if sdk_name == "openai-agents":
        raise NotImplementedError(
            f"Agent SDK '{sdk_name}' is not yet supported. "
            "Only 'claude-code' and 'codex' are implemented."
        )
    raise ValueError(f"Unknown agent SDK: {sdk_name}")


def _route_stream(
    model: str,
    messages: list[dict[str, Any]],
    *,
    hooks: Hooks | None = None,
    timeout: int = 300,
    **kwargs: Any,
) -> AgentStream | CodexStream:
    """Route a sync agent stream to the appropriate SDK."""
    timeout = _normalize_timeout(timeout, caller="_route_stream", logger=logger)
    sdk_name, _ = _parse_agent_model(model)
    if sdk_name == "codex":
        return _stream_codex(model, messages, hooks=hooks, timeout=timeout, **kwargs)
    if sdk_name == "claude-code":
        return _stream_agent(model, messages, hooks=hooks, timeout=timeout, **kwargs)
    if sdk_name == "openai-agents":
        raise NotImplementedError(
            f"Agent SDK '{sdk_name}' is not yet supported. "
            "Only 'claude-code' and 'codex' are implemented."
        )
    raise ValueError(f"Unknown agent SDK: {sdk_name}")


async def _route_astream(
    model: str,
    messages: list[dict[str, Any]],
    *,
    hooks: Hooks | None = None,
    timeout: int = 300,
    **kwargs: Any,
) -> AsyncAgentStream | AsyncCodexStream:
    """Route an async agent stream to the appropriate SDK."""
    timeout = _normalize_timeout(timeout, caller="_route_astream", logger=logger)
    sdk_name, _ = _parse_agent_model(model)
    if sdk_name == "codex":
        return await _astream_codex(model, messages, hooks=hooks, timeout=timeout, **kwargs)
    if sdk_name == "claude-code":
        return await _astream_agent(model, messages, hooks=hooks, timeout=timeout, **kwargs)
    if sdk_name == "openai-agents":
        raise NotImplementedError(
            f"Agent SDK '{sdk_name}' is not yet supported. "
            "Only 'claude-code' and 'codex' are implemented."
        )
    raise ValueError(f"Unknown agent SDK: {sdk_name}")

