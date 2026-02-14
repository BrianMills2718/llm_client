"""Agent SDK routing for llm_client.

Routes agent model strings (e.g., "claude-code", "claude-code/opus") to the
Claude Agent SDK instead of litellm. Provides the same LLMCallResult interface.

Agent models are detected by prefix:
- "claude-code" or "claude-code/<model>" → Claude Agent SDK
- "openai-agents/<model>" → Reserved (NotImplementedError)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

from llm_client.client import LLMCallResult

logger = logging.getLogger(__name__)


def _parse_agent_model(model: str) -> tuple[str, str | None]:
    """Parse an agent model string into (sdk_name, underlying_model).

    Examples:
        "claude-code"         → ("claude-code", None)
        "claude-code/opus"    → ("claude-code", "opus")
        "openai-agents/gpt-5" → ("openai-agents", "gpt-5")
    """
    if "/" in model:
        sdk, _, underlying = model.partition("/")
        return (sdk.lower(), underlying)
    return (model.lower(), None)


# kwargs consumed by agent SDKs (not passed through)
_AGENT_KWARGS = frozenset({
    "allowed_tools", "cwd", "max_turns", "permission_mode", "max_budget_usd",
})


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


async def _acall_agent(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> LLMCallResult:
    """Call an agent SDK and return an LLMCallResult.

    Lazily imports the agent SDK. Currently supports claude-agent-sdk only.
    Hooks are NOT fired here — the caller (call_llm/acall_llm) handles them.
    """
    sdk_name, underlying_model = _parse_agent_model(model)

    if sdk_name in ("openai-agents",):
        raise NotImplementedError(
            f"Agent SDK '{sdk_name}' is not yet supported. "
            "Only 'claude-code' is implemented."
        )

    if sdk_name != "claude-code":
        raise ValueError(f"Unknown agent SDK: {sdk_name}")

    try:
        from claude_agent_sdk import (  # type: ignore[import-untyped]
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )
    except ImportError:
        raise ImportError(
            "claude-agent-sdk is required for agent models. "
            "Install with: pip install llm_client[agents]"
        ) from None

    prompt, system_prompt = _messages_to_agent_prompt(messages)

    # Separate agent kwargs from others
    agent_kw: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _AGENT_KWARGS:
            agent_kw[k] = v
        else:
            logger.debug("Ignoring kwarg %r for agent model %s", k, model)

    # Build ClaudeAgentOptions
    options_kw: dict[str, Any] = {}
    if underlying_model is not None:
        options_kw["model"] = underlying_model
    if system_prompt is not None:
        options_kw["system_prompt"] = system_prompt
    for key in ("allowed_tools", "cwd", "max_turns", "permission_mode", "max_budget_usd"):
        if key in agent_kw:
            options_kw[key] = agent_kw[key]

    options = ClaudeAgentOptions(**options_kw)

    # Run the agent and collect output
    text_parts: list[str] = []
    result_msg: ResultMessage | None = None

    async def _run() -> None:
        nonlocal result_msg
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                result_msg = message

    if timeout > 0:
        await asyncio.wait_for(_run(), timeout=float(timeout))
    else:
        await _run()

    content = "\n".join(text_parts) if text_parts else ""
    cost = (
        result_msg.total_cost_usd
        if result_msg and result_msg.total_cost_usd is not None
        else 0.0
    )
    usage = result_msg.usage if result_msg and result_msg.usage else {}
    is_error = result_msg.is_error if result_msg else True

    return LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
        finish_reason="error" if is_error else "stop",
        raw_response=result_msg,
    )


def _call_agent(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> LLMCallResult:
    """Sync wrapper for _acall_agent."""
    coro = _acall_agent(model, messages, timeout=timeout, **kwargs)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
