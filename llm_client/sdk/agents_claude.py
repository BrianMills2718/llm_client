"""Claude Agent SDK adapter for llm_client.

Handles all Claude Code agent interactions: basic calls, structured output,
and streaming. Converts Claude Agent SDK types (AssistantMessage, ResultMessage,
ToolUseBlock, etc.) into the unified LLMCallResult interface.

This module is an implementation detail of agents.py. All public names are
re-exported from agents.py for backward compatibility.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import queue
import threading
import time
from typing import Any, Callable, cast

from pydantic import BaseModel

from llm_client.core.client import Hooks, LLMCallResult
from llm_client.core.data_types import TurnEvent
from llm_client.execution.timeout_policy import normalize_timeout as _normalize_timeout

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SDK import
# ---------------------------------------------------------------------------


def _import_sdk() -> tuple[Any, ...]:
    """Lazily import claude_agent_sdk components.

    Returns:
        (AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, ToolUseBlock, query)
    """
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            ToolUseBlock,
            query,
        )
    except ImportError:
        raise ImportError(
            "claude-agent-sdk is required for agent models. "
            "Install with: pip install llm_client[agents]"
        ) from None
    return AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, ToolUseBlock, query


# ---------------------------------------------------------------------------
# Option building
# ---------------------------------------------------------------------------


def _build_agent_options(
    model: str,
    messages: list[dict[str, Any]],
    *,
    output_format: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[str, Any, Any]:
    """Build ClaudeAgentOptions from model string + kwargs.

    Returns:
        (prompt, options, sdk_components) where sdk_components is
        (AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, ToolUseBlock, query)
    """
    import os

    from llm_client.sdk.agents import (
        _AGENT_KWARGS,
        _apply_agent_yolo_defaults,
        _messages_to_agent_prompt,
        _parse_agent_model,
    )

    _, underlying_model = _parse_agent_model(model)

    sdk = _import_sdk()
    _, ClaudeAgentOptions, _, _, _, _ = sdk

    prompt, system_prompt = _messages_to_agent_prompt(messages)

    # Separate agent kwargs from others
    agent_kw: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _AGENT_KWARGS:
            agent_kw[k] = v
        else:
            logger.debug("Ignoring kwarg %r for agent model %s", k, model)
    agent_kw = _apply_agent_yolo_defaults("claude-code", agent_kw)

    # Build ClaudeAgentOptions
    options_kw: dict[str, Any] = {}
    if underlying_model is not None:
        options_kw["model"] = underlying_model
    if system_prompt is not None:
        options_kw["system_prompt"] = system_prompt
    if output_format is not None:
        options_kw["output_format"] = output_format
    for key in ("allowed_tools", "cwd", "max_turns", "permission_mode", "max_budget_usd", "mcp_servers"):
        if key in agent_kw:
            options_kw[key] = agent_kw[key]

    # Agent subprocesses need a clean env:
    # 1. CLAUDECODE causes the child CLI to refuse to start (nested session detection)
    # 2. Auto-loaded API keys (e.g. ANTHROPIC_API_KEY from ~/.secrets/api_keys.env)
    #    cause the bundled CLI to use the wrong auth mechanism instead of OAuth.
    env = options_kw.get("env", {})
    if os.environ.get("CLAUDECODE"):
        env.setdefault("CLAUDECODE", "")
    from llm_client import _auto_loaded_keys
    for key in _auto_loaded_keys:
        env.setdefault(key, "")
    if env:
        options_kw["env"] = env

    options = ClaudeAgentOptions(**options_kw)
    return prompt, options, sdk


# ---------------------------------------------------------------------------
# Result building
# ---------------------------------------------------------------------------


def _result_from_agent(
    model: str,
    last_message_text: list[str],
    all_text: list[str],
    result_msg: Any,
    agent_tool_calls: list[dict[str, Any]] | None = None,
) -> LLMCallResult:
    """Build LLMCallResult from collected agent output.

    Args:
        last_message_text: Text blocks from the final assistant message only.
        all_text: Text blocks from ALL assistant messages (full conversation).
        result_msg: The ResultMessage from the SDK.
        agent_tool_calls: Tool use records captured from ToolUseBlock in messages.
    """
    from llm_client.sdk.agents import _agent_billing_mode

    content = "\n".join(last_message_text) if last_message_text else ""
    full_text = "\n".join(all_text) if all_text else ""
    billing_mode = _agent_billing_mode()
    if billing_mode == "api":
        cost = (
            result_msg.total_cost_usd
            if result_msg and result_msg.total_cost_usd is not None
            else 0.0
        )
        cost_source = "provider_reported" if result_msg and result_msg.total_cost_usd is not None else "unavailable"
        effective_billing_mode = "api_metered"
    else:
        # Claude Code OAuth subscription plans are not metered per-call in local API logs.
        cost = 0.0
        cost_source = "subscription_included"
        effective_billing_mode = "subscription_included"
    usage: dict[str, Any] = dict(result_msg.usage) if result_msg and result_msg.usage else {}
    is_error = result_msg.is_error if result_msg else True

    # Enrich usage with timing and session data from ResultMessage
    if result_msg:
        for attr in ("duration_ms", "duration_api_ms", "num_turns", "session_id"):
            val = getattr(result_msg, attr, None)
            if val is not None:
                usage[attr] = val

    return LLMCallResult(
        content=content,
        usage=usage,
        cost=cost,
        model=model,
        tool_calls=agent_tool_calls or [],
        finish_reason="error" if is_error else "stop",
        raw_response=result_msg,
        full_text=full_text if full_text != content else None,
        cost_source=cost_source,
        billing_mode=effective_billing_mode,
    )


def _capture_tool_result(
    message: Any,
    conversation_trace: list[dict[str, Any]],
    agent_tool_calls: list[dict[str, Any]],
    tool_id_to_idx: dict[str, int],
    ToolResultBlock: type,
) -> None:
    """Extract tool result data from UserMessage or other message types.

    Claude Agent SDK sends UserMessage with:
      - content: list[TextBlock | ToolResultBlock | ...] — the actual tool result data
      - tool_use_result: dict — a summary dict (not needed when content has ToolResultBlock)
    """
    # UserMessage.content is a list of blocks — extract ToolResultBlock instances
    content_list = getattr(message, "content", None)
    if isinstance(content_list, list):
        for block in content_list:
            if isinstance(block, ToolResultBlock):
                tool_use_id = getattr(block, "tool_use_id", "") or ""
                raw_content = getattr(block, "content", "")
                is_error = getattr(block, "is_error", False) or False
                content_str = str(raw_content)[:2000] if raw_content else ""
                conversation_trace.append({
                    "role": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content_str,
                    "is_error": is_error,
                })
                if tool_use_id in tool_id_to_idx:
                    idx = tool_id_to_idx[tool_use_id]
                    agent_tool_calls[idx]["result_preview"] = content_str[:500]
                    agent_tool_calls[idx]["is_error"] = is_error
        return

    # Standalone ToolResultBlock (not wrapped in UserMessage)
    tool_use_id = getattr(message, "tool_use_id", None)
    if tool_use_id is not None:
        raw_content = getattr(message, "content", None)
        is_error = getattr(message, "is_error", False) or False
        content_str = str(raw_content)[:2000] if raw_content else ""
        conversation_trace.append({
            "role": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content_str,
            "is_error": is_error,
        })
        if tool_use_id in tool_id_to_idx:
            idx = tool_id_to_idx[tool_use_id]
            agent_tool_calls[idx]["result_preview"] = content_str[:500]
            agent_tool_calls[idx]["is_error"] = is_error


# ---------------------------------------------------------------------------
# Basic calls
# ---------------------------------------------------------------------------


async def _acall_agent(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    on_turn: Callable[[Any], None] | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Call Claude Agent SDK and return an LLMCallResult."""
    timeout = _normalize_timeout(timeout, caller="_acall_agent", logger=logger)
    run_started = time.monotonic()
    prompt, options, sdk = _build_agent_options(model, messages, **kwargs)
    AssistantMessage, _, ResultMessage, TextBlock, ToolUseBlock, query_fn = sdk
    from claude_agent_sdk import ToolResultBlock

    # Track text per assistant message so we can return only the final one
    all_messages_text: list[list[str]] = []
    current_msg_text: list[str] = []
    agent_tool_calls: list[dict[str, Any]] = []
    conversation_trace: list[dict[str, Any]] = []
    result_msg: Any = None
    # Map tool_use_id → index in agent_tool_calls for pairing results
    _tool_id_to_idx: dict[str, int] = {}

    async def _run() -> None:
        nonlocal result_msg, current_msg_text
        async for message in query_fn(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                current_msg_text = []
                step: dict[str, Any] = {"role": "assistant", "content": []}
                for block in message.content:
                    if isinstance(block, TextBlock):
                        current_msg_text.append(block.text)
                        step["content"].append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUseBlock):
                        tc_record = {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": block.input,
                            },
                        }
                        _tool_id_to_idx[block.id] = len(agent_tool_calls)
                        agent_tool_calls.append(tc_record)
                        step["content"].append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })
                if step["content"]:
                    conversation_trace.append(step)
                if current_msg_text:
                    all_messages_text.append(current_msg_text)
                if on_turn is not None:
                    # Count tool_use blocks in this assistant message
                    turn_tool_blocks = [
                        b for b in message.content if isinstance(b, ToolUseBlock)
                    ]
                    n_turn_tools = len(turn_tool_blocks)
                    turn_tcs = (
                        [{"id": tc["id"], "name": tc["function"]["name"],
                          "arguments": tc["function"]["arguments"]}
                         for tc in agent_tool_calls[-n_turn_tools:]]
                        if n_turn_tools > 0 else []
                    )
                    on_turn(TurnEvent(
                        turn=len(conversation_trace),
                        elapsed_s=round(time.monotonic() - run_started, 3),
                        tool_calls=turn_tcs,
                        text_preview=("".join(current_msg_text))[:200],
                    ))
            elif isinstance(message, ResultMessage):
                result_msg = message
            else:
                # Capture tool results from UserMessage or ToolResultBlock
                _capture_tool_result(message, conversation_trace,
                                     agent_tool_calls, _tool_id_to_idx,
                                     ToolResultBlock)

    if timeout > 0:
        await asyncio.wait_for(_run(), timeout=float(timeout))
    else:
        await _run()

    # content = last assistant message only; full_text = everything
    last_text = all_messages_text[-1] if all_messages_text else []
    all_text = [t for parts in all_messages_text for t in parts]
    result = _result_from_agent(model, last_text, all_text, result_msg, agent_tool_calls)
    result.raw_response = {
        "result_message": result_msg,
        "conversation_trace": conversation_trace,
    }
    return result


def _call_agent(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> LLMCallResult:
    """Sync wrapper for _acall_agent."""
    from llm_client.sdk.agents import _run_sync

    coro = _acall_agent(model, messages, timeout=timeout, **kwargs)
    return cast(LLMCallResult, _run_sync(coro))


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


async def _acall_agent_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Call Claude Agent SDK with structured output (JSON schema)."""
    timeout = _normalize_timeout(timeout, caller="_acall_agent_structured", logger=logger)
    schema = response_model.model_json_schema()
    output_format = {"type": "json_schema", "schema": schema}

    prompt, options, sdk = _build_agent_options(
        model, messages, output_format=output_format, **kwargs,
    )
    AssistantMessage, _, ResultMessage, TextBlock, ToolUseBlock, query_fn = sdk

    text_parts: list[str] = []
    result_msg: Any = None

    async def _run() -> None:
        nonlocal result_msg
        async for message in query_fn(prompt=prompt, options=options):
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

    # Parse structured output: prefer SDK's structured_output, else parse text
    parsed_data: Any = None
    if result_msg and hasattr(result_msg, "structured_output") and result_msg.structured_output is not None:
        parsed_data = result_msg.structured_output
    else:
        raw_text = "\n".join(text_parts) if text_parts else ""
        if not raw_text.strip():
            raise ValueError("Empty response from agent — no structured output")
        parsed_data = _json.loads(raw_text)

    validated = response_model.model_validate(parsed_data)

    llm_result = _result_from_agent(model, text_parts, text_parts, result_msg)
    # Override content with the validated JSON for consistency
    llm_result.content = validated.model_dump_json()

    return validated, llm_result


def _call_agent_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Sync wrapper for _acall_agent_structured."""
    from llm_client.sdk.agents import _run_sync

    coro = _acall_agent_structured(
        model, messages, response_model, timeout=timeout, **kwargs,
    )
    return cast(tuple[BaseModel, LLMCallResult], _run_sync(coro))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class AsyncAgentStream:
    """Async streaming wrapper for Claude Agent SDK. Yields text chunks per AssistantMessage."""

    def __init__(
        self,
        model: str,
        messages: list[dict[str, Any]],
        hooks: Hooks | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._hooks = hooks
        self._text_parts: list[str] = []
        self._result_msg: Any = None
        self._result: LLMCallResult | None = None
        self._queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
        self._task_handle: asyncio.Task[None] | None = None
        self._timeout = timeout
        self._messages = messages
        self._log_task = kwargs.pop("task", None)
        self._trace_id = kwargs.pop("trace_id", None)
        self._t0 = time.monotonic()
        self._kwargs = kwargs

    async def _produce(self) -> None:
        """Run the agent query and push text chunks to the queue."""
        try:
            prompt, options, sdk = _build_agent_options(
                self._model, self._messages, **self._kwargs,
            )
            AssistantMessage, _, ResultMessage, TextBlock, ToolUseBlock, query_fn = sdk

            async for message in query_fn(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            self._text_parts.append(block.text)
                            await self._queue.put(block.text)
                elif isinstance(message, ResultMessage):
                    self._result_msg = message

            await self._queue.put(None)  # sentinel
        except Exception as e:
            await self._queue.put(e)

    async def _ensure_started(self) -> None:
        if self._task_handle is None:
            self._task_handle = asyncio.create_task(self._produce())

    def __aiter__(self) -> AsyncAgentStream:
        return self

    async def __anext__(self) -> str:
        await self._ensure_started()
        item = await self._queue.get()
        if item is None:
            self._finalize()
            raise StopAsyncIteration
        if isinstance(item, Exception):
            self._finalize()
            raise item
        return item

    def _finalize(self) -> None:
        self._result = _result_from_agent(
            self._model, self._text_parts, self._text_parts, self._result_msg,
        )
        if self._hooks and self._hooks.after_call:
            self._hooks.after_call(self._result)
        import llm_client.io_log as _io_log
        _io_log.log_call(
            model=self._model, messages=self._messages, result=self._result,
            latency_s=time.monotonic() - self._t0,
            caller="astream_agent", task=self._log_task, trace_id=self._trace_id,
        )

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result


class AgentStream:
    """Sync streaming wrapper for Claude Agent SDK. Wraps AsyncAgentStream in a background thread."""

    def __init__(
        self,
        model: str,
        messages: list[dict[str, Any]],
        hooks: Hooks | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._hooks = hooks
        self._result: LLMCallResult | None = None
        self._queue: queue.Queue[str | None | Exception] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._timeout = timeout
        self._messages = messages
        self._kwargs = kwargs

    def _run_async(self) -> None:
        """Run the async stream in a new event loop on a background thread."""
        async def _consume() -> None:
            astream = AsyncAgentStream(
                self._model, self._messages,
                hooks=self._hooks, timeout=self._timeout, **self._kwargs,
            )
            try:
                async for chunk in astream:
                    self._queue.put(chunk)
                self._result = astream.result
                self._queue.put(None)  # sentinel
            except Exception as e:
                self._queue.put(e)

        asyncio.run(_consume())

    def _ensure_started(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_async, daemon=True)
            self._thread.start()

    def __iter__(self) -> AgentStream:
        return self

    def __next__(self) -> str:
        self._ensure_started()
        item = self._queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            # Drain remaining items if thread is still running
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            if self._result is None:
                raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result


async def _astream_agent(
    model: str,
    messages: list[dict[str, Any]],
    *,
    hooks: Hooks | None = None,
    timeout: int = 300,
    **kwargs: Any,
) -> AsyncAgentStream:
    """Create an async Claude agent stream."""
    if hooks and hooks.before_call:
        hooks.before_call(model, messages, kwargs)
    return AsyncAgentStream(model, messages, hooks=hooks, timeout=timeout, **kwargs)


def _stream_agent(
    model: str,
    messages: list[dict[str, Any]],
    *,
    hooks: Hooks | None = None,
    timeout: int = 300,
    **kwargs: Any,
) -> AgentStream:
    """Create a sync Claude agent stream."""
    if hooks and hooks.before_call:
        hooks.before_call(model, messages, kwargs)
    return AgentStream(model, messages, hooks=hooks, timeout=timeout, **kwargs)
