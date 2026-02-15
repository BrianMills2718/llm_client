"""Agent SDK routing for llm_client.

Routes agent model strings (e.g., "claude-code", "claude-code/opus") to the
Claude Agent SDK instead of litellm. Provides the same LLMCallResult interface.

Supports:
- Basic calls (_acall_agent / _call_agent)
- Structured output (_acall_agent_structured / _call_agent_structured)
- Streaming (AsyncAgentStream / AgentStream, message-level granularity)

Agent models are detected by prefix:
- "claude-code" or "claude-code/<model>" → Claude Agent SDK
- "openai-agents/<model>" → Reserved (NotImplementedError)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json as _json
import logging
import queue
import threading
from typing import Any

from pydantic import BaseModel

from llm_client.client import Hooks, LLMCallResult

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
    prompt, options, sdk = _build_agent_options(model, messages, **kwargs)
    AssistantMessage, _, ResultMessage, TextBlock, query_fn = sdk

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

    return _result_from_agent(model, text_parts, result_msg)


def _call_agent(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> LLMCallResult:
    """Sync wrapper for _acall_agent."""
    coro = _acall_agent(model, messages, timeout=timeout, **kwargs)
    return _run_sync(coro)


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


def _import_sdk() -> tuple[Any, ...]:
    """Lazily import claude_agent_sdk components.

    Returns:
        (AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query)
    """
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
    return AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query


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
        (AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query)
    """
    sdk_name, underlying_model = _parse_agent_model(model)

    if sdk_name in ("openai-agents",):
        raise NotImplementedError(
            f"Agent SDK '{sdk_name}' is not yet supported. "
            "Only 'claude-code' is implemented."
        )
    if sdk_name != "claude-code":
        raise ValueError(f"Unknown agent SDK: {sdk_name}")

    sdk = _import_sdk()
    _, ClaudeAgentOptions, _, _, _ = sdk

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
    if output_format is not None:
        options_kw["output_format"] = output_format
    for key in ("allowed_tools", "cwd", "max_turns", "permission_mode", "max_budget_usd"):
        if key in agent_kw:
            options_kw[key] = agent_kw[key]

    options = ClaudeAgentOptions(**options_kw)
    return prompt, options, sdk


def _result_from_agent(
    model: str,
    text_parts: list[str],
    result_msg: Any,
) -> LLMCallResult:
    """Build LLMCallResult from collected agent output."""
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
    """Call an agent SDK with structured output (JSON schema).

    Uses the SDK's output_format to request JSON conforming to the Pydantic
    model's schema. Parses and validates the result.

    Returns:
        Tuple of (validated Pydantic model instance, LLMCallResult)
    """
    schema = response_model.model_json_schema()
    output_format = {"type": "json_schema", "schema": schema}

    prompt, options, sdk = _build_agent_options(
        model, messages, output_format=output_format, **kwargs,
    )
    AssistantMessage, _, ResultMessage, TextBlock, query_fn = sdk

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

    llm_result = _result_from_agent(model, text_parts, result_msg)
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
    coro = _acall_agent_structured(
        model, messages, response_model, timeout=timeout, **kwargs,
    )
    return _run_sync(coro)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class AsyncAgentStream:
    """Async streaming wrapper for agent SDK. Yields text chunks per AssistantMessage.

    Granularity is message-level (each TextBlock from AssistantMessage), not
    token-level. This is coarser than litellm streaming but still useful for
    seeing results as the agent produces them.

    Example::

        stream = await _astream_agent("claude-code", messages)
        async for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.result.cost)
    """

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
        self._task: asyncio.Task[None] | None = None
        self._timeout = timeout
        self._messages = messages
        self._kwargs = kwargs

    async def _produce(self) -> None:
        """Run the agent query and push text chunks to the queue."""
        try:
            prompt, options, sdk = _build_agent_options(
                self._model, self._messages, **self._kwargs,
            )
            AssistantMessage, _, ResultMessage, TextBlock, query_fn = sdk

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
        if self._task is None:
            self._task = asyncio.create_task(self._produce())

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
            self._model, self._text_parts, self._result_msg,
        )
        if self._hooks and self._hooks.after_call:
            self._hooks.after_call(self._result)

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result


class AgentStream:
    """Sync streaming wrapper for agent SDK. Wraps AsyncAgentStream in a background thread.

    Example::

        stream = _stream_agent("claude-code", messages)
        for chunk in stream:
            print(chunk, end="", flush=True)
        print(stream.result.cost)
    """

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
    """Create an async agent stream. Returns immediately; iteration drives execution."""
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
    """Create a sync agent stream. Returns immediately; iteration drives execution."""
    if hooks and hooks.before_call:
        hooks.before_call(model, messages, kwargs)
    return AgentStream(model, messages, hooks=hooks, timeout=timeout, **kwargs)
