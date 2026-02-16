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
import queue
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_client.client import Hooks, LLMCallResult

logger = logging.getLogger(__name__)


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
    return (model.lower(), None)


# kwargs consumed by agent SDKs (not passed through)
_AGENT_KWARGS = frozenset({
    # Claude Agent SDK
    "allowed_tools", "cwd", "max_turns", "permission_mode", "max_budget_usd",
    # Codex SDK
    "sandbox_mode", "working_directory", "approval_policy",
    "model_reasoning_effort", "network_access_enabled", "web_search_enabled",
    "additional_directories", "skip_git_repo_check",
    "api_key", "base_url",
    # Codex MCP server control
    "codex_home", "mcp_servers",
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


# ===========================================================================
# Claude Agent SDK
# ===========================================================================


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
    _, underlying_model = _parse_agent_model(model)

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


def _result_from_agent(
    model: str,
    last_message_text: list[str],
    all_text: list[str],
    result_msg: Any,
) -> LLMCallResult:
    """Build LLMCallResult from collected agent output.

    Args:
        last_message_text: Text blocks from the final assistant message only.
        all_text: Text blocks from ALL assistant messages (full conversation).
        result_msg: The ResultMessage from the SDK.
    """
    content = "\n".join(last_message_text) if last_message_text else ""
    full_text = "\n".join(all_text) if all_text else ""
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
        full_text=full_text if full_text != content else None,
    )


async def _acall_agent(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> LLMCallResult:
    """Call Claude Agent SDK and return an LLMCallResult."""
    prompt, options, sdk = _build_agent_options(model, messages, **kwargs)
    AssistantMessage, _, ResultMessage, TextBlock, query_fn = sdk

    # Track text per assistant message so we can return only the final one
    all_messages_text: list[list[str]] = []
    current_msg_text: list[str] = []
    result_msg: Any = None

    async def _run() -> None:
        nonlocal result_msg, current_msg_text
        async for message in query_fn(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                current_msg_text = []
                for block in message.content:
                    if isinstance(block, TextBlock):
                        current_msg_text.append(block.text)
                if current_msg_text:
                    all_messages_text.append(current_msg_text)
            elif isinstance(message, ResultMessage):
                result_msg = message

    if timeout > 0:
        await asyncio.wait_for(_run(), timeout=float(timeout))
    else:
        await _run()

    # content = last assistant message only; full_text = everything
    last_text = all_messages_text[-1] if all_messages_text else []
    all_text = [t for parts in all_messages_text for t in parts]
    return _result_from_agent(model, last_text, all_text, result_msg)


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


# ---------------------------------------------------------------------------
# Claude structured output
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
    coro = _acall_agent_structured(
        model, messages, response_model, timeout=timeout, **kwargs,
    )
    return _run_sync(coro)


# ---------------------------------------------------------------------------
# Claude streaming
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
            self._model, self._text_parts, self._text_parts, self._result_msg,
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


# ===========================================================================
# Codex SDK
# ===========================================================================


def _create_codex_home(mcp_servers: dict[str, Any]) -> str:
    """Create a temp directory with .codex/config.toml containing only specified MCP servers.

    Symlinks auth.json and other credential files from the real ~/.codex/ so
    authentication continues to work.

    Args:
        mcp_servers: Dict of server_name -> {command, args?, cwd?, env?}.

    Returns:
        Path to the temp directory (acts as HOME for Codex CLI).
    """
    tmp_dir = tempfile.mkdtemp(prefix="codex_home_")
    config_dir = Path(tmp_dir) / ".codex"
    config_dir.mkdir()

    # Symlink auth and other essential files from real .codex dir
    real_codex = Path.home() / ".codex"
    if real_codex.is_dir():
        for fname in ("auth.json", "version.json", ".personality_migration"):
            src = real_codex / fname
            if src.exists():
                (config_dir / fname).symlink_to(src)

    # Write minimal config.toml with only specified MCP servers
    lines: list[str] = []
    for name, cfg in mcp_servers.items():
        lines.append(f'[mcp_servers."{name}"]')
        lines.append(f'command = "{cfg["command"]}"')
        if "args" in cfg:
            args_str = ", ".join(f'"{a}"' for a in cfg["args"])
            lines.append(f"args = [{args_str}]")
        if "cwd" in cfg:
            lines.append(f'cwd = "{cfg["cwd"]}"')
        if "env" in cfg:
            lines.append(f'[mcp_servers."{name}".env]')
            for k, v in cfg["env"].items():
                lines.append(f'{k} = "{v}"')
        lines.append("")

    (config_dir / "config.toml").write_text("\n".join(lines))
    return tmp_dir


def _prepare_codex_mcp(kwargs: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Process mcp_servers kwarg into codex_home.

    Returns:
        (updated_kwargs, tmp_dir_to_cleanup_or_None)
    """
    if "mcp_servers" not in kwargs:
        return kwargs, None
    if "codex_home" in kwargs:
        raise ValueError("Cannot specify both 'mcp_servers' and 'codex_home'")
    mcp_servers = kwargs.pop("mcp_servers")
    tmp_dir = _create_codex_home(mcp_servers)
    kwargs["codex_home"] = tmp_dir
    return kwargs, tmp_dir


def _cleanup_tmp(tmp_dir: str | None) -> None:
    """Remove a temporary codex home directory if it exists."""
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)


_codex_patched = False


def _patch_codex_buffer_limit() -> None:
    """Increase asyncio subprocess buffer from 64KB to 4MB in CodexExec.run.

    The Codex SDK uses asyncio.create_subprocess_exec with the default 64KB
    buffer limit. MCP tool results (large JSON payloads) easily exceed this,
    causing asyncio.LimitOverrunError ("Separator is not found, and chunk
    exceed the limit"). This patches the run method to use a 4MB limit.
    """
    global _codex_patched
    if _codex_patched:
        return
    try:
        from openai_codex_sdk.exec import CodexExec  # type: ignore[import-untyped]
        import asyncio as _aio
        import functools

        _original_run = CodexExec.run

        @functools.wraps(_original_run)
        async def _patched_run(self: Any, args: Any) -> Any:
            # Monkey-patch create_subprocess_exec to inject limit=4MB
            _orig_create = _aio.create_subprocess_exec

            async def _create_with_limit(*a: Any, **kw: Any) -> Any:
                kw.setdefault("limit", 4 * 1024 * 1024)
                return await _orig_create(*a, **kw)

            _aio.create_subprocess_exec = _create_with_limit  # type: ignore[assignment]
            try:
                async for line in _original_run(self, args):
                    yield line
            finally:
                _aio.create_subprocess_exec = _orig_create  # type: ignore[assignment]

        CodexExec.run = _patched_run  # type: ignore[assignment]
        _codex_patched = True
        logging.getLogger(__name__).debug("Patched CodexExec buffer limit to 4MB")
    except Exception:
        pass  # SDK not installed or API changed — skip silently


def _import_codex_sdk() -> tuple[Any, ...]:
    """Lazily import openai_codex_sdk components.

    Returns:
        (Codex, CodexOptions, ThreadOptions, TurnOptions, Turn, StreamedTurn,
         AgentMessageItem, ItemCompletedEvent, TurnCompletedEvent, Usage)
    """
    try:
        from openai_codex_sdk import (  # type: ignore[import-untyped]
            AgentMessageItem,
            Codex,
            ItemCompletedEvent,
            StreamedTurn,
            ThreadOptions,
            Turn,
            TurnCompletedEvent,
            TurnOptions,
            Usage,
        )
        from openai_codex_sdk.codex import CodexOptions  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "openai-codex-sdk is required for codex agent models. "
            "Install with: pip install llm_client[codex]"
        ) from None
    _patch_codex_buffer_limit()
    return (
        Codex, CodexOptions, ThreadOptions, TurnOptions, Turn, StreamedTurn,
        AgentMessageItem, ItemCompletedEvent, TurnCompletedEvent, Usage,
    )


def _build_codex_options(
    model: str,
    messages: list[dict[str, Any]],
    *,
    output_schema: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[str, Any, Any, Any, tuple[Any, ...]]:
    """Build Codex options from model string + kwargs.

    Returns:
        (prompt, codex_options, thread_options, turn_options, sdk_components)
    """
    _, underlying_model = _parse_agent_model(model)
    sdk = _import_codex_sdk()
    Codex, CodexOptions, ThreadOptions, TurnOptions = sdk[0], sdk[1], sdk[2], sdk[3]

    prompt, system_prompt = _messages_to_agent_prompt(messages)

    # Codex has no system_prompt param — prepend to prompt
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"

    # Separate recognized kwargs
    agent_kw: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _AGENT_KWARGS:
            agent_kw[k] = v
        else:
            logger.debug("Ignoring kwarg %r for codex model %s", k, model)

    # CodexOptions (connection-level)
    codex_kw: dict[str, Any] = {}
    if "api_key" in agent_kw:
        codex_kw["api_key"] = agent_kw["api_key"]
    if "base_url" in agent_kw:
        codex_kw["base_url"] = agent_kw["base_url"]
    if "codex_home" in agent_kw:
        # Override HOME so Codex CLI reads config from codex_home/.codex/config.toml
        env_dict = dict(os.environ)
        env_dict["HOME"] = str(agent_kw["codex_home"])
        codex_kw["env"] = env_dict
    codex_opts = CodexOptions(**codex_kw) if codex_kw else None

    # ThreadOptions (session-level)
    thread_kw: dict[str, Any] = {}
    if underlying_model is not None:
        thread_kw["model"] = underlying_model
    for key in (
        "sandbox_mode", "working_directory", "approval_policy",
        "model_reasoning_effort", "network_access_enabled", "web_search_enabled",
        "additional_directories", "skip_git_repo_check",
    ):
        if key in agent_kw:
            thread_kw[key] = agent_kw[key]
    # Defaults for safety
    thread_kw.setdefault("sandbox_mode", "workspace-write")
    thread_kw.setdefault("approval_policy", "never")
    thread_opts = ThreadOptions(**thread_kw)

    # TurnOptions (per-turn)
    turn_kw: dict[str, Any] = {}
    if output_schema is not None:
        turn_kw["output_schema"] = output_schema
    turn_opts = TurnOptions(**turn_kw) if turn_kw else None

    return prompt, codex_opts, thread_opts, turn_opts, sdk


def _estimate_codex_cost(model: str, usage: Any) -> float:
    """Estimate USD cost from Codex Usage via litellm. Approximate."""
    _, underlying = _parse_agent_model(model)
    lookup_model = underlying or "gpt-4o"  # best guess for bare "codex"
    try:
        import litellm
        cost = litellm.completion_cost(
            model=lookup_model,
            prompt_tokens=getattr(usage, "input_tokens", 0),
            completion_tokens=getattr(usage, "output_tokens", 0),
        )
        return float(cost)
    except Exception:
        return 0.0


def _result_from_codex(
    model: str,
    final_response: str,
    usage: Any,
    raw_turn: Any,
    is_error: bool = False,
) -> LLMCallResult:
    """Build LLMCallResult from Codex Turn output."""
    usage_dict: dict[str, Any] = {}
    if usage is not None:
        usage_dict = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "cached_input_tokens": getattr(usage, "cached_input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        }

    cost = _estimate_codex_cost(model, usage) if usage else 0.0

    return LLMCallResult(
        content=final_response,
        usage=usage_dict,
        cost=cost,
        model=model,
        finish_reason="error" if is_error else "stop",
        raw_response=raw_turn,
    )


async def _acall_codex(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> LLMCallResult:
    """Call Codex SDK and return an LLMCallResult."""
    kwargs, tmp_dir = _prepare_codex_mcp(kwargs)
    try:
        prompt, codex_opts, thread_opts, turn_opts, sdk = _build_codex_options(
            model, messages, **kwargs,
        )
        Codex = sdk[0]

        codex = Codex(options=codex_opts)
        thread = codex.start_thread(options=thread_opts)

        async def _run() -> Any:
            return await thread.run(prompt, turn_opts)

        if timeout > 0:
            turn = await asyncio.wait_for(_run(), timeout=float(timeout))
        else:
            turn = await _run()

        return _result_from_codex(model, turn.final_response, turn.usage, turn)
    finally:
        _cleanup_tmp(tmp_dir)


def _call_codex(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> LLMCallResult:
    """Sync wrapper for _acall_codex."""
    coro = _acall_codex(model, messages, timeout=timeout, **kwargs)
    return _run_sync(coro)


# ---------------------------------------------------------------------------
# Codex structured output
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    """Strip markdown code fences (safety net for JSON parsing)."""
    text = text.strip()
    text = re.sub(r"^```(?:json|python|xml|text)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text)
    return text.strip()


async def _acall_codex_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Call Codex SDK with structured output (JSON schema)."""
    kwargs, tmp_dir = _prepare_codex_mcp(kwargs)
    try:
        schema = response_model.model_json_schema()
        prompt, codex_opts, thread_opts, turn_opts, sdk = _build_codex_options(
            model, messages, output_schema=schema, **kwargs,
        )
        Codex = sdk[0]

        codex = Codex(options=codex_opts)
        thread = codex.start_thread(options=thread_opts)

        async def _run() -> Any:
            return await thread.run(prompt, turn_opts)

        if timeout > 0:
            turn = await asyncio.wait_for(_run(), timeout=float(timeout))
        else:
            turn = await _run()

        # Parse JSON from final_response
        raw_text = turn.final_response or ""
        if not raw_text.strip():
            raise ValueError("Empty response from Codex — no structured output")

        try:
            parsed_data = _json.loads(raw_text)
        except _json.JSONDecodeError:
            # Safety net: strip code fences and retry
            parsed_data = _json.loads(_strip_fences(raw_text))

        validated = response_model.model_validate(parsed_data)

        llm_result = _result_from_codex(model, turn.final_response, turn.usage, turn)
        llm_result.content = validated.model_dump_json()

        return validated, llm_result
    finally:
        _cleanup_tmp(tmp_dir)


def _call_codex_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Sync wrapper for _acall_codex_structured."""
    coro = _acall_codex_structured(
        model, messages, response_model, timeout=timeout, **kwargs,
    )
    return _run_sync(coro)


# ---------------------------------------------------------------------------
# Codex streaming
# ---------------------------------------------------------------------------


class AsyncCodexStream:
    """Async streaming wrapper for Codex SDK. Yields text from AgentMessageItem events."""

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
        self._usage: Any = None
        self._result: LLMCallResult | None = None
        self._queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._timeout = timeout
        self._messages = messages
        self._kwargs, self._tmp_dir = _prepare_codex_mcp(kwargs)

    async def _produce(self) -> None:
        """Run the Codex streamed turn and push text chunks to the queue."""
        try:
            prompt, codex_opts, thread_opts, turn_opts, sdk = _build_codex_options(
                self._model, self._messages, **self._kwargs,
            )
            Codex = sdk[0]
            AgentMessageItem = sdk[6]
            ItemCompletedEvent = sdk[7]
            TurnCompletedEvent = sdk[8]

            codex = Codex(options=codex_opts)
            thread = codex.start_thread(options=thread_opts)
            streamed_turn = await thread.run_streamed(prompt, turn_opts)

            async for event in streamed_turn.events:
                if isinstance(event, ItemCompletedEvent):
                    if isinstance(event.item, AgentMessageItem):
                        text = event.item.text
                        self._text_parts.append(text)
                        await self._queue.put(text)
                elif isinstance(event, TurnCompletedEvent):
                    self._usage = event.usage

            await self._queue.put(None)  # sentinel
        except Exception as e:
            await self._queue.put(e)

    async def _ensure_started(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._produce())

    def __aiter__(self) -> AsyncCodexStream:
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
        final_response = "\n".join(self._text_parts) if self._text_parts else ""
        self._result = _result_from_codex(
            self._model, final_response, self._usage, None,
        )
        if self._hooks and self._hooks.after_call:
            self._hooks.after_call(self._result)
        _cleanup_tmp(self._tmp_dir)

    @property
    def result(self) -> LLMCallResult:
        """The accumulated result. Available after the stream is fully consumed."""
        if self._result is None:
            raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result


class CodexStream:
    """Sync streaming wrapper for Codex SDK. Wraps AsyncCodexStream in a background thread."""

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
            astream = AsyncCodexStream(
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

    def __iter__(self) -> CodexStream:
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
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            if self._result is None:
                raise RuntimeError("Stream not yet consumed. Iterate first.")
        return self._result


async def _astream_codex(
    model: str,
    messages: list[dict[str, Any]],
    *,
    hooks: Hooks | None = None,
    timeout: int = 300,
    **kwargs: Any,
) -> AsyncCodexStream:
    """Create an async Codex stream."""
    if hooks and hooks.before_call:
        hooks.before_call(model, messages, kwargs)
    return AsyncCodexStream(model, messages, hooks=hooks, timeout=timeout, **kwargs)


def _stream_codex(
    model: str,
    messages: list[dict[str, Any]],
    *,
    hooks: Hooks | None = None,
    timeout: int = 300,
    **kwargs: Any,
) -> CodexStream:
    """Create a sync Codex stream."""
    if hooks and hooks.before_call:
        hooks.before_call(model, messages, kwargs)
    return CodexStream(model, messages, hooks=hooks, timeout=timeout, **kwargs)


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
    sdk_name, _ = _parse_agent_model(model)
    if sdk_name == "codex":
        return _call_codex(model, messages, timeout=timeout, **kwargs)
    if sdk_name == "claude-code":
        return _call_agent(model, messages, timeout=timeout, **kwargs)
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
    sdk_name, _ = _parse_agent_model(model)
    if sdk_name == "codex":
        return await _acall_codex(model, messages, timeout=timeout, **kwargs)
    if sdk_name == "claude-code":
        return await _acall_agent(model, messages, timeout=timeout, **kwargs)
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
