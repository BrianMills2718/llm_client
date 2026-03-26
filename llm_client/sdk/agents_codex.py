"""Codex SDK adapter for llm_client.

Handles all Codex agent interactions: SDK and CLI transports, process isolation,
structured output, streaming, MCP server configuration, and timeout enforcement.
Converts Codex SDK types into the unified LLMCallResult interface.

This module is an implementation detail of agents.py. All public names are
re-exported from agents.py for backward compatibility.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import multiprocessing as _mp
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import BaseModel

from llm_client.sdk.agents_codex_process import (
    _collect_process_tree_snapshot,
    _codex_exec_diagnostics,
    _codex_timeout_message,
    _compact_json,
    _process_exists,
    _safe_error_text,
    _safe_line_preview,
    _terminate_pid_tree,
)
from llm_client.core.client import Hooks, LLMCallResult
from llm_client.core.data_types import TurnEvent
from llm_client.execution.timeout_policy import normalize_timeout as _normalize_timeout

logger = logging.getLogger(__name__)

# Env-var constants duplicated from agents.py to avoid circular import.
_CODEX_PROCESS_ISOLATION_ENV = "LLM_CLIENT_CODEX_PROCESS_ISOLATION"
_CODEX_PROCESS_START_METHOD_ENV = "LLM_CLIENT_CODEX_PROCESS_START_METHOD"
_CODEX_PROCESS_GRACE_ENV = "LLM_CLIENT_CODEX_PROCESS_GRACE_S"
_CODEX_ALLOW_MINIMAL_EFFORT_ENV = "LLM_CLIENT_CODEX_ALLOW_MINIMAL_EFFORT"
_CODEX_TRANSPORT_ENV = "LLM_CLIENT_CODEX_TRANSPORT"


def _agents_mod() -> Any:
    """Lazy import of llm_client.agents to break circular dependency."""
    import llm_client.sdk.agents as _m
    return _m


def _codex_process_isolation_enabled(kwargs: dict[str, Any]) -> bool:
    """Resolve Codex process isolation policy.

    Explicit kwarg wins; otherwise use env default.
    """
    if "codex_process_isolation" in kwargs:
        return _as_bool(kwargs.get("codex_process_isolation"), default=False)
    return _as_bool(os.environ.get(_CODEX_PROCESS_ISOLATION_ENV), default=False)


def _codex_process_start_method(kwargs: dict[str, Any]) -> str:
    available = set(_mp.get_all_start_methods())
    requested = str(
        kwargs.get("codex_process_start_method")
        or os.environ.get(_CODEX_PROCESS_START_METHOD_ENV, "")
    ).strip().lower()
    if requested in available:
        return requested
    if "fork" in available:
        return "fork"
    return "spawn"


def _codex_process_grace_s(kwargs: dict[str, Any]) -> float:
    raw = kwargs.get("codex_process_grace_s", os.environ.get(_CODEX_PROCESS_GRACE_ENV, 3.0))
    try:
        return max(0.5, float(raw))
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid codex_process_grace_s=%r, using default 3.0: %s", raw, exc)
        return 3.0


def _codex_transport(kwargs: dict[str, Any]) -> str:
    """Resolve Codex transport selection."""

    raw = str(
        kwargs.get("codex_transport")
        or os.environ.get(_CODEX_TRANSPORT_ENV, "sdk")
    ).strip().lower()
    if raw in {"sdk", "cli", "auto"}:
        return raw
    return "sdk"


def _codex_cli_path(kwargs: dict[str, Any]) -> str:
    """Resolve the Codex CLI executable."""

    raw = str(kwargs.get("codex_cli_path") or "codex").strip()
    return raw or "codex"


def _agent_hard_timeout(kwargs: dict[str, Any], default_timeout: int) -> int:
    """Resolve a transport-level hard timeout independent of provider request timeout policy."""

    raw = kwargs.get("agent_hard_timeout")
    if raw is None:
        return max(0, int(default_timeout))
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return max(0, int(default_timeout))
    return max(0, parsed)



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


def _as_bool(value: Any, *, default: bool = False) -> bool:
    """Parse a value as boolean with common string coercions."""
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
        from openai_codex_sdk.exec import CodexExec
        import asyncio as _aio
        import functools

        _original_run = CodexExec.run

        @functools.wraps(_original_run)
        async def _patched_run(self: Any, args: Any) -> Any:
            start_mono = time.monotonic()
            run_diag: dict[str, Any] = {
                "started_s": 0.0,
                "lines_seen": 0,
                "first_line_s": None,
                "last_line_s": None,
                "line_tail": [],
                "proc_pid": None,
                "proc_argv": None,
                "proc_started_s": None,
                "exception_type": None,
                "exception": None,
                "ended_s": None,
            }
            setattr(self, "_llmc_last_run_diag", run_diag)
            # Monkey-patch create_subprocess_exec to inject limit=4MB
            _orig_create = _aio.create_subprocess_exec

            async def _create_with_limit(*a: Any, **kw: Any) -> Any:
                desired_limit = 4 * 1024 * 1024
                current_limit = kw.get("limit")
                if not isinstance(current_limit, int) or current_limit < desired_limit:
                    kw["limit"] = desired_limit
                proc = await _orig_create(*a, **kw)
                run_diag["proc_pid"] = int(getattr(proc, "pid", 0) or 0) or None
                run_diag["proc_argv"] = [str(x) for x in a[:8]]
                run_diag["proc_started_s"] = round(time.monotonic() - start_mono, 3)
                return proc

            _aio.create_subprocess_exec = cast(Any, _create_with_limit)
            try:
                async for line in _original_run(self, args):
                    now_s = round(time.monotonic() - start_mono, 3)
                    run_diag["lines_seen"] = int(run_diag.get("lines_seen", 0) or 0) + 1
                    if run_diag["first_line_s"] is None:
                        run_diag["first_line_s"] = now_s
                    run_diag["last_line_s"] = now_s
                    line_tail = cast(list[str], run_diag.setdefault("line_tail", []))
                    line_tail.append(_safe_line_preview(line, max_chars=220))
                    if len(line_tail) > 8:
                        del line_tail[:-8]
                    yield line
            except BaseException as exc:
                run_diag["exception_type"] = type(exc).__name__
                run_diag["exception"] = _safe_error_text(exc)
                raise
            finally:
                run_diag["ended_s"] = round(time.monotonic() - start_mono, 3)
                _aio.create_subprocess_exec = cast(Any, _orig_create)

        setattr(CodexExec, "run", _patched_run)
        _codex_patched = True
        logging.getLogger(__name__).debug("Patched CodexExec buffer limit to 4MB")
    except Exception:
        logging.getLogger(__name__).debug("Codex buffer patch skipped (SDK not installed or API changed)", exc_info=True)


def _import_codex_sdk() -> tuple[Any, ...]:
    """Lazily import openai_codex_sdk components.

    Returns:
        (Codex, CodexOptions, ThreadOptions, TurnOptions, Turn, StreamedTurn,
         AgentMessageItem, ItemCompletedEvent, TurnCompletedEvent, Usage)
    """
    try:
        from openai_codex_sdk import (
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
        from openai_codex_sdk.codex import CodexOptions  # type: ignore[attr-defined]
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
    _, underlying_model = _agents_mod()._parse_agent_model(model)
    sdk = _import_codex_sdk()
    Codex, CodexOptions, ThreadOptions, TurnOptions = sdk[0], sdk[1], sdk[2], sdk[3]

    prompt, system_prompt = _agents_mod()._messages_to_agent_prompt(messages)

    # Codex has no system_prompt param — prepend to prompt
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"

    # Separate recognized kwargs
    agent_kw: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _agents_mod()._AGENT_KWARGS:
            agent_kw[k] = v
        else:
            logger.debug("Ignoring kwarg %r for codex model %s", k, model)
    agent_kw = _agents_mod()._apply_agent_yolo_defaults("codex", agent_kw)

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
    thread_kw["model_reasoning_effort"] = _agents_mod()._normalize_codex_reasoning_effort(
        thread_kw.get("model_reasoning_effort")
    )
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
    _, underlying = _agents_mod()._parse_agent_model(model)
    lookup_model = underlying or "gpt-4o"  # best guess for bare "codex"
    try:
        import litellm
        cost = litellm.completion_cost(
            model=lookup_model,
            prompt_tokens=getattr(usage, "input_tokens", 0),
            completion_tokens=getattr(usage, "output_tokens", 0),
        )
        return float(cost)
    except Exception as exc:
        logger.warning("Codex cost extraction failed (returning 0.0): %s", exc)
        return 0.0


def _extract_codex_tool_calls(raw_turn: Any) -> list[dict[str, Any]]:
    """Extract MCP tool-call records from Codex Turn.items into OpenAI-like shape."""
    items = getattr(raw_turn, "items", None)
    if not isinstance(items, list):
        return []

    out: list[dict[str, Any]] = []
    for item in items:
        item_type = str(getattr(item, "type", "") or "").strip().lower()
        if item_type != "mcp_tool_call":
            continue
        tool_name = str(getattr(item, "tool", "") or "")
        if not tool_name:
            continue
        arguments = getattr(item, "arguments", None)
        if not isinstance(arguments, dict):
            arguments = {}
        status = str(getattr(item, "status", "") or "")
        error = getattr(item, "error", None)
        result_obj = getattr(item, "result", None)
        result_preview = _safe_line_preview(result_obj, max_chars=500) if result_obj is not None else ""
        out.append(
            {
                "id": str(getattr(item, "id", "") or ""),
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                },
                "server": str(getattr(item, "server", "") or ""),
                "status": status,
                "result_preview": result_preview,
                "is_error": bool(error) or status.lower() in {"failed", "error"},
                "error": _safe_line_preview(error, max_chars=300) if error else "",
            }
        )
    return out


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

    billing_mode = _agents_mod()._agent_billing_mode()
    if billing_mode == "api":
        cost = _estimate_codex_cost(model, usage) if usage else 0.0
        cost_source = "estimated_from_usage"
        effective_billing_mode = "api_metered"
    else:
        # Codex ChatGPT subscription/OAuth mode should not map usage tokens to API USD.
        cost = 0.0
        cost_source = "subscription_included"
        effective_billing_mode = "subscription_included"
    tool_calls = _extract_codex_tool_calls(raw_turn)

    return LLMCallResult(
        content=final_response,
        usage=usage_dict,
        cost=cost,
        model=model,
        tool_calls=tool_calls,
        finish_reason="error" if is_error else "stop",
        raw_response=raw_turn,
        cost_source=cost_source,
        billing_mode=effective_billing_mode,
    )


def _serialize_llm_result(result: LLMCallResult) -> dict[str, Any]:
    """Serialize LLMCallResult for transfer across process boundaries."""
    raw_summary: dict[str, Any] | None = None
    raw = result.raw_response
    if raw is not None:
        raw_summary = {"type": type(raw).__name__}
        for key in ("num_turns",):
            if hasattr(raw, key):
                raw_summary[key] = getattr(raw, key)

    return {
        "content": result.content,
        "usage": result.usage,
        "cost": result.cost,
        "model": result.model,
        "requested_model": result.requested_model,
        "resolved_model": result.resolved_model,
        "execution_model": result.execution_model,
        "routing_trace": result.routing_trace,
        "tool_calls": result.tool_calls,
        "finish_reason": result.finish_reason,
        "warnings": result.warnings,
        "warning_records": result.warning_records,
        "full_text": result.full_text,
        "cost_source": result.cost_source,
        "billing_mode": result.billing_mode,
        "marginal_cost": result.marginal_cost,
        "cache_hit": result.cache_hit,
        "raw_response_summary": raw_summary,
    }


def _deserialize_llm_result(payload: dict[str, Any]) -> LLMCallResult:
    """Deserialize LLMCallResult from a process-safe payload."""
    return LLMCallResult(
        content=str(payload.get("content", "")),
        usage=cast(dict[str, Any], payload.get("usage", {})),
        cost=float(payload.get("cost", 0.0) or 0.0),
        model=str(payload.get("model", "")),
        requested_model=cast(str | None, payload.get("requested_model")),
        resolved_model=cast(str | None, payload.get("resolved_model")),
        execution_model=cast(str | None, payload.get("execution_model")),
        routing_trace=cast(dict[str, Any] | None, payload.get("routing_trace")),
        tool_calls=cast(list[dict[str, Any]], payload.get("tool_calls", [])),
        finish_reason=str(payload.get("finish_reason", "")),
        raw_response=payload.get("raw_response_summary"),
        warnings=cast(list[str], payload.get("warnings", [])),
        warning_records=cast(list[dict[str, Any]], payload.get("warning_records", [])),
        full_text=cast(str | None, payload.get("full_text")),
        cost_source=str(payload.get("cost_source", "unspecified")),
        billing_mode=str(payload.get("billing_mode", "unknown")),
        marginal_cost=cast(float | None, payload.get("marginal_cost")),
        cache_hit=bool(payload.get("cache_hit", False)),
    )


def _build_codex_cli_command(
    model: str,
    prompt: str,
    *,
    output_schema: dict[str, Any] | None,
    kwargs: dict[str, Any],
    output_path: str,
    schema_path: str | None,
) -> tuple[list[str], dict[str, str], str]:
    """Build a direct `codex exec` command for one prompt."""

    kwargs = _agents_mod()._apply_agent_yolo_defaults("codex", dict(kwargs))
    _, underlying_model = _agents_mod()._parse_agent_model(model)
    cli_path = _codex_cli_path(kwargs)
    working_directory = str(kwargs.get("working_directory") or os.getcwd())
    sandbox_mode = str(kwargs.get("sandbox_mode") or "workspace-write")
    approval_policy = str(kwargs.get("approval_policy") or "never")
    reasoning_effort = _agents_mod()._normalize_codex_reasoning_effort(
        kwargs.get("model_reasoning_effort")
    )
    command = [
        cli_path,
        "exec",
        "--color",
        "never",
        "-C",
        working_directory,
        "-s",
        sandbox_mode,
        "-o",
        output_path,
    ]
    if approval_policy == "never":
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.extend(["-a", approval_policy])
    if kwargs.get("skip_git_repo_check"):
        command.append("--skip-git-repo-check")
    if underlying_model:
        command.extend(["--model", underlying_model])
    if reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    for add_dir in kwargs.get("additional_directories", []) or []:
        command.extend(["--add-dir", str(add_dir)])
    if schema_path is not None and output_schema is not None:
        command.extend(["--output-schema", schema_path])
    command.append("-")

    env = dict(os.environ)
    codex_home = kwargs.get("codex_home")
    if codex_home:
        env["HOME"] = str(codex_home)
    return command, env, prompt


def _result_from_codex_cli(
    model: str,
    final_response: str,
    *,
    transport: str,
    warning: str | None = None,
) -> LLMCallResult:
    """Build an `LLMCallResult` from direct Codex CLI output."""

    result = LLMCallResult(
        content=final_response,
        usage={},
        cost=0.0,
        model=model,
        finish_reason="stop",
        raw_response={"transport": transport},
        cost_source="subscription_included",
        billing_mode="subscription_included",
    )
    if warning:
        result.warnings.append(warning)
    return result


def _call_codex_via_cli(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    output_schema: dict[str, Any] | None = None,
    fallback_warning: str | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Run Codex through the local CLI with enforced subprocess timeouts."""

    prompt, system_prompt = _agents_mod()._messages_to_agent_prompt(messages)
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    hard_timeout = _agent_hard_timeout(kwargs, timeout)
    with tempfile.TemporaryDirectory(prefix="llm_client_codex_cli_") as tmp_dir:
        output_path = str(Path(tmp_dir) / "last_message.txt")
        schema_path: str | None = None
        if output_schema is not None:
            schema_path = str(Path(tmp_dir) / "output_schema.json")
            Path(schema_path).write_text(_json.dumps(output_schema))
        command, env, stdin_payload = _build_codex_cli_command(
            model,
            prompt,
            output_schema=output_schema,
            kwargs=kwargs,
            output_path=output_path,
            schema_path=schema_path,
        )
        try:
            completed = subprocess.run(
                command,
                input=stdin_payload,
                text=True,
                capture_output=True,
                check=False,
                timeout=(float(hard_timeout) if hard_timeout > 0 else None),
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            diagnostics = {
                "phase": "codex_cli_exec",
                "command": command[:10],
                "timeout_s": hard_timeout,
                "stdout_tail": _safe_line_preview(exc.stdout, max_chars=400),
                "stderr_tail": _safe_line_preview(exc.stderr, max_chars=400),
            }
            raise TimeoutError(
                _codex_timeout_message(
                    model=model,
                    timeout_s=max(hard_timeout, 0),
                    working_directory=kwargs.get("working_directory"),
                    sandbox_mode=kwargs.get("sandbox_mode"),
                    approval_policy=kwargs.get("approval_policy"),
                    diagnostics=diagnostics,
                    structured=output_schema is not None,
                )
            ) from exc

        if completed.returncode != 0:
            diagnostics = {
                "phase": "codex_cli_exec",
                "returncode": completed.returncode,
                "stdout_tail": _safe_line_preview(completed.stdout, max_chars=500),
                "stderr_tail": _safe_line_preview(completed.stderr, max_chars=500),
            }
            raise RuntimeError(
                "CODEX_CLI_ERROR "
                f"(model={model}, returncode={completed.returncode}) "
                f"diagnostics={_compact_json(diagnostics)}"
            )

        final_response = Path(output_path).read_text() if Path(output_path).exists() else ""
        final_response = final_response.strip()
        if not final_response:
            raise ValueError("Empty response from Codex CLI")
        return _result_from_codex_cli(
            model,
            final_response,
            transport="codex_cli",
            warning=fallback_warning,
        )


async def _acall_codex_via_cli(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    output_schema: dict[str, Any] | None = None,
    fallback_warning: str | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Async wrapper for direct Codex CLI execution."""

    return await asyncio.to_thread(
        _call_codex_via_cli,
        model,
        messages,
        timeout=timeout,
        output_schema=output_schema,
        fallback_warning=fallback_warning,
        **kwargs,
    )


def _runtime_mod() -> Any:
    """Lazy import of ``agents_codex_runtime`` to keep wrappers monkeypatchable."""

    import llm_client.sdk.agents_codex_runtime as _runtime

    return _runtime


async def _await_codex_turn_with_hard_timeout(
    turn_coro: Any,
    *,
    timeout_s: int,
    cancel_grace_s: float = 2.0,
) -> tuple[Any, dict[str, Any]]:
    """Compatibility wrapper for the extracted Codex hard-timeout helper."""

    return await _runtime_mod()._await_codex_turn_with_hard_timeout(
        turn_coro,
        timeout_s=timeout_s,
        cancel_grace_s=cancel_grace_s,
    )


async def _acall_codex_inproc(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    on_turn: Callable[[Any], None] | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Compatibility wrapper for extracted in-process Codex text execution."""

    return await _runtime_mod()._acall_codex_inproc(
        model,
        messages,
        timeout=timeout,
        on_turn=on_turn,
        **kwargs,
    )


def _codex_text_worker_entry(
    conn: Any,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    kwargs: dict[str, Any],
) -> None:
    """Compatibility wrapper for the isolated-process text worker entrypoint."""

    _runtime_mod()._codex_text_worker_entry(conn, model, messages, timeout, kwargs)


def _call_codex_in_isolated_process(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    kwargs: dict[str, Any],
) -> LLMCallResult:
    """Compatibility wrapper for extracted isolated-process text execution."""

    return _runtime_mod()._call_codex_in_isolated_process(
        model,
        messages,
        timeout=timeout,
        kwargs=kwargs,
    )


async def _acall_codex(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    on_turn: Callable[[Any], None] | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Call Codex SDK and return an LLMCallResult."""
    timeout = _normalize_timeout(timeout, caller="_acall_codex", logger=logger)
    transport = _codex_transport(kwargs)
    hard_timeout = _agent_hard_timeout(kwargs, timeout)
    if transport == "cli":
        return await _acall_codex_via_cli(model, messages, timeout=timeout, **kwargs)
    if transport == "auto" and timeout <= 0 and hard_timeout > 0:
        warning = (
            "CODEX_TRANSPORT_AUTO[sdk->cli]: provider timeout unavailable; "
            f"using CLI transport with agent_hard_timeout={hard_timeout}s"
        )
        logger.warning(warning)
        return await _acall_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            fallback_warning=warning,
            **kwargs,
        )
    if _codex_process_isolation_enabled(kwargs):
        async def _sdk_call() -> LLMCallResult:
            return await asyncio.to_thread(
                _call_codex_in_isolated_process,
                model,
                messages,
                timeout=timeout,
                kwargs=dict(kwargs),
            )
    else:
        async def _sdk_call() -> LLMCallResult:
            return await _acall_codex_inproc(model, messages, timeout=timeout, on_turn=on_turn, **kwargs)

    if transport != "auto":
        return await _sdk_call()
    return await _agents_mod()._acall_codex_auto_fallback(
        _sdk_call,
        lambda warning: _acall_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            fallback_warning=warning,
            **kwargs,
        ),
    )


def _call_codex(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    on_turn: Callable[[Any], None] | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Sync wrapper for _acall_codex."""
    timeout = _normalize_timeout(timeout, caller="_call_codex", logger=logger)
    transport = _codex_transport(kwargs)
    hard_timeout = _agent_hard_timeout(kwargs, timeout)
    if transport == "cli":
        return _call_codex_via_cli(model, messages, timeout=timeout, **kwargs)
    if transport == "auto" and timeout <= 0 and hard_timeout > 0:
        warning = (
            "CODEX_TRANSPORT_AUTO[sdk->cli]: provider timeout unavailable; "
            f"using CLI transport with agent_hard_timeout={hard_timeout}s"
        )
        logger.warning(warning)
        return _call_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            fallback_warning=warning,
            **kwargs,
        )
    if _codex_process_isolation_enabled(kwargs):
        def _sdk_call() -> LLMCallResult:
            return _call_codex_in_isolated_process(
                model,
                messages,
                timeout=timeout,
                kwargs=dict(kwargs),
            )
    else:
        def _sdk_call() -> LLMCallResult:
            return cast(LLMCallResult, _agents_mod()._run_sync(_acall_codex_inproc(model, messages, timeout=timeout, on_turn=on_turn, **kwargs)))

    if transport != "auto":
        return _sdk_call()
    return _agents_mod()._call_codex_auto_fallback(
        _sdk_call,
        lambda warning: _call_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            fallback_warning=warning,
            **kwargs,
        ),
    )


# ---------------------------------------------------------------------------
# Codex structured output
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    """Compatibility wrapper for extracted fenced-JSON cleanup."""

    return _runtime_mod()._strip_fences(text)


async def _acall_codex_structured_inproc(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Compatibility wrapper for extracted in-process structured Codex execution."""

    return await _runtime_mod()._acall_codex_structured_inproc(
        model,
        messages,
        response_model,
        timeout=timeout,
        **kwargs,
    )


def _codex_structured_worker_entry(
    conn: Any,
    model: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    timeout: int,
    kwargs: dict[str, Any],
) -> None:
    """Compatibility wrapper for the isolated-process structured worker entrypoint."""

    _runtime_mod()._codex_structured_worker_entry(conn, model, messages, schema, timeout, kwargs)


def _call_codex_structured_in_isolated_process(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int,
    kwargs: dict[str, Any],
) -> tuple[BaseModel, LLMCallResult]:
    """Compatibility wrapper for extracted isolated-process structured execution."""

    return _runtime_mod()._call_codex_structured_in_isolated_process(
        model,
        messages,
        response_model,
        timeout=timeout,
        kwargs=kwargs,
    )


async def _acall_codex_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Call Codex SDK with structured output (JSON schema)."""
    timeout = _normalize_timeout(timeout, caller="_acall_codex_structured", logger=logger)
    transport = _codex_transport(kwargs)
    hard_timeout = _agent_hard_timeout(kwargs, timeout)
    if transport == "cli":
        llm_result = await _acall_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            output_schema=response_model.model_json_schema(),
            **kwargs,
        )
        parsed = response_model.model_validate_json(llm_result.content)
        llm_result.content = parsed.model_dump_json()
        return parsed, llm_result
    if transport == "auto" and timeout <= 0 and hard_timeout > 0:
        warning = (
            "CODEX_TRANSPORT_AUTO[sdk->cli]: provider timeout unavailable; "
            f"using CLI transport with agent_hard_timeout={hard_timeout}s"
        )
        logger.warning(warning)
        llm_result = await _acall_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            output_schema=response_model.model_json_schema(),
            fallback_warning=warning,
            **kwargs,
        )
        parsed = response_model.model_validate_json(llm_result.content)
        llm_result.content = parsed.model_dump_json()
        return parsed, llm_result

    if _codex_process_isolation_enabled(kwargs):
        async def _sdk_call() -> tuple[BaseModel, LLMCallResult]:
            return await asyncio.to_thread(
                _call_codex_structured_in_isolated_process,
                model,
                messages,
                response_model,
                timeout=timeout,
                kwargs=dict(kwargs),
            )
    else:
        async def _sdk_call() -> tuple[BaseModel, LLMCallResult]:
            return await _acall_codex_structured_inproc(
                model,
                messages,
                response_model,
                timeout=timeout,
                **kwargs,
            )

    if transport != "auto":
        return await _sdk_call()
    async def _cli_from_warning(warning: str) -> tuple[BaseModel, LLMCallResult]:
        llm_result = await _acall_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            output_schema=response_model.model_json_schema(),
            fallback_warning=warning,
            **kwargs,
        )
        parsed = response_model.model_validate_json(llm_result.content)
        llm_result.content = parsed.model_dump_json()
        return parsed, llm_result
    return await _agents_mod()._acall_codex_auto_fallback(_sdk_call, _cli_from_warning)


def _call_codex_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Sync wrapper for _acall_codex_structured."""
    timeout = _normalize_timeout(timeout, caller="_call_codex_structured", logger=logger)
    transport = _codex_transport(kwargs)
    hard_timeout = _agent_hard_timeout(kwargs, timeout)
    if transport == "cli":
        llm_result = _call_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            output_schema=response_model.model_json_schema(),
            **kwargs,
        )
        parsed = response_model.model_validate_json(llm_result.content)
        llm_result.content = parsed.model_dump_json()
        return parsed, llm_result
    if transport == "auto" and timeout <= 0 and hard_timeout > 0:
        warning = (
            "CODEX_TRANSPORT_AUTO[sdk->cli]: provider timeout unavailable; "
            f"using CLI transport with agent_hard_timeout={hard_timeout}s"
        )
        logger.warning(warning)
        llm_result = _call_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            output_schema=response_model.model_json_schema(),
            fallback_warning=warning,
            **kwargs,
        )
        parsed = response_model.model_validate_json(llm_result.content)
        llm_result.content = parsed.model_dump_json()
        return parsed, llm_result
    if transport != "auto":
        return cast(
            tuple[BaseModel, LLMCallResult],
            _agents_mod()._run_sync(
                _acall_codex_structured(
                    model,
                    messages,
                    response_model,
                    timeout=timeout,
                    **kwargs,
                )
            ),
        )
    def _sdk_call() -> tuple[BaseModel, LLMCallResult]:
        return cast(
            tuple[BaseModel, LLMCallResult],
            _agents_mod()._run_sync(
                _acall_codex_structured(
                    model,
                    messages,
                    response_model,
                    timeout=timeout,
                    **kwargs,
                )
            ),
        )

    def _cli_from_warning(warning: str) -> tuple[BaseModel, LLMCallResult]:
        llm_result = _call_codex_via_cli(
            model,
            messages,
            timeout=timeout,
            output_schema=response_model.model_json_schema(),
            fallback_warning=warning,
            **kwargs,
        )
        parsed = response_model.model_validate_json(llm_result.content)
        llm_result.content = parsed.model_dump_json()
        return parsed, llm_result
    return _agents_mod()._call_codex_auto_fallback(_sdk_call, _cli_from_warning)


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
        self._task_handle: asyncio.Task[None] | None = None
        self._timeout = timeout
        self._messages = messages
        self._log_task = kwargs.pop("task", None)
        self._trace_id = kwargs.pop("trace_id", None)
        self._t0 = time.monotonic()
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
        if self._task_handle is None:
            self._task_handle = asyncio.create_task(self._produce())

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
        import llm_client.io_log as _io_log
        _io_log.log_call(
            model=self._model, messages=self._messages, result=self._result,
            latency_s=time.monotonic() - self._t0,
            caller="astream_codex", task=self._log_task, trace_id=self._trace_id,
        )
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
