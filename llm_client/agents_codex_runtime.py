"""Codex SDK runtime execution helpers.

This module owns the concrete SDK execution paths for Codex text and
structured calls, including in-process execution, isolated-process worker
entrypoints, and the hard-timeout helper. ``agents_codex`` keeps the
orchestration logic and compatibility wrappers, while this module concentrates
the runtime implementation details that were making that file oversized.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import multiprocessing as _mp
import time
import traceback
from typing import Any, Callable, cast

from pydantic import BaseModel

from llm_client.agents_codex_process import (
    _codex_exec_diagnostics,
    _codex_timeout_message,
    _compact_json,
    _safe_error_text,
    _terminate_pid_tree,
)
from llm_client.client import LLMCallResult
from llm_client.data_types import TurnEvent
from llm_client.timeout_policy import normalize_timeout as _normalize_timeout

logger = logging.getLogger(__name__)


def _codex_mod() -> Any:
    """Lazy import ``agents_codex`` so runtime helpers can call shared adapters."""

    import llm_client.agents_codex as _codex

    return _codex


async def _await_codex_turn_with_hard_timeout(
    turn_coro: Any,
    *,
    timeout_s: int,
    cancel_grace_s: float = 2.0,
) -> tuple[Any, dict[str, Any]]:
    """Await one Codex turn with a hard timeout and bounded cancel grace."""

    start_mono = time.monotonic()
    turn_task = asyncio.create_task(turn_coro)
    done, _pending = await asyncio.wait(
        {turn_task},
        timeout=float(timeout_s),
        return_when=asyncio.FIRST_COMPLETED,
    )
    if turn_task in done:
        return await turn_task, {
            "elapsed_s": round(time.monotonic() - start_mono, 3),
            "timed_out": False,
        }

    turn_task.cancel()
    cancel_started = time.monotonic()
    done_after_cancel, _pending_after_cancel = await asyncio.wait(
        {turn_task},
        timeout=cancel_grace_s,
        return_when=asyncio.FIRST_COMPLETED,
    )
    cancel_completed = turn_task in done_after_cancel
    if not cancel_completed:
        logger.warning(
            "Codex turn cancellation exceeded grace window (%.1fs); proceeding with timeout",
            cancel_grace_s,
        )
    raise TimeoutError(
        _compact_json(
            {
                "timed_out": True,
                "timeout_s": int(timeout_s),
                "elapsed_s": round(time.monotonic() - start_mono, 3),
                "cancel_grace_s": round(cancel_grace_s, 3),
                "cancel_wait_s": round(time.monotonic() - cancel_started, 3),
                "cancel_completed": cancel_completed,
                "task_done": turn_task.done(),
            },
            max_chars=700,
        )
    )


async def _acall_codex_inproc(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int = 300,
    on_turn: Callable[[Any], None] | None = None,
    **kwargs: Any,
) -> LLMCallResult:
    """Execute one Codex text call in-process via the SDK."""

    timeout = _normalize_timeout(timeout, caller="_acall_codex_inproc", logger=logger)
    codex_mod = _codex_mod()
    kwargs, tmp_dir = codex_mod._prepare_codex_mcp(kwargs)
    try:
        prompt, codex_opts, thread_opts, turn_opts, sdk = codex_mod._build_codex_options(
            model, messages, **kwargs,
        )
        Codex = sdk[0]

        codex = Codex(options=codex_opts)
        thread = codex.start_thread(options=thread_opts)

        async def _run() -> Any:
            return await thread.run(prompt, turn_opts)

        run_started = time.monotonic()
        if timeout > 0:
            try:
                turn, _ = await _await_codex_turn_with_hard_timeout(
                    _run(),
                    timeout_s=int(timeout),
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as exc:
                timeout_diag: dict[str, Any] = {
                    "phase": "await_thread_run",
                    "elapsed_s": round(time.monotonic() - run_started, 3),
                }
                payload = _safe_error_text(exc).strip()
                if payload.startswith("{") and payload.endswith("}"):
                    try:
                        parsed = _json.loads(payload)
                        if isinstance(parsed, dict):
                            timeout_diag["hard_timeout"] = parsed
                    except Exception:
                        timeout_diag["hard_timeout_raw"] = payload[:300]
                else:
                    timeout_diag["hard_timeout_raw"] = payload[:300]
                exec_diag = _codex_exec_diagnostics(thread)
                if exec_diag:
                    timeout_diag["exec"] = exec_diag
                    hard = timeout_diag.get("hard_timeout")
                    if (
                        isinstance(hard, dict)
                        and hard.get("cancel_completed") is False
                        and isinstance(exec_diag.get("proc_pid"), int)
                        and int(exec_diag["proc_pid"]) > 0
                    ):
                        timeout_diag["forced_terminate"] = _terminate_pid_tree(int(exec_diag["proc_pid"]))
                logger.warning("Codex timeout diagnostics: %s", _compact_json(timeout_diag, max_chars=2500))
                raise TimeoutError(
                    _codex_timeout_message(
                        model=model,
                        timeout_s=timeout,
                        working_directory=getattr(thread_opts, "working_directory", None),
                        sandbox_mode=getattr(thread_opts, "sandbox_mode", None),
                        approval_policy=getattr(thread_opts, "approval_policy", None),
                        diagnostics=timeout_diag,
                        structured=False,
                    )
                ) from exc
        else:
            turn = await _run()

        final_response = (turn.final_response or "").strip()
        if not final_response:
            turn_count = getattr(turn, "num_turns", None)
            raise ValueError(
                "Empty response from Codex SDK"
                + (f" (num_turns={turn_count})" if turn_count is not None else "")
            )

        if on_turn is not None:
            on_turn(TurnEvent(
                turn=getattr(turn, "num_turns", 1) or 1,
                elapsed_s=round(time.monotonic() - run_started, 3),
                tool_calls=codex_mod._extract_codex_tool_calls(turn),
                text_preview=(turn.final_response or "")[:200],
            ))

        return codex_mod._result_from_codex(model, final_response, turn.usage, turn)
    finally:
        codex_mod._cleanup_tmp(tmp_dir)


def _codex_text_worker_entry(
    conn: Any,
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    kwargs: dict[str, Any],
) -> None:
    """Worker entrypoint for isolated-process Codex text calls."""

    try:
        local_kwargs = dict(kwargs)
        local_kwargs["codex_process_isolation"] = False
        result = _codex_mod()._agents_mod()._run_sync(
            _acall_codex_inproc(model, messages, timeout=timeout, **local_kwargs)
        )
        conn.send({"ok": True, "result": _codex_mod()._serialize_llm_result(result)})
    except BaseException as exc:
        conn.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": _safe_error_text(exc),
                "traceback": traceback.format_exc(limit=30),
            }
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _call_codex_in_isolated_process(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: int,
    kwargs: dict[str, Any],
) -> LLMCallResult:
    """Execute one Codex text call in a child process."""

    codex_mod = _codex_mod()
    start_method = codex_mod._codex_process_start_method(kwargs)
    grace_s = codex_mod._codex_process_grace_s(kwargs)
    ctx = _mp.get_context(start_method)
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process_factory = getattr(ctx, "Process")
    proc = process_factory(
        target=_codex_text_worker_entry,
        args=(send_conn, model, messages, int(timeout), dict(kwargs)),
        daemon=True,
    )
    proc.start()
    send_conn.close()

    wait_s = (float(timeout) if timeout > 0 else 3600.0) + grace_s
    if not recv_conn.poll(wait_s):
        forced: dict[str, Any] | None = None
        if proc.is_alive() and isinstance(proc.pid, int) and proc.pid > 0:
            forced = _terminate_pid_tree(int(proc.pid), grace_s=max(0.3, grace_s / 2))
        try:
            proc.join(timeout=max(0.5, grace_s))
        except Exception:
            pass
        timeout_diag = {
            "phase": "isolated_worker_wait",
            "start_method": start_method,
            "wait_s": round(wait_s, 3),
            "worker_pid": proc.pid,
            "worker_exitcode": proc.exitcode,
            "forced_terminate": forced,
        }
        raise TimeoutError(
            _codex_timeout_message(
                model=model,
                timeout_s=max(int(timeout), 0),
                working_directory=kwargs.get("working_directory"),
                sandbox_mode=kwargs.get("sandbox_mode"),
                approval_policy=kwargs.get("approval_policy"),
                diagnostics=timeout_diag,
                structured=False,
            )
        )

    payload: dict[str, Any]
    try:
        payload = cast(dict[str, Any], recv_conn.recv())
    except EOFError as exc:
        raise RuntimeError(
            f"CODEX_WORKER_EOF[start_method={start_method}, pid={proc.pid}, exitcode={proc.exitcode}]"
        ) from exc
    finally:
        try:
            recv_conn.close()
        except Exception:
            pass
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass
        if proc.is_alive() and isinstance(proc.pid, int) and proc.pid > 0:
            _terminate_pid_tree(int(proc.pid), grace_s=0.5)

    if payload.get("ok") is True:
        result_payload = cast(dict[str, Any], payload.get("result", {}))
        return codex_mod._deserialize_llm_result(result_payload)

    err_type = str(payload.get("error_type") or "RuntimeError")
    err_message = str(payload.get("error_message") or "Codex worker failed")
    err_trace = str(payload.get("traceback") or "")
    diagnostics = {
        "phase": "isolated_worker_result",
        "start_method": start_method,
        "worker_pid": proc.pid,
        "worker_exitcode": proc.exitcode,
        "worker_error_type": err_type,
    }
    if err_type in {"TimeoutError", "CancelledError"}:
        raise TimeoutError(
            _codex_timeout_message(
                model=model,
                timeout_s=max(int(timeout), 0),
                working_directory=kwargs.get("working_directory"),
                sandbox_mode=kwargs.get("sandbox_mode"),
                approval_policy=kwargs.get("approval_policy"),
                diagnostics=diagnostics,
                structured=False,
            )
            + f" worker_error={err_message}"
        )
    raise RuntimeError(
        "CODEX_WORKER_ERROR"
        f"[{err_type}, start_method={start_method}, pid={proc.pid}, exitcode={proc.exitcode}] "
        f"{err_message}"
        + (f"\n{err_trace}" if err_trace else "")
    )


def _strip_fences(text: str) -> str:
    """Strip markdown code fences as a safety net for JSON parsing."""

    import re

    text = text.strip()
    text = re.sub(r"^```(?:json|python|xml|text)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text)
    return text.strip()


async def _acall_codex_structured_inproc(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int = 300,
    **kwargs: Any,
) -> tuple[BaseModel, LLMCallResult]:
    """Execute one Codex structured call in-process via the SDK."""

    timeout = _normalize_timeout(timeout, caller="_acall_codex_structured_inproc", logger=logger)
    codex_mod = _codex_mod()
    kwargs, tmp_dir = codex_mod._prepare_codex_mcp(kwargs)
    try:
        schema = response_model.model_json_schema()
        prompt, codex_opts, thread_opts, turn_opts, sdk = codex_mod._build_codex_options(
            model, messages, output_schema=schema, **kwargs,
        )
        Codex = sdk[0]

        codex = Codex(options=codex_opts)
        thread = codex.start_thread(options=thread_opts)

        async def _run() -> Any:
            return await thread.run(prompt, turn_opts)

        run_started = time.monotonic()
        if timeout > 0:
            try:
                turn, _ = await _await_codex_turn_with_hard_timeout(
                    _run(),
                    timeout_s=int(timeout),
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as exc:
                timeout_diag: dict[str, Any] = {
                    "phase": "await_thread_run",
                    "elapsed_s": round(time.monotonic() - run_started, 3),
                }
                payload = _safe_error_text(exc).strip()
                if payload.startswith("{") and payload.endswith("}"):
                    try:
                        parsed = _json.loads(payload)
                        if isinstance(parsed, dict):
                            timeout_diag["hard_timeout"] = parsed
                    except Exception:
                        timeout_diag["hard_timeout_raw"] = payload[:300]
                else:
                    timeout_diag["hard_timeout_raw"] = payload[:300]
                exec_diag = _codex_exec_diagnostics(thread)
                if exec_diag:
                    timeout_diag["exec"] = exec_diag
                    hard = timeout_diag.get("hard_timeout")
                    if (
                        isinstance(hard, dict)
                        and hard.get("cancel_completed") is False
                        and isinstance(exec_diag.get("proc_pid"), int)
                        and int(exec_diag["proc_pid"]) > 0
                    ):
                        timeout_diag["forced_terminate"] = _terminate_pid_tree(int(exec_diag["proc_pid"]))
                logger.warning("Codex structured timeout diagnostics: %s", _compact_json(timeout_diag, max_chars=2500))
                raise TimeoutError(
                    _codex_timeout_message(
                        model=model,
                        timeout_s=timeout,
                        working_directory=getattr(thread_opts, "working_directory", None),
                        sandbox_mode=getattr(thread_opts, "sandbox_mode", None),
                        approval_policy=getattr(thread_opts, "approval_policy", None),
                        diagnostics=timeout_diag,
                        structured=True,
                    )
                ) from exc
        else:
            turn = await _run()

        raw_text = turn.final_response or ""
        if not raw_text.strip():
            raise ValueError("Empty response from Codex — no structured output")

        try:
            parsed_data = _json.loads(raw_text)
        except _json.JSONDecodeError:
            parsed_data = _json.loads(_strip_fences(raw_text))

        validated = response_model.model_validate(parsed_data)
        llm_result = codex_mod._result_from_codex(model, turn.final_response, turn.usage, turn)
        llm_result.content = validated.model_dump_json()
        return validated, llm_result
    finally:
        codex_mod._cleanup_tmp(tmp_dir)


def _codex_structured_worker_entry(
    conn: Any,
    model: str,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    timeout: int,
    kwargs: dict[str, Any],
) -> None:
    """Worker entrypoint for isolated-process Codex structured calls."""

    try:
        local_kwargs = dict(kwargs)
        local_kwargs["codex_process_isolation"] = False
        codex_mod = _codex_mod()
        kwargs2, tmp_dir = codex_mod._prepare_codex_mcp(local_kwargs)
        try:
            prompt, codex_opts, thread_opts, turn_opts, sdk = codex_mod._build_codex_options(
                model, messages, output_schema=schema, **kwargs2,
            )
            Codex = sdk[0]
            codex = Codex(options=codex_opts)
            thread = codex.start_thread(options=thread_opts)

            async def _run() -> Any:
                return await thread.run(prompt, turn_opts)

            if timeout > 0:
                turn, _ = codex_mod._agents_mod()._run_sync(
                    _await_codex_turn_with_hard_timeout(_run(), timeout_s=int(timeout))
                )
            else:
                turn = codex_mod._agents_mod()._run_sync(_run())

            raw_text = (turn.final_response or "").strip()
            if not raw_text:
                raise ValueError("Empty response from Codex — no structured output")
            llm_result = codex_mod._result_from_codex(model, turn.final_response, turn.usage, turn)
            conn.send(
                {
                    "ok": True,
                    "raw_text": raw_text,
                    "llm_result": codex_mod._serialize_llm_result(llm_result),
                }
            )
        finally:
            codex_mod._cleanup_tmp(tmp_dir)
    except BaseException as exc:
        conn.send(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": _safe_error_text(exc),
                "traceback": traceback.format_exc(limit=30),
            }
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _call_codex_structured_in_isolated_process(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    timeout: int,
    kwargs: dict[str, Any],
) -> tuple[BaseModel, LLMCallResult]:
    """Execute one Codex structured call in a child process."""

    codex_mod = _codex_mod()
    start_method = codex_mod._codex_process_start_method(kwargs)
    grace_s = codex_mod._codex_process_grace_s(kwargs)
    ctx = _mp.get_context(start_method)
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    schema = response_model.model_json_schema()
    process_factory = getattr(ctx, "Process")
    proc = process_factory(
        target=_codex_structured_worker_entry,
        args=(send_conn, model, messages, schema, int(timeout), dict(kwargs)),
        daemon=True,
    )
    proc.start()
    send_conn.close()

    wait_s = (float(timeout) if timeout > 0 else 3600.0) + grace_s
    if not recv_conn.poll(wait_s):
        forced: dict[str, Any] | None = None
        if proc.is_alive() and isinstance(proc.pid, int) and proc.pid > 0:
            forced = _terminate_pid_tree(int(proc.pid), grace_s=max(0.3, grace_s / 2))
        try:
            proc.join(timeout=max(0.5, grace_s))
        except Exception:
            pass
        timeout_diag = {
            "phase": "isolated_worker_wait",
            "start_method": start_method,
            "wait_s": round(wait_s, 3),
            "worker_pid": proc.pid,
            "worker_exitcode": proc.exitcode,
            "forced_terminate": forced,
        }
        raise TimeoutError(
            _codex_timeout_message(
                model=model,
                timeout_s=max(int(timeout), 0),
                working_directory=kwargs.get("working_directory"),
                sandbox_mode=kwargs.get("sandbox_mode"),
                approval_policy=kwargs.get("approval_policy"),
                diagnostics=timeout_diag,
                structured=True,
            )
        )

    payload: dict[str, Any]
    try:
        payload = cast(dict[str, Any], recv_conn.recv())
    except EOFError as exc:
        raise RuntimeError(
            f"CODEX_STRUCTURED_WORKER_EOF[start_method={start_method}, pid={proc.pid}, exitcode={proc.exitcode}]"
        ) from exc
    finally:
        try:
            recv_conn.close()
        except Exception:
            pass
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass
        if proc.is_alive() and isinstance(proc.pid, int) and proc.pid > 0:
            _terminate_pid_tree(int(proc.pid), grace_s=0.5)

    if payload.get("ok") is True:
        raw_text = str(payload.get("raw_text", ""))
        try:
            parsed_data = _json.loads(raw_text)
        except _json.JSONDecodeError:
            parsed_data = _json.loads(_strip_fences(raw_text))
        validated = response_model.model_validate(parsed_data)
        llm_payload = cast(dict[str, Any], payload.get("llm_result", {}))
        llm_result = codex_mod._deserialize_llm_result(llm_payload)
        llm_result.content = validated.model_dump_json()
        return validated, llm_result

    err_type = str(payload.get("error_type") or "RuntimeError")
    err_message = str(payload.get("error_message") or "Codex structured worker failed")
    diagnostics = {
        "phase": "isolated_worker_result",
        "start_method": start_method,
        "worker_pid": proc.pid,
        "worker_exitcode": proc.exitcode,
        "worker_error_type": err_type,
    }
    if err_type in {"TimeoutError", "CancelledError"}:
        raise TimeoutError(
            _codex_timeout_message(
                model=model,
                timeout_s=max(int(timeout), 0),
                working_directory=kwargs.get("working_directory"),
                sandbox_mode=kwargs.get("sandbox_mode"),
                approval_policy=kwargs.get("approval_policy"),
                diagnostics=diagnostics,
                structured=True,
            )
            + f" worker_error={err_message}"
        )
    raise RuntimeError(
        "CODEX_STRUCTURED_WORKER_ERROR"
        f"[{err_type}, start_method={start_method}, pid={proc.pid}, exitcode={proc.exitcode}] "
        f"{err_message}"
    )
