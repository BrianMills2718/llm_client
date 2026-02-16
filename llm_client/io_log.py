"""Persistent I/O logging for LLM calls.

Appends one JSONL record per LLM call to:
    {DATA_ROOT}/{PROJECT}/{PROJECT}_llm_client_data/calls.jsonl

Configured via env vars (library convention — llm_client already auto-loads
from ~/.secrets/api_keys.env):

    LLM_CLIENT_LOG_ENABLED  — "1" (default) or "0" to disable
    LLM_CLIENT_DATA_ROOT    — base dir (default: ~/projects/data)
    LLM_CLIENT_PROJECT      — project name (default: basename(os.getcwd()))

Or override at runtime via configure().
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_enabled: bool = os.environ.get("LLM_CLIENT_LOG_ENABLED", "1") == "1"
_data_root: Path = Path(os.environ.get("LLM_CLIENT_DATA_ROOT", str(Path.home() / "projects" / "data")))
_project: str | None = os.environ.get("LLM_CLIENT_PROJECT")


def _get_project() -> str:
    """Get project name, lazily resolving cwd if not configured."""
    if _project is not None:
        return _project
    return Path.cwd().name


def _log_dir() -> Path:
    return _data_root / _get_project() / f"{_get_project()}_llm_client_data"


def configure(
    *,
    enabled: bool | None = None,
    data_root: str | Path | None = None,
    project: str | None = None,
) -> None:
    """Override logging config at runtime."""
    global _enabled, _data_root, _project
    if enabled is not None:
        _enabled = enabled
    if data_root is not None:
        _data_root = Path(data_root)
    if project is not None:
        _project = project


def log_call(
    *,
    model: str,
    messages: list[dict[str, Any]] | None = None,
    result: Any = None,
    error: Exception | None = None,
    latency_s: float | None = None,
    caller: str = "call_llm",
    task: str | None = None,
) -> None:
    """Append one JSONL record. Never raises — logging must not break calls."""
    if not _enabled:
        return
    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)

        # Extract fields from result if available
        response_content = None
        usage = None
        cost = None
        finish_reason = None
        if result is not None:
            response_content = getattr(result, "content", None)
            if isinstance(response_content, str) and len(response_content) > 5000:
                response_content = response_content[:5000] + f"...[truncated {len(result.content)} chars]"
            usage = getattr(result, "usage", None)
            cost = getattr(result, "cost", None)
            finish_reason = getattr(result, "finish_reason", None)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "messages": _truncate_messages(messages),
            "response": response_content,
            "usage": usage,
            "cost": cost,
            "finish_reason": finish_reason,
            "latency_s": round(latency_s, 3) if latency_s is not None else None,
            "error": str(error) if error else None,
            "caller": caller,
            "task": task,
        }
        with open(d / "calls.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Never break LLM calls for logging
        logger.debug("io_log.log_call failed", exc_info=True)


def _truncate_messages(
    messages: list[dict[str, Any]] | None,
    max_content: int = 2000,
) -> list[dict[str, Any]] | None:
    """Truncate message content for storage."""
    if messages is None:
        return None
    out: list[dict[str, Any]] = []
    for m in messages:
        m2 = dict(m)
        content = m2.get("content")
        if isinstance(content, str) and len(content) > max_content:
            m2["content"] = content[:max_content] + f"...[truncated {len(content)} chars]"
        out.append(m2)
    return out
