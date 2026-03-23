"""Codex process-diagnostics and forced-termination helpers.

This module owns the Codex-specific process snapshot, timeout message, and
best-effort termination helpers that support isolated-process execution.
Keeping them separate from ``agents_codex`` makes the main adapter focus on
transport orchestration while this module handles OS-level diagnostics and
kill-path behavior.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any, cast


def _safe_error_text(exc: BaseException) -> str:
    """Extract stable error text, falling back to the exception type name."""

    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__


def _safe_line_preview(value: Any, *, max_chars: int = 240) -> str:
    """Render one compact single-line preview for Codex subprocess diagnostics."""

    import json as _json

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
    """Render one compact JSON diagnostic string with bounded length."""

    import json as _json

    try:
        rendered = _json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        rendered = str(payload)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars] + "...(truncated)"


def _collect_process_tree_snapshot(root_pid: int, *, max_nodes: int = 20) -> list[dict[str, Any]]:
    """Collect a best-effort process-tree snapshot rooted at one pid."""

    if root_pid <= 0:
        return []
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,stat=,etime=,pcpu=,pmem=,command="],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
        )
    except Exception:
        return []

    nodes: dict[int, dict[str, Any]] = {}
    children: dict[int, list[int]] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except Exception:
            continue
        rec = {
            "pid": pid,
            "ppid": ppid,
            "stat": parts[2],
            "etime": parts[3],
            "pcpu": parts[4],
            "pmem": parts[5],
            "command": parts[6][:220],
        }
        nodes[pid] = rec
        children.setdefault(ppid, []).append(pid)

    if root_pid not in nodes:
        return []

    out_nodes: list[dict[str, Any]] = []
    q: list[int] = [root_pid]
    seen: set[int] = set()
    while q and len(out_nodes) < max_nodes:
        pid = q.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        node = nodes.get(pid)
        if node is None:
            continue
        out_nodes.append(node)
        q.extend(children.get(pid, []))
    return out_nodes


def _process_exists(pid: int) -> bool:
    """Return whether one process id currently exists."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _terminate_pid_tree(root_pid: int, *, grace_s: float = 0.8) -> dict[str, Any]:
    """Best-effort terminate one process tree, preferring children first."""

    snapshot = _collect_process_tree_snapshot(root_pid, max_nodes=64)
    pids = [int(n["pid"]) for n in snapshot if isinstance(n.get("pid"), int)]
    if root_pid not in pids:
        pids.append(root_pid)
    pids = list(dict.fromkeys(reversed(pids)))

    result: dict[str, Any] = {
        "root_pid": root_pid,
        "target_pids": pids,
        "term_sent": [],
        "kill_sent": [],
        "alive_after": [],
    }
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            cast(list[int], result["term_sent"]).append(pid)
        except Exception:
            pass

    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        alive = [pid for pid in pids if _process_exists(pid)]
        if not alive:
            break
        time.sleep(0.05)

    alive_after_term = [pid for pid in pids if _process_exists(pid)]
    for pid in alive_after_term:
        try:
            os.kill(pid, signal.SIGKILL)
            cast(list[int], result["kill_sent"]).append(pid)
        except Exception:
            pass

    result["alive_after"] = [pid for pid in pids if _process_exists(pid)]
    return result


def _codex_timeout_message(
    *,
    model: str,
    timeout_s: int,
    working_directory: Any,
    sandbox_mode: Any,
    approval_policy: Any,
    diagnostics: dict[str, Any] | None,
    structured: bool,
) -> str:
    """Build the shared timeout message for Codex text or structured calls."""

    call_kind = "codex_structured" if structured else "codex_call"
    wd = str(working_directory or "<unset>")
    sandbox = str(sandbox_mode or "<unset>")
    approval = str(approval_policy or "<unset>")
    message = (
        f"CODEX_TIMEOUT[{call_kind}] after {int(timeout_s)}s "
        f"(model={model}, working_directory={wd}, sandbox_mode={sandbox}, "
        f"approval_policy={approval})"
    )
    if diagnostics:
        message += f" diagnostics={_compact_json(diagnostics)}"
    return message


def _codex_exec_diagnostics(thread: Any) -> dict[str, Any]:
    """Collect Codex exec diagnostics plus a process-tree snapshot when possible."""

    exec_obj = getattr(thread, "_exec", None)
    if exec_obj is None:
        return {}
    raw = getattr(exec_obj, "_llmc_last_run_diag", None)
    if not isinstance(raw, dict):
        return {}
    diag = dict(raw)
    pid = diag.get("proc_pid")
    if isinstance(pid, int) and pid > 0:
        diag["process_tree"] = _collect_process_tree_snapshot(pid)
    return diag
