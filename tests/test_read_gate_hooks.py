"""Integration-style tests for llm_client Claude read-gating hooks.

These tests exercise the real shell hooks via subprocess so the repo validates
the same path Claude Code uses at runtime. They prove that the existing custom
gate still blocks/permits edits correctly after the governed-repo alignment,
and that hook decisions are now observable through JSONL logging.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_GATE_HOOK = REPO_ROOT / ".claude" / "hooks" / "gate-edit.sh"
LOCAL_TRACK_HOOK = REPO_ROOT / ".claude" / "hooks" / "track-reads.sh"
FILE_CONTEXT_SCRIPT = REPO_ROOT / "scripts" / "meta" / "file_context.py"
LINK_CHECKER = REPO_ROOT / "scripts" / "check_markdown_links.py"
TARGET_FILE = "llm_client/agents.py"
REQUIRED_DOCS = [
    "CLAUDE.md",
    "docs/adr/0005-reason-code-registry-governance.md",
    "docs/adr/0006-actor-id-issuance-policy.md",
    "docs/adr/0010-cross-project-runtime-substrate.md",
]


def _run_hook(
    script: Path,
    payload: dict[str, object],
    tmp_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute one hook with isolated reads/log files."""

    env = os.environ.copy()
    env["CLAUDE_SESSION_READS_FILE"] = str(tmp_path / "session_reads.txt")
    env["CLAUDE_HOOK_LOG_FILE"] = str(tmp_path / "hook_log.jsonl")
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(script)],
        cwd=str(REPO_ROOT),
        env=env,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )


def _read_log_entries(tmp_path: Path) -> list[dict[str, object]]:
    """Return parsed JSONL hook-log entries for one isolated test run."""

    log_file = tmp_path / "hook_log.jsonl"
    if not log_file.exists():
        return []
    return [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_file_context_includes_default_and_coupled_required_reads() -> None:
    """The local file-context resolver should include defaults and coupled ADR docs."""

    result = subprocess.run(
        [sys.executable, str(FILE_CONTEXT_SCRIPT), TARGET_FILE, "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["files"][0]["required_reads"] == REQUIRED_DOCS


def test_gate_edit_blocks_and_logs_missing_required_reads(tmp_path: Path) -> None:
    """The hook should block governed edits until the required docs were read."""

    result = _run_hook(
        LOCAL_GATE_HOOK,
        {
            "tool_name": "Edit",
            "tool_input": {"file_path": TARGET_FILE},
        },
        tmp_path,
    )

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["decision"] == "block"
    assert "blocked edit" in payload["reason"].lower()

    entries = _read_log_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["hook"] == "gate-edit"
    assert entries[0]["decision"] == "block"
    assert entries[0]["file_path"] == TARGET_FILE
    assert entries[0]["required_reads"] == REQUIRED_DOCS
    assert entries[0]["reads_completed"] == []
    assert entries[0]["reads_file"] == str((tmp_path / "session_reads.txt").resolve())
    assert entries[0]["context_emitted"] is False
    assert entries[0]["context_bytes"] == 0


def test_gate_edit_allows_and_logs_after_required_reads(tmp_path: Path) -> None:
    """The hook should allow once required docs are present in the reads file."""

    reads_file = tmp_path / "session_reads.txt"
    reads_file.write_text("\n".join(REQUIRED_DOCS) + "\n", encoding="utf-8")

    result = _run_hook(
        LOCAL_GATE_HOOK,
        {
            "tool_name": "Edit",
            "tool_input": {"file_path": TARGET_FILE},
        },
        tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout == ""

    entries = _read_log_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["hook"] == "gate-edit"
    assert entries[0]["decision"] == "allow"
    assert entries[0]["required_reads"] == REQUIRED_DOCS
    assert entries[0]["reads_completed"] == REQUIRED_DOCS
    assert entries[0]["missing_reads"] == []
    assert entries[0]["reads_file"] == str((tmp_path / "session_reads.txt").resolve())
    assert entries[0]["context_emitted"] is False
    assert entries[0]["context_bytes"] == 0


def test_track_reads_records_session_file_and_log(tmp_path: Path) -> None:
    """The read hook should append the read path and emit one log entry."""

    result = _run_hook(
        LOCAL_TRACK_HOOK,
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "CLAUDE.md"},
        },
        tmp_path,
        extra_env={
            "CLAUDE_HOOK_EXPERIMENT_ID": "ctx-exp-1",
            "CLAUDE_HOOK_VARIANT_ID": "rich-context",
            "CLAUDE_HOOK_DOWNSTREAM_RUN_ID": "run_ctx_eval_1",
        },
    )

    assert result.returncode == 0
    reads_file = tmp_path / "session_reads.txt"
    assert reads_file.read_text(encoding="utf-8").splitlines() == ["CLAUDE.md"]

    entries = _read_log_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0]["hook"] == "track-reads"
    assert entries[0]["decision"] == "recorded"
    assert entries[0]["file_path"] == "CLAUDE.md"
    assert entries[0]["reads_file"] == str((tmp_path / "session_reads.txt").resolve())
    assert entries[0]["experiment_id"] == "ctx-exp-1"
    assert entries[0]["variant_id"] == "rich-context"
    assert entries[0]["downstream_run_id"] == "run_ctx_eval_1"


def test_markdown_link_checker_passes_for_governance_entrypoints() -> None:
    """The local markdown-link checker should pass on the main governance docs."""

    result = subprocess.run(
        [
            sys.executable,
            str(LINK_CHECKER),
            "CLAUDE.md",
            "docs/plans/CLAUDE.md",
            "scripts/CLAUDE.md",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
