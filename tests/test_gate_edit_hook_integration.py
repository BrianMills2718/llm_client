from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SCRIPT = REPO_ROOT / ".claude" / "hooks" / "gate-edit.sh"


def _run_gate_hook(
    *,
    file_path: str,
    mode: str | None = None,
    reads: list[str] | None = None,
    tool_name: str = "Edit",
) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        reads_file = Path(tmp.name)
        if reads:
            tmp.write("\n".join(reads) + "\n")

    try:
        env = dict(os.environ)
        env["LLM_CLIENT_READS_FILE"] = str(reads_file)
        if mode is not None:
            env["LLM_CLIENT_READ_GATE_MODE"] = mode

        payload = json.dumps({
            "tool_name": tool_name,
            "tool_input": {"file_path": file_path},
        })
        return subprocess.run(
            ["bash", str(HOOK_SCRIPT)],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
            env=env,
        )
    finally:
        reads_file.unlink(missing_ok=True)


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is required by gate-edit.sh")
def test_gate_edit_hook_blocks_missing_reads_in_strict_mode() -> None:
    proc = _run_gate_hook(file_path="llm_client/client.py", mode="strict", reads=[])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert '"decision": "block"' in proc.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is required by gate-edit.sh")
def test_gate_edit_hook_warn_mode_allows_with_context_output() -> None:
    proc = _run_gate_hook(file_path="llm_client/client.py", mode="warn", reads=[])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '"hookSpecificOutput"' in proc.stdout
    assert "Read gate warning for" in proc.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq is required by gate-edit.sh")
def test_gate_edit_hook_skips_non_source_files() -> None:
    proc = _run_gate_hook(
        file_path="docs/adr/0001-model-identity-v0.md",
        mode="strict",
        reads=[],
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == ""
