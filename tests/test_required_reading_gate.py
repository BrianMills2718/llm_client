from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "meta" / "check_required_reading.py"
RELATIONSHIPS_FILE = REPO_ROOT / "scripts" / "relationships.yaml"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("check_required_reading", CHECK_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_docs_for(target_file: str) -> list[str]:
    gate = _load_gate_module()
    relationships = gate._load_yaml_dict(RELATIONSHIPS_FILE)
    required_docs, _ = gate._required_docs_for_target(target_file, relationships)
    return required_docs


def _run_gate(
    *,
    target_file: str,
    mode: str | None = None,
    reads: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
        reads_file = Path(tmp.name)
        if reads:
            tmp.write("\n".join(reads) + "\n")

    try:
        env = dict(os.environ)
        if mode is not None:
            env["LLM_CLIENT_READ_GATE_MODE"] = mode
        return subprocess.run(
            [sys.executable, str(CHECK_SCRIPT), target_file, "--reads-file", str(reads_file)],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
            env=env,
        )
    finally:
        reads_file.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("mode", "expected_rc", "expected_fragment"),
    [
        ("strict", 1, "blocked edit"),
        ("warn", 0, "warning for"),
        ("off", 0, ""),
    ],
)
def test_required_reading_gate_modes(mode: str, expected_rc: int, expected_fragment: str) -> None:
    proc = _run_gate(target_file="llm_client/core/client.py", mode=mode, reads=[])
    assert proc.returncode == expected_rc, proc.stdout + proc.stderr
    if expected_fragment:
        assert expected_fragment in proc.stdout


def test_uncoupled_file_defaults_to_strict_mode() -> None:
    proc = _run_gate(target_file="llm_client/errors.py", reads=[])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "blocked edit" in proc.stdout


def test_client_coupling_includes_background_polling_adr() -> None:
    assert "docs/adr/0009-long-thinking-background-polling.md" in _required_docs_for(
        "llm_client/core/client.py"
    )


def test_coupled_file_passes_when_required_docs_are_read() -> None:
    proc = _run_gate(
        target_file="llm_client/core/client.py",
        reads=_required_docs_for("llm_client/core/client.py"),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
