from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "meta" / "validate_relationships.py"


def _run_validator(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )


def test_default_relationships_config_is_valid() -> None:
    proc = _run_validator("--strict")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "validation passed" in proc.stdout.lower()


def test_validator_rejects_missing_doc(tmp_path: Path) -> None:
    cfg = tmp_path / "relationships.yaml"
    cfg.write_text(
        "\n".join(
            [
                "required_reading:",
                "  defaults:",
                "    - CLAUDE.md",
                "couplings:",
                "  - sources:",
                "      - llm_client/client.py",
                "    docs:",
                "      - docs/adr/DOES_NOT_EXIST.md",
                "    description: test coupling",
                "    required_reading: true",
            ]
        ),
        encoding="utf-8",
    )

    proc = _run_validator("--strict", "--config", str(cfg))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "does not exist" in proc.stdout
