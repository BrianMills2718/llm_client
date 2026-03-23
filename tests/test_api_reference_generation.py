"""Regression tests for the generated browser API reference."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "scripts" / "meta" / "generate_api_reference.py"


def load_generator_module():
    """Load the generator script as a module for direct unit testing."""
    spec = importlib.util.spec_from_file_location("generate_api_reference", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generator module from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generator_discovers_public_package_modules() -> None:
    """The generator should cover the package graph, not a hand-picked slice."""
    generator = load_generator_module()

    module_names = generator.discover_module_names("llm_client")

    assert len(module_names) > 20
    assert "llm_client" in module_names
    assert "llm_client.client" in module_names
    assert "llm_client.io_log" in module_names
    assert "llm_client.models" in module_names


def test_generator_emits_docstrings_and_signatures() -> None:
    """The generated docs should reflect real code docstrings and typed signatures."""
    generator = load_generator_module()

    markdown_text, html_text = generator.generate_documents("llm_client")

    assert "Browser view:" in markdown_text
    assert "Generated from package docstrings and typed signatures." in markdown_text
    assert "llm_client.client" in html_text
    assert "Call any LLM. Routes by model string" in html_text
    assert "Persistent I/O logging for LLM calls and embeddings." in html_text
    assert "render_prompt" in html_text
    assert "call_llm" in html_text


def test_generator_check_detects_drift() -> None:
    """The generator's check mode should fail when checked-in docs drift."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        html_path = tmp_path / "API_REFERENCE.html"
        markdown_path = tmp_path / "API_REFERENCE.md"

        write_proc = subprocess.run(
            [
                sys.executable,
                str(GENERATOR_PATH),
                "--write",
                "--html-path",
                str(html_path),
                "--markdown-path",
                str(markdown_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )
        assert write_proc.returncode == 0, write_proc.stdout + write_proc.stderr

        html_path.write_text(html_path.read_text(encoding="utf-8") + "\n<!-- drift -->\n", encoding="utf-8")

        check_proc = subprocess.run(
            [
                sys.executable,
                str(GENERATOR_PATH),
                "--check",
                "--html-path",
                str(html_path),
                "--markdown-path",
                str(markdown_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )

        assert check_proc.returncode == 1, check_proc.stdout + check_proc.stderr
        assert "stale html output" in check_proc.stdout
