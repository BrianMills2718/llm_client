"""Tests for llm_client subtree instruction files."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "llm_client"


def test_package_subtree_docs_exist() -> None:
    """The selected package subdirs should each have a local instruction file."""

    for relpath in [
        "CLAUDE.md",
        "cli/CLAUDE.md",
        "observability/CLAUDE.md",
        "prompts/CLAUDE.md",
        "rubrics/CLAUDE.md",
    ]:
        assert (PACKAGE_ROOT / relpath).exists()


def test_agents_mirror_claude_symlinks() -> None:
    """Nested AGENTS.md files should mirror their CLAUDE.md counterparts."""

    for relpath in [
        "",
        "cli",
        "observability",
        "prompts",
        "rubrics",
    ]:
        agents_path = PACKAGE_ROOT / relpath / "AGENTS.md"
        claude_path = PACKAGE_ROOT / relpath / "CLAUDE.md"
        assert agents_path.is_symlink()
        assert agents_path.resolve() == claude_path.resolve()


def test_package_root_routes_to_local_surfaces() -> None:
    """The package-root router should point agents to the local subtrees."""

    text = (PACKAGE_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "cli/" in text
    assert "observability/" in text
    assert "prompts/" in text
    assert "rubrics/" in text
