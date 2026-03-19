from __future__ import annotations

from pathlib import Path

from llm_client.model_policy_audit import scan_paths


def test_scan_paths_flags_direct_literal_call(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    source = project / "service.py"
    source.write_text(
        'from llm_client import call_llm\n'
        'call_llm("openrouter/openai/gpt-5-mini", messages)\n',
        encoding="utf-8",
    )

    violations = scan_paths([project])

    assert len(violations) == 1
    assert violations[0].kind == "direct_call_literal"


def test_scan_paths_allows_override_fields(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    config = project / "config.yaml"
    config.write_text(
        'selection_task: graph_building\n'
        'fallback_model: "openrouter/openai/gpt-5-mini"\n',
        encoding="utf-8",
    )

    violations = scan_paths([project])

    assert violations == []


def test_scan_paths_allows_inline_bypass_comment(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    source = project / "service.py"
    source.write_text(
        'MODEL = "openrouter/openai/gpt-5-mini"  # model-policy: allow-raw-model\n',
        encoding="utf-8",
    )

    violations = scan_paths([project])

    assert violations == []
