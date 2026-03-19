from __future__ import annotations

import argparse
import json

import pytest

from llm_client.cli.tool_lint import cmd_tool_lint, register_parser


def test_cmd_tool_lint_json_success(tmp_path, capsys) -> None:
    module_path = tmp_path / "demo_tools.py"
    module_path.write_text(
        """
def search_entities(query: str, limit: int = 5) -> str:
    \"\"\"Search entities.\"\"\"
    return query

search_entities.__tool_input_examples__ = [{"query": "alan turing", "limit": 3}]

TOOLS = [search_entities]
TOOL_CONTRACTS = {
    "search_entities": {
        "produces": [{"kind": "ENTITY_SET", "ref_type": "name"}],
    }
}
""",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        module=str(module_path),
        tool=[],
        tool_list_var="TOOLS",
        tool_map_var=None,
        contracts_var="TOOL_CONTRACTS",
        allow_missing_examples=False,
        allow_missing_contracts=False,
        format="json",
        warn_only=False,
        fail_on_warning=False,
        error_exit_code=2,
        warning_exit_code=3,
    )

    cmd_tool_lint(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_tools"] == 1
    assert payload["n_errors"] == 0
    assert payload["n_warnings"] == 0


def test_cmd_tool_lint_fail_on_warning(tmp_path, capsys) -> None:
    module_path = tmp_path / "demo_tools_warn.py"
    module_path.write_text(
        """
def search_entities(query: str, limit: int = 5) -> str:
    \"\"\"Search entities.\"\"\"
    return query

TOOLS = [search_entities]
""",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        module=str(module_path),
        tool=[],
        tool_list_var="TOOLS",
        tool_map_var=None,
        contracts_var=None,
        allow_missing_examples=False,
        allow_missing_contracts=False,
        format="json",
        warn_only=False,
        fail_on_warning=True,
        error_exit_code=2,
        warning_exit_code=7,
    )

    with pytest.raises(SystemExit) as excinfo:
        cmd_tool_lint(args)
    assert int(excinfo.value.code) == 7
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_errors"] == 0
    assert payload["n_warnings"] >= 1


def test_tool_lint_register_parser_sets_handler() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parser(subparsers)
    args = parser.parse_args(["tool-lint", "--module", "example.module"])
    assert args.command == "tool-lint"
    assert callable(args.handler)
    assert args.error_exit_code == 2
    assert args.warning_exit_code == 3
