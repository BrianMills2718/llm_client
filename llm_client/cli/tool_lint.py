"""CLI command for direct-tool registry linting."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


def _load_module(module_ref: str) -> types.ModuleType:
    candidate = Path(module_ref)
    if candidate.exists():
        spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
        if spec is None or spec.loader is None:
            raise ValueError(f"Unable to load module from path: {module_ref}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_ref)


def _collect_tool_callables(module: types.ModuleType, args: argparse.Namespace) -> list[Any]:
    selected: list[Any] = []

    for name in args.tool or []:
        if not hasattr(module, name):
            raise ValueError(f"Module {module.__name__} has no attribute {name!r}")
        selected.append(getattr(module, name))

    if args.tool_list_var:
        if not hasattr(module, args.tool_list_var):
            raise ValueError(f"Module {module.__name__} has no attribute {args.tool_list_var!r}")
        values = getattr(module, args.tool_list_var)
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{args.tool_list_var!r} must be a list/tuple of callables")
        selected.extend(values)

    if args.tool_map_var:
        if not hasattr(module, args.tool_map_var):
            raise ValueError(f"Module {module.__name__} has no attribute {args.tool_map_var!r}")
        values = getattr(module, args.tool_map_var)
        if not isinstance(values, dict):
            raise ValueError(f"{args.tool_map_var!r} must be a dict of name->callable")
        selected.extend(values.values())

    deduped: list[Any] = []
    seen_ids: set[int] = set()
    for item in selected:
        if not callable(item):
            raise ValueError(f"Selected tool object is not callable: {item!r}")
        marker = id(item)
        if marker in seen_ids:
            continue
        seen_ids.add(marker)
        deduped.append(item)

    if not deduped:
        raise ValueError(
            "No tools selected. Use --tool, --tool-list-var, or --tool-map-var to select callables."
        )
    return deduped


def _load_contracts(module: types.ModuleType, var_name: str | None) -> dict[str, Any]:
    if not var_name:
        return {}
    if not hasattr(module, var_name):
        raise ValueError(f"Module {module.__name__} has no attribute {var_name!r}")
    value = getattr(module, var_name)
    if not isinstance(value, dict):
        raise ValueError(f"{var_name!r} must be a dict of tool_name->contract")
    return value


def cmd_tool_lint(args: argparse.Namespace) -> None:
    from llm_client.tool_utils import lint_tool_registry

    try:
        module = _load_module(args.module)
        tools = _collect_tool_callables(module, args)
        contracts = _load_contracts(module, args.contracts_var)
        report = lint_tool_registry(
            tools,
            tool_contracts=contracts,
            require_examples_for_nontrivial=not args.allow_missing_examples,
            require_contract_for_nontrivial=not args.allow_missing_contracts,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print("\nTool Lint:")
        print("─" * 72)
        print(f"Module:      {args.module}")
        print(f"Tools:       {report.get('n_tools', 0)}")
        print(f"Errors:      {report.get('n_errors', 0)}")
        print(f"Warnings:    {report.get('n_warnings', 0)}")
        findings = report.get("findings") or []
        if findings:
            print("\nFindings:")
            for finding in findings:
                print(
                    f"  [{finding.get('severity', '?')}] "
                    f"{finding.get('tool_name', '?')} "
                    f"{finding.get('code', '?')}: {finding.get('message', '')}"
                )

    n_errors = int(report.get("n_errors", 0))
    n_warnings = int(report.get("n_warnings", 0))
    if n_errors > 0 and not args.warn_only:
        sys.exit(args.error_exit_code)
    if n_errors == 0 and n_warnings > 0 and args.fail_on_warning and not args.warn_only:
        sys.exit(args.warning_exit_code)


def register_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "tool-lint",
        help="Lint direct Python tool registries for descriptions/examples/contracts",
    )
    parser.add_argument(
        "--module",
        required=True,
        help="Import path or filesystem path to the Python module containing tool callables",
    )
    parser.add_argument(
        "--tool",
        action="append",
        default=[],
        help="Specific callable attribute name to lint (repeatable)",
    )
    parser.add_argument(
        "--tool-list-var",
        help="Module attribute name containing a list/tuple of tool callables",
    )
    parser.add_argument(
        "--tool-map-var",
        help="Module attribute name containing a dict of name->callable",
    )
    parser.add_argument(
        "--contracts-var",
        help="Module attribute name containing dict tool_name->contract",
    )
    parser.add_argument(
        "--allow-missing-examples",
        action="store_true",
        help="Do not warn when nontrivial tools lack input examples",
    )
    parser.add_argument(
        "--allow-missing-contracts",
        action="store_true",
        help="Do not warn when nontrivial tools lack declarative contracts",
    )
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Never exit non-zero for findings; print/report only",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit non-zero on warnings when there are no errors",
    )
    parser.add_argument(
        "--error-exit-code",
        type=int,
        default=2,
        help="Exit code used when lint errors are present",
    )
    parser.add_argument(
        "--warning-exit-code",
        type=int,
        default=3,
        help="Exit code used when --fail-on-warning trips on warnings",
    )
    parser.set_defaults(handler=cmd_tool_lint)
