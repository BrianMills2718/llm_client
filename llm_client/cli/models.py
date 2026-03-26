"""CLI commands for inspecting the model registry.

Read-only commands for listing models, showing details, and viewing task
profiles with their preferred model selections.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from llm_client.core.models import (
    ModelInfo,
    TaskProfile,
    _load_config,
    _load_registry_models,
    _load_task_profile,
    _sort_by_prefer,
)


def _format_context(ctx: int) -> str:
    """Format context window size compactly (e.g. 128K, 1M, 2M)."""
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M"
    if ctx >= 1_000:
        return f"{ctx // 1_000}K"
    return str(ctx)


def _yes_no(val: bool) -> str:
    """Format a boolean as a compact yes/no marker."""
    return "yes" if val else "no"


def _is_available(model: ModelInfo) -> bool:
    """Check if a model's API key env var is set."""
    return bool(os.environ.get(model.api_key_env))


def cmd_list(args: argparse.Namespace) -> None:
    """List all registered models in a compact table sorted by intelligence."""
    config = _load_config()
    models = _load_registry_models(config)

    if not args.all:
        models = [m for m in models if _is_available(m)]

    # Sort by intelligence descending
    models.sort(key=lambda m: m.intelligence, reverse=True)

    if not models:
        print("No models found.", file=sys.stderr)
        sys.exit(1)

    headers = ["Name", "Provider", "Intel", "Speed", "Cost", "Context", "Struct", "Tools"]
    rows: list[tuple[str, ...]] = []
    for m in models:
        rows.append((
            m.name,
            m.provider,
            str(m.intelligence),
            str(m.speed),
            f"${m.cost:.2f}",
            _format_context(m.context),
            _yes_no(m.structured_output),
            _yes_no(m.tool_calling),
        ))

    col_widths = [max(len(h), 4) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("─" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        print(fmt.format(*row))

    suffix = "" if args.all else "  (use --all to include unavailable models)"
    print(f"\n{len(rows)} models{suffix}")


def cmd_show(args: argparse.Namespace) -> None:
    """Show full details for a specific model."""
    config = _load_config()
    models = _load_registry_models(config)

    match = [m for m in models if m.name == args.name or m.litellm_id == args.name]
    if not match:
        names = sorted(m.name for m in models)
        print(f"Unknown model: {args.name!r}", file=sys.stderr)
        print(f"Available: {', '.join(names)}", file=sys.stderr)
        sys.exit(1)

    m = match[0]
    available = _is_available(m)
    lines = [
        f"Name:              {m.name}",
        f"LiteLLM ID:        {m.litellm_id}",
        f"Provider:          {m.provider}",
        f"API key env:       {m.api_key_env}  ({'set' if available else 'NOT SET'})",
        f"Intelligence:      {m.intelligence}",
        f"Speed:             {m.speed}",
        f"Cost:              ${m.cost:.3f}/1M tokens",
        f"Context:           {_format_context(m.context)} tokens",
        f"Structured output: {_yes_no(m.structured_output)}",
        f"Tool calling:      {_yes_no(m.tool_calling)}",
        f"Tags:              {', '.join(m.tags) if m.tags else '-'}",
    ]
    print("\n".join(lines))


def cmd_tasks(args: argparse.Namespace) -> None:
    """Show task profiles with their top preferred models."""
    config = _load_config()
    tasks_cfg = config["tasks"]
    models = _load_registry_models(config)

    task_names = sorted(tasks_cfg.keys())

    for task_name in task_names:
        profile = _load_task_profile(config, task_name)

        # Find qualifying models (available_only=False to show all candidates)
        candidates = [
            m for m in models
            if _qualifies(m, profile)
        ]
        candidates = _sort_by_prefer(candidates, profile.prefer)
        top3 = candidates[:3]

        # Format requirements
        reqs: list[str] = []
        if profile.require.structured_output:
            reqs.append("structured_output")
        if profile.require.min_intelligence > 0:
            reqs.append(f"intel>={profile.require.min_intelligence}")
        if profile.require.min_context > 0:
            reqs.append(f"ctx>={_format_context(profile.require.min_context)}")
        req_str = ", ".join(reqs) if reqs else "-"

        prefer_str = ", ".join(profile.prefer) if profile.prefer else "-"

        print(f"{task_name}")
        print(f"  {profile.description}")
        print(f"  require: {req_str}  prefer: {prefer_str}")
        if top3:
            model_strs = [
                f"{m.name} (${m.cost:.2f})"
                + ("" if _is_available(m) else " [unavail]")
                for m in top3
            ]
            print(f"  top models: {', '.join(model_strs)}")
        else:
            print("  top models: (none qualify)")
        print()


def _qualifies(model: ModelInfo, profile: TaskProfile) -> bool:
    """Check if a model meets a task profile's hard requirements (ignoring availability)."""
    req = profile.require
    if req.structured_output and not model.structured_output:
        return False
    if model.intelligence < req.min_intelligence:
        return False
    if model.context < req.min_context:
        return False
    return True


def register_parser(subparsers: Any) -> None:
    """Register the 'models' command group with list/show/tasks subcommands."""
    parser = subparsers.add_parser("models", help="Inspect the model registry")
    model_sub = parser.add_subparsers(dest="models_command", help="Model registry commands")

    # models list
    list_parser = model_sub.add_parser("list", help="List all registered models")
    list_parser.add_argument(
        "--all", action="store_true",
        help="Include models whose API key is not set",
    )
    list_parser.set_defaults(handler=cmd_list)

    # models show <name>
    show_parser = model_sub.add_parser("show", help="Show full details for a model")
    show_parser.add_argument("name", help="Model name or litellm_id")
    show_parser.set_defaults(handler=cmd_show)

    # models tasks
    tasks_parser = model_sub.add_parser("tasks", help="Show task profiles with preferred models")
    tasks_parser.set_defaults(handler=cmd_tasks)

    # If no subcommand given, print help
    parser.set_defaults(handler=lambda _args: parser.print_help())
