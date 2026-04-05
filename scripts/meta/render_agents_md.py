#!/usr/bin/env python3
"""Render a generated AGENTS.md projection from canonical repo governance.

This tool keeps Codex-facing instructions aligned with repo-local governance
without making ``AGENTS.md`` a second hand-maintained authority.

Canonical inputs:
- ``CLAUDE.md`` for human-readable governance and workflow policy
- ``scripts/relationships.yaml`` for machine-readable coupling, ADR, and
  required-reading rules

The renderer is intentionally deterministic. It does not use an LLM or any
summarization heuristic beyond extracting a fixed set of sections from
``CLAUDE.md`` and recording a sync marker for ``scripts/relationships.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enforced_planning.agents_rendering import CanonicalInputs
from enforced_planning.agents_rendering import build_renderer
from enforced_planning.agents_rendering import extract_overview as _extract_overview
from enforced_planning.agents_rendering import extract_section as _extract_section
from enforced_planning.agents_rendering import extract_title as _extract_title
from enforced_planning.agents_rendering import parse_args as _parse_args


_RUNTIME = build_renderer(SCRIPT_PATH)
REPO_ROOT = _RUNTIME.repo_root
DEFAULT_TEMPLATE = _RUNTIME.default_template


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return a stable path string for generated output provenance."""

    return _RUNTIME.repo_relative(path, repo_root)


def resolve_inputs(
    repo_root: Path,
    claude_file: str = "CLAUDE.md",
    relationships_file: str = "scripts/relationships.yaml",
    output_file: str = "AGENTS.md",
    template_path: Path = DEFAULT_TEMPLATE,
) -> CanonicalInputs:
    """Resolve canonical input paths and fail loudly when missing."""
    return _RUNTIME.resolve_inputs(
        repo_root=repo_root,
        claude_file=claude_file,
        relationships_file=relationships_file,
        output_file=output_file,
        template_path=template_path,
    )


def render_agents_markdown(inputs: CanonicalInputs) -> str:
    """Render the generated ``AGENTS.md`` content for a repo."""
    return _RUNTIME.render_agents_markdown(inputs)


def render_agents_md(claude_path: Path) -> str:
    """Backwards-compatible wrapper around the explicit rendering API."""

    inputs = resolve_inputs(
        repo_root=claude_path.resolve().parent,
        claude_file=claude_path.name,
    )
    return render_agents_markdown(inputs)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the renderer."""
    return _parse_args(DEFAULT_TEMPLATE)


def main() -> int:
    """Render ``AGENTS.md`` and write or print the result."""

    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    template_path = Path(args.template).resolve()
    try:
        inputs = resolve_inputs(
            repo_root=repo_root,
            claude_file=args.claude_file,
            relationships_file=args.relationships_file,
            output_file=args.output_file,
            template_path=template_path,
        )
        rendered = render_agents_markdown(inputs)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    if args.stdout:
        print(rendered, end="")
        return 0

    inputs.output_path.write_text(rendered, encoding="utf-8")
    print(f"Rendered {inputs.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
