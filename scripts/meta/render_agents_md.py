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
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_CANDIDATES = (
    ROOT / "meta-process" / "templates" / "agents.md.template",
    ROOT.parent / "enforced-planning" / "templates" / "agents.md.template",
    ROOT.parents[1] / "enforced-planning" / "templates" / "agents.md.template",
)


def _resolve_default_template() -> Path:
    """Return the first existing AGENTS template candidate.

    `project-meta` historically carried the template locally under
    `meta-process/templates/`. After the framework extraction, clean branches
    may rely on the portable `~/projects/enforced-planning/templates/` copy
    instead. Resolve deterministically so generated AGENTS surfaces keep working
    across both layouts without hardcoded home-directory strings.
    """

    for candidate in DEFAULT_TEMPLATE_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_TEMPLATE_CANDIDATES[0]


DEFAULT_TEMPLATE = _resolve_default_template()

SECTION_RE = re.compile(
    r"^##\s+(?P<heading>[^\n]+)\n(?P<body>.*?)(?=^##\s|\Z)",
    re.MULTILINE | re.DOTALL,
)

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "Commands": ("Commands", "Quick Reference - Commands"),
    "Principles": ("Principles", "Design Principles"),
    "Workflow": ("Workflow",),
    "References": ("References",),
}


@dataclass(frozen=True)
class CanonicalInputs:
    """Resolved canonical inputs used to render ``AGENTS.md``."""

    repo_root: Path
    claude_path: Path
    relationships_path: Path
    output_path: Path
    template_path: Path


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return a stable path string for generated output provenance.

    Prefer a repo-relative path when the referenced file lives inside the target
    repo. When the generator itself lives outside the target repo, fall back to
    a path relative to the project-meta root so generated output still has a
    stable, non-absolute provenance marker.
    """

    resolved = path.resolve()
    repo_root_resolved = repo_root.resolve()
    root_resolved = ROOT.resolve()
    if resolved.is_relative_to(repo_root_resolved):
        return str(resolved.relative_to(repo_root_resolved))
    if resolved.is_relative_to(root_resolved):
        return str(resolved.relative_to(root_resolved))
    return path.name


def _extract_title(markdown: str) -> str:
    """Return the first H1 heading from a markdown document."""

    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if not match:
        raise ValueError("CLAUDE.md is missing a top-level title")
    return match.group(1).strip()


def _extract_overview(markdown: str) -> str:
    """Return the intro block between the title and the first thematic break.

    Some governed repos have no prose overview before the first thematic break.
    In that case, return a deterministic fallback sentence rather than failing
    generation for an otherwise valid governance file.
    """

    lines = markdown.splitlines()
    title_seen = False
    collected: list[str] = []
    for line in lines:
        if not title_seen:
            if line.startswith("# "):
                title_seen = True
            continue
        if line.strip() == "---" or line.startswith("## "):
            break
        collected.append(line)

    overview = "\n".join(collected).strip()
    if not overview:
        return (
            f"{_extract_title(markdown)} uses `CLAUDE.md` as canonical repo "
            "governance and workflow policy."
        )
    return overview


def _extract_section(markdown: str, heading: str) -> str:
    """Return the body for a named H2 section or supported heading alias."""

    accepted_headings = SECTION_ALIASES.get(heading, (heading,))
    for match in SECTION_RE.finditer(markdown):
        current_heading = match.group("heading").strip()
        if current_heading in accepted_headings:
            lines = match.group("body").splitlines()
            while lines and not lines[-1].strip():
                lines.pop()
            while lines and lines[-1].strip() == "---":
                lines.pop()
                while lines and not lines[-1].strip():
                    lines.pop()
            body = "\n".join(lines).strip()
            if not body:
                raise ValueError(f"CLAUDE.md section {current_heading!r} is empty")
            return body
    raise ValueError(
        "CLAUDE.md is missing required section: "
        f"{heading} (accepted headings: {', '.join(accepted_headings)})"
    )


def resolve_inputs(
    repo_root: Path,
    claude_file: str = "CLAUDE.md",
    relationships_file: str = "scripts/relationships.yaml",
    output_file: str = "AGENTS.md",
    template_path: Path = DEFAULT_TEMPLATE,
) -> CanonicalInputs:
    """Resolve canonical input paths and fail loudly when missing."""

    claude_path = repo_root / claude_file
    relationships_path = repo_root / relationships_file
    output_path = repo_root / output_file

    if not claude_path.exists():
        raise FileNotFoundError(f"Missing canonical governance file: {claude_path}")
    if not relationships_path.exists():
        raise FileNotFoundError(
            f"Missing machine-readable governance file: {relationships_path}"
        )
    if not template_path.exists():
        raise FileNotFoundError(f"Missing AGENTS template: {template_path}")
    if output_path.is_symlink() and output_path.resolve() == claude_path.resolve():
        raise ValueError(
            "AGENTS output path is a symlink to CLAUDE.md. Remove the symlink "
            "before rendering so generated output cannot overwrite canonical governance."
        )

    return CanonicalInputs(
        repo_root=repo_root,
        claude_path=claude_path,
        relationships_path=relationships_path,
        output_path=output_path,
        template_path=template_path,
    )


def render_agents_markdown(inputs: CanonicalInputs) -> str:
    """Render the generated ``AGENTS.md`` content for a repo."""

    claude_text = inputs.claude_path.read_text(encoding="utf-8")
    relationships_text = inputs.relationships_path.read_text(encoding="utf-8")
    template_text = inputs.template_path.read_text(encoding="utf-8")
    relationships_sha256 = hashlib.sha256(
        relationships_text.encode("utf-8")
    ).hexdigest()[:12]

    generator_relpath = _repo_relative(Path(__file__).resolve(), inputs.repo_root)
    sync_checker_relpath = _repo_relative(
        ROOT / "scripts" / "meta" / "check_agents_sync.py",
        inputs.repo_root,
    )
    claude_relpath = _repo_relative(inputs.claude_path, inputs.repo_root)
    relationships_relpath = _repo_relative(inputs.relationships_path, inputs.repo_root)

    machine_governance_note = (
        f"`{relationships_relpath}` is the source of truth for machine-readable "
        "governance in this repo: ADR coupling, required-reading edges, and "
        "doc-code linkage. This generated file does not inline that graph; it "
        "records the canonical path and sync marker, then points operators and "
        "validators back to the source graph. Prefer deterministic validators "
        "over prompt-only memory when those scripts are available."
    )

    rendered = template_text.format(
        title=_extract_title(claude_text),
        generator_relpath=generator_relpath,
        sync_checker_relpath=sync_checker_relpath,
        claude_relpath=claude_relpath,
        relationships_relpath=relationships_relpath,
        relationships_sha256=relationships_sha256,
        overview=_extract_overview(claude_text),
        commands=_extract_section(claude_text, "Commands"),
        principles=_extract_section(claude_text, "Principles"),
        workflow=_extract_section(claude_text, "Workflow"),
        references=_extract_section(claude_text, "References"),
        machine_governance_note=machine_governance_note,
    )
    return rendered.strip() + "\n"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the renderer."""

    parser = argparse.ArgumentParser(
        description="Render a generated AGENTS.md from canonical repo governance",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root containing CLAUDE.md and scripts/relationships.yaml",
    )
    parser.add_argument(
        "--claude-file",
        default="CLAUDE.md",
        help="Repo-relative path to canonical CLAUDE.md",
    )
    parser.add_argument(
        "--relationships-file",
        default="scripts/relationships.yaml",
        help="Repo-relative path to canonical relationships.yaml",
    )
    parser.add_argument(
        "--output-file",
        default="AGENTS.md",
        help="Repo-relative output path for generated AGENTS.md",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help="Path to the AGENTS markdown template",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print rendered markdown instead of writing the output file",
    )
    return parser.parse_args()


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
