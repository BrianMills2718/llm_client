#!/usr/bin/env python3
"""Validate a plan against the unified documentation graph.

This gate checks:

* Files the plan claims to affect
* ADR governance attached to those files
* Required coupling and architecture documentation references
* Whether required reads/documentation were captured in the plan

Exit code:
  - 0 when the plan is validated
  - 1 when blocking gaps are found
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PLANS_DIR = ROOT / "docs" / "plans"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from file_context import collect_context, load_relationships


PATH_CLEAN_RE = re.compile(r"[,;:.()]$")


def normalize(path: str) -> str:
    return str(path).replace("\\", "/").strip()


def get_current_plan_number() -> int | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    branch = result.stdout.strip()
    match = re.match(r".*plan-(\d+).*", branch)
    if not match:
        return None
    return int(match.group(1))


def find_plan_file(plan_number: int, plans_dir: Path) -> Path | None:
    patterns = [
        f"{plan_number:02d}_*.md",
        f"{plan_number}_*.md",
    ]
    for pattern in patterns:
        matches = sorted(plans_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_section(content: str, heading: str) -> str:
    pattern = re.compile(
        rf"^##\s*{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        return ""
    return match.group(1).strip()


def split_lines(section: str) -> list[str]:
    lines: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(stripped)
    return lines


def looks_like_file_path(value: str) -> bool:
    if not value:
        return False
    value = value.strip().strip("`\"'()[],")
    if value in {"None", "none", "N/A", "n/a", "-"}:
        return False
    if value.startswith("#") or value.startswith("http://") or value.startswith("https://"):
        return False
    return bool("/" in value or "\\" in value or re.search(r"\.[A-Za-z0-9]{1,8}$", value))


def extract_inline_paths(line: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"\b([A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,8})\b", line):
        value = match.group(1)
        if looks_like_file_path(value):
            paths.append(normalize(value))

    for match in re.finditer(r"`([^`]+)`", line):
        value = match.group(1).strip()
        if ":" in value:
            candidate = value.split(":", 1)[0]
        else:
            candidate = value
        candidate = PATH_CLEAN_RE.sub("", candidate).strip()
        if looks_like_file_path(candidate):
            paths.append(normalize(candidate))

    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", line):
        value = match.group(1).strip()
        if looks_like_file_path(value):
            paths.append(normalize(value))

    return paths


def extract_paths(section: str) -> list[str]:
    result: list[str] = []
    for line in split_lines(section):
        if line.startswith("|") and line.endswith("|"):
            # Skip empty / table spacer rows
            if re.match(r"^\|[-\s|:]+\|$", line):
                continue
        result.extend(extract_inline_paths(line))
    dedupe: list[str] = []
    seen: set[str] = set()
    for path in result:
        if path and path not in seen:
            seen.add(path)
            dedupe.append(path)
    return dedupe


def parse_files_affected(content: str) -> list[str]:
    section = extract_section(content, "Files Affected")
    paths = extract_paths(section)
    return paths


def parse_references_reviewed(content: str) -> list[str]:
    section = extract_section(content, "References Reviewed")
    paths = extract_paths(section)
    return paths


def parse_uncertainty_register(content: str) -> list[str]:
    section = extract_section(content, "Uncertainty Register")
    if not section:
        return []
    items: list[str] = []
    for line in split_lines(section):
        if line.startswith("- ") or line.startswith("* "):
            value = line[2:].strip()
            if value and value != "-":
                items.append(value)
    return items


def parse_plan_status(content: str) -> tuple[str, str]:
    status = "Unknown"
    if m := re.search(r"\*{1,2}Status:?\*?\s*:?\s*([^\n]+)", content, re.IGNORECASE):
        status = m.group(1).strip()
    title = "Plan"
    if first := re.search(r"^#\s*([^\n]+)", content, re.MULTILINE):
        title = first.group(1).strip()
    return title, status


def parse_mentioned_adrs(content: str) -> set[int]:
    result: set[int] = set()
    for match in re.finditer(r"\bADR[-_](\d{1,4})\b", content, re.IGNORECASE):
        try:
            result.add(int(match.group(1)))
        except ValueError:
            continue
    return result


def get_plan_file(plan_number: int | None, plans_dir: Path, plan_file: str | None) -> Path:
    if plan_file:
        candidate = Path(plan_file)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.exists():
            print(f"Plan file not found: {candidate}")
            raise SystemExit(1)
        return candidate

    if plan_number is None:
        print("No plan specified and branch name does not contain plan-<N>.")
        print("Use --plan <N> or --plan-file <path>.")
        raise SystemExit(1)

    plan_path = find_plan_file(plan_number, plans_dir)
    if not plan_path:
        print(f"Plan file not found for plan #{plan_number}")
        raise SystemExit(1)
    return plan_path


def collect_plan_requirements(
    file_paths: list[str],
    relationships: dict[str, Any],
) -> tuple[set[str], set[str], set[str], list[tuple[str, int, str]]]:
    required_docs_strict: set[str] = set()
    required_docs_soft: set[str] = set()
    required_adrs: dict[int, str] = {}
    governance: list[tuple[str, int, str]] = []

    for file_path in file_paths:
        ctx = collect_context(file_path, relationships)
        if not ctx.governance and not ctx.current_arch_docs and not ctx.coupled_docs:
            # No relationships configured for this file in the graph.
            continue

        for adr in ctx.governance:
            required_adrs[adr["adr"]] = adr.get("title", f"ADR-{adr['adr']:04d}")
            governance.append((adr["path"], adr["adr"], adr["title"]))

        required_docs_strict.update(ctx.current_arch_docs)
        required_docs_strict.update(ctx.target_arch_docs)
        required_docs_strict.update(ctx.gap_docs)
        required_docs_strict.update(ctx.plan_refs)

        for coupling in ctx.coupled_docs:
            target = normalize(coupling["path"])
            if coupling.get("soft"):
                required_docs_soft.add(target)
            else:
                required_docs_strict.add(target)

    return required_docs_strict, required_docs_soft, set(required_adrs), governance


@dataclass
class ValidationResult:
    plan_number: int | None
    plan_file: Path
    title: str
    status: str
    affected_files: list[str]
    references_reviewed: list[str]
    uncertainties: list[str]
    required_docs_strict: set[str]
    required_docs_soft: set[str]
    missing_strict: set[str]
    missing_soft: set[str]
    missing_adrs: list[tuple[int, str]]
    governance: list[tuple[str, int, str]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "plan_number": self.plan_number,
            "plan_file": str(self.plan_file),
            "title": self.title,
            "status": self.status,
            "affected_files": self.affected_files,
            "references_reviewed": self.references_reviewed,
            "uncertainties_count": len(self.uncertainties),
            "required_docs": {
                "strict": sorted(self.required_docs_strict),
                "soft": sorted(self.required_docs_soft),
                "missing_strict": sorted(self.missing_strict),
                "missing_soft": sorted(self.missing_soft),
            },
            "governance": [
                {"source": source, "adr": adr, "title": title}
                for source, adr, title in self.governance
            ],
            "missing_adrs": [
                {"adr": adr, "title": title}
                for adr, title in self.missing_adrs
            ],
        }


def validate_plan(plan_file: Path, plan_number: int | None, relationships: dict[str, Any]) -> ValidationResult:
    content = read_text(plan_file)
    title, status = parse_plan_status(content)
    affected = parse_files_affected(content)
    if not affected:
        # Fallback: read from task pack lines with inline file references.
        affected_section = extract_section(content, "Task Pack")
        affected = extract_paths(affected_section)

    references = parse_references_reviewed(content)
    uncertainties = parse_uncertainty_register(content)
    covered = {normalize(p) for p in set(affected) | set(references)}

    required_strict, required_soft, adr_nums, governance = collect_plan_requirements(
        affected,
        relationships,
    )
    required_strict_norm = {normalize(p) for p in required_strict}
    required_soft_norm = {normalize(p) for p in required_soft}

    mentioned_adrs = parse_mentioned_adrs(content)
    adrs_meta = {int(k): v for k, v in relationships.get("adrs", {}).items()}

    missing_adrs: list[tuple[int, str]] = []
    for adr_num in sorted(adr_nums):
        adr_ref = f"ADR-{adr_num:04d}"
        if adr_num in mentioned_adrs:
            continue
        adr_file = normalize(adrs_meta.get(adr_num, {}).get("file", ""))
        if adr_file and adr_file not in covered:
            missing_adrs.append((adr_num, adrs_meta.get(adr_num, {}).get("title", adr_ref)))

    missing_strict = {path for path in required_strict_norm if path not in covered}
    missing_soft = {path for path in required_soft_norm if path not in covered}

    return ValidationResult(
        plan_number=plan_number,
        plan_file=plan_file,
        title=title,
        status=status,
        affected_files=sorted(affected),
        references_reviewed=sorted(references),
        uncertainties=uncertainties,
        required_docs_strict=required_strict_norm,
        required_docs_soft=required_soft_norm,
        missing_strict=missing_strict,
        missing_soft=missing_soft,
        missing_adrs=missing_adrs,
        governance=governance,
    )


def print_summary(result: ValidationResult) -> None:
    print(f"Plan validation: {result.title}")
    if result.plan_number is not None:
        print(f"  Number: #{result.plan_number}")
    print(f"  File: {result.plan_file}")
    print(f"  Status: {result.status}")
    print(f"  Affected files: {len(result.affected_files)}")
    if not result.affected_files:
        print("  (no affected files discovered)")
    for path in result.affected_files:
        print(f"    - {path}")

    print("\nGOVERNANCE:")
    if result.governance:
        seen = set()
        for source, adr, title in result.governance:
            key = (source, adr, title)
            if key in seen:
                continue
            seen.add(key)
            print(f"  - ADR-{adr:04d}: {title} ({source})")
    else:
        print("  (none)")

    print("\nREQUIRED DOCUMENTATION (strict):")
    if result.required_docs_strict:
        for path in sorted(result.required_docs_strict):
            print(f"  - {path}")
    else:
        print("  (none)")

    if result.required_docs_soft:
        print("\nREQUIRED DOCUMENTATION (soft):")
        for path in sorted(result.required_docs_soft):
            print(f"  - {path}")

    if result.missing_strict or result.missing_adrs or result.missing_soft:
        print("\nGAPS FOUND:")
        if result.missing_strict:
            print("  Strict documentation gaps:")
            for path in sorted(result.missing_strict):
                print(f"    - {path}")
        if result.missing_adrs:
            print("  Missing governance ADR coverage in plan text:")
            for adr_num, title in result.missing_adrs:
                print(f"    - ADR-{adr_num:04d}: {title}")
        if result.missing_soft:
            print("  Soft documentation gaps:")
            for path in sorted(result.missing_soft):
                print(f"    - {path}")
    else:
        print("\nNo documentation gaps found.")

    if result.uncertainties:
        print(f"\nUncertainty register entries: {len(result.uncertainties)} (not a blocker)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a plan against the documentation relationship graph"
    )
    parser.add_argument("--plan", type=int, help="Plan number (e.g. 28)")
    parser.add_argument("--plan-file", help="Plan file path (overrides --plan)")
    parser.add_argument(
        "--plans-dir",
        default=str(PLANS_DIR),
        help="Directory containing plan files",
    )
    parser.add_argument(
        "--config",
        default="scripts/relationships.yaml",
        help="Path to relationship configuration",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Do not fail when gaps are found",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable result",
    )
    args = parser.parse_args()

    plans_dir = Path(args.plans_dir)
    if not plans_dir.is_absolute():
        plans_dir = ROOT / plans_dir

    plan_number = args.plan
    if plan_number is None and not args.plan_file:
        plan_number = get_current_plan_number()

    plan_path = get_plan_file(plan_number, plans_dir, args.plan_file)
    if not args.plan and plan_number is None:
        # Fallback by parsing the plan number from file name
        match = re.match(r"(\d+)_", plan_path.name)
        if match:
            plan_number = int(match.group(1))

    relationships = load_relationships(config_path=args.config)

    result = validate_plan(plan_path, plan_number, relationships)

    if args.json:
        print(json.dumps(result.to_payload(), indent=2))
    else:
        print_summary(result)

    if args.warn_only:
        return 0

    if result.missing_strict or result.missing_adrs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
