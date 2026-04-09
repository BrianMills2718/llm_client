#!/usr/bin/env python3
"""Validate reviewed dead-code dispositions against fresh detector output.

This validator enforces that every current dead-code finding has an explicit,
still-valid reviewed disposition before strict dead-code enforcement may ignore
it. It fails loudly on stale suppressions, incomplete review placeholders, and
planned-feature justifications that are no longer backed by an active plan.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import check_dead_code

ALLOWED_RETAINED_DISPOSITIONS = set(check_dead_code.RETAINED_FINDING_DISPOSITIONS)
PLAN_REF_RE = re.compile(r"Plan\s+#(?P<number>\d+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for audit validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=".",
        help="Repo root to validate (defaults to current working directory).",
    )
    parser.add_argument(
        "--audit-file",
        help="Override the audit file path relative to the repo root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the validation result as JSON.",
    )
    return parser.parse_args(argv)


def _find_plan_file(repo_root: Path, plan_number: int) -> Path | None:
    """Return the numbered plan file if it exists."""

    plans_dir = repo_root / "docs" / "plans"
    if not plans_dir.is_dir():
        return None
    matches = sorted(plans_dir.glob(f"{plan_number}_*.md"))
    if not matches:
        return None
    return matches[0]


def _plan_status(plan_file: Path) -> str | None:
    """Return the plan's Status field when present."""

    text = plan_file.read_text(encoding="utf-8")
    match = re.search(r"\*\*Status:\*\*\s*([^\n]+)", text)
    if not match:
        return None
    return match.group(1).strip()


def validate_dead_code_audit(repo_root: Path, audit_file: str) -> dict[str, Any]:
    """Validate the repo's reviewed dead-code dispositions."""

    config = check_dead_code._load_config(repo_root)
    if not config.get("enabled"):
        return {
            "passed": True,
            "skipped": True,
            "reason": "dead_code not enabled in meta-process.yaml",
            "errors": [],
        }

    raw_result = check_dead_code.scan_dead_code(repo_root)
    if raw_result.error:
        return {
            "passed": False,
            "skipped": False,
            "reason": "",
            "errors": [raw_result.error],
        }

    payload = check_dead_code._load_audit_payload(repo_root, audit_file)
    entries = payload.get("findings", [])
    if not isinstance(entries, list):
        return {
            "passed": False,
            "skipped": False,
            "reason": "",
            "errors": [f"{audit_file} field 'findings' must be a list"],
        }

    current_by_signature = {
        check_dead_code.finding_signature(finding, raw_result.detector): finding
        for finding in raw_result.findings
    }

    errors: list[str] = []
    audit_by_signature: dict[str, dict[str, Any]] = {}

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"{audit_file} finding #{index} must be an object")
            continue

        detector = entry.get("detector")
        file_path = entry.get("file")
        kind = entry.get("kind")
        name = entry.get("name")
        if not all(isinstance(value, str) and value for value in (detector, file_path, kind, name)):
            errors.append(
                f"{audit_file} finding #{index} must include non-empty detector/file/kind/name"
            )
            continue
        signature = "::".join([detector, file_path, kind, name])
        if signature in audit_by_signature:
            errors.append(f"{audit_file} contains duplicate reviewed finding: {signature}")
            continue
        audit_by_signature[signature] = entry

        if signature not in current_by_signature:
            errors.append(
                f"{audit_file} contains stale reviewed finding with no current detector match: {signature}"
            )
            continue

        disposition = entry.get("disposition")
        if disposition not in ALLOWED_RETAINED_DISPOSITIONS:
            errors.append(
                f"{audit_file} finding {signature} must use one of {sorted(ALLOWED_RETAINED_DISPOSITIONS)}"
            )
            continue

        note = entry.get("note")
        if not isinstance(note, str) or not note.strip():
            errors.append(f"{audit_file} finding {signature} requires a non-empty note")

        plan_ref = entry.get("plan_ref")
        if disposition == "keep_planned_feature":
            if not isinstance(plan_ref, str) or not plan_ref.strip():
                errors.append(f"{audit_file} finding {signature} requires plan_ref")
                continue
            match = PLAN_REF_RE.fullmatch(plan_ref.strip())
            if not match:
                errors.append(
                    f"{audit_file} finding {signature} has invalid plan_ref {plan_ref!r}; expected 'Plan #N'"
                )
                continue
            plan_file = _find_plan_file(repo_root, int(match.group("number")))
            if plan_file is None:
                errors.append(
                    f"{audit_file} finding {signature} references missing plan {plan_ref}"
                )
                continue
            status = _plan_status(plan_file)
            if status and "complete" in status.lower():
                errors.append(
                    f"{audit_file} finding {signature} references completed plan {plan_ref}"
                )
        elif plan_ref not in (None, "") and not isinstance(plan_ref, str):
            errors.append(f"{audit_file} finding {signature} has non-string plan_ref")

    for signature in current_by_signature:
        if signature not in audit_by_signature:
            errors.append(
                f"Current dead-code finding has no reviewed disposition in {audit_file}: {signature}"
            )

    return {
        "passed": not errors,
        "skipped": False,
        "reason": "",
        "errors": errors,
        "current_findings_count": len(current_by_signature),
        "reviewed_findings_count": len(audit_by_signature),
        "audit_file": audit_file,
    }


def main(argv: list[str] | None = None) -> int:
    """Validate the current repo's dead-code review file."""

    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    config = check_dead_code._load_config(repo_root)
    audit_file = args.audit_file or str(config.get("audit_file", "dead_code_audit.json"))
    payload = validate_dead_code_audit(repo_root, audit_file)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif payload.get("skipped"):
        print(payload["reason"])
    elif payload["passed"]:
        print(f"Dead-code audit valid: {audit_file}")
    else:
        print(f"Dead-code audit invalid: {audit_file}")
        for error in payload["errors"]:
            print(f"- {error}")

    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
