#!/usr/bin/env python3
"""Generate or refresh the reviewed dead-code audit file for one repo.

This script records every current detector finding in a deterministic JSON file.
Reviewers fill in dispositions only for findings that are intentionally retained.
Unjustified findings stay as ``review_required`` and continue to block strict
dead-code enforcement until the code is deleted or integrated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import check_dead_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for dead-code audit generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=".",
        help="Repo root to audit (defaults to current working directory).",
    )
    parser.add_argument(
        "--output",
        help="Override the audit output path relative to the repo root.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the merged audit file instead of printing JSON to stdout.",
    )
    return parser.parse_args(argv)


def _merge_findings(
    raw_findings: list[check_dead_code.Finding],
    *,
    detector: str,
    existing_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge current detector output with any existing reviewer annotations."""

    existing_by_signature: dict[str, dict[str, Any]] = {}
    for entry in existing_entries:
        if not isinstance(entry, dict):
            continue
        detector_name = entry.get("detector")
        file_path = entry.get("file")
        kind = entry.get("kind")
        name = entry.get("name")
        if not all(
            isinstance(value, str) and value
            for value in (detector_name, file_path, kind, name)
        ):
            continue
        signature = "::".join([detector_name, file_path, kind, name])
        existing_by_signature[signature] = entry

    merged: list[dict[str, Any]] = []
    for finding in sorted(
        raw_findings,
        key=lambda item: (item.file, item.line, item.kind, item.name),
    ):
        signature = check_dead_code.finding_signature(finding, detector)
        existing = existing_by_signature.get(signature, {})
        merged.append(
            {
                "file": finding.file,
                "line": finding.line,
                "name": finding.name,
                "kind": finding.kind,
                "detector": detector,
                "confidence": finding.confidence,
                "disposition": existing.get("disposition", "review_required"),
                "note": existing.get("note", ""),
                "plan_ref": existing.get("plan_ref"),
            }
        )
    return merged


def build_audit_payload(repo_root: Path, output_file: str) -> dict[str, Any]:
    """Return the merged dead-code audit payload for one repo."""

    raw_result = check_dead_code.scan_dead_code(repo_root)
    if raw_result.error:
        raise RuntimeError(raw_result.error)
    if not raw_result.detector:
        raise RuntimeError("dead-code detector did not report which tool was used")

    existing_payload = check_dead_code._load_audit_payload(repo_root, output_file)
    existing_entries = existing_payload.get("findings", [])
    if not isinstance(existing_entries, list):
        raise ValueError(f"{output_file} field 'findings' must be a list")

    merged_findings = _merge_findings(
        raw_result.findings,
        detector=raw_result.detector,
        existing_entries=existing_entries,
    )
    return {
        "version": 1,
        "generated_at": check_dead_code._current_timestamp(),
        "detector": raw_result.detector,
        "findings_count": len(merged_findings),
        "audit_file": output_file,
        "findings": merged_findings,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate or refresh the current repo's dead-code audit file."""

    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    config = check_dead_code._load_config(repo_root)
    if not config.get("enabled"):
        print(
            json.dumps(
                {
                    "skipped": True,
                    "reason": "dead_code not enabled in meta-process.yaml",
                }
            )
        )
        return 0

    output_file = args.output or str(config.get("audit_file", "dead_code_audit.json"))
    payload = build_audit_payload(repo_root, output_file)

    if args.write:
        output_path = repo_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")
        return 0

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
