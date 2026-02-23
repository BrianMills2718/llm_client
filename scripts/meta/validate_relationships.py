#!/usr/bin/env python3
"""Validate relationships config used by read-gate and quiz tooling."""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required (pip install pyyaml)")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML value must be a mapping")
    return data


def _is_glob_pattern(text: str) -> bool:
    return any(ch in text for ch in ("*", "?", "["))


def _validate_required_reading(
    config: dict,
    repo_root: Path,
    errors: list[str],
) -> None:
    rr = config.get("required_reading", {})
    if not isinstance(rr, dict):
        errors.append("required_reading must be a mapping")
        return
    defaults = rr.get("defaults", [])
    if not isinstance(defaults, list):
        errors.append("required_reading.defaults must be a list")
        return
    for idx, doc in enumerate(defaults):
        if not isinstance(doc, str) or not doc.strip():
            errors.append(f"required_reading.defaults[{idx}] must be a non-empty string")
            continue
        doc_path = repo_root / doc
        if not doc_path.exists():
            errors.append(f"required_reading default doc does not exist: {doc}")


def _validate_couplings(
    config: dict,
    repo_root: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    couplings = config.get("couplings", [])
    if not isinstance(couplings, list):
        errors.append("couplings must be a list")
        return
    if not couplings:
        warnings.append("couplings list is empty")
        return

    for idx, coupling in enumerate(couplings):
        prefix = f"couplings[{idx}]"
        if not isinstance(coupling, dict):
            errors.append(f"{prefix} must be a mapping")
            continue

        sources = coupling.get("sources")
        docs = coupling.get("docs")
        description = coupling.get("description")
        required_reading = coupling.get("required_reading", True)

        if not isinstance(sources, list) or not sources:
            errors.append(f"{prefix}.sources must be a non-empty list")
        else:
            for s_idx, source in enumerate(sources):
                if not isinstance(source, str) or not source.strip():
                    errors.append(f"{prefix}.sources[{s_idx}] must be a non-empty string")
                    continue
                if not _is_glob_pattern(source) and not (repo_root / source).exists():
                    warnings.append(f"{prefix}.sources[{s_idx}] references missing path: {source}")

        if not isinstance(docs, list) or not docs:
            errors.append(f"{prefix}.docs must be a non-empty list")
        else:
            for d_idx, doc in enumerate(docs):
                if not isinstance(doc, str) or not doc.strip():
                    errors.append(f"{prefix}.docs[{d_idx}] must be a non-empty string")
                    continue
                doc_path = repo_root / doc
                if not doc_path.exists():
                    errors.append(f"{prefix}.docs[{d_idx}] does not exist: {doc}")

        if not isinstance(description, str) or not description.strip():
            errors.append(f"{prefix}.description must be a non-empty string")

        if not isinstance(required_reading, bool):
            errors.append(f"{prefix}.required_reading must be a boolean")

        uncertainty = coupling.get("uncertainty")
        if uncertainty is not None and (not isinstance(uncertainty, str) or not uncertainty.strip()):
            errors.append(f"{prefix}.uncertainty must be a non-empty string when present")


def _summarize_matches(config: dict, sample_paths: list[str]) -> list[str]:
    couplings = config.get("couplings", [])
    if not isinstance(couplings, list):
        return []
    lines: list[str] = []
    for path in sample_paths:
        matched = 0
        for coupling in couplings:
            if not isinstance(coupling, dict):
                continue
            sources = coupling.get("sources", [])
            if not isinstance(sources, list):
                continue
            if any(isinstance(pattern, str) and fnmatch.fnmatch(path, pattern) for pattern in sources):
                matched += 1
        lines.append(f"{path}: {matched} matching coupling(s)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="scripts/relationships.yaml",
        help="Path to relationships YAML (default: scripts/relationships.yaml)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings in addition to errors",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Optional sample path(s) to print coupling match counts",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found: {args.config}")
        return 1

    try:
        config = _load_yaml(config_path)
    except Exception as exc:
        print(f"ERROR: Failed to parse {args.config}: {exc}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    _validate_required_reading(config, repo_root, errors)
    _validate_couplings(config, repo_root, errors, warnings)

    if errors:
        print("Relationships validation errors:")
        for err in errors:
            print(f"  - {err}")
    if warnings:
        print("Relationships validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if args.sample:
        print("Sample coupling matches:")
        for line in _summarize_matches(config, args.sample):
            print(f"  - {line}")

    if errors:
        return 1
    if args.strict and warnings:
        return 1

    print("Relationships validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
