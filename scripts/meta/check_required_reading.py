#!/usr/bin/env python3
"""Enforce required reading before editing production source files.

Used by `.claude/hooks/gate-edit.sh`.
The script checks whether docs coupled to a target file were read in the
current session (tracked by `.claude/hooks/track-reads.sh`).
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _norm(path: str) -> str:
    p = path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def _rel_from_repo(path: Path, repo_root: Path) -> str | None:
    try:
        return _norm(path.resolve().relative_to(repo_root.resolve()).as_posix())
    except Exception:
        return None


def _aliases(path_text: str, repo_root: Path) -> set[str]:
    """Return normalized + resolved aliases for a path string."""
    aliases: set[str] = set()
    raw = _norm(path_text)
    if not raw:
        return aliases
    aliases.add(raw)
    as_path = Path(raw)
    if as_path.is_absolute():
        rel = _rel_from_repo(as_path, repo_root)
        if rel:
            aliases.add(rel)
    else:
        rel = _rel_from_repo(repo_root / as_path, repo_root)
        if rel:
            aliases.add(rel)
    return aliases


def _load_relationships(repo_root: Path) -> dict:
    rel_path = repo_root / "scripts" / "relationships.yaml"
    if not rel_path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(rel_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _required_docs_for_target(target: str, relationships: dict) -> list[str]:
    """Resolve required docs from relationships config for a target file."""
    required_reading = relationships.get("required_reading", {})
    defaults = required_reading.get("defaults", [])
    docs: set[str] = {
        _norm(str(x))
        for x in defaults
        if isinstance(x, str) and _norm(str(x))
    }

    # Safe baseline if config exists but doesn't provide defaults.
    if not docs:
        docs.add("CLAUDE.md")

    for coupling in relationships.get("couplings", []):
        if not isinstance(coupling, dict):
            continue
        if coupling.get("required_reading", True) is False:
            continue
        if coupling.get("soft", False):
            continue
        sources = coupling.get("sources", [])
        cdocs = coupling.get("docs", [])
        if not isinstance(sources, list) or not isinstance(cdocs, list):
            continue
        if not any(
            isinstance(pattern, str) and fnmatch.fnmatch(target, _norm(pattern))
            for pattern in sources
        ):
            continue
        for doc in cdocs:
            if isinstance(doc, str) and _norm(doc):
                docs.add(_norm(doc))

    return sorted(docs)


def _load_read_set(reads_file: Path, repo_root: Path) -> set[str]:
    if not reads_file.exists():
        return set()
    read_set: set[str] = set()
    for line in reads_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        read_set.update(_aliases(line.strip(), repo_root))
    return read_set


def _missing_docs(required_docs: list[str], read_set: set[str], repo_root: Path) -> list[str]:
    missing: list[str] = []
    for doc in required_docs:
        if _aliases(doc, repo_root) & read_set:
            continue
        missing.append(doc)
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_file", help="Repo-relative file being edited")
    parser.add_argument(
        "--reads-file",
        default="/tmp/.claude_session_reads",
        help="Session file tracking read paths (default: /tmp/.claude_session_reads)",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    target = _norm(args.target_file)
    relationships = _load_relationships(repo_root)
    required_docs = _required_docs_for_target(target, relationships)
    read_set = _load_read_set(Path(args.reads_file), repo_root)
    missing = _missing_docs(required_docs, read_set, repo_root)

    if missing:
        print(f"Read gate blocked edit to: {target}")
        print("Required docs not read in this session:")
        for doc in missing:
            print(f"  - {doc}")
        print("")
        print("Read the missing docs first, then retry the edit.")
        print("Temporary bypass: SKIP_READ_GATE=1")
        return 1

    print(f"Read gate OK for: {target}")
    print("Required docs read:")
    for doc in required_docs:
        print(f"  - {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
