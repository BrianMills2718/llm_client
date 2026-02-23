#!/usr/bin/env python3
"""Enforce required reading before editing production source files.

Used by `.claude/hooks/gate-edit.sh`.
The script checks whether docs coupled to a target file were read in the
current session (tracked by `.claude/hooks/track-reads.sh`).
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None

VALID_MODES = {"strict", "warn", "off"}
DEFAULT_GATE_SETTINGS = {
    "enabled": True,
    "mode": "strict",
    "uncoupled_mode": "strict",
    "config_file": "scripts/relationships.yaml",
    "show_success": False,
}


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


def _load_yaml_dict(path: Path) -> dict:
    if not path.exists() or yaml is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_mode(raw: str | None, fallback: str) -> str:
    mode = (raw or "").strip().lower()
    if mode in VALID_MODES:
        return mode
    return fallback


def _load_gate_settings(repo_root: Path) -> dict:
    settings = dict(DEFAULT_GATE_SETTINGS)

    meta_config = _load_yaml_dict(repo_root / "meta-process.yaml")
    meta_process = meta_config.get("meta_process", {})
    quality = meta_process.get("quality", {}) if isinstance(meta_process, dict) else {}
    required_reading = quality.get("required_reading", {}) if isinstance(quality, dict) else {}
    if isinstance(required_reading, dict):
        settings.update({k: v for k, v in required_reading.items() if k in settings})

    env_enabled = _parse_bool(os.getenv("LLM_CLIENT_READ_GATE_ENABLED"))
    if env_enabled is not None:
        settings["enabled"] = env_enabled

    env_mode = os.getenv("LLM_CLIENT_READ_GATE_MODE")
    if env_mode:
        settings["mode"] = env_mode

    env_uncoupled_mode = os.getenv("LLM_CLIENT_READ_GATE_UNCOUPLED_MODE")
    if env_uncoupled_mode:
        settings["uncoupled_mode"] = env_uncoupled_mode

    env_show_success = _parse_bool(os.getenv("LLM_CLIENT_READ_GATE_SHOW_SUCCESS"))
    if env_show_success is not None:
        settings["show_success"] = env_show_success

    env_config_file = os.getenv("LLM_CLIENT_READ_GATE_CONFIG")
    if env_config_file:
        settings["config_file"] = env_config_file

    settings["mode"] = _normalize_mode(str(settings.get("mode")), DEFAULT_GATE_SETTINGS["mode"])
    settings["uncoupled_mode"] = _normalize_mode(
        str(settings.get("uncoupled_mode")),
        settings["mode"],
    )
    settings["enabled"] = bool(settings.get("enabled", True))
    settings["show_success"] = bool(settings.get("show_success", False))
    settings["config_file"] = _norm(str(settings.get("config_file", DEFAULT_GATE_SETTINGS["config_file"])))
    return settings


def _load_relationships(repo_root: Path, config_file: str) -> dict:
    return _load_yaml_dict(repo_root / config_file)


def _required_docs_for_target(target: str, relationships: dict) -> tuple[list[str], bool]:
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

    matched_any = False
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
        matched_any = True
        for doc in cdocs:
            if isinstance(doc, str) and _norm(doc):
                docs.add(_norm(doc))

    return sorted(docs), matched_any


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
    settings = _load_gate_settings(repo_root)

    if not settings["enabled"]:
        if settings["show_success"]:
            print(f"Read gate disabled for: {target}")
        return 0

    relationships = _load_relationships(repo_root, settings["config_file"])
    required_docs, matched_coupling = _required_docs_for_target(target, relationships)
    read_set = _load_read_set(Path(args.reads_file), repo_root)
    missing = _missing_docs(required_docs, read_set, repo_root)

    effective_mode = settings["mode"] if matched_coupling else settings["uncoupled_mode"]
    if effective_mode == "off":
        if settings["show_success"]:
            scope = "coupled" if matched_coupling else "uncoupled"
            print(f"Read gate off ({scope}) for: {target}")
        return 0

    if missing:
        if effective_mode == "warn":
            print(f"Read gate warning for: {target}")
            print("Required docs not read in this session:")
            for doc in missing:
                print(f"  - {doc}")
            print("")
            print("Continuing because mode=warn.")
            print("To enforce blocking: set mode=strict.")
            return 0

        print(f"Read gate blocked edit to: {target}")
        print("Required docs not read in this session:")
        for doc in missing:
            print(f"  - {doc}")
        print("")
        print("Read the missing docs first, then retry the edit.")
        print("Temporary bypass: SKIP_READ_GATE=1")
        print("Config: set mode=warn/off in meta-process.yaml to relax.")
        return 1

    if settings["show_success"]:
        print(f"Read gate OK for: {target}")
        print("Required docs read:")
        for doc in required_docs:
            print(f"  - {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
