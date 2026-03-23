"""Git utilities for correlating scores with code changes.

Thin wrappers around git subprocess calls + path-based diff classification.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import PurePosixPath

logger = logging.getLogger(__name__)

# Change type constants
PROMPT_CHANGE = "PROMPT_CHANGE"
RUBRIC_CHANGE = "RUBRIC_CHANGE"
CODE_CHANGE = "CODE_CHANGE"
CONFIG_CHANGE = "CONFIG_CHANGE"
TEST_CHANGE = "TEST_CHANGE"


def get_git_head(cwd: str | None = None) -> str | None:
    """Return short SHA of HEAD, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def get_diff_files(commit_a: str, commit_b: str, cwd: str | None = None) -> list[str]:
    """Return list of files changed between two commits."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", commit_a, commit_b],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
        )
        if result.returncode != 0:
            logger.debug("git diff failed: %s", result.stderr.strip())
            return []
        return [f for f in result.stdout.strip().splitlines() if f]
    except Exception:
        logger.debug("get_diff_files failed", exc_info=True)
        return []


def get_working_tree_files(cwd: str | None = None) -> list[str]:
    """Return files changed in the current working tree (staged/unstaged/untracked)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=cwd,
        )
        if result.returncode != 0:
            logger.debug("git status failed: %s", result.stderr.strip())
            return []

        files: list[str] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.rstrip()
            if len(line) < 4:
                continue
            path_part = line[3:]
            # Rename/Copies appear as "old/path -> new/path"; keep destination path.
            if " -> " in path_part:
                path_part = path_part.split(" -> ", 1)[1]
            if path_part:
                files.append(path_part)

        # Preserve order while de-duplicating.
        return list(dict.fromkeys(files))
    except Exception:
        logger.debug("get_working_tree_files failed", exc_info=True)
        return []


def is_git_dirty(cwd: str | None = None) -> bool:
    """Return True when the working tree has local modifications or untracked files."""
    return bool(get_working_tree_files(cwd=cwd))


def classify_diff_files(files: list[str]) -> set[str]:
    """Classify changed files into change type categories by path heuristics."""
    categories: set[str] = set()
    for f in files:
        p = PurePosixPath(f)
        parts = p.parts
        name = p.name

        # Prompt changes: files in prompts/ dir or .yaml in a prompts dir
        if "prompts" in parts:
            categories.add(PROMPT_CHANGE)
            continue

        # Rubric changes
        if "rubrics" in parts:
            categories.add(RUBRIC_CHANGE)
            continue

        # Test changes
        if "tests" in parts or name.startswith("test_"):
            categories.add(TEST_CHANGE)
            continue

        # Python code
        if name.endswith(".py"):
            categories.add(CODE_CHANGE)
            continue

        # Config files
        if name.endswith((".yaml", ".yml", ".json", ".toml")):
            categories.add(CONFIG_CHANGE)
            continue

    return categories
