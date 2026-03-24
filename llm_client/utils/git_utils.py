"""Git utilities for llm_client observability."""

import subprocess


def get_git_head() -> str | None:
    """Return the current git HEAD commit hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
