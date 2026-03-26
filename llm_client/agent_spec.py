"""Compatibility shim — agent_spec was extracted to project-meta (Plan #17).

This module re-exports ``load_agent_spec`` so that in-package imports
(e.g. ``llm_client.observability.experiments``) continue to work.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# project-meta/scripts/meta/ is the canonical home after Plan #17.
_PROJECT_META_SCRIPTS = Path.home() / "projects" / "project-meta" / "scripts" / "meta"

try:
    # Try importing from project-meta if it's on the path
    from scripts.meta.agent_spec import load_agent_spec  # type: ignore[import-untyped]
except ImportError:
    # Add project-meta root to sys.path and retry
    _project_meta_root = str(Path.home() / "projects" / "project-meta")
    if _project_meta_root not in sys.path:
        sys.path.insert(0, _project_meta_root)
    try:
        from scripts.meta.agent_spec import load_agent_spec  # type: ignore[import-untyped]
    except ImportError as exc:
        _import_err = exc

        def load_agent_spec(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise ImportError(
                "agent_spec was extracted from llm_client to project-meta/scripts/meta/ "
                "(Plan #17). Ensure ~/projects/project-meta exists."
            ) from _import_err

__all__ = ["load_agent_spec"]
