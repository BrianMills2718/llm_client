"""Prompt loading and rendering from YAML/Jinja2 templates.

Convention: every project stores LLM prompts as YAML files with Jinja2
templates in a ``prompts/`` directory.  This module provides the single
entry point for loading and rendering them.

YAML format::

    name: pass3_diagnostic_test
    version: "1.0"
    description: Van Evera diagnostic testing
    messages:
      - role: system
        content: |
          You are a process tracing analyst.
      - role: user
        content: |
          {% for h in hypotheses %}
          ## {{ h.label }}
          {% endfor %}

Usage::

    from llm_client import render_prompt, call_llm_structured

    messages = render_prompt("prompts/pass3_test.yaml", hypotheses=hyps)
    result, meta = call_llm_structured("gemini-flash", messages, response_model=Out)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from jinja2 import BaseLoader, Environment, StrictUndefined, TemplateNotFound

logger = logging.getLogger(__name__)


class _YAMLInlineLoader(BaseLoader):
    """Jinja2 loader for inline strings (no filesystem template inheritance)."""

    def get_source(
        self, environment: Environment, template: str
    ) -> tuple[str, str | None, None]:
        raise TemplateNotFound(template)


# Single shared environment — StrictUndefined so missing vars fail loud.
_env = Environment(loader=_YAMLInlineLoader(), undefined=StrictUndefined)


def render_prompt(
    template_path: str | Path,
    **context: Any,
) -> list[dict[str, str]]:
    """Load a YAML prompt template and render Jinja2 placeholders.

    Args:
        template_path: Path to the YAML file (absolute, or relative to cwd).
        **context: Variables to substitute into Jinja2 templates.

    Returns:
        List of message dicts (OpenAI chat format): [{"role": ..., "content": ...}]

    Raises:
        FileNotFoundError: If template_path doesn't exist.
        yaml.YAMLError: If YAML is malformed.
        jinja2.UndefinedError: If a template variable is missing from context.
        ValueError: If YAML structure is invalid (no messages key, bad format).
    """
    path = Path(template_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError(f"Prompt YAML must be a mapping, got {type(raw).__name__}: {path}")

    messages_raw = raw.get("messages")
    if not messages_raw:
        raise ValueError(f"Prompt YAML missing 'messages' key: {path}")

    if not isinstance(messages_raw, list):
        raise ValueError(f"'messages' must be a list, got {type(messages_raw).__name__}: {path}")

    messages: list[dict[str, str]] = []
    for i, msg in enumerate(messages_raw):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError(
                f"Message {i} must have 'role' and 'content' keys: {path}"
            )

        template = _env.from_string(str(msg["content"]))
        rendered = template.render(**context).strip()

        messages.append({"role": str(msg["role"]), "content": rendered})

    logger.debug(
        "Rendered prompt %s (%d messages, %d total chars)",
        path.name,
        len(messages),
        sum(len(m["content"]) for m in messages),
    )

    return messages
