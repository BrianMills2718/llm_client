"""Prompt loading and rendering from YAML/Jinja2 templates.

Convention: every project stores LLM prompts as YAML files with Jinja2
templates in a ``prompts/`` directory.  This module provides the single
entry point for loading and rendering them. The same renderer also supports
explicit shared prompt asset references via ``prompt_ref=...``.

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
from llm_client.prompt_assets import resolve_prompt_asset

logger = logging.getLogger(__name__)


class _YAMLInlineLoader(BaseLoader):
    """Jinja2 loader for inline strings (no filesystem template inheritance)."""

    def get_source(
        self, _environment: Environment, template: str
    ) -> tuple[str, str | None, None]:
        raise TemplateNotFound(template)


# Single shared environment — StrictUndefined so missing vars fail loud.
_env = Environment(loader=_YAMLInlineLoader(), undefined=StrictUndefined)


def render_prompt(
    template_path: str | Path | None = None,
    *,
    prompt_ref: str | None = None,
    **context: Any,
) -> list[dict[str, str]]:
    """Load a YAML prompt template or prompt asset and render Jinja2 placeholders.

    Args:
        template_path: Path to the YAML file (absolute, or relative to cwd).
            Mutually exclusive with ``prompt_ref``.
        prompt_ref: Explicit shared prompt asset reference such as
            ``shared.summarize.concise@1``. Mutually exclusive with
            ``template_path``.
        **context: Variables to substitute into Jinja2 templates.

    Returns:
        List of message dicts (OpenAI chat format): [{"role": ..., "content": ...}]

    Raises:
        FileNotFoundError: If the template or prompt asset doesn't exist.
        yaml.YAMLError: If YAML is malformed.
        jinja2.UndefinedError: If a template variable is missing from context.
        ValueError: If the render request or YAML structure is invalid.
    """
    if (template_path is None) == (prompt_ref is None):
        raise ValueError("Provide exactly one of template_path or prompt_ref.")

    if prompt_ref is not None:
        resolved_asset = resolve_prompt_asset(prompt_ref)
        path = resolved_asset.template_path
    else:
        assert template_path is not None  # narrow for type-checking
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
