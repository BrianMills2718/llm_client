"""Context window management for the MCP agent loop.

Provides message-history sizing, compaction of verbose tool outputs, and
proactive clearing of older tool-result payloads to keep prompt size under
the configured cap.
"""

from __future__ import annotations

import json as _json
from typing import Any

from llm_client.foundation import sha256_text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_MESSAGE_CHARS: int = 260_000
"""Soft cap for serialized message-history size; old tool outputs are compacted above this."""

DEFAULT_TOOL_RESULT_KEEP_RECENT: int = 12
"""Keep this many most-recent tool result payloads fully in-context; older payloads are cleared."""

DEFAULT_TOOL_RESULT_CONTEXT_PREVIEW_CHARS: int = 200
"""Chars of preview retained when older tool payloads are cleared from active prompt context."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _message_char_length(message: dict[str, Any]) -> int:
    """Best-effort serialized length for one chat message."""
    try:
        return len(_json.dumps(message, ensure_ascii=False, default=str))
    except Exception:
        return len(str(message))


def _trim_text(value: str, max_chars: int) -> str:
    """Truncate text to max_chars with an ellipsis suffix when shortened."""
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

def _compact_tool_history_for_context(
    messages: list[dict[str, Any]],
    max_message_chars: int,
) -> tuple[int, int, int]:
    """Compact verbose historical tool outputs when message history grows too large.

    Returns (compacted_message_count, chars_saved, resulting_chars).
    """
    if max_message_chars <= 0:
        total = sum(_message_char_length(m) for m in messages if isinstance(m, dict))
        return 0, 0, total

    total_chars = sum(_message_char_length(m) for m in messages if isinstance(m, dict))
    if total_chars <= max_message_chars:
        return 0, 0, total_chars

    target_chars = max(int(max_message_chars * 0.75), 32_000)
    compacted = 0
    saved = 0
    replacement = (
        '{"notice":"Earlier tool result compacted to fit context window. '
        'Re-run the tool if this evidence is needed again."}'
    )
    replacement_len = len(replacement)

    for msg in messages:
        if total_chars <= target_chars:
            break
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or len(content) <= 512:
            continue
        old_len = len(content)
        if old_len <= replacement_len:
            continue
        msg["content"] = replacement
        delta = old_len - replacement_len
        total_chars -= delta
        saved += delta
        compacted += 1

    return compacted, saved, total_chars


def _clear_old_tool_results_for_context(
    messages: list[dict[str, Any]],
    keep_recent: int,
    preview_chars: int,
    tool_result_metadata_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[int, int]:
    """Replace older tool payloads with compact stubs while keeping recent results verbatim.

    This proactively limits prompt growth on long tool traces without losing
    traceability: each cleared payload retains tool_call_id, preview, and hash.

    Returns (cleared_message_count, chars_saved).
    """
    if keep_recent < 0:
        return 0, 0

    tool_indices: list[int] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        if not isinstance(msg.get("content"), str):
            continue
        tool_indices.append(idx)

    if len(tool_indices) <= keep_recent:
        return 0, 0

    keep_set = set(tool_indices[-keep_recent:]) if keep_recent > 0 else set()
    preview_limit = max(40, int(preview_chars))
    cleared = 0
    saved = 0

    for idx in tool_indices:
        if idx in keep_set:
            continue
        msg = messages[idx]
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue
        # Idempotent clearing: do not keep rewriting previously-cleared stubs.
        if '"notice":"Tool result cleared from active context' in content:
            continue

        content_one_line = " ".join(content.strip().split())
        stub = _json.dumps(
            {
                "notice": "Tool result cleared from active context to reduce prompt size.",
                "tool_call_id": str(msg.get("tool_call_id", "")).strip(),
                "content_sha256": sha256_text(content).replace("sha256:", ""),
                "preview": content_one_line[:preview_limit],
                **(
                    tool_result_metadata_by_id.get(str(msg.get("tool_call_id", "")).strip(), {})
                    if isinstance(tool_result_metadata_by_id, dict)
                    else {}
                ),
            },
            ensure_ascii=False,
        )
        old_len = len(content)
        new_len = len(stub)
        msg["content"] = stub
        if old_len > new_len:
            saved += old_len - new_len
        cleared += 1

    return cleared, saved
