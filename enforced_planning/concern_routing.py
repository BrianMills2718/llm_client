"""Review-claim creation and concern routing for coordination-aware review work.

These helpers make review intent visible in the claim registry and route
concerns to the most durable surface available: PR comments once a branch is
published, otherwise the local inbox channel.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from enforced_planning import coordination_claims
from enforced_planning import push_safety


@dataclass(frozen=True)
class ConcernRoute:
    """Describe where one concern was delivered."""

    route: str
    destination: str
    target_branch: str
    recipient: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for CLI use."""

        return asdict(self)


def _split_path_values(raw_values: list[str] | None) -> list[str]:
    """Normalize repeated delimiter-packed path lists into stable write paths."""

    values: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values or []:
        for chunk in raw_value.replace(",", "|").replace(";", "|").split("|"):
            text = coordination_claims._normalize_repo_path(chunk)
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
    return values


def _candidate_review_scope(target_branch: str) -> str:
    """Derive a stable review-claim scope for one target branch."""

    return f"review-{target_branch}"


def _current_branch(repo_root: Path) -> str | None:
    """Return the current branch when HEAD is attached, else ``None``."""

    try:
        return push_safety.current_branch(repo_root)
    except RuntimeError:
        return None


def create_review_claim(
    *,
    repo_root: str | Path,
    agent: str,
    project: str,
    target_branch: str,
    intent: str,
    write_paths: list[str],
    plan_ref: str | None = None,
    ttl_hours: float = coordination_claims.DEFAULT_TTL_HOURS,
    notes: str | None = None,
    scope: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Create a review claim that makes cross-lane inspection visible."""

    resolved_repo_root = push_safety.resolve_repo_root(repo_root)
    normalized_write_paths = _split_path_values(write_paths)
    if not normalized_write_paths:
        raise ValueError("Review claims require at least one write path.")
    current_branch = _current_branch(resolved_repo_root)
    resolved_scope = scope or _candidate_review_scope(target_branch)
    ok, message = coordination_claims.create_claim(
        agent=agent,
        project=project,
        scope=resolved_scope,
        intent=intent,
        plan_ref=plan_ref,
        ttl_hours=ttl_hours,
        claim_type="review",
        write_paths=normalized_write_paths,
        branch=current_branch,
        worktree_path=str(resolved_repo_root),
        parent_scope=target_branch,
        session_id=session_id,
        notes=notes,
    )
    if not ok:
        raise ValueError(message)
    return {
        "ok": True,
        "message": message,
        "scope": resolved_scope,
        "target_branch": target_branch,
        "write_paths": normalized_write_paths,
    }


def _run_gh(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one GitHub CLI command and capture stdout/stderr without guessing."""

    return subprocess.run(
        ["gh", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )


def _open_pr_for_branch(repo_root: Path, branch: str) -> dict[str, Any] | None:
    """Return the first open PR for a head branch when one exists."""

    result = _run_gh(repo_root, ["pr", "list", "--head", branch, "--state", "open", "--json", "number,url,headRefName"])
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "gh pr list failed")
    payload = json.loads(result.stdout or "[]")
    if not isinstance(payload, list):
        raise RuntimeError("gh pr list must return a JSON array")
    if not payload:
        return None
    pr = payload[0]
    if not isinstance(pr, dict):
        raise RuntimeError("gh pr list returned an invalid PR record")
    return pr


def _best_recipient(project: str, target_branch: str, explicit_recipient: str | None) -> str:
    """Choose the local inbox recipient for a concern that has no PR yet."""

    if explicit_recipient:
        return explicit_recipient
    matching_claims = [
        claim
        for claim in coordination_claims.check_claims(project)
        if claim.branch == target_branch
    ]
    for claim in matching_claims:
        if claim.session_name:
            return claim.session_name
    for claim in matching_claims:
        if claim.scope:
            return claim.scope
    return target_branch


def _message_sender(repo_root: Path, agent: str) -> str:
    """Derive a stable sender label for local inbox messages."""

    return _current_branch(repo_root) or coordination_claims.resolve_session_id(agent) or agent


def _message_id(sender: str, timestamp: datetime) -> str:
    """Return a compact unique message identifier."""

    payload = f"{sender}-{timestamp.isoformat()}-{os.urandom(4).hex()}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"msg-{timestamp.strftime('%Y%m%d-%H%M%S')}-{digest}"


def _write_local_message(
    *,
    repo_root: Path,
    sender: str,
    recipient: str,
    subject: str,
    content: str,
) -> Path:
    """Persist a concern to the repo-local inbox using the existing message format."""

    timestamp = datetime.now(timezone.utc)
    inbox_dir = repo_root / ".claude" / "messages" / "inbox" / recipient
    inbox_dir.mkdir(parents=True, exist_ok=True)
    path = inbox_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_from-{sender}_suggestion.md"
    message = "\n".join(
        [
            "---",
            f"id: {_message_id(sender, timestamp)}",
            f"from: {sender}",
            f"to: {recipient}",
            f"timestamp: {timestamp.isoformat().replace('+00:00', 'Z')}",
            "type: suggestion",
            f"subject: {subject}",
            "status: unread",
            "---",
            "",
            "## Content",
            "",
            content,
            "",
            "## Requested Action",
            "",
            "- [ ] Review the concern",
            "- [ ] Reply or update the lane state if it changes direction",
            "",
        ]
    )
    path.write_text(message, encoding="utf-8")
    return path


def route_concern(
    *,
    repo_root: str | Path,
    agent: str,
    project: str,
    target_branch: str,
    subject: str,
    content: str,
    recipient: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Route a concern to PR comment when published, otherwise to local inbox."""

    resolved_repo_root = push_safety.resolve_repo_root(repo_root)
    pr = _open_pr_for_branch(resolved_repo_root, target_branch)
    if pr is not None:
        pr_number = pr.get("number")
        pr_url = pr.get("url")
        if not isinstance(pr_number, int) or not isinstance(pr_url, str):
            raise RuntimeError("Open PR record is missing number/url")
        pr_body = f"## {subject}\n\n{content}"
        if dry_run:
            route = ConcernRoute(
                route="pr_comment",
                destination=pr_url,
                target_branch=target_branch,
                recipient=str(pr_number),
            )
            return {"ok": True, **route.to_dict()}
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write(pr_body)
            temp_path = handle.name
        try:
            result = _run_gh(resolved_repo_root, ["pr", "comment", str(pr_number), "--body-file", temp_path])
        finally:
            Path(temp_path).unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip() or "gh pr comment failed")
        route = ConcernRoute(
            route="pr_comment",
            destination=pr_url,
            target_branch=target_branch,
            recipient=str(pr_number),
        )
        return {"ok": True, **route.to_dict()}

    resolved_recipient = _best_recipient(project, target_branch, recipient)
    sender = _message_sender(resolved_repo_root, agent)
    if dry_run:
        route = ConcernRoute(
            route="local_message",
            destination=str(resolved_repo_root / ".claude" / "messages" / "inbox" / resolved_recipient),
            target_branch=target_branch,
            recipient=resolved_recipient,
        )
        return {"ok": True, **route.to_dict()}
    message_path = _write_local_message(
        repo_root=resolved_repo_root,
        sender=sender,
        recipient=resolved_recipient,
        subject=subject,
        content=content,
    )
    route = ConcernRoute(
        route="local_message",
        destination=str(message_path),
        target_branch=target_branch,
        recipient=resolved_recipient,
    )
    return {"ok": True, **route.to_dict()}
