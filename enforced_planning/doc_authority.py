"""Authority-drift reconciliation storage and validation.

This module keeps authority drift machine-visible without requiring opportunistic
edits to separately claimed authority surfaces. The first implementation slice
targets indexed authority surfaces such as plan indexes.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
import re
import uuid
from typing import Any

import yaml  # type: ignore[import-untyped]

from enforced_planning import coordination_claims
from enforced_planning.worktree_paths import resolve_canonical_repo_root


DEFAULT_DOC_AUTHORITY_CONFIG = Path("scripts/doc_authority.yaml")
AUTHORITY_OBLIGATIONS_DIR = Path.home() / ".claude" / "coordination" / "authority_obligations"
STATUS_RE = re.compile(r"\*\*Status:\*\*\s*(.+?)(?:\n|$)")
PLAN_NUMBER_RE = re.compile(r"^(\d+)_")


@dataclass(frozen=True)
class AuthorityRule:
    """One configured authority rule for a governed repo."""

    concern: str
    kind: str
    authority_surface: str
    source_glob: str
    resolution_mode: str


@dataclass(frozen=True)
class AuthorityObligation:
    """One unresolved or resolved reconciliation obligation."""

    obligation_id: str
    project: str
    concern: str
    authority_surface: str
    artifact_path: str
    required_action: str
    created_by_agent: str
    created_by_scope: str
    plan_ref: str | None
    owner_scope: str | None
    notes: str | None
    status: str
    created_at: str
    resolved_at: str | None
    source_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe payload."""

        payload = asdict(self)
        payload.pop("source_file", None)
        return payload


@dataclass(frozen=True)
class AuthorityIssue:
    """One validation finding for doc-authority drift."""

    code: str
    severity: str
    concern: str
    authority_surface: str
    artifact_path: str
    message: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe issue payload."""

        return asdict(self)


def _normalize_repo_path(path: str) -> str:
    """Normalize a repo-relative path for overlap checks."""

    normalized = path.replace("\\", "/").strip()
    normalized = normalized.lstrip("./")
    return str(Path(normalized).as_posix())


def _paths_overlap(left: str, right: str) -> bool:
    """Return whether two repo-relative paths overlap."""

    left_norm = _normalize_repo_path(left)
    right_norm = _normalize_repo_path(right)
    return (
        left_norm == right_norm
        or left_norm.startswith(f"{right_norm}/")
        or right_norm.startswith(f"{left_norm}/")
    )


def _parse_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML mapping from disk."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _severity_rank(value: str) -> int:
    """Return deterministic severity sorting order."""

    return {"info": 0, "warn": 1, "fail": 2}.get(value, 2)


def _parse_plan_doc_status(plan_path: Path) -> tuple[str, str] | None:
    """Return one plan number plus normalized status emoji."""

    match = PLAN_NUMBER_RE.match(plan_path.name)
    if not match:
        return None
    plan_number = str(int(match.group(1)))
    status_match = STATUS_RE.search(plan_path.read_text(encoding="utf-8"))
    if not status_match:
        return None
    raw_status = status_match.group(1).strip()
    for emoji in ("✅", "🚧", "📋", "⏸️", "❌"):
        if emoji in raw_status:
            return plan_number, emoji
    lowered = raw_status.lower()
    if "complete" in lowered:
        return plan_number, "✅"
    if "progress" in lowered:
        return plan_number, "🚧"
    if "planned" in lowered:
        return plan_number, "📋"
    if "blocked" in lowered:
        return plan_number, "⏸️"
    if "needs plan" in lowered:
        return plan_number, "❌"
    return plan_number, raw_status


def _parse_plan_index(index_path: Path) -> dict[str, str]:
    """Return plan-number to status-emoji mapping from the plan index table."""

    statuses: dict[str, str] = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) < 4:
            continue
        plan_text = parts[0]
        if not plan_text.isdigit():
            continue
        status_cell = parts[3]
        for emoji in ("✅", "🚧", "📋", "⏸️", "❌"):
            if emoji in status_cell:
                statuses[plan_text] = emoji
                break
    return statuses


def _repo_project_name(repo_root: Path) -> str:
    """Return the canonical project name for a repo or worktree root."""

    return resolve_canonical_repo_root(repo_root).name


def load_authority_rules(repo_root: Path, *, config_path: Path | None = None) -> list[AuthorityRule]:
    """Load doc-authority rules for one repo."""

    canonical_repo_root = resolve_canonical_repo_root(repo_root)
    if config_path is not None:
        resolved_config_path = config_path
    else:
        local_config_path = repo_root / DEFAULT_DOC_AUTHORITY_CONFIG
        canonical_config_path = canonical_repo_root / DEFAULT_DOC_AUTHORITY_CONFIG
        resolved_config_path = local_config_path if local_config_path.exists() else canonical_config_path
    payload = _parse_yaml_mapping(resolved_config_path)
    rules: list[AuthorityRule] = []
    for item in payload.get("indexed_authority_surfaces", []):
        if not isinstance(item, dict):
            continue
        concern = item.get("concern")
        kind = item.get("kind")
        authority_surface = item.get("authority_surface")
        source_glob = item.get("source_glob")
        resolution_mode = item.get("resolution_mode", "manual")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (concern, kind, authority_surface, source_glob, resolution_mode)
        ):
            raise ValueError(f"Invalid doc-authority rule in {resolved_config_path}")
        rules.append(
            AuthorityRule(
                concern=concern.strip(),
                kind=kind.strip(),
                authority_surface=_normalize_repo_path(authority_surface),
                source_glob=source_glob.strip(),
                resolution_mode=resolution_mode.strip(),
            )
        )
    return rules


def list_obligations(
    *,
    project: str | None = None,
    concern: str | None = None,
    status: str | None = None,
) -> list[AuthorityObligation]:
    """Return authority obligations filtered by project/concern/status."""

    if not AUTHORITY_OBLIGATIONS_DIR.exists():
        return []
    obligations: list[AuthorityObligation] = []
    for obligation_file in sorted(AUTHORITY_OBLIGATIONS_DIR.glob("*.yaml")):
        payload = _parse_yaml_mapping(obligation_file)
        record = AuthorityObligation(
            obligation_id=str(payload.get("obligation_id", "")).strip(),
            project=str(payload.get("project", "")).strip(),
            concern=str(payload.get("concern", "")).strip(),
            authority_surface=_normalize_repo_path(str(payload.get("authority_surface", "")).strip()),
            artifact_path=_normalize_repo_path(str(payload.get("artifact_path", "")).strip()),
            required_action=str(payload.get("required_action", "")).strip(),
            created_by_agent=str(payload.get("created_by_agent", "")).strip(),
            created_by_scope=str(payload.get("created_by_scope", "")).strip(),
            plan_ref=str(payload.get("plan_ref", "")).strip() or None,
            owner_scope=str(payload.get("owner_scope", "")).strip() or None,
            notes=str(payload.get("notes", "")).strip() or None,
            status=str(payload.get("status", "open")).strip() or "open",
            created_at=str(payload.get("created_at", "")).strip(),
            resolved_at=str(payload.get("resolved_at", "")).strip() or None,
            source_file=str(obligation_file),
        )
        if not record.obligation_id:
            continue
        if project and record.project != project:
            continue
        if concern and record.concern != concern:
            continue
        if status and record.status != status:
            continue
        obligations.append(record)
    return obligations


def record_obligation(
    *,
    project: str,
    concern: str,
    authority_surface: str,
    artifact_path: str,
    required_action: str,
    created_by_agent: str,
    created_by_scope: str,
    plan_ref: str | None = None,
    owner_scope: str | None = None,
    notes: str | None = None,
) -> AuthorityObligation:
    """Create or return one open authority reconciliation obligation."""

    authority_surface_norm = _normalize_repo_path(authority_surface)
    artifact_path_norm = _normalize_repo_path(artifact_path)
    for existing in list_obligations(project=project, concern=concern, status="open"):
        if (
            existing.authority_surface == authority_surface_norm
            and existing.artifact_path == artifact_path_norm
        ):
            return existing

    now = datetime.now(timezone.utc).isoformat()
    obligation_id = f"{project}-{uuid.uuid4().hex[:12]}"
    record = AuthorityObligation(
        obligation_id=obligation_id,
        project=project,
        concern=concern,
        authority_surface=authority_surface_norm,
        artifact_path=artifact_path_norm,
        required_action=required_action.strip(),
        created_by_agent=created_by_agent.strip(),
        created_by_scope=created_by_scope.strip(),
        plan_ref=plan_ref.strip() if isinstance(plan_ref, str) and plan_ref.strip() else None,
        owner_scope=owner_scope.strip() if isinstance(owner_scope, str) and owner_scope.strip() else None,
        notes=notes.strip() if isinstance(notes, str) and notes.strip() else None,
        status="open",
        created_at=now,
        resolved_at=None,
    )
    AUTHORITY_OBLIGATIONS_DIR.mkdir(parents=True, exist_ok=True)
    obligation_path = AUTHORITY_OBLIGATIONS_DIR / f"{obligation_id}.yaml"
    obligation_path.write_text(
        yaml.safe_dump(record.to_dict(), default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return AuthorityObligation(**{**record.to_dict(), "source_file": str(obligation_path)})


def resolve_obligation(*, obligation_id: str, notes: str | None = None) -> AuthorityObligation:
    """Mark one obligation resolved and return the updated record."""

    obligation_path = AUTHORITY_OBLIGATIONS_DIR / f"{obligation_id}.yaml"
    if not obligation_path.exists():
        raise ValueError(f"Unknown authority obligation: {obligation_id}")
    payload = _parse_yaml_mapping(obligation_path)
    payload["status"] = "resolved"
    payload["resolved_at"] = datetime.now(timezone.utc).isoformat()
    if notes:
        payload["notes"] = notes
    obligation_path.write_text(
        yaml.safe_dump(payload, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return AuthorityObligation(**{**payload, "source_file": str(obligation_path)})


def _owner_claims_for_surface(project: str, authority_surface: str) -> list[coordination_claims.ClaimRecord]:
    """Return live write claims that own one authority surface."""

    claims = coordination_claims.check_claims(project)
    owners: list[coordination_claims.ClaimRecord] = []
    for claim in claims:
        if claim.claim_type != "write":
            continue
        if any(_paths_overlap(authority_surface, write_path) for write_path in claim.write_paths):
            owners.append(claim)
    return owners


def _matching_open_obligations(
    *,
    project: str,
    concern: str,
    authority_surface: str,
    artifact_path: str,
) -> list[AuthorityObligation]:
    """Return open obligations for one concrete drift item."""

    matches: list[AuthorityObligation] = []
    for obligation in list_obligations(project=project, concern=concern, status="open"):
        if (
            obligation.authority_surface == _normalize_repo_path(authority_surface)
            and obligation.artifact_path == _normalize_repo_path(artifact_path)
        ):
            matches.append(obligation)
    return matches


def _issue(
    *,
    code: str,
    severity: str,
    concern: str,
    authority_surface: str,
    artifact_path: str,
    message: str,
    evidence: dict[str, Any],
) -> AuthorityIssue:
    """Build one authority validation issue."""

    return AuthorityIssue(
        code=code,
        severity=severity,
        concern=concern,
        authority_surface=_normalize_repo_path(authority_surface),
        artifact_path=_normalize_repo_path(artifact_path),
        message=message,
        evidence=evidence,
    )


def _validate_plan_index_rule(repo_root: Path, rule: AuthorityRule) -> list[AuthorityIssue]:
    """Validate one indexed plan authority surface."""

    canonical_repo_root = resolve_canonical_repo_root(repo_root)
    authority_surface_path = canonical_repo_root / rule.authority_surface
    if not authority_surface_path.exists():
        raise ValueError(f"Authority surface does not exist: {authority_surface_path}")
    index_statuses = _parse_plan_index(authority_surface_path)
    issues: list[AuthorityIssue] = []
    project = _repo_project_name(canonical_repo_root)
    for plan_path in sorted(canonical_repo_root.glob(rule.source_glob)):
        parsed = _parse_plan_doc_status(plan_path)
        if parsed is None:
            continue
        plan_number, file_status = parsed
        artifact_path = _normalize_repo_path(str(plan_path.relative_to(canonical_repo_root)))
        issue_code: str | None = None
        issue_message: str | None = None
        if plan_number not in index_statuses:
            issue_code = "authority_surface_missing_artifact"
            issue_message = (
                f"{rule.authority_surface} does not index {artifact_path}."
            )
        elif index_statuses[plan_number] != file_status:
            issue_code = "authority_surface_status_mismatch"
            issue_message = (
                f"{rule.authority_surface} reports Plan #{plan_number} as {index_statuses[plan_number]} "
                f"but {artifact_path} is {file_status}."
            )
        if issue_code is None or issue_message is None:
            continue

        owners = _owner_claims_for_surface(project, rule.authority_surface)
        obligations = _matching_open_obligations(
            project=project,
            concern=rule.concern,
            authority_surface=rule.authority_surface,
            artifact_path=artifact_path,
        )
        evidence = {
            "plan_number": plan_number,
            "authority_surface": rule.authority_surface,
            "artifact_path": artifact_path,
            "owner_scopes": [claim.scope for claim in owners],
            "obligation_ids": [obligation.obligation_id for obligation in obligations],
            "resolution_mode": rule.resolution_mode,
        }
        if rule.resolution_mode == "generated":
            issues.append(
                _issue(
                    code="generated_authority_surface_requires_regeneration",
                    severity="fail",
                    concern=rule.concern,
                    authority_surface=rule.authority_surface,
                    artifact_path=artifact_path,
                    message=(
                        f"{issue_message} This surface is configured as generated, so regenerate it instead of leaving drift."
                    ),
                    evidence=evidence,
                )
            )
            continue
        if obligations:
            issues.append(
                _issue(
                    code="recorded_reconciliation_obligation",
                    severity="info",
                    concern=rule.concern,
                    authority_surface=rule.authority_surface,
                    artifact_path=artifact_path,
                    message=(
                        f"{issue_message} Drift is recorded as reconciliation obligation "
                        f"{obligations[0].obligation_id}."
                    ),
                    evidence=evidence,
                )
            )
            continue
        if owners:
            issues.append(
                _issue(
                    code="missing_reconciliation_obligation",
                    severity="fail",
                    concern=rule.concern,
                    authority_surface=rule.authority_surface,
                    artifact_path=artifact_path,
                    message=(
                        f"{issue_message} Authority surface owner(s) exist, so the landing lane "
                        "must record a reconciliation obligation instead of leaving silent drift."
                    ),
                    evidence=evidence,
                )
            )
            continue
        issues.append(
            _issue(
                code="unowned_authority_drift",
                severity="fail",
                concern=rule.concern,
                authority_surface=rule.authority_surface,
                artifact_path=artifact_path,
                message=(
                    f"{issue_message} No active claim owns {rule.authority_surface}, so the drift is unowned."
                ),
                evidence=evidence,
            )
        )
    return issues


def validate_doc_authority(
    repo_root: Path,
    *,
    config_path: Path | None = None,
) -> list[AuthorityIssue]:
    """Return sorted authority validation issues for one repo."""

    issues: list[AuthorityIssue] = []
    for rule in load_authority_rules(repo_root, config_path=config_path):
        if rule.kind == "plan_index":
            issues.extend(_validate_plan_index_rule(repo_root, rule))
            continue
        raise ValueError(f"Unsupported authority rule kind: {rule.kind}")
    return sorted(
        issues,
        key=lambda item: (
            -_severity_rank(item.severity),
            item.concern,
            item.artifact_path,
        ),
    )


def unresolved_owned_obligations(claim: coordination_claims.ClaimRecord) -> list[AuthorityObligation]:
    """Return unresolved obligations against surfaces owned by one claim."""

    project = claim.primary_project()
    if not project or not claim.write_paths:
        return []
    obligations = list_obligations(project=project, status="open")
    matches: list[AuthorityObligation] = []
    for obligation in obligations:
        if any(_paths_overlap(obligation.authority_surface, write_path) for write_path in claim.write_paths):
            matches.append(obligation)
    return matches


def assert_no_unresolved_owned_obligations(claim: coordination_claims.ClaimRecord) -> None:
    """Fail loud when one lane owns an authority surface with open debt."""

    matches = unresolved_owned_obligations(claim)
    if not matches:
        return
    formatted = "; ".join(
        f"{item.concern}::{item.authority_surface} <- {item.artifact_path} ({item.obligation_id})"
        for item in matches
    )
    raise ValueError(
        "Cannot close this lane because it owns authority surfaces with unresolved "
        f"reconciliation obligations: {formatted}"
    )
