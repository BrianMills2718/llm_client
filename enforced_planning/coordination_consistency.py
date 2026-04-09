"""Check whether live coordination claims, registry outputs, and git worktrees agree."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from enforced_planning import active_work_registry
from enforced_planning import coordination_claims


DEFAULT_WORKSPACE_ROOT = Path.home() / "projects"


@dataclass(frozen=True)
class WorktreeRecord:
    """One linked git worktree plus the minimal state needed for consistency checks."""

    repo: str
    repo_root: str
    path: str
    branch: str | None
    head: str | None
    detached: bool
    bare: bool
    prunable_reason: str | None
    exists_on_disk: bool
    is_main_worktree: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe worktree record."""
        return asdict(self)


@dataclass(frozen=True)
class ConsistencyIssue:
    """One concrete consistency finding from the coordination-state audit."""

    severity: str
    code: str
    repo: str | None
    message: str
    claim_scope: str | None = None
    claim_agent: str | None = None
    worktree_path: str | None = None
    branch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe issue record."""
        return asdict(self)


def run_git_worktree_list(repo_root: Path) -> str:
    """Return ``git worktree list --porcelain`` output for one repo."""
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Unable to list worktrees for {repo_root}: {(result.stderr or result.stdout).strip()}"
        )
    return result.stdout


def parse_worktree_list(*, repo: str, repo_root: Path, porcelain: str) -> list[WorktreeRecord]:
    """Parse porcelain worktree output into deterministic records."""
    records: list[WorktreeRecord] = []
    current: dict[str, Any] | None = None

    def flush(record: dict[str, Any] | None) -> None:
        if record is None:
            return
        path = Path(record["path"])
        records.append(
            WorktreeRecord(
                repo=repo,
                repo_root=str(repo_root),
                path=str(path),
                branch=record.get("branch"),
                head=record.get("head"),
                detached=bool(record.get("detached", False)),
                bare=bool(record.get("bare", False)),
                prunable_reason=record.get("prunable_reason"),
                exists_on_disk=path.exists(),
                is_main_worktree=path.resolve() == repo_root.resolve(),
            )
        )

    for raw_line in porcelain.splitlines():
        if raw_line.startswith("worktree "):
            flush(current)
            current = {"path": raw_line[9:]}
            continue
        if current is None:
            continue
        if raw_line.startswith("HEAD "):
            current["head"] = raw_line[5:]
        elif raw_line.startswith("branch "):
            current["branch"] = raw_line[7:].removeprefix("refs/heads/")
        elif raw_line == "detached":
            current["detached"] = True
        elif raw_line == "bare":
            current["bare"] = True
        elif raw_line.startswith("prunable "):
            current["prunable_reason"] = raw_line[9:]

    flush(current)
    return records


def load_repo_worktrees(repo_roots: dict[str, Path]) -> list[WorktreeRecord]:
    """Load linked git worktrees for each repo in scope."""
    records: list[WorktreeRecord] = []
    for repo, repo_root in repo_roots.items():
        records.extend(parse_worktree_list(repo=repo, repo_root=repo_root, porcelain=run_git_worktree_list(repo_root)))
    return records


def _canonicalize_registry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop volatile fields before comparing two registry payloads."""
    canonical = dict(payload)
    canonical.pop("generated_at_utc", None)
    return canonical


def _normalize_registry_markdown(text: str) -> str:
    """Drop volatile timestamp lines before comparing rendered markdown."""
    lines = [line for line in text.splitlines() if not line.startswith("Generated: `")]
    return "\n".join(lines).strip() + "\n"


def _claim_key(claim: coordination_claims.ClaimRecord) -> tuple[str, str | None, str | None]:
    """Return the stable key used to match claims against worktrees."""
    return (claim.agent, claim.scope, claim.primary_project())


def _worktree_matches_claim(
    claim: coordination_claims.ClaimRecord,
    worktree: WorktreeRecord,
) -> bool:
    """Return whether a worktree plausibly corresponds to one live claim."""
    if claim.worktree_path:
        try:
            if Path(claim.worktree_path).expanduser().resolve() == Path(worktree.path).resolve():
                return True
        except FileNotFoundError:
            pass
    if claim.branch and claim.branch == worktree.branch:
        return True
    return False


def build_consistency_report(
    *,
    claims: list[coordination_claims.ClaimRecord],
    repo_roots: dict[str, Path],
    registry_json_path: Path | None = None,
    registry_markdown_path: Path | None = None,
) -> dict[str, Any]:
    """Return a structured coordination consistency report."""
    worktrees = load_repo_worktrees(repo_roots)
    worktrees_by_repo: dict[str, list[WorktreeRecord]] = {}
    for worktree in worktrees:
        worktrees_by_repo.setdefault(worktree.repo, []).append(worktree)

    expected_registry = active_work_registry.build_registry_payload(claims=claims)
    issues: list[ConsistencyIssue] = []
    matched_worktrees: set[tuple[str, str]] = set()

    for claim in claims:
        project = claim.primary_project()
        if not project:
            issues.append(
                ConsistencyIssue(
                    severity="hard",
                    code="claim-missing-project",
                    repo=None,
                    message=f"Claim {claim.agent}:{claim.scope} has no primary project.",
                    claim_scope=claim.scope,
                    claim_agent=claim.agent,
                    branch=claim.branch,
                    worktree_path=claim.worktree_path,
                )
            )
            continue
        if project not in repo_roots:
            issues.append(
                ConsistencyIssue(
                    severity="hard",
                    code="claim-project-out-of-scope",
                    repo=project,
                    message=f"Claim {claim.agent}:{claim.scope} points at repo '{project}', which is not in the consistency scope.",
                    claim_scope=claim.scope,
                    claim_agent=claim.agent,
                    branch=claim.branch,
                    worktree_path=claim.worktree_path,
                )
            )
            continue

        repo_worktrees = worktrees_by_repo.get(project, [])
        claim_match = next((wt for wt in repo_worktrees if _worktree_matches_claim(claim, wt)), None)

        if claim.worktree_path:
            claim_path = Path(claim.worktree_path).expanduser()
            if not claim_path.exists():
                issues.append(
                    ConsistencyIssue(
                        severity="hard",
                        code="claim-worktree-missing",
                        repo=project,
                        message=f"Claim {claim.agent}:{claim.scope} references missing worktree path {claim.worktree_path}.",
                        claim_scope=claim.scope,
                        claim_agent=claim.agent,
                        branch=claim.branch,
                        worktree_path=claim.worktree_path,
                    )
                )
            elif claim_match is None:
                issues.append(
                    ConsistencyIssue(
                        severity="hard",
                        code="claim-worktree-unlinked",
                        repo=project,
                        message=f"Claim {claim.agent}:{claim.scope} points at {claim.worktree_path}, but git does not list that path as a linked worktree for {project}.",
                        claim_scope=claim.scope,
                        claim_agent=claim.agent,
                        branch=claim.branch,
                        worktree_path=claim.worktree_path,
                    )
                )

        if claim.branch and claim_match is None:
            issues.append(
                ConsistencyIssue(
                    severity="hard",
                    code="claim-branch-unlinked",
                    repo=project,
                    message=f"Claim {claim.agent}:{claim.scope} references branch {claim.branch}, but no linked worktree in {project} is attached to that branch.",
                    claim_scope=claim.scope,
                    claim_agent=claim.agent,
                    branch=claim.branch,
                    worktree_path=claim.worktree_path,
                )
            )

        if claim_match is not None:
            matched_worktrees.add((claim_match.repo, claim_match.path))
            if claim.branch and claim_match.branch and claim.branch != claim_match.branch:
                issues.append(
                    ConsistencyIssue(
                        severity="hard",
                        code="claim-branch-mismatch",
                        repo=project,
                        message=f"Claim {claim.agent}:{claim.scope} says branch {claim.branch}, but linked worktree {claim_match.path} is on {claim_match.branch}.",
                        claim_scope=claim.scope,
                        claim_agent=claim.agent,
                        branch=claim.branch,
                        worktree_path=claim_match.path,
                    )
                )

    for worktree in worktrees:
        if worktree.prunable_reason:
            issues.append(
                ConsistencyIssue(
                    severity="hard",
                    code="worktree-prunable",
                    repo=worktree.repo,
                    message=f"Linked worktree {worktree.path} is prunable/broken: {worktree.prunable_reason}",
                    branch=worktree.branch,
                    worktree_path=worktree.path,
                )
            )
            continue
        if not worktree.exists_on_disk:
            issues.append(
                ConsistencyIssue(
                    severity="hard",
                    code="worktree-missing-on-disk",
                    repo=worktree.repo,
                    message=f"Linked worktree {worktree.path} does not exist on disk.",
                    branch=worktree.branch,
                    worktree_path=worktree.path,
                )
            )
            continue
        if worktree.is_main_worktree:
            continue
        if (worktree.repo, worktree.path) not in matched_worktrees:
            issues.append(
                ConsistencyIssue(
                    severity="warning",
                    code="worktree-unclaimed",
                    repo=worktree.repo,
                    message=f"Linked worktree {worktree.path} has no matching live v2 claim.",
                    branch=worktree.branch,
                    worktree_path=worktree.path,
                )
            )

    if registry_json_path is not None:
        if not registry_json_path.exists():
            issues.append(
                ConsistencyIssue(
                    severity="hard",
                    code="registry-json-missing",
                    repo=None,
                    message=f"Registry JSON file is missing: {registry_json_path}",
                )
            )
        else:
            actual_json = json.loads(registry_json_path.read_text(encoding="utf-8"))
            if _canonicalize_registry_payload(actual_json) != _canonicalize_registry_payload(expected_registry):
                issues.append(
                    ConsistencyIssue(
                        severity="hard",
                        code="registry-json-drift",
                        repo=None,
                        message=f"Registry JSON {registry_json_path} is out of sync with the live claim state.",
                    )
                )

    if registry_markdown_path is not None:
        if not registry_markdown_path.exists():
            issues.append(
                ConsistencyIssue(
                    severity="hard",
                    code="registry-markdown-missing",
                    repo=None,
                    message=f"Registry markdown file is missing: {registry_markdown_path}",
                )
            )
        else:
            actual_markdown = registry_markdown_path.read_text(encoding="utf-8")
            expected_markdown = active_work_registry.render_markdown(expected_registry)
            if _normalize_registry_markdown(actual_markdown) != _normalize_registry_markdown(expected_markdown):
                issues.append(
                    ConsistencyIssue(
                        severity="hard",
                        code="registry-markdown-drift",
                        repo=None,
                        message=f"Registry markdown {registry_markdown_path} is out of sync with the live claim state.",
                    )
                )

    return {
        "workspace_root": str(next(iter(repo_roots.values())).parent if repo_roots else DEFAULT_WORKSPACE_ROOT),
        "repos": {name: str(path) for name, path in sorted(repo_roots.items())},
        "claim_count": len(claims),
        "worktree_count": len(worktrees),
        "issues": [issue.to_dict() for issue in issues],
        "hard_issue_count": sum(1 for issue in issues if issue.severity == "hard"),
        "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
        "claims": [claim.to_dict() for claim in claims],
        "worktrees": [worktree.to_dict() for worktree in worktrees],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for the coordination consistency checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-root",
        default=str(DEFAULT_WORKSPACE_ROOT),
        help=f"Workspace root containing the repo directories (default: {DEFAULT_WORKSPACE_ROOT})",
    )
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        help="Repo name under the workspace root to include in the consistency check. Repeat as needed.",
    )
    parser.add_argument(
        "--claims-dir",
        help="Override the live coordination claims directory instead of ~/.claude/coordination/claims/.",
    )
    parser.add_argument(
        "--registry-json",
        help="Optional registry JSON file to compare against the live claim state.",
    )
    parser.add_argument(
        "--registry-markdown",
        help="Optional registry markdown file to compare against the live claim state.",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the coordination consistency checker."""
    args = parse_args(argv)
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    repo_roots = {repo: (workspace_root / repo).resolve() for repo in args.repo}

    if args.claims_dir:
        coordination_claims.CLAIMS_DIR = Path(args.claims_dir).expanduser().resolve()

    claims = coordination_claims.check_claims()
    report = build_consistency_report(
        claims=claims,
        repo_roots=repo_roots,
        registry_json_path=Path(args.registry_json).expanduser().resolve() if args.registry_json else None,
        registry_markdown_path=Path(args.registry_markdown).expanduser().resolve() if args.registry_markdown else None,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("Coordination consistency report")
        print(f"Repos checked: {', '.join(sorted(report['repos']))}")
        print(f"Live claims: {report['claim_count']}")
        print(f"Linked worktrees: {report['worktree_count']}")
        print(f"Hard issues: {report['hard_issue_count']}")
        print(f"Warnings: {report['warning_count']}")
        if report["issues"]:
            print("")
            for issue in report["issues"]:
                label = "ERROR" if issue["severity"] == "hard" else "WARN"
                repo = issue["repo"] or "-"
                print(f"[{label}] {issue['code']} ({repo}) {issue['message']}")
        else:
            print("No consistency issues detected.")

    return 1 if report["hard_issue_count"] else 0
