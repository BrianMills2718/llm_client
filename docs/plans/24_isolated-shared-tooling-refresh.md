# Plan #24: isolated shared-tooling refresh

**Status:** In Progress
**Type:** implementation  <!-- implementation | design -->
**Priority:** High
**Blocked By:** None
**Blocks:** None

---

## Gap

**Current:** Canonical `llm_client` is governed but behind the authoritative shared plan-coordination/tooling surface. The canonical root is also dirty in unrelated runtime/agent code and tests, so the sanctioned refresh cannot be applied there blindly. This clean worktree starts from the same `HEAD` but without the unrelated canonical dirt, which makes it the right place to prepare and verify the shared-tooling refresh slice in isolation.

**Target:** Apply the sanctioned shared-tooling refresh in this clean worktree, verify the refreshed slice mechanically, clear stale-tooling warnings here, and produce an exact patch set for the cross-repo Plan 65 landing-safety comparison against canonical `llm_client`.

**Why:** `llm_client` is shared infrastructure. Leaving it behind the authoritative coordination/runtime tooling creates ecosystem drift, but landing into the dirty canonical root without measured overlap would violate the same governed landing-safety policy now enforced elsewhere.

---

## References Reviewed

> **REQUIRED:** Cite specific code/docs reviewed before planning.

- `CLAUDE.md` - repo-local workflow and shared-infrastructure boundary
- `docs/plans/CLAUDE.md` - local plan namespace and completed Plan 23 authoritative coordination rollout
- `docs/plans/23_authoritative-coordination-wave-1-rollout.md` - existing governed baseline
- `/home/brian/projects/project-meta_worktrees/plan-65-llm-client-isolated-refresh/docs/plans/65_llm-client-isolated-shared-tooling-refresh-and-blocker-reconciliation.md` - cross-repo umbrella plan and landing-safety criteria
- `/home/brian/projects/project-meta_worktrees/plan-65-llm-client-isolated-refresh/scripts/meta/install_governed_repo.py` - authoritative sanctioned refresh surface
- `/home/brian/projects/project-meta_worktrees/plan-65-llm-client-isolated-refresh/scripts/meta/audit_governed_repo.py` - authoritative stale-tooling/governed audit surface
- `git status --short --branch` from canonical `~/projects/llm_client` - dirty-root blocker evidence
- `CLAUDE.md` - project conventions

---

## Files Affected

> **REQUIRED:** Declare upfront what files will be touched.

- docs/plans/24_isolated-shared-tooling-refresh.md (modify)
- docs/plans/CLAUDE.md (modify)
- scripts/check_truth_surface_drift.py (create)
- scripts/render_truth_surface_status.py (create)
- scripts/check_markdown_links.py (modify)
- scripts/meta/check_agents_sync.py (modify)
- scripts/meta/check_coordination_claims.py (modify)
- scripts/meta/create_plan.py (modify)
- scripts/meta/file_context.py (modify)
- scripts/meta/plan_reservations.py (modify)
- scripts/meta/render_agents_md.py (modify)
- scripts/meta/validate_plan.py (modify)
- scripts/meta/worktree-coordination/check_claims.py (modify)
- scripts/meta/worktree-coordination/safe_worktree_remove.py (modify)
- AGENTS.md (modify if sync required by post-refresh audit)

---

## Plan

### Steps

1. Run the authoritative governed audit against this clean worktree to confirm the pre-refresh stale-tooling/AGENTS state.
2. Apply the sanctioned refresh from authoritative `project-meta` using `install_governed_repo.py`.
3. Re-run the worktree governed audit and mechanical checks (`git diff --check`, `py_compile`, `check_agents_sync.py --check`).
4. If AGENTS drift remains after the refresh, regenerate/sync `AGENTS.md` in this worktree and re-run the audit.
5. Commit the isolated refresh slice so Plan 65 can compare its exact patch footprint against the dirty canonical root.

---

## Required Tests

### Existing Verification (Must Pass)

| Command | Why |
|---------|-----|
| `git diff --check` | No malformed patch or whitespace regression in the refreshed slice |
| `python -m py_compile scripts/check_markdown_links.py scripts/check_truth_surface_drift.py scripts/render_truth_surface_status.py scripts/meta/check_agents_sync.py scripts/meta/check_coordination_claims.py scripts/meta/create_plan.py scripts/meta/file_context.py scripts/meta/plan_reservations.py scripts/meta/render_agents_md.py scripts/meta/validate_plan.py scripts/meta/worktree-coordination/check_claims.py scripts/meta/worktree-coordination/safe_worktree_remove.py` | Refreshed Python tooling parses cleanly |
| `python scripts/meta/check_agents_sync.py --check` | Local AGENTS mirror is truthful if touched |
| `python /home/brian/projects/project-meta_worktrees/plan-65-llm-client-isolated-refresh/scripts/meta/audit_governed_repo.py --repo-root . --json` | Worktree governed/stale-tooling state is measured with the authoritative audit surface |

---

## Acceptance Criteria

- [ ] The clean worktree reproduces the expected pre-refresh governed/stale-tooling state.
- [ ] The sanctioned refresh is applied in the worktree.
- [ ] Worktree governed audit clears stale-tooling warnings or narrows residual issues to explicitly classified non-stale signals.
- [ ] Mechanical verification commands pass in the refreshed worktree.
- [ ] `AGENTS.md` is in sync if the post-refresh audit requires it.
- [ ] A committed isolated refresh patch exists for Plan 65 landing-safety comparison.

---

## Notes

- This plan intentionally does not decide canonical replay. That decision belongs to `project-meta` Plan 65 after measured overlap comparison against the dirty canonical root.
- If this worktree refresh succeeds but canonical replay is blocked, this worktree commit is still the rollback-safe success artifact for the isolated slice.
