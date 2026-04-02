# Plan #23: Authoritative coordination wave-1 rollout

**Status:** ✅ Complete

**Verified:** 2026-04-02T11:52:32Z
**Verification Evidence:**
```yaml
completed_by: scripts/complete_plan.py
timestamp: 2026-04-02T11:52:32Z
tests:
  unit: 1246 passed, 3 skipped, 1 deselected, 9 warnings in 62.53s (0:01:02)
  e2e_smoke: skipped (--skip-e2e)
  e2e_real: skipped (--skip-real-e2e)
  doc_coupling: passed
commit: 10c13e7
```
**Type:** implementation
**Priority:** Critical
**Blocked By:** None
**Blocks:** truthful local coordination rollout in `prompt_eval`

---

## Gap

**Current:** `llm_client` is already mechanically governed and worktree-ready,
but it still lacked the sanctioned local plan-coordination entrypoints required
by the authoritative coordination rollout:

- `scripts/meta/check_coordination_claims.py`
- `scripts/meta/create_plan.py`
- `scripts/meta/plan_reservations.py`

The measured governed audit already passed otherwise, so the remaining gap is
deliberately narrow.

**Target:** `llm_client` exposes the sanctioned local plan-coordination
entrypoints, records this rollout under a local plan, and finishes with the
same governed status plus no plan-coordination warnings.

**Why:** `llm_client` is a central shared-infrastructure repo. If the
authoritative coordination contract cannot land here cleanly, the rollout is
not ready for broader ecosystem use.

---

## References Reviewed

- `CLAUDE.md` - local governance and commands
- `meta-process.yaml` - local coordination expectation and capability ownership
- `docs/plans/CLAUDE.md` - local plan index and numbering
- `scripts/meta/check_agents_sync.py` - local AGENTS sync surface already present
- `/home/brian/projects/project-meta_worktrees/plan-58-authoritative-registry-rollout/docs/plans/60_wave-1-authoritative-coordination-adoption-and-digimon-prerequisite-remediation.md`
  - cross-repo rollout contract driving this local slice
- `/home/brian/projects/project-meta_worktrees/plan-58-authoritative-registry-rollout/scripts/meta/install_governed_repo.py`
  - sanctioned narrow bootstrap path
- `python /home/brian/projects/project-meta_worktrees/plan-58-authoritative-registry-rollout/scripts/meta/audit_governed_repo.py --repo-root /home/brian/projects/llm_client_worktrees/plan-60-llm-client-coordination --json`
  - pre-bootstrap state: governed, but all three plan-coordination scripts missing
- same audit after bootstrap
  - post-bootstrap state: governed, no plan-coordination warnings

---

## Files Affected

- `docs/plans/23_authoritative-coordination-wave-1-rollout.md` (create/modify)
- `docs/plans/CLAUDE.md` (modify)
- `scripts/meta/check_coordination_claims.py` (create)
- `scripts/meta/complete_plan.py` (modify if closeout tooling is not venv-safe)
- `scripts/meta/create_plan.py` (create)
- `scripts/meta/plan_reservations.py` (create)

---

## Plan

### Steps

1. Bootstrap the sanctioned local plan-coordination entrypoints via the narrow
   installer mode.
2. Create this repo-local plan immediately after bootstrap and confirm the diff
   stays bounded to the three scripts plus plan/index files.
3. Re-run governed audit and local coordination smoke checks.
4. Commit the rollout as one rollback point with explicit verification evidence.
5. If formal closeout is blocked by non-venv-safe plan tooling, fix that
   tooling in-repo, rerun closeout, and keep the change bounded to the local
   coordination surface.

---

## Required Tests

### Existing Checks (Must Pass)

| Command | Why |
|---------|-----|
| `python /home/brian/projects/project-meta_worktrees/plan-58-authoritative-registry-rollout/scripts/meta/audit_governed_repo.py --repo-root /home/brian/projects/llm_client_worktrees/plan-60-llm-client-coordination --json` | proves the repo remains governed and the plan-coordination warnings are gone |
| `python scripts/meta/check_coordination_claims.py --check --project llm_client --json` | proves the local coordination CLI runs in-repo |
| `python scripts/meta/create_plan.py --dry-run --title "coordination smoke" --no-fetch` | proves the local plan allocation path executes |
| `git diff --check` | proves the rollout slice is syntactically clean |

### New Tests

None. The rollout copies the already-tested sanctioned scripts from
`project-meta`; repo-local verification is smoke/audit based.

---

## Acceptance Criteria

- [ ] the only production files added in this slice are the three sanctioned
      plan-coordination scripts
- [ ] governed audit still returns `status=PASS` and `classification=governed`
- [ ] governed audit no longer reports missing plan-coordination scripts
- [ ] `python scripts/meta/check_coordination_claims.py --check --project llm_client --json` succeeds
- [ ] `python scripts/meta/create_plan.py --dry-run --title "coordination smoke" --no-fetch` succeeds
- [ ] the worktree ends this slice at a clean rollback commit

---

## Notes

- This rollout is intentionally bounded. Do not use this plan as license to
  refresh unrelated governance surfaces in `llm_client`.
- AGENTS mirror hygiene is already covered indirectly by the governed audit in
  this repo; this slice does not widen scope to sync missing helper scripts
  outside the sanctioned plan-coordination entrypoints.
- The cross-repo orchestration stays in `project-meta`, but this repo owns the
  local proof that the sanctioned entrypoints work here.
