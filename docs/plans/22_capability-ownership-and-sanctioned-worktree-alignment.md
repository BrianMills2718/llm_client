# Plan 22: Capability Ownership And Sanctioned Worktree Alignment

**Status:** In Progress
**Type:** implementation
**Priority:** High
**Blocked By:** 21
**Blocks:** truthful shared capability ownership and sanctioned worktree policy

---

## Gap

**Current:** `llm_client` is already a governed shared-infrastructure repo, but
two governance truths still disagree:

1. `meta-process.yaml` enables claims and worktrees, while the repo lacks the
   sanctioned Makefile targets and coordination scripts expected by that config
2. `llm_client` already appears in the shared capability-ownership registry,
   but it still has no repo-local capability ownership source of record

**Target:** give `llm_client` one repo-local ownership source of record, make
its local docs point at that source, and bring the declared sanctioned worktree
policy into line with the actual repo surface.

**Why:** `llm_client` is the core shared runtime substrate. It should not be the
repo that proves the registry can drift from repo-local ownership truth or that
declared worktree policy can remain partly imaginary.

---

## References Reviewed

- `CLAUDE.md`
- `README.md`
- `docs/plans/CLAUDE.md`
- `scripts/CLAUDE.md`
- `scripts/relationships.yaml`
- `meta-process.yaml`
- `Makefile`
- `~/projects/project-meta/docs/plans/46_llm-client-capability-ownership-and-worktree-alignment.md`
- `~/projects/project-meta/docs/ops/GOVERNED_REPO_CONTRACT.md`
- `~/projects/project-meta/docs/ops/CANONICAL_SOURCES_AND_CONSUMER_REPOS.md`
- `~/projects/project-meta/scripts/capability_ownership_registry.yaml`
- `~/projects/project-meta/scripts/meta/audit_governed_repo.py`
- `~/projects/project-meta/scripts/meta/install_governed_repo.py`

---

## Pre-Made Decisions

1. Use the shared governed-repo installer/sync tooling first to repair missing
   sanctioned worktree entrypoints instead of hand-rolling them locally.
2. The first repo-local ownership source will be a dedicated
   `docs/ops/CAPABILITY_DECOMPOSITION.md` file, not just README prose.
3. This wave will not expand the registry taxonomy beyond a bounded local
   source-of-record update unless real evidence demands it.
4. README truthfulness is part of the wave; the local ownership doc must be
   discoverable from the top-level surface.

---

## Files Affected

- `docs/plans/22_capability-ownership-and-sanctioned-worktree-alignment.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `README.md` (modify)
- `Makefile` (modify)
- `meta-process.yaml` (modify)
- `scripts/CLAUDE.md` (modify)
- `docs/ops/CAPABILITY_DECOMPOSITION.md` (create)
- `KNOWLEDGE.md` (modify)

---

## Plan

### Step 1: Declare repo-local capability ownership

- create `docs/ops/CAPABILITY_DECOMPOSITION.md`
- add `meta_process.capability_ownership` and point it at that source
- update README so the ownership source is discoverable

### Step 2: Align sanctioned worktree policy

- use the shared installer/sync path to add or repair sanctioned Makefile
  entrypoints and coordination scripts if possible
- only patch repo-local residual gaps after the shared path is tried

### Step 3: Update local workflow docs

- refresh `scripts/CLAUDE.md` so the new sanctioned worktree and ownership
  surfaces are discoverable
- record durable findings in `KNOWLEDGE.md`

### Step 4: Hand off to shared registry alignment

- leave the repo ready for the matching `project-meta` registry/source updates

---

## Required Tests

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `python ~/projects/project-meta/scripts/meta/audit_governed_repo.py --repo-root . --json` | governed/capability/worktree truth stays explicit |
| `python scripts/meta/sync_plan_status.py --check` | local plan surface stays truthful |
| `python scripts/check_markdown_links.py README.md CLAUDE.md docs/plans/CLAUDE.md docs/plans/22_capability-ownership-and-sanctioned-worktree-alignment.md docs/ops/CAPABILITY_DECOMPOSITION.md scripts/CLAUDE.md KNOWLEDGE.md` | updated local docs remain navigable |
| `python scripts/meta/check_agents_sync.py --check` | AGENTS projection stays in sync if touched surfaces require refresh |

---

## Acceptance Criteria

- [ ] `llm_client` declares repo-local capability ownership with a real source
      of record
- [ ] README and local workflow docs make that ownership source discoverable
- [ ] the repo's declared sanctioned worktree policy matches the actual
      Makefile/scripts surface
- [ ] the shared installer path was used first or an explicit bounded gap was
      recorded if it could not fully repair the worktree surface

---

## Notes

- this plan intentionally stops at the repo boundary; the matching shared
  registry and canonical-source updates happen under `project-meta` Plan 46
