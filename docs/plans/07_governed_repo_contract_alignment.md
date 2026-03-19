# Plan 07: Governed Repo Contract Alignment

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** Tier 0 governed rollout completion in `project-meta` Plan 09

---

## Gap

**Current:** `llm_client` already has a stronger custom required-reading gate
than most repos, but it is still only a partial governed repo. `AGENTS.md` is a
symlink to `CLAUDE.md`, `CLAUDE.md` is too thin to serve as canonical
governance, several canonical validators are missing locally, doc-coupling
still points at the legacy `doc_coupling.yaml` path, and the hook path is not
logging decisions.

**Target:** `llm_client` satisfies the governed-repo contract while preserving
its stronger local read-gate behavior. Canonical governance is local and
deterministic: `CLAUDE.md` is authoritative, `AGENTS.md` is generated,
`scripts/relationships.yaml` is the unified graph, required validators exist
locally, doc-coupling reads the unified graph, and hook decisions are logged.

**Why:** `llm_client` is a Tier 0 active-stack repo. OpenClaw and the governed
rollout cannot honestly claim the core stack is installed if the runtime
substrate itself still has split-brain governance or broken local tool paths.

---

## References Reviewed

- `CLAUDE.md` - current thin repo-governance placeholder
- `README.md` - repo purpose and runtime-substrate positioning
- `docs/plans/CLAUDE.md` - current plan inventory
- `meta-process.yaml` - repo-local governance settings
- `scripts/relationships.yaml` - current required-reading and ADR/doc coupling graph
- `scripts/meta/check_required_reading.py` - existing custom read-gate logic
- `.claude/hooks/gate-edit.sh` - existing edit-gate hook
- `.claude/hooks/track-reads.sh` - existing read-tracking hook
- `.claude/settings.json` - active hook wiring
- `tests/test_required_reading_gate.py` - current gate behavior coverage
- `scripts/meta/check_doc_coupling.py` - current doc-coupling implementation
- `scripts/meta/complete_plan.py` - current completion path and broken top-level script assumption
- `~/projects/project-meta/docs/ops/GOVERNED_REPO_CONTRACT.md` - governed-repo contract
- `~/projects/project-meta/docs/plans/09_governed-repo-rollout-and-active-stack-installation.md` - active rollout plan
- `~/projects/project-meta/scripts/meta/render_agents_md.py` - canonical AGENTS renderer
- `~/projects/project-meta/scripts/meta/file_context.py` - canonical file-context resolver
- `~/projects/project-meta/scripts/meta/validate_plan.py` - canonical plan validator
- `~/projects/project-meta/scripts/check_markdown_links.py` - canonical markdown-link validator
- `~/projects/project-meta/scripts/meta/hook_log.py` - canonical hook logger

---

## Files Affected

- `CLAUDE.md` (modify)
- `AGENTS.md` (replace symlink with generated file)
- `docs/plans/07_governed_repo_contract_alignment.md` (create)
- `docs/plans/CLAUDE.md` (modify)
- `meta-process.yaml` (modify)
- `scripts/CLAUDE.md` (modify)
- `scripts/check_doc_coupling.py` (create)
- `scripts/check_required_reading.py` (create)
- `scripts/check_markdown_links.py` (create)
- `scripts/meta/check_doc_coupling.py` (modify)
- `scripts/meta/file_context.py` (create)
- `scripts/meta/hook_log.py` (create)
- `scripts/meta/validate_plan.py` (create)
- `.claude/hooks/gate-edit.sh` (modify)
- `.claude/hooks/track-reads.sh` (modify)
- `.gitignore` (modify)
- `tests/test_read_gate_hooks.py` (create)

---

## Plan

### Step 1: Canonicalize repo governance

- expand `CLAUDE.md` into the canonical repo-governance file
- add this plan to the index
- replace the `AGENTS.md -> CLAUDE.md` symlink with generated output

### Step 2: Install missing governed-repo validators locally

- vendor `file_context.py`, `validate_plan.py`, and `check_markdown_links.py`
- add top-level script wrappers where the repo already expects them
- make doc-coupling point at `scripts/relationships.yaml`

### Step 3: Keep the custom read gate but make it observable and fail loud

- add structured hook logging
- support the standard hook env vars used elsewhere in the stack
- stop silently allowing edits when the required-reading checker is missing

### Step 4: Verify governed status

- run the local validator/test suite for the governance slice
- run the canonical governed-repo audit
- confirm `llm_client` upgrades from `partial` to `governed`

---

## Required Tests

### New Tests

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_read_gate_hooks.py` | `test_gate_edit_blocks_and_logs_missing_required_reads` | Hook blocks governed edits and logs the block decision |
| `tests/test_read_gate_hooks.py` | `test_gate_edit_allows_and_logs_after_required_reads` | Hook allows only after required docs are read and logs the allow decision |
| `tests/test_read_gate_hooks.py` | `test_track_reads_records_session_file_and_log` | Read hook records the read path and writes a JSONL log entry |
| `tests/test_read_gate_hooks.py` | `test_file_context_includes_default_and_coupled_required_reads` | Local file-context resolver includes `CLAUDE.md` defaults and coupled ADR docs |

### Existing Tests

| Test Pattern | Why |
|--------------|-----|
| `tests/test_required_reading_gate.py` | Existing custom gate behavior must stay intact |
| `tests/test_relationships_validation.py` | Unified relationships graph must remain valid |

---

## Acceptance Criteria

- [x] `CLAUDE.md` is canonical enough to generate `AGENTS.md`
- [x] `AGENTS.md` is a real generated file, not a symlink
- [x] local `file_context.py`, `validate_plan.py`, and `check_markdown_links.py` exist
- [x] doc-coupling reads `scripts/relationships.yaml`, not `scripts/doc_coupling.yaml`
- [x] custom Claude hooks log block/allow/read events to `.claude/hook_log.jsonl`
- [x] hook dependencies fail loud instead of silently allowing governed edits
- [x] `pytest -q tests/test_required_reading_gate.py tests/test_read_gate_hooks.py tests/test_relationships_validation.py` passes
- [x] `python scripts/meta/validate_relationships.py --strict` passes
- [x] `python scripts/meta/validate_plan.py --plan-file docs/plans/07_governed_repo_contract_alignment.md` passes
- [x] `python scripts/check_markdown_links.py CLAUDE.md docs/plans/CLAUDE.md scripts/CLAUDE.md` passes
- [x] `python ~/projects/project-meta/scripts/audit_governed_repo.py --repo-root "$PWD" --strict-governed --json` reports `governed`

**Verified:** 2026-03-19
**Evidence:**

- `pytest -q tests/test_required_reading_gate.py tests/test_read_gate_hooks.py tests/test_relationships_validation.py`
- `python scripts/meta/validate_relationships.py --strict`
- `python scripts/meta/validate_plan.py --plan-file docs/plans/07_governed_repo_contract_alignment.md`
- `python scripts/meta/sync_plan_status.py --check`
- `python scripts/check_markdown_links.py CLAUDE.md docs/plans/CLAUDE.md scripts/CLAUDE.md`
- `python scripts/check_doc_coupling.py --validate-config`
- `python ~/projects/project-meta/scripts/audit_governed_repo.py --repo-root "$PWD" --strict-governed --json`

---

## Notes

- This slice preserves the stronger local read-gate policy; it does not replace
  it with the generic generated gate.
- `llm_client/errors.py` already has a staged unrelated change in the worktree.
  This plan must not absorb or revert it.
- Top-level wrappers are acceptable here because existing repo tooling already
  calls `scripts/check_doc_coupling.py` and the hook prefers
  `scripts/check_required_reading.py`.
