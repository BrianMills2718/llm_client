# Plan 09: Replay And Divergence Diagnosis

**Status:** Complete
**Type:** implementation
**Priority:** High
**Blocked By:** None
**Blocks:** trustworthy operational-readiness debugging across consuming repos

---

## Gap

**Current:** `llm_client` observability can show call rows, lifecycle events,
trace trees, and experiment summaries, but it cannot yet answer the shared
operator question: "are these two operational surfaces actually issuing the same
call contract, and if not where do they diverge?" Current tooling requires
repo-local reconstruction when live and proxy paths disagree.

**Target:** `llm_client` provides a shared call-snapshot substrate with:

1. stable request fingerprints,
2. compact call diff reports,
3. call-level replay under fresh trace/project tags,
4. an operator-facing CLI surface for compare/replay workflows.

**Why:** this is a cross-project observability gap, not an `onto-canon6`-local
prompt issue. The workspace rules require the active plan to be updated when new
evidence changes what counts as proof or the next useful step. Shared replay and
diff tooling has higher leverage than more project-local mismatch debugging.

---

## References Reviewed

- `/home/brian/projects/.claude/CLAUDE.md:13-23` - top-down planning and proof-first execution rule
- `/home/brian/projects/.claude/CLAUDE.md:41-45` - durable issue incorporation and stale-plan prohibition
- `/home/brian/projects/llm_client/docs/plans/01_master-roadmap.md:31-47` - roadmap update rule when default slice changes
- `/home/brian/projects/llm_client/docs/plans/01_master-roadmap.md:149-177` - Program E is the active execution program
- `/home/brian/projects/llm_client/docs/plans/06_simplification-and-observability.md:25-45` - active observability modernization gap
- `/home/brian/projects/llm_client/docs/adr/0007-observability-contract-boundary.md:13-23` - canonical observability boundary and ADR governance
- `/home/brian/projects/llm_client/docs/adr/0013-stream-lifecycle-heartbeat-observability.md:21-45` - client-observed truth vs provider inference discipline
- `/home/brian/projects/llm_client/llm_client/client.py:783-977` - current call + lifecycle observability emission points
- `/home/brian/projects/llm_client/llm_client/observability/query.py:19-184` - current lookup and trace-query limits
- `/home/brian/projects/llm_client/llm_client/__main__.py:29-51` - current CLI command registration
- `/home/brian/projects/llm_client/llm_client/cli/traces.py:12-100` - trace summary CLI is present, compare/replay CLI is absent

---

## Files Affected

> This section declares the intended implementation surface for this plan. New
> slices must amend this section before touching additional files.

- llm_client/client.py (modify — emit normalized replayable snapshot metadata)
- llm_client/io_log.py (modify — persist snapshot fingerprints and artifact refs through compatibility facade)
- llm_client/observability/query.py (modify — lookup/diff helpers for captured call snapshots)
- llm_client/observability/replay.py (create — snapshot normalization, fingerprinting, diff, replay)
- llm_client/observability/__init__.py (modify — export shared replay/diff surface)
- llm_client/cli/replay.py (create — compare/replay operator CLI)
- llm_client/__main__.py (modify — register compare/replay command)
- tests/test_observability_replay.py (create — fingerprint/diff/replay contract tests)
- tests/test_cli_replay.py (create — CLI behavior tests)
- docs/API_REFERENCE.md (modify — document snapshot, diff, and replay surfaces)
- docs/adr/0014-call-replay-and-divergence-diagnosis-boundary.md (create)
- scripts/relationships.yaml (modify — govern new observability ADR linkage)

---

## Definition Of Done

This plan is done only when all of the following are true:

1. A captured call has a stable normalized snapshot identity suitable for
   cross-surface comparison.
2. Two captured calls can be compared with a compact report that explains the
   material caller-visible differences without dumping raw blobs by default.
3. A captured call snapshot can be replayed through `llm_client` under a fresh
   trace/project tag.
4. Full replayable payloads are preserved without truncation, either directly
   or through explicit artifact references.
5. The shared surface is reusable by consuming repos without copying
   comparison/replay logic into project-local code.

---

## Plan

## Thin Slices

### Phase 1: Call Snapshot Contract

**Purpose:** define the normalized replayable call contract before any storage
or CLI work.

**Input -> Output:** existing ad hoc call rows -> explicit snapshot schema and
fingerprint rules

**Passes if:**

- the snapshot contract names exactly which fields are part of replay identity
- ephemeral metadata is explicitly excluded
- the fingerprint rule is deterministic and documented
- ADR 0014 and this plan agree on the shared/local boundary

**Fails if:**

- fingerprint semantics depend on timestamps, call ids, cost, or latency
- the snapshot contract relies on project-local workflow state
- the design requires silent truncation

### Phase 2: Snapshot Persistence And Lookup

**Purpose:** persist replayable snapshot metadata and expose lookup helpers.

**Input -> Output:** ordinary call rows -> call rows plus snapshot identity and
retrieval path

**Passes if:**

- new calls record snapshot fingerprints
- full replayable payload lookup works without truncation
- existing observability compatibility tests keep passing

**Fails if:**

- compatibility imports break
- snapshot persistence silently drops oversized content
- operators cannot retrieve the stored snapshot deterministically

### Phase 3: Compact Call Diff

**Purpose:** compare two captured calls and show only the material differences.

**Input -> Output:** two call ids / trace-linked calls -> deterministic compact
diff report

**Passes if:**

- message, schema, routing, and result differences are visible
- unchanged fields are suppressed by default
- output is readable enough for CLI and run-note usage

**Fails if:**

- diff output requires reading raw JSON blobs to understand the disagreement
- formatting is unstable across identical comparisons

### Phase 4: Call-Level Replay

**Purpose:** replay one captured snapshot through the shared runtime.

**Input -> Output:** call snapshot -> fresh replay run with explicit new
trace/project tags

**Passes if:**

- replay uses the shared `llm_client` runtime rather than project-local logic
- replay never mutates or overwrites the original record
- replay results can be compared back to the source call

**Fails if:**

- replay depends on hidden workflow context that is not in the snapshot
- replay only works for one consuming repo

### Phase 5: Operator CLI And Proof Slice

**Purpose:** make the capability usable and prove it on one known divergence.

**Input -> Output:** internal replay/diff helpers -> operator-facing CLI and one
documented proof case

**Passes if:**

- `python -m llm_client replay compare ...` or equivalent surfaces the compact diff
- `python -m llm_client replay rerun ...` or equivalent triggers replay cleanly
- one known live-vs-proxy mismatch is diagnosed with the shared surface

**Fails if:**

- the feature is only callable from internal Python
- the proof still depends on repo-local ad hoc diff logic

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_observability_replay.py` | `test_snapshot_fingerprint_ignores_ephemeral_metadata` | timestamps, call ids, and latency do not perturb identity |
| `tests/test_observability_replay.py` | `test_snapshot_fingerprint_changes_for_material_request_delta` | meaningful caller-visible request changes alter the diagnosis output |
| `tests/test_observability_replay.py` | `test_compare_call_snapshots_reports_compact_differences` | compact diff highlights only relevant changes |
| `tests/test_observability_replay.py` | `test_replay_call_snapshot_uses_new_trace_and_preserves_original_record` | replay is non-destructive and auditable |
| `tests/test_cli_replay.py` | `test_replay_compare_cli_outputs_compact_report` | operator CLI exposes compare cleanly |
| `tests/test_cli_replay.py` | `test_replay_rerun_cli_creates_fresh_replay_trace` | operator CLI exposes replay cleanly |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_io_log.py` | observability persistence and compatibility must stay stable |
| `tests/test_io_log_compat.py` | `io_log` facade delegation must remain truthful |
| `tests/test_client.py` | core call path logging and transport behavior must not regress |
| `tests/test_client_lifecycle.py` | lifecycle observability contract must remain intact |
| `tests/test_public_surface.py` | new exports/CLI registration must preserve package surface expectations |

---

## Acceptance Criteria

- [x] ADR 0014 is accepted and linked in governance surfaces
- [x] Snapshot identity and replay boundary are documented before implementation
- [x] New fingerprint, diff, and replay tests pass
- [x] Existing observability and client tests pass
- [x] Operator CLI exists for compare/replay workflows
- [x] One real known divergence case is diagnosed through the shared surface

---

## Completion Evidence

The implementation and proof slice completed on 2026-03-22.

### Fresh Cross-Project Proof Case

The proof case used the known `onto-canon6` chunk-003 live-vs-parity mismatch
under fresh observability project labels so the rows carried call snapshots and
fingerprints:

1. live extraction call: `201474`
   - project: `onto-canon6-plan09-proof-live`
   - prompt ref:
     `onto_canon6.extraction.text_to_candidate_assertions_compact_v4@1`
2. parity prompt-eval call: `201477`
   - project: `onto-canon6-plan09-proof-prompt-eval`
   - prompt ref:
     `onto_canon6.extraction.prompt_eval_text_to_candidate_assertions_compact_operational_parity@2`
3. replayed live call: `201480`
   - project: `onto-canon6-plan09-proof-replay`
   - trace id: `plan09-proof-replay-live-201474`

### What The Shared Surface Proved

`python -m llm_client replay compare --left-call-id 201474 --right-call-id 201477`
showed the live/parity divergence through shared infrastructure only:

1. fingerprints do not match;
2. the parity call includes explicit `temperature=0.0` while the live call does
   not;
3. the user wrapper text differs (`source text` vs `source material`, plus
   prompt-eval case wrapper text); and
4. the result payloads differ.

`python -m llm_client replay rerun --call-id 201474 ...` proved
non-destructive replay under a fresh trace/project tag. Comparing the original
call to replay call `201480` showed:

1. `fingerprints_match=True`;
2. `request: no request differences`; and
3. `result.response` still differed, proving why replay plus compare is needed
   instead of fingerprinting alone.

### Verification

Code verification already completed during implementation:

1. `python -m py_compile ...`
2. `pytest -q tests/test_observability_replay.py tests/test_cli_replay.py tests/test_io_log.py tests/test_public_surface.py`
3. `pytest -q tests/test_io_log_compat.py tests/test_client_lifecycle.py`
4. `pytest -q tests/test_client.py`
5. `python scripts/meta/validate_relationships.py`
6. `python scripts/meta/generate_api_reference.py --write`

Proof-slice verification:

1. fresh live `onto-canon6` extraction call captured with snapshot fields
2. fresh `prompt_eval` parity call captured with snapshot fields
3. `python -m llm_client replay compare --left-call-id 201474 --right-call-id 201477 --format text`
4. `python -m llm_client replay rerun --call-id 201474 --trace-id plan09-proof-replay-live-201474 --project onto-canon6-plan09-proof-replay --task budget_extraction --max-budget 0.0 --format text`
5. `python -m llm_client replay compare --left-call-id 201474 --right-call-id 201480 --format text`

---

## Notes

- Shared infrastructure should own call-level truth. It should not absorb
  arbitrary workflow reconstruction from consuming repos.
- The first proof case can come from `onto-canon6`, but the resulting machinery
  must not be `onto-canon6`-specific.
- If full snapshot persistence forces artifact-backed storage, keep the DB row
  compact and query-friendly, but never truncate the replayable payload.
