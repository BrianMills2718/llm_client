# ADR 0014: Call Replay And Divergence Diagnosis Boundary

Status: Accepted
Date: 2026-03-22
Last verified: 2026-04-01

## Context

`llm_client` already owns the shared observability boundary for cross-project
LLM execution, but the current surface is stronger at proving that two
operational paths disagree than at explaining why they disagree.

Today we can inspect:

1. call rows and trace rollups,
2. lifecycle events such as `started`, `progress`, `stalled`, `completed`, and
   `failed`,
3. high-level result payloads and experiment aggregates.

That is enough to detect a mismatch between, for example, a proxy evaluation
lane and a live operational path. It is not yet enough to answer the higher
leverage questions cleanly:

1. Did both paths issue the same semantic request?
2. If not, which caller-visible inputs diverged?
3. Can we replay the exact captured call contract through the shared runtime?
4. Which parts of the problem belong in `llm_client`, and which remain
   workflow-specific project logic?

ADR 0007 keeps canonical observability in `llm_client/observability/*` and
requires ADR-level governance for payload-shape changes. ADR 0013 already makes
an important distinction between client-observed truth and provider-internal
inference; the same discipline is needed here.

## Decision

1. `llm_client` owns the shared call-level replay and divergence-diagnosis
   substrate.

2. The canonical shared unit is a **call snapshot**:
   - a normalized representation of one `llm_client` call contract at the
     boundary immediately before provider dispatch,
   - including caller-visible request inputs needed for comparison and replay,
   - excluding purely ephemeral observability metadata such as timestamps,
     call ids, latency, and cost.

3. `llm_client` must expose a stable **request fingerprint** derived from the
   normalized call snapshot:
   - identical semantic call contracts should produce the same fingerprint even
     across different traces or projects,
   - observability-only metadata must not perturb the fingerprint,
   - meaningful caller-visible request differences must appear either in the
     fingerprint or in the compact diff report.

4. `llm_client` must expose compact **call diff** surfaces that compare two
   captured call snapshots and report only the differences needed for the next
   operator decision:
   - rendered messages / content blocks,
   - structured-output schema or response-format identity,
   - model and routing inputs,
   - relevant transport-affecting kwargs,
   - observed result / error summaries.

5. `llm_client` must expose **call-level replay** over captured snapshots:
   - replay reissues the captured call through the shared `llm_client` runtime,
   - replay must run under an explicit new trace/project tag rather than
     mutating or overwriting the original record,
   - replay is about the call contract, not about reconstructing arbitrary
     workflow state.

6. Workflow-specific reconstruction remains project-local:
   - if reproducing a call requires rebuilding domain workflow state before the
     call boundary, that adapter belongs in the consuming project,
   - once the project can hand `llm_client` a prepared call snapshot or an
     equivalent call contract, comparison and replay belong in shared
     infrastructure.

7. Persistence remains safe-by-default and never truncates:
   - the database remains the primary query index,
   - replayable payloads may be stored directly or by artifact reference, but
     not truncated,
   - compact metadata and fingerprints must remain query-friendly even when the
     full snapshot lives out-of-row.

## Consequences

Positive:
1. Cross-project debugging stops reinventing request comparison logic.
2. Live-vs-proxy disagreements can be localized with shared tools instead of
   ad hoc repo-local scripts.
3. The observability layer becomes more operationally useful without turning
   `llm_client` into a workflow engine.
4. Shared fingerprints and diffs give prompt, schema, and routing work a more
   truthful operational-readiness signal.

Negative:
1. Observability payload shape grows and must be governed carefully.
2. Exact replay increases storage pressure if snapshots are large.
3. Call-level replay cannot prove equivalence of higher-level workflows that
   diverge before the `llm_client` call boundary.

## Testing Contract

1. Snapshot normalization tests must prove that ephemeral metadata does not
   perturb the fingerprint.
2. Diff tests must prove that meaningful caller-visible request changes are
   reported compactly and deterministically.
3. Replay tests must prove that a captured snapshot can be reissued under a new
   trace/project without mutating the original record.
4. Existing observability compatibility tests must continue passing.
5. Any artifact-backed snapshot persistence must prove "no truncation" and
   explicit lookup of the full replayable payload.
