# ADR 0017: Exclude Same-Host Orphaned Calls From the Active-Call Query

Status: Accepted
Date: 2026-03-20

Related: Builds on ADR 0013 (timeouts are not the default liveness mechanism),
ADR 0014 (lifecycle heartbeats), and ADR 0016 (progress-aware idle detection).

## Context

Plans 09 and 12 gave `llm_client` a useful lifecycle and active-call query:

1. live calls emit `started`, `heartbeat`, `progress`, `stalled`, `completed`,
   and `failed`, and
2. `get_active_llm_calls()` returns the latest known non-terminal state.

That works while the originating process stays alive and keeps emitting
terminal events. A real `onto-canon6` consumer run exposed the remaining gap:
if the process is interrupted locally before it emits a terminal lifecycle
event, the observability store still shows the last non-terminal row as active.

This is not a timeout problem. It is an orphan-detection problem:

1. the call is no longer running,
2. the store has no terminal event, and
3. the query currently has no same-host process identity to prove the process
   is gone.

Without a fix, the active-call query is truthful while a process is alive, but
it can over-report interrupted local calls as still active.

## Decision

1. Lifecycle events must carry additive same-host process identity:
   - `host_name`
   - `process_id`
   - `process_start_token`
2. `process_start_token` should be a best-effort token that distinguishes PID
   reuse on Linux, using `/proc/<pid>/stat` start ticks when available.
3. `get_active_llm_calls()` must exclude a non-terminal record when it can
   prove, on the same host, that the originating process is gone or no longer
   matches the original process identity.
4. When liveness cannot be determined honestly, the query must keep the record
   and report `process_alive=None` rather than guessing.
5. This slice is limited to same-host orphan detection. It does not claim
   remote-process liveness across machines.

## Consequences

Positive:
1. Interrupted local processes no longer linger as false active calls when the
   query can prove they are gone.
2. Active-call views become more trustworthy for long-running consumer
   workflows like `onto-canon6` extraction experiments.
3. The fix stays additive and does not depend on provider cooperation.

Negative:
1. Process-liveness checks are best-effort and platform-sensitive.
2. Older lifecycle rows without process identity cannot be retroactively fixed.
3. Cross-host orchestration still needs a different contract if remote liveness
   ever matters.

## Non-Goals

1. This ADR does not add automatic cancellation.
2. This ADR does not require a distributed worker registry.
3. This ADR does not infer liveness for remote hosts.
4. This ADR does not promise perfect PID-reuse detection on every platform.

## Testing Contract

1. Lifecycle event validation must accept the new process-identity fields.
2. Public lifecycle emission must populate process identity on started and
   terminal events.
3. `get_active_llm_calls()` must exclude same-host orphaned calls when
   process-liveness proves they are gone.
4. `get_active_llm_calls()` must keep records when process liveness is unknown.
