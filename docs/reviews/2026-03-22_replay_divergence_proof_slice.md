# Replay And Divergence Diagnosis Proof Slice

This note records the real cross-project proof case used to complete
Plan 09.

## Purpose

Prove that the shared `llm_client` replay/diff surface can diagnose one real
operational mismatch without repo-local comparison code, and can replay one
captured call non-destructively under a fresh trace/project tag.

## Proof Case

Source mismatch: the known `onto-canon6` chunk-003 compact extraction gap.

Fresh instrumented rows:

1. live call `201474`
   - project: `onto-canon6-plan09-proof-live`
   - trace: `onto_canon6.extract.158723600359dde1`
   - prompt ref:
     `onto_canon6.extraction.text_to_candidate_assertions_compact_v4@1`
2. parity call `201477`
   - project: `onto-canon6-plan09-proof-prompt-eval`
   - trace:
     `prompt_eval/b6fc156237de/compact_operational_parity/r0/psyop_017_full_chunk003_analytical_context_strict_omit`
   - prompt ref:
     `onto_canon6.extraction.prompt_eval_text_to_candidate_assertions_compact_operational_parity@2`
3. replay call `201480`
   - project: `onto-canon6-plan09-proof-replay`
   - trace: `plan09-proof-replay-live-201474`

## Compare Output

`python -m llm_client replay compare --left-call-id 201474 --right-call-id 201477 --format text`

Key result:

1. `fingerprints_match=False`
2. request diff shows:
   - `request.kwargs.temperature: missing on left; right=0.0`
   - user-message wrapper text differs (`source text` vs `source material`,
     plus prompt-eval case wrapper text)
   - prompt refs differ
3. result diff shows the response payloads differ

This is the shared-infra answer we needed. The mismatch is no longer "we know
behavior differs but do not know where." The compare surface shows the
request-boundary differences directly.

## Replay Output

`python -m llm_client replay rerun --call-id 201474 --trace-id plan09-proof-replay-live-201474 --project onto-canon6-plan09-proof-replay --task budget_extraction --max-budget 0.0 --format text`

Result:

1. replay completed without mutating the original call row
2. replay produced fresh row `201480`
3. `call_fingerprint(201474) == call_fingerprint(201480)`

Follow-up compare:

`python -m llm_client replay compare --left-call-id 201474 --right-call-id 201480 --format text`

Key result:

1. `fingerprints_match=True`
2. `request: no request differences`
3. `result.response` still differs

That last point matters. It proves request-identity equality is not enough to
explain live behavior by itself. The shared surface must support both
fingerprinting and replay/diff, not only one of them.

## Verification

Implementation verification was already completed before this proof slice:

1. `pytest -q tests/test_observability_replay.py tests/test_cli_replay.py tests/test_io_log.py tests/test_public_surface.py`
2. `pytest -q tests/test_io_log_compat.py tests/test_client_lifecycle.py`
3. `pytest -q tests/test_client.py`
4. `python scripts/meta/validate_relationships.py`

Proof-slice commands:

1. fresh live `onto-canon6` extraction rerun with isolated `LLM_CLIENT_PROJECT`
2. fresh one-case `onto-canon6` prompt experiment with isolated
   `LLM_CLIENT_PROJECT`
3. shared `llm_client replay compare`
4. shared `llm_client replay rerun`
5. shared original-vs-replay `llm_client replay compare`

## Conclusion

Plan 09 is now proved on a real consuming-repo mismatch.

`llm_client` owns:

1. captured request identity,
2. compact cross-call diffs, and
3. call-level replay under fresh trace/project tags.

Consuming repos still own workflow reconstruction up to the `llm_client`
boundary, but they no longer need repo-local tooling to compare or replay calls
once that boundary is reached.
