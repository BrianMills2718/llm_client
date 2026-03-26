# Experiment Observability

## Recording experiments

```python
from llm_client import start_run, log_item, finish_run

run_id = start_run(
    name="extraction_v2",
    condition_id="baseline",
    scenario_id="phase1",
    phase="phase1",
)

for item in dataset:
    result = await acall_llm_structured(...)
    log_item(run_id, item_id=item.id, result=result, score=score)

finish_run(run_id)
```

## CLI commands

```bash
# List experiments
python -m llm_client experiments

# Filter by condition/scenario
python -m llm_client experiments --condition-id forced_off --scenario-id phase1

# Compare two runs
python -m llm_client experiments --compare RUN_BASE RUN_CANDIDATE

# Compare cohorts
python -m llm_client experiments --compare-cohorts baseline forced_reduced forced_off \
    --baseline-condition-id baseline --scenario-id phase1 --phase phase1

# Detailed run view with triage
python -m llm_client experiments --detail RUN_ID

# Deterministic checks
python -m llm_client experiments --detail RUN_ID --det-checks default

# Rubric-based LLM review
python -m llm_client experiments --detail RUN_ID --review-rubric extraction_quality

# Policy gates (CI-friendly)
python -m llm_client experiments --detail RUN_ID \
    --gate-policy '{"pass_if":{"avg_llm_em_gte":80}}' \
    --gate-fail-exit-code
```

## Adoption gates

For long-thinking adoption telemetry:

```python
from llm_client import get_background_mode_adoption

summary = get_background_mode_adoption(
    experiments_path="~/projects/data/task_graph/experiments.jsonl",
    run_id_prefix="nightly_",
)
print(summary["background_mode_rate_among_reasoning"])
```

CLI gate (cron/CI-friendly):

```bash
python -m llm_client adoption --run-id-prefix nightly_ --format table
python -m llm_client adoption --run-id-prefix nightly_ --since 2026-02-20 \
    --min-rate 0.95 --metric among_reasoning --min-samples 20

# Or via wrapper script:
./scripts/adoption_gate.sh
```

## Eval helpers

```python
from prompt_eval.experiment_eval import (
    build_gate_signals,
    extract_agent_outcome,
    summarize_agent_outcomes,
)

outcome = extract_agent_outcome(item_result)
summary = summarize_agent_outcomes(run_items)
signals = build_gate_signals(run_info=run_info, items=run_items)
```
