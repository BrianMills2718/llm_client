# Public Surface Audit

**Status:** Complete (all four candidate groups audited and deprecated)
**Date:** 2026-03-18
**Last verified:** 2026-03-18

## Purpose

This document classifies the current top-level `llm_client` package surface
before any export slimming work.

It exists to answer one question plainly:

Which names should remain first-class top-level substrate APIs, which names can
stay temporarily for compatibility, and which names should eventually move off
the top-level package surface?

This is an audit only. It does not remove exports.

## Requirements

Any future export change must satisfy all of the following:

1. No silent public API break.
2. Core substrate imports remain simple for project code.
3. Optional agent/eval/workflow concerns stop looking equal to the core.
4. Downstream usage is measured before any removal.
5. Import-time side effects are treated as compatibility constraints, not as
   incidental behavior.

## Evidence Reviewed

- [llm_client/__init__.py](../llm_client/__init__.py)
- [README.md](../README.md)
- workspace-wide `from llm_client import ...` usage scan across Python repos
- workspace-wide `import llm_client` usage scan across Python repos

## Current Facts

### Top-level export size

- `llm_client.__all__` currently exposes **146** names.
- The package imports **167** names from internal modules before curating
  `__all__`.
- Several additional top-level attributes exist because `__init__.py` imports
  them, but they are not part of the declared `__all__` surface. Treat those
  as de facto compatibility surface, not as intentionally curated exports.

### High-usage top-level imports

From a workspace-wide Python scan, the most common direct imports are:

- `render_prompt`: ~75 uses
- `call_llm`: ~37 uses
- `acall_llm`: ~33 uses
- `call_llm_structured`: ~32 uses
- `acall_llm_structured`: ~25 uses
- `embed`: ~20 uses
- `strip_fences`: ~6 uses
- `LLMCallResult`: ~6 uses

These are high-risk removal candidates and should be treated as stable until a
real migration is planned.

### Important compatibility constraints

- Several repos use `import llm_client` specifically for import-time API key
  loading side effects.
- Some repos import package submodules via `from llm_client import io_log`,
  `rate_limit`, or `models`. Those are not part of `__all__`, but they are
  still part of the de facto public package surface.
- Generated code in `agent_ontology` references both `import llm_client` and
  `import llm_client.llm_client`, which raises the migration cost of any
  package-topology change.

## Classification

### Keep At Top Level

These are the substrate surface and should remain easy to import directly:

- Core call APIs:
  `call_llm`, `acall_llm`, `call_llm_structured`, `acall_llm_structured`,
  `call_llm_with_tools`, `acall_llm_with_tools`,
  `call_llm_batch`, `acall_llm_batch`,
  `call_llm_structured_batch`, `acall_llm_structured_batch`,
  `stream_llm`, `astream_llm`, `stream_llm_with_tools`,
  `astream_llm_with_tools`, `embed`, `aembed`
- Core result/runtime types:
  `LLMCallResult`, `LLMStream`, `AsyncLLMStream`,
  `Hooks`, `RetryPolicy`, `CachePolicy`, `AsyncCachePolicy`,
  `LRUCache`, `EmbeddingResult`
- Error taxonomy:
  `LLMError` family, `classify_error`, `wrap_error`
- Core config/routing/model-selection:
  `ClientConfig`, `CallRequest`, `ResolvedCallPlan`, `resolve_call`,
  `get_model`, `list_models`, `query_performance`,
  `ResolvedModelSelection`, `ResolvedModelChain`,
  `resolve_model_selection`, `resolve_model_chain`, `strict_model_policy`
- Prompt surface:
  `render_prompt`, `load_prompt_asset`, `parse_prompt_ref`,
  `resolve_prompt_asset`, prompt-asset manifest/ref/result types
- Shared observability substrate:
  `start_run`, `log_item`, `finish_run`, `get_runs`, `get_run`,
  `get_run_items`, `compare_runs`, `compare_cohorts`, `get_cost`,
  `get_trace_tree`, `log_embedding`, `log_foundation_event`,
  `log_experiment_aggregate`, `get_experiment_aggregates`,
  `import_jsonl`, `lookup_result`

### Hold At Top Level For Now

These are real exported surfaces today, but they should be treated as
compatibility holds rather than areas to expand:

- `strip_fences`
- `configure_rate_limit`
- agent-runtime convenience types:
  `MCPAgentResult`, `MCPSessionPool`, `MCPToolCallRecord`,
  `DEFAULT_*` MCP constants
- tool-registry helpers:
  `callable_to_openai_tool`, `lint_tool_callable`,
  `lint_tool_registry`, `prepare_direct_tools`
- validator and agent-spec helpers:
  `ValidationResult`, `run_validators`, `register_validator`, `spec_hash`,
  `AgentSpecValidationError`, `REQUIRED_AGENT_SPEC_SECTIONS`,
  `load_agent_spec`, `validate_agent_spec`
- feature-profile/adoption helpers already built on shared observability:
  `ActiveFeatureProfile`, `ActiveExperimentRun`, `ExperimentRun`,
  `activate_feature_profile`, `activate_experiment_run`,
  `configure_feature_profile`, `experiment_run`,
  `configure_experiment_enforcement`, `configure_agent_spec_enforcement`,
  `enforce_agent_spec`, `get_active_experiment_run_id`,
  `get_active_feature_profile`, `get_background_mode_adoption`,
  `get_completed_traces`

Rule for this category:
keep them working, but stop broadening the top-level surface around them by
default.

### Candidate Move-Off-Top-Level Groups

These names are not necessarily wrong to keep in the repo, but they should not
remain indistinguishable from the core substrate forever:

- Difficulty/model-floor helpers:
  `DifficultyTier`, `get_model_for_difficulty`,
  `get_model_candidates_for_difficulty`, `get_effective_tier`,
  `load_model_floors`, `save_model_floors`
- Task-graph helpers:
  `ExecutionReport`, `ExperimentRecord`, `GraphMeta`, `TaskDef`,
  `TaskGraph`, `TaskResult`, `TaskStatus`, `load_graph`, `run_graph`,
  `toposort_waves`
- Analyzer helpers:
  `AnalysisReport`, `IssueCategory`, `Proposal`,
  `analyze_history`, `analyze_run`, `analyze_scores`,
  `check_scorer_reliability`
- Git/diff helpers:
  `CODE_CHANGE`, `CONFIG_CHANGE`, `PROMPT_CHANGE`, `RUBRIC_CHANGE`,
  `TEST_CHANGE`, `classify_diff_files`, `get_diff_files`, `get_git_head`,
  `get_working_tree_files`, `is_git_dirty`
- Scoring/rubric helpers:
  `CriterionScore`, `Rubric`, `RubricCriterion`, `ScoreResult`,
  `ascore_output`, `list_rubrics`, `load_rubric`, `score_output`
- Experiment-eval/gating helpers:
  `DEFAULT_DETERMINISTIC_CHECKS`, `run_deterministic_checks_for_item`,
  `run_deterministic_checks_for_items`, `review_items_with_rubric`,
  `load_gate_policy`, `extract_agent_outcome`, `extract_adoption_profile`,
  `summarize_agent_outcomes`, `summarize_adoption_profiles`,
  `build_gate_signals`, `evaluate_gate_policy`, `triage_items`

These should move toward explicit module imports or a sibling package boundary
later. They should not be removed until downstream usage is audited at the
symbol level.

## Recommendations

1. Freeze export growth for non-core groups.
2. Keep the core runtime substrate easy to import from the top level.
3. Do not remove optional groups until a real migration map exists.
4. Treat `import llm_client` side-effect users as a first-class compatibility
   constraint.
5. Prefer migration by namespace, not by sudden deletion.

## Completed Symbol-Level Audit: Difficulty Group

### Scope

This audit covered the difficulty-related helpers currently exposed either via
declared `__all__` or de facto package attributes:

- `DifficultyTier`
- `get_model_for_difficulty`
- `get_effective_tier`
- `load_model_floors`
- `save_model_floors`
- `get_model_candidates_for_difficulty`

### Evidence

- workspace-wide import scan found **no non-`llm_client` repo imports** of the
  difficulty helpers from either `llm_client` top level or
  `llm_client.difficulty`
- live code consumers are currently internal to `llm_client`:
  - [`llm_client/task_graph.py`](../llm_client/task_graph.py)
  - [`llm_client/analyzer.py`](../llm_client/analyzer.py)
- direct test/doc references exist in:
  - [`tests/test_difficulty.py`](../tests/test_difficulty.py)
  - [`tests/test_analyzer.py`](../tests/test_analyzer.py)
  - [`docs/TASK_GRAPH_DESIGN.md`](TASK_GRAPH_DESIGN.md)
  - [`CLAUDE.md`](../CLAUDE.md)

### Interpretation

The difficulty group is not a substrate-wide convenience surface in the way
`call_llm`, `render_prompt`, or `embed` are. It is a narrow policy layer tied
mainly to:

- the simple `task_graph` runner
- analyzer/model-floor review logic
- internal docs/tests

That makes it the first credible pilot group for future top-level export
slimming.

### Current Recommendation

Treat the difficulty group as a **low-risk future namespace move**, with this
sequence:

1. keep the `llm_client.difficulty` module stable,
2. stop expanding the difficulty-related top-level surface,
3. when a real migration slice is scheduled, deprecate top-level re-exports
   before removing them,
4. update internal docs/tests at the same time as the deprecation.

This audit did not justify immediate removal, but it did justify a compatibility
preserving deprecation pilot. That pilot is now implemented:

- `llm_client.difficulty` remains the stable import path
- top-level `llm_client` access still resolves
- top-level access now emits explicit deprecation warnings
- internal docs now prefer the module-level import

The remaining decision is removal timing after the deprecation window, not
whether the group is coupled to external project code.

## Completed Symbol-Level Audit: Git-Utils Group

### Scope

This audit covered the git/diff helpers currently exposed from the package
root:

- `CODE_CHANGE`
- `CONFIG_CHANGE`
- `PROMPT_CHANGE`
- `RUBRIC_CHANGE`
- `TEST_CHANGE`
- `classify_diff_files`
- `get_diff_files`
- `get_git_head`
- `get_working_tree_files`
- `is_git_dirty`

### Evidence

- workspace-wide scans found no non-`llm_client` project imports of these names
  from the package root
- live consumers are internal to `llm_client`, mainly:
  - observability helpers
  - analyzer logic
  - CLI experiment tooling
  - `task_graph`
  - tests

### Interpretation

The git-utils group is not a core substrate convenience surface. It is a
supporting internal utility layer that can be namespaced cleanly without
changing the main runtime contract.

### Current Recommendation

Treat the git-utils group as the second low-risk pilot for top-level slimming.
That pilot is now implemented:

- `llm_client.git_utils` remains the stable import path
- top-level `llm_client` access still resolves
- top-level access now emits explicit deprecation warnings
- focused surface tests cover both function and constant compatibility

The remaining decision is the removal window after downstream docs and internal
usage have had time to converge on module imports.

## Completed Symbol-Level Audit: Scoring and Experiment-Eval Groups

### Scope

This audit covered the scoring/rubric helpers and experiment-eval helpers that
still appear at the package root:

- scoring:
  `CriterionScore`, `Rubric`, `RubricCriterion`, `ScoreResult`,
  `ascore_output`, `list_rubrics`, `load_rubric`, `score_output`
- experiment-eval:
  `DEFAULT_DETERMINISTIC_CHECKS`, `run_deterministic_checks_for_item`,
  `run_deterministic_checks_for_items`, `review_items_with_rubric`,
  `load_gate_policy`, `extract_agent_outcome`, `extract_adoption_profile`,
  `summarize_agent_outcomes`, `summarize_adoption_profiles`,
  `build_gate_signals`, `evaluate_gate_policy`, `triage_items`

### Evidence

- workspace-wide scans found no non-`llm_client` Python consumers of these
  groups
- live references were limited to:
  - `llm_client` docs/examples
  - `llm_client` tests
  - internal module-to-module use

### Interpretation

These groups are strategically optional and currently low-risk to namespace.
They are good candidates for moving off the package root before higher-risk
surfaces such as `task_graph` or analyzer.

### Current Recommendation

Treat scoring and experiment-eval as the third low-risk top-level migration
pilot. That pilot is now implemented:

- `llm_client.scoring` and `llm_client.experiment_eval` remain the stable homes
- top-level package access still resolves
- top-level access now emits explicit deprecation warnings
- current docs/examples in `README.md`, `CLAUDE.md`, and `scoring.py` now teach
  module-level imports

The remaining decision is the eventual removal window after the deprecation
period, not whether these groups are still coupled to external project code.

At the module-namespace level, the decision is now explicit as well:

- `llm_client.scoring` remains a stable optional eval module
- `llm_client.experiment_eval` remains a stable optional eval module
- neither module is part of the core call/routing/runtime substrate story
- any future physical move must preserve those module contracts first

## Completed Symbol-Level Audit: Task-Graph And Analyzer Top-Level Exports

### Evidence

- live external Python consumers exist at the module level, for example:
  - `project-meta/ops/openclaw/run_task.py`
- no live external Python consumers were found for package-root imports such as
  `from llm_client import load_graph` or `from llm_client import analyze_run`
- internal docs in `CLAUDE.md` were still teaching the deprecated package-root
  imports before this migration slice

### Interpretation

These groups are still strategically important and still have real consumers,
but those consumers already use `llm_client.task_graph` and
`llm_client.analyzer`. That makes the **top-level re-exports** low-risk to
deprecate even though the underlying modules are not candidates for removal.

### Current Recommendation

Treat task-graph and analyzer as two-layer decisions:

- top-level `llm_client` re-exports: now deprecated through compatibility shims
- module namespaces (`llm_client.task_graph`, `llm_client.analyzer`): keep
  stable and treat as separate boundary questions tied to workflow/analyzer
  architecture

This preserves existing real consumers while stopping the package root from
teaching these optional subsystems as first-class substrate APIs.

At the module-namespace level, the current explicit rule is:

- `llm_client.task_graph` remains the stable optional simple-runner module
- `llm_client.analyzer` remains the stable optional analyzer module
- neither module should be taught as core substrate API
- any deeper move/removal must wait on the workflow/analyzer architecture plans

## Recommended Next Step

Before removing any top-level exports:

1. add explicit grouped sections to `__init__.py` for `core`, `compat_hold`,
   and `candidate_move`,
2. generate a downstream usage list for each candidate move group,
3. only then design a deprecation or namespace-migration slice.
