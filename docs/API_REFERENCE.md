# API Reference

This document is the stable entry point into the `llm_client` API surface.
It is an index, not a second full README.

## Start Here

1. [README.md](../README.md) for installation, usage, routing, and examples.
2. [AGENTS.md](../AGENTS.md) for repo operating rules and architectural
   boundaries.
3. [docs/plans/01_master-roadmap.md](plans/01_master-roadmap.md) for the
   current long-term program state.

## Core Runtime Surface

Use these as the primary top-level imports:

- text calls:
  `call_llm`, `acall_llm`, `stream_llm`, `astream_llm`
- structured calls:
  `call_llm_structured`, `acall_llm_structured`
- tool calls:
  `call_llm_with_tools`, `acall_llm_with_tools`,
  `stream_llm_with_tools`, `astream_llm_with_tools`
- batch and embeddings:
  `call_llm_batch`, `acall_llm_batch`, `embed`, `aembed`
- model selection:
  `get_model`, `resolve_model_selection`, `resolve_model_chain`
- prompt assets:
  `render_prompt`, `load_prompt_asset`, `parse_prompt_ref`,
  `resolve_prompt_asset`

Required call kwargs on real project calls:

- `task=`
- `trace_id=`
- `max_budget=`

## Observability And Experiments

Authoritative shared observability remains in `llm_client`:

- run lifecycle:
  `start_run`, `log_item`, `finish_run`
- queries:
  `get_runs`, `get_run`, `get_run_items`, `compare_runs`,
  `compare_cohorts`, `get_cost`, `get_trace_tree`, `lookup_result`
- aggregates:
  `log_experiment_aggregate`, `get_experiment_aggregates`

Reference docs:

- [docs/ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md](ECOSYSTEM_TOP_DOWN_ARCHITECTURE.md)
- [docs/ECOSYSTEM_UNCERTAINTIES.md](ECOSYSTEM_UNCERTAINTIES.md)
- [docs/plans/05_eval-boundary-cleanup.md](plans/05_eval-boundary-cleanup.md)

## Optional Module Namespaces

These are stable module namespaces, but they are not the core substrate:

- workflow:
  `llm_client.workflow_langgraph`
- task DAG runner:
  `llm_client.task_graph`
- eval helpers:
  `llm_client.experiment_eval`, `llm_client.scoring`, `llm_client.analyzer`
- compatibility guidance:
  `llm_client.difficulty`

For boundary rationale, see:

- [docs/PUBLIC_SURFACE_AUDIT.md](PUBLIC_SURFACE_AUDIT.md)
- [docs/plans/02_client-boundary-hardening.md](plans/02_client-boundary-hardening.md)
- [docs/plans/04_workflow-layer-boundary.md](plans/04_workflow-layer-boundary.md)

## Prompt Assets

Prompt assets are explicit data with identity and lineage.

Reference docs:

- [docs/PROMPT_ASSET_LAYER.md](PROMPT_ASSET_LAYER.md)
- [docs/adr/0011-prompt-assets-explicit-identity.md](adr/0011-prompt-assets-explicit-identity.md)

## CLI Surface

Primary entry point:

```bash
python -m llm_client --help
```

Notable subcommands:

- `cost`
- `traces`
- `scores`
- `experiments`
- `adoption`
- `backfill`
- `tool-lint`

## Package Extras

Install only what you need:

- `pip install -e "~/projects/llm_client[structured]"`
- `pip install -e "~/projects/llm_client[workflow]"`
- `pip install -e "~/projects/llm_client[agents]"`
- `pip install -e "~/projects/llm_client[codex]"`
- `pip install -e "~/projects/llm_client[mcp]"`

## Source Of Truth Rule

When this file and the code disagree:

1. `pyproject.toml` is authoritative for package metadata and extras.
2. module docstrings and public function signatures are authoritative for code
   behavior.
3. the roadmap and ADRs are authoritative for architectural boundaries.
