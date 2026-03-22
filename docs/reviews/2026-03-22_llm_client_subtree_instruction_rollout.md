# `llm_client` Subtree Instruction Rollout Review Packet

This packet summarizes the docs-only rollout that added subtree-local `CLAUDE.md` / `AGENTS.md` files in `llm_client`.

Use this file as the review surface for another LLM. The goal is to verify whether the subtree boundaries, file routing, and plan bookkeeping are consistent with the repo's instruction hierarchy.

## What Changed

- Added a package-root router at `llm_client/CLAUDE.md`.
- Added subtree-local instruction files for:
  - `llm_client/cli/CLAUDE.md`
  - `llm_client/observability/CLAUDE.md`
  - `llm_client/prompts/CLAUDE.md`
  - `llm_client/rubrics/CLAUDE.md`
- Added matching `AGENTS.md` symlink mirrors for each of the above and for `llm_client/AGENTS.md`.
- Added a new plan at `docs/plans/08_llm_client-subtree-instructions.md`.
- Added a focused regression test at `tests/test_llm_client_subtree_instructions.py`.
- Updated the plan index in `docs/plans/CLAUDE.md`.

## Why This Exists

The repo already had strong repo-root guidance, but the package surfaces inside `llm_client/llm_client/` are operationally distinct:

- `cli/` is command-entry and inspection surface
- `observability/` is event/run/query plumbing
- `prompts/` is prompt asset data
- `rubrics/` is rubric YAML data

The rollout makes those surfaces load local instructions automatically so agents do not need to infer subtree boundaries from the root docs alone.

## Validation Already Run

- `pytest -q tests/test_llm_client_subtree_instructions.py`
- `python scripts/meta/sync_plan_status.py --check`
- `python /home/brian/projects/project-meta/scripts/check_markdown_links.py --repo-root /home/brian/projects/llm_client docs/plans/CLAUDE.md docs/plans/08_llm_client-subtree-instructions.md llm_client/CLAUDE.md llm_client/cli/CLAUDE.md llm_client/observability/CLAUDE.md llm_client/prompts/CLAUDE.md llm_client/rubrics/CLAUDE.md`

All of the above passed before commit `21f6572`.

## What to Review

1. Are the chosen subtree boundaries the right ones for the current `llm_client` architecture?
2. Do the local `CLAUDE.md` files stay delta-only, or do any of them repeat parent policy unnecessarily?
3. Are the `AGENTS.md` symlinks the right mirror strategy for this repo?
4. Is the package-root `llm_client/CLAUDE.md` routing to the right local surfaces?
5. Does the new plan file reflect a useful rollout slice, or should the next phase be split more narrowly?

## Files to Inspect First

- `llm_client/CLAUDE.md`
- `llm_client/cli/CLAUDE.md`
- `llm_client/observability/CLAUDE.md`
- `llm_client/prompts/CLAUDE.md`
- `llm_client/rubrics/CLAUDE.md`
- `tests/test_llm_client_subtree_instructions.py`
- `docs/plans/08_llm_client-subtree-instructions.md`
- `docs/plans/CLAUDE.md`

## Reviewer Prompt

Review this rollout for:

- subtree specificity
- instruction hierarchy consistency
- AGENTS/CLAUDE mirror correctness
- any missing or overbroad scope
- any documentation drift relative to the repo's plan/index conventions

Return concrete findings only. If there are no issues, say that the rollout looks structurally consistent and call out any residual risks.
