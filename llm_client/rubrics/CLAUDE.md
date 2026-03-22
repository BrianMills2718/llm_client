# Rubrics

This subtree stores versioned scoring rubrics for `llm_client`.

## Purpose

Rubrics are reusable evaluation data. They should stay small, explicit, and
stable enough for prompt evaluation, scoring, and analysis workflows to
reference by filename.

## What Lives Here

- `analysis_quality.yaml`
- `code_quality.yaml`
- `extraction_quality.yaml`
- `research_quality.yaml`
- `summarization_quality.yaml`

## Local Rules

1. Keep rubric names stable once they are used by tests or workflows.
2. Update weights, scales, and descriptions together so the rubric remains
   coherent.
3. Treat rubric changes as evaluation-contract changes, not casual content
   edits.
