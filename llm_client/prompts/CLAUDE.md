# Prompt Assets

This subtree stores prompt assets for `llm_client`.

## Purpose

Prompt assets are data, not inline source strings. Keep them explicit,
versioned, and easy for agents to find without reading code that only consumes
them.

## What Lives Here

- `rubric_judge.yaml`, the structured scoring prompt used by rubric evaluation

## Local Rules

1. Keep prompt templates declarative and data-driven.
2. Update prompt-related docs or schemas when the prompt contract changes.
3. Do not move prompt text into Python source unless a test or loader requires
   it for a specific compatibility reason.
