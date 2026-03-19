# Prompt Asset Layer

This document defines the minimal shared prompt asset contract implemented in
`llm_client`.

## Goal

Provide one deterministic, explicit way to reference reusable prompts across
projects without introducing overrides, shadowing, or prompt lookup ambiguity.

## Reference Format

Prompt assets are addressed by explicit references:

```text
<asset_id>@<positive_integer_version>
```

Example:

```text
shared.summarize.concise@1
```

`asset_id` is dotted and becomes nested directories under the package prompt
asset root.

## Directory Layout

```text
llm_client/prompt_assets/
  shared/
    summarize/
      concise/
        v1/
          manifest.yaml
          template.yaml
```

The resolver maps `shared.summarize.concise@1` to:

```text
llm_client/prompt_assets/shared/summarize/concise/v1/manifest.yaml
```

## Manifest Schema

Prompt asset manifests are strict YAML mappings with these fields:

```yaml
id: shared.summarize.concise
version: 1
kind: chat_prompt
status: canonical
description: Concise reusable summarization prompt for general text inputs.
derived_from: null
tags: [summarize, reusable, general]
template_path: template.yaml
input_schema: prompt_eval.text_input.v1
output_schema: text.summary.v1
```

Rules:

1. `id` and `version` must match the explicit `prompt_ref`.
2. `template_path` is resolved relative to the manifest directory.
3. `derived_from`, when present, must itself be a valid explicit `prompt_ref`.
4. Extra manifest keys are rejected.

## Render Contract

`llm_client.render_prompt()` now accepts exactly one source:

1. `template_path=...`
2. `prompt_ref=...`

Examples:

```python
from llm_client import render_prompt

messages = render_prompt("prompts/local_prompt.yaml", topic="prompt assets")
shared_messages = render_prompt(prompt_ref="shared.summarize.concise@1", style="bullet")
```

The renderer fails loudly when:

1. both `template_path` and `prompt_ref` are supplied,
2. neither source is supplied,
3. the prompt ref is malformed,
4. the manifest does not exist,
5. the manifest is malformed or inconsistent with the ref,
6. the template file does not exist,
7. required Jinja variables are missing.

## Current Proving Slice

The initial built-in shared prompt family is:

1. `shared.summarize.concise@1`
2. `shared.summarize.bullet@1` derived from `shared.summarize.concise@1`

`prompt_eval` can materialize these assets into `PromptVariant` objects via
`build_prompt_variant_from_ref(...)` and records the explicit `prompt_ref` in
shared observability.
