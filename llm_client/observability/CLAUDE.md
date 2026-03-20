# llm_client/observability

This subtree contains observability storage and reconstruction surfaces for
`llm_client`.

## Use This Directory For

- shared run/item/aggregate persistence
- query and reconstruction helpers
- operator review surfaces for what the runtime actually did

## Working Rules

- Treat observability as part of the runtime contract, not optional logging.
- Changes here must preserve reconstruction value for downstream tools such as
  `prompt_eval` and agent tracing.
