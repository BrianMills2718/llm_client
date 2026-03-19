# ADR 0011: Prompt Assets Use Explicit Identity and Lineage

Status: Accepted  
Date: 2026-03-17

## Context

The project direction is to treat prompts as data, not as ad hoc inline strings
inside application code. We also want prompts to be reusable across projects and
easy for coding agents to inspect, compare, and improve.

The main risk in a shared prompt system is ambiguity. Project-local overrides,
implicit resolution rules, and symlink-based indirection make it hard to answer
basic questions such as:

1. which prompt actually ran,
2. where the source of truth lives,
3. whether one prompt is a modified copy of another,
4. how an agent should compare prompt variants across projects.

## Decision

1. Prompts remain data assets and should be loaded through explicit prompt
   rendering APIs instead of inline Python f-strings.
2. Reusable prompts belong in a shared prompt asset layer rather than being
   owned by `prompt_eval`.
3. Each prompt asset must have explicit identity:
   - asset ID,
   - version,
   - namespace or owner,
   - metadata sufficient for provenance.
4. Customized prompts create new prompt assets with lineage metadata such as
   `derived_from`; they do not silently override an existing shared prompt.
5. Hidden runtime override resolution is not the default architecture. The
   system should prefer explicit prompt references over project-local shadowing.
6. `llm_client.render_prompt()` should resolve prompt inputs deterministically
   from explicit references or explicit file paths.
7. `prompt_eval` evaluates prompt assets and records their identity in run
   metadata; it does not become the canonical prompt registry.

## Consequences

Positive:
1. Deterministic provenance for prompt runs.
2. Better reuse and comparison across projects.
3. Lower ambiguity for coding agents inspecting or modifying prompt behavior.
4. Cleaner promotion path from local prompt experiments to shared prompt assets.

Negative:
1. A shared prompt asset layer needs its own metadata and versioning discipline.
2. Teams lose the convenience of implicit overrides and must create explicit new
   prompt assets for meaningful changes.
3. Existing project-local prompt layouts will need a gradual migration path.

## Testing Contract

1. Prompt rendering tests must prove deterministic resolution for explicit
   prompt references.
2. Observability and experiment logging should record prompt asset identity when
   the caller supplies it.
3. Prompt comparison workflows must treat prompt lineage as explicit metadata,
   not inferred from override order.
