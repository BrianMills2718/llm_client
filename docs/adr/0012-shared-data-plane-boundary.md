# ADR 0012: Shared Data Plane Boundary

Status: Accepted  
Date: 2026-03-17

## Context

The broader ecosystem direction is to standardize reusable data across projects
so coding agents can rely on common datasets, artifacts, schemas, and
provenance instead of each repo inventing its own storage conventions.

The risk is that `llm_client` could grow from a shared runtime and observability
layer into an undifferentiated data platform that stores every raw payload,
corpus, and derived object itself.

We need a clear line between:

1. shared execution and experiment metadata,
2. shared datasets and artifacts,
3. project-specific derived stores and caches.

## Decision

1. Canonical reusable datasets, artifacts, schema definitions, and lineage
   metadata should be standardized across projects rather than being
   independently owned by each project repo.
2. `llm_client` remains the control plane, not the entire data plane. It owns:
   - execution events,
   - observability,
   - run metadata,
   - scoring records,
   - provenance links to datasets and artifacts.
3. The shared data plane should be a separate architectural layer that can host
   dataset registries, artifact registries, schema metadata, and storage
   adapters.
4. `llm_client` should persist stable references such as `dataset_id`,
   `artifact_id`, `uri`, `content_hash`, `schema_name`, `schema_version`, and
   lineage links instead of trying to inline every raw dataset or artifact.
5. Project-local stores are usually derived views, caches, or specialized
   runtimes. They are sources of truth only when they represent genuinely
   project-unique state that cannot reasonably live in the shared data plane.
6. Embedding work is split across the layers:
   - `llm_client` logs the embedding event and its provenance,
   - vectors, indexes, and large embedding artifacts live in the data plane or
     in project-specific derived stores that are linked back to that event.

## Consequences

Positive:
1. Cross-project reuse becomes an explicit architectural goal instead of an
   accident.
2. `llm_client` can stay focused on runtime and observability contracts.
3. Provenance becomes easier to query without forcing all payloads into one
   database schema.

Negative:
1. The ecosystem now needs a separate shared data layer with its own
   operational decisions.
2. Some current project-local data handling will need to be reclassified as
   either canonical shared data or derived local materialization.
3. Artifact and dataset identity must be designed carefully to avoid a weak
   registry that adds ceremony without enough value.

## Testing Contract

1. Observability tests should prove that run metadata can link cleanly to
   external dataset and artifact identifiers.
2. Provenance and lineage features must fail loudly when required references are
   missing or inconsistent.
3. New storage integrations must preserve the distinction between shared
   metadata in `llm_client` and bulk payloads in the data plane.
