# Changelog

## 0.1.0 — unreleased

First extraction of a design proven in production in a commercial product.

- Ingest → chunk → embed → extract → resolve → apply → retrieve, on Postgres +
  pgvector.
- De-duplication: canonical-name roster shown to the extractor, exact-title and
  alias matching, type-gated merging, LLM adjudication in the uncertain band,
  in-run memo, non-lossy merges with a decision trail.
- Governed writes: additive operations apply, truth operations gate for agents
  and apply-with-audit for humans, classified server-side.
- Bi-temporal edges; contradictions surfaced as open questions rather than
  resolved; nothing deleted.
- Four protocols for host integration: Embedder, StructuredLLM, Meter,
  CanonicalResolver.
- MCP server with six tools; scope is server config, never a tool argument.
