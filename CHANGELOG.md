# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## 0.1.0 — unreleased

First extraction of a design proven in production in a commercial product.

### Added

- Ingest → chunk → embed → extract → resolve → apply → retrieve, on Postgres +
  pgvector.
- De-duplication: canonical-name roster shown to the extractor, exact-title and
  alias matching, lexical near-duplicate nomination, type-gated merging, LLM
  adjudication in the uncertain band, in-run memo, non-lossy merges with a
  decision trail.
- Governed writes: additive operations apply, truth operations gate for agents
  and apply-with-audit for humans, classified server-side.
- Bi-temporal edges; contradictions surfaced as open questions rather than
  resolved; nothing deleted.
- Four protocols for host integration: `Embedder`, `StructuredLLM`, `Meter`,
  `CanonicalResolver`.
- MCP server with six tools; scope is server config, never a tool argument.
- `health()` reports `sources_unindexed` — text captured while the embedder was
  unavailable, which nothing else in the system would ever mention.
- Documentation: architecture, extending, MCP integration, operations.

### Security

- **Cross-tenant isolation on every read.** `RetrievalService`, the roster and
  resolution filtered on `graph_id` alone, so two tenants sharing a `graph_id`
  could read each other's nodes, source spans, evidence and health counts —
  and, worse, a new claim could merge into the other tenant's node. Scope is now
  bound at construction and every scoped statement filters both columns.
  Regression coverage in `tests/test_integration.py::TestTenantIsolation`.
- **`restructure()` now verifies scope.** It accepted `tenant_id`/`graph_id` and
  ignored them, so any source could be re-extracted by UUID across tenants. It
  raises `ScopeViolationError` instead.
- **Edge invalidation is tenant-scoped.** `SUPERSEDE` and `REMOVE_NODE`
  invalidated edges filtering on `graph_id` alone.
- **The write gate no longer trusts the caller.** `GraphService.apply` honoured
  the `risk` field on the incoming operation, and the server-side `classify()`
  had no call sites — so an operation could label itself "additive" and retire a
  claim with no review. Risk is now recomputed from the op kind and the actor,
  and the changeset records the server's verdict so a reviewer cannot be shown a
  decision the gate did not make. This also makes the documented human
  apply-with-audit path work for the first time.

### Fixed

- **The raw source survived nothing.** Capture and embedding shared one
  uncommitted transaction, so an embedder outage rolled back the source row —
  the pipeline losing the one artefact it exists to never lose, at exactly the
  moment it was under stress. `add()` now commits the captured text before any
  remote call; a failed index leaves retryable, counted rows.
- **Lexical near-duplicates were never nominated.** `titles_match()` was
  documented and tested as a nomination for adjudication but had no call site,
  so a near-identical same-type title whose summary was worded differently could
  score below the adjudication band and silently become a second node.
- `IngestionService.ingest` is split into `capture` and `index`, which take
  plain values rather than a possibly-expired ORM instance.

### Tests

326 tests, 75% coverage, run against a real Postgres with pgvector on 3.11,
3.12 and 3.13. Notable additions:

- `TestTenantIsolation` — every read surface, the roster, the merge path and
  `restructure`, each asserted separately, because isolation is only as good as
  the one query that forgets it.
- `test_governance.py` — the gate reaches its verdict without consulting the
  caller, in both directions.
- `test_chunker.py` — the citation invariant
  (`source[start:end] == chunk.text`) across 12 awkward corpora × 4 configs,
  including unicode, CRLF, whitespace-free blobs and tokens longer than the
  window; plus coverage that no text is silently dropped between chunks.
- `test_mcp.py` — scope is absent from every tool schema and discarded when it
  arrives in arguments; `remember` reports merges and gated operations honestly.
- `TestTheSourceIsNeverLost` — the raw text outlives an embedder outage.
