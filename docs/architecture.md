# Architecture

How a piece of text becomes a claim you can query, and where each decision is
made.

## The pipeline

```
add(tenant, graph, text)
  │
  ├─ capture ──────────► cg_sources          raw text, verbatim, committed first
  │                                          ─────────────────────────────────
  ├─ index ────────────► cg_segments         chunks + embeddings (retrieval index)
  │
  ├─ roster ◄─────────── cg_nodes            "here is what this graph already knows"
  │
  ├─ extract ──────────► candidate nodes/edges       ← the only unbounded LLM step
  │
  ├─ resolve ──────────► ResolvedOp[]         merge? supersede? contradict? new?
  │
  ├─ apply ────────────► cg_nodes, cg_edges   additive ops only
  │                    ► cg_changesets        everything, with the server's verdict
  │
  └─ meter ────────────► your billing system  (runs even when the run failed)
```

Two commits, deliberately. The first lands the raw text before any remote call
is made; everything after it is derived and rebuildable. See
[Durability](#durability).

## The six tables

All prefixed `cg_`. The library is a guest in your database: its own Alembic
version table (`cg_alembic_version`), no foreign keys into your schema, drops
cleanly.

| Table | Holds | Notes |
| --- | --- | --- |
| `cg_runs` | One structuring pass | The provenance spine. `metrics` records model, tokens, nodes created/merged, ops gated. |
| `cg_sources` | Raw evidence, verbatim | The one thing that cannot be rebuilt. Everything else cites a span in here. |
| `cg_segments` | Embedded chunks | **Search plumbing, not meaning.** Collapsing these into nodes turns the graph into a chunk store with extra steps. |
| `cg_nodes` | Claims and entities | `aliases` holds every surface form that ever resolved here. `properties` holds `summary`, `summaries`, `citations`, `merges`. |
| `cg_edges` | Typed, bi-temporal relations | `valid_at`/`invalid_at` is when the fact was true; `created_at`/`expired_at` is when the row existed. |
| `cg_changesets` | Proposed mutations and the gate | The part with no equivalent in comparable systems, which write unilaterally at ingest. |

`tenant_id` and `graph_id` are `String(64)` and carry no foreign key on purpose:
the library must never assume your id scheme.

## Where each decision lives

### Extraction — `services/extraction.py`

Bounded input, loud failure. A source longer than `max_source_chars` is
segmented and extracted in batches rather than truncated, because a truncated
model response is indistinguishable from "this source was empty" — permanently,
with no signal.

A provider error raises `ExtractionError`. It is never reported as an empty
result, which would mark the source `done` and lose the content while looking
successful.

### The roster — `services/roster.py`

The highest-leverage component, and the least obvious one.

Measured on a real corpus: **10 of 11 near-duplicate pairs were cross-run**. The
extractor had never been shown the graph, so each pass re-invented a surface
form for something already stored. Showing it the existing titles moves
de-duplication from *detection* to *prevention* — a reused title matches on
exact equality and needs no threshold at all.

It is passed as a **separate system block, after the static prompt**, so the
stable prefix stays cacheable while the volatile part changes per graph.

### Resolution — `services/resolution.py`

The make-or-break step, and the rules are asymmetric on purpose:

> a duplicate is annoying and reversible
> an over-merge destroys a distinct fact and is not

Decision order:

1. **Same run.** Ops apply only after the whole extraction resolves, so two
   candidates naming the same thing cannot see each other in the database.
   Without an in-run memo the pipeline manufactures duplicates inside the
   de-duplication pass itself.
2. **High cosine, same type** (≥ `merge_threshold`, 0.90). Cheap and
   unambiguous — when it fires. Measured: it fired *zero* times on a real
   100-node corpus.
3. **Exact title or alias, same type.** Decisive, and never gated behind an
   optional resolver. This is the most common real duplicate and the vector path
   does not catch it: two byte-identical titles measured **0.798**, because the
   embedded text is `title\nsummary` and differing summaries pull identical
   titles apart.
4. **Canonical entity.** If the host owns this entity, adopt its name.
5. **Adjudication.** Two ways in: cosine in the band `[related_low, merge_threshold)`,
   or a same-type title with ≥0.8 token overlap. Both are *nominations*. The
   model decides, biased toward "distinct".
6. **New node.**

**Type equality gates every merge.** Cosine cannot see the type system, and the
single highest-scoring pair in the measured corpus was a screen and a flow
sharing a name — exactly the merge that must never happen. A cross-type
"duplicate" verdict from the adjudicator is refused in code rather than trusted
to the prompt.

### The write gate — `services/graph.py`

Every write to `cg_nodes`/`cg_edges` funnels through `GraphService`. That is not
tidiness; a second write path is a hole in the gate.

Two axes decide an operation's fate:

| | human actor | agent actor |
| --- | --- | --- |
| presentation (`SET_LAYOUT`) | applies | applies |
| additive (`ADD_NODE`, `ADD_EDGE`, `MERGE_NODE`, `CONTRADICT`) | applies | applies |
| truth (`UPDATE_NODE`, `SUPERSEDE`, `DISCONNECT`, `REMOVE_NODE`) | applies, with audit | **gated** |
| anything unrecognised | gated | gated |

The classification is computed **server-side from the op kind**, by
`classify()`. `ResolvedOp.risk` is a proposal and is never consulted — otherwise
anything able to construct an op could label a supersede "additive" and retire a
claim nobody reviewed. The changeset row is written from the server's verdicts,
so what a reviewer reads and what the gate did cannot disagree.

### Retrieval — `services/retrieval.py`

Two shapes for two questions. `search_segments` answers "what did the source
material say" — verbatim spans with provenance. `retrieve_subgraph` answers
"what do we know about this, and how does it connect" — locate by similarity,
then expand.

Expansion caps at 1–2 hops. Graph structure pays for itself on genuinely
multi-hop compositional questions and is otherwise a slower way to do vector
search; the dominant real access pattern is one hop.

## Scoping

`tenant_id` and `graph_id` are required on every public call and never inferred
from ambient state.

`RetrievalService` takes both **at construction** and has no method that accepts
a different one, which makes "the caller passed a tenant but the query only
filtered by graph" unrepresentable rather than merely absent. Writes assert
scope again in SQL at the point of write (`_assert_nodes_in_scope`) — the route
guard is the first line, but one missing predicate in hand-written SQL is a
cross-tenant leak.

Why `graph_id` alone is not enough: it is an opaque string the host chooses, so
two tenants can pick the same one. A read scoped by graph alone returns another
tenant's rows with no error. The write path is worse — resolution decides what a
claim *merges into*, so an unscoped candidate lookup joins two customers' graphs
with a write nothing downstream can distinguish from a legitimate merge.

## Durability

The one guarantee: **the raw source survives everything downstream of it.**

`add()` commits the captured text before making any remote call. Past that
commit, every failure costs derived data that can be rebuilt by re-running
structuring:

| Failure | Outcome |
| --- | --- |
| Embedder down at capture | Source stored, unindexed. `health()["sources_unindexed"]` counts it. |
| Extraction fails | Source `failed` with the error; `raw_content` untouched; retryable. |
| Adjudicator fails | Fails open to "distinct" — a duplicate, which a human can merge. Set `fail_closed=True` to route to the gate instead. |
| Metering fails | Logged and ignored. The tokens are already spent; a metering failure must not roll back real work. |

Structuring runs inside a `SAVEPOINT` so a failure undoes the graph writes
without taking the source row — or the caller's transaction — with it.

An embedder **never** substitutes a vector on failure. A fabricated vector is
indistinguishable from a real one once stored and silently corrupts every
similarity decision made against it thereafter, including merges, which are
irreversible.

## Indexing

No `ivfflat` index ships in the migration, on purpose. Its centroids are trained
at `CREATE INDEX` time — on an empty table, for a fresh migration — and are
never retrained. With `lists=100 / probes=1` plus a selective `graph_id`
post-filter, recall collapses *silently*: the query returns fewer rows than
asked for and reports success.

At moderate scale an exact scan over the filtered set is sub-millisecond at 100%
recall. Past roughly 50k nodes in one graph, add HNSW — it builds incrementally,
needs no training pass, and degrades gracefully. The migration ships the exact
statements as comments.

## Further reading

- [Extending contextgraph](extending.md) — the four Protocols and the ontology
- [MCP server](mcp.md) — running the graph as agent tools
- [Operations](operations.md) — migrations, health, and what to alert on
