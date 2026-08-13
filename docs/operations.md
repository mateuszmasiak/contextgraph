# Operations

Running contextgraph in production: migrations, monitoring, and the failures
worth alerting on.

## Migrations

The schema is a guest in your database. Six tables prefixed `cg_`, its own
Alembic version table (`cg_alembic_version`), no foreign keys into your schema.

```bash
export DATABASE_URL=postgresql://user:pass@host/yourdb
alembic -c alembic.ini upgrade head
```

Requires the `vector` extension; the migration runs
`CREATE EXTENSION IF NOT EXISTS vector`, which needs a role with sufficient
privilege the first time.

Three things in `migrations/env.py` exist because of the guest status:

- **Its own version table.** A host that already runs Alembic owns
  `alembic_version`; sharing it means whichever tool stamps last convinces the
  other that migrations it has never seen are applied.
- **Autogenerate is fenced to `cg_` objects.** Without the `include_object`
  filter, an autogenerate run reads every table it does not know about as
  "dropped" and emits `drop_table` for your data.
- **The URL is coerced to asyncpg.** `DATABASE_URL` in the wild is whatever your
  cloud provider or `psql` wanted; libpq-shaped URLs reach asyncpg as a
  connect-time `TypeError` that reads like a driver bug.

`downgrade` drops the six tables and their indexes. It does **not** drop the
`vector` extension — that is database-wide and may be in use by your own tables.

## Health

```python
await graph.health(tenant_id, graph_id)
```

```python
{
  "nodes": 1284,
  "nodes_without_embedding": 0,
  "active_edges": 3106,
  "sources_pending": 0,
  "sources_failed": 0,
  "sources_unindexed": 0,
  "changesets_awaiting_review": 3,
}
```

Every one of these is a signal that fails *silently* otherwise.

### `nodes_without_embedding` — alert on any non-zero value

The leading indicator, and the worst one. A node with no embedding is invisible
to retrieval **and to de-duplication**, permanently. It will never be found by
`recall`, and every future extraction of the same fact will create yet another
node because the candidate lookup cannot see it. They accumulate quietly and
nothing else in the system will ever mention them.

Cause is almost always an embedder failure during `apply_gated` — a changeset
approved while the provider was down. Backfill:

```sql
SELECT id, title FROM cg_nodes
WHERE graph_id = $1 AND embedding IS NULL AND deleted_at IS NULL;
```

Then re-embed `title\nsummary` for each and `UPDATE ... SET embedding = ...`.

### `sources_unindexed` — alert if it stays non-zero

Text captured while the embedder was unavailable. This is a *deliberate*
outcome: `add()` commits the raw source before making any remote call, so an
embedding outage costs the search index rather than the evidence. The rows are
retryable — but nothing retries them for you.

```sql
SELECT id FROM cg_sources src
WHERE graph_id = $1
  AND coalesce(raw_content,'') <> ''
  AND NOT EXISTS (SELECT 1 FROM cg_segments WHERE source_id = src.id);
```

Re-run `restructure(tenant, graph, source_id)`, or re-ingest.

### `sources_pending` — alert if it grows

Sources captured but never structured. A steady non-zero count means
structuring is not running at all. A brief spike during an outage is normal.

### `sources_failed` — investigate, then retry

`cg_sources.extraction_error` holds the message. `raw_content` is untouched, so
these are always retryable:

```python
await graph.restructure(tenant_id, graph_id, source_id)
```

### `changesets_awaiting_review` — a queue, not an error

Operations that would change or retire existing knowledge, waiting for a human.
Alert on **age**, not count: a changeset nobody looked at for a week means the
review loop is not staffed, and the graph is quietly diverging from what agents
believe they wrote.

```python
for cs in await graph.pending(tenant_id, graph_id):
    print(cs["id"], cs["summary"], cs["created_at"], len(cs["ops"]))

await graph.review(tenant_id, graph_id, changeset_id,
                   approve=True, reviewer_id="alice")
```

`review()` takes a row lock, so two reviewers hitting approve concurrently
cannot both apply the same operations; the loser gets
`ConcurrentModificationError`.

## Cost

Every run records what it spent, in `cg_runs.metrics`:

```sql
SELECT
  date_trunc('day', created_at) AS day,
  metrics->>'model'                        AS model,
  sum((metrics->>'llm_calls')::int)        AS calls,
  sum((metrics->>'input_tokens')::int)     AS input_tokens,
  sum((metrics->>'output_tokens')::int)    AS output_tokens,
  sum((metrics->>'embed_tokens')::int)     AS embed_tokens,
  sum((metrics->>'nodes_created')::int)    AS created,
  sum((metrics->>'nodes_merged')::int)     AS merged
FROM cg_runs
WHERE tenant_id = $1 AND status = 'completed'
GROUP BY 1, 2 ORDER BY 1 DESC;
```

For per-tenant billing, implement the [`Meter`](extending.md#meter) protocol
rather than querying this table. Note it is called for **failed** runs too: the
provider was still paid for whatever was consumed before the failure.

**What drives cost.** One extraction call per source (or per 200k-char batch),
plus one adjudication call per candidate that lands in the uncertain band. If
adjudication calls dominate, raise `resolution.related_low`. The roster adds
roughly 1.1k–4.5k input tokens per extraction, and it is worth every one — it is
what stops the graph filling with near-duplicates.

Both bundled LLM adapters order system blocks stable-prefix-first so provider
prompt caching applies to the ontology and instructions.

## Scaling

**Vector indexes.** None ship by default, deliberately — see
[architecture](architecture.md#indexing). Past roughly 50k nodes in a single
graph:

```sql
CREATE INDEX CONCURRENTLY ix_cg_nodes_embedding_hnsw
    ON cg_nodes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);
```

```sql
CREATE INDEX CONCURRENTLY ix_cg_segments_embedding_hnsw
    ON cg_segments USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);
```

Then, for the filtered queries this schema issues (pgvector ≥ 0.8):

```sql
SET hnsw.iterative_scan = relaxed_order;
```

Use HNSW, not ivfflat. Match the operator class to the distance operator —
`vector_cosine_ops` serves `<=>`; an index built for `<->` is simply not used,
with no error.

**Edge growth.** Nothing is deleted, so invalidated edges accumulate without
bound while the live set stays roughly constant. The partial index
`ix_cg_edges_graph_live` covers this, and Postgres will only use it if the query
spells the predicate `invalid_at IS NULL` — which the library does.

**The same-type title scan.** `_match_by_title` scans all active nodes of one
type per candidate. The set is small by construction; if a graph grows a type
with tens of thousands of nodes, add a generated normalised-title column and
index it.

## Backup and deletion

`cg_sources.raw_content` is the only irreplaceable data. Nodes, edges, segments
and changesets can all be rebuilt by re-running structuring over the sources —
slowly, and at provider cost, but completely.

Tenant deletion, in dependency order:

```sql
DELETE FROM cg_edges      WHERE tenant_id = $1;
DELETE FROM cg_nodes      WHERE tenant_id = $1;
DELETE FROM cg_segments   WHERE tenant_id = $1;
DELETE FROM cg_changesets WHERE tenant_id = $1;
DELETE FROM cg_sources    WHERE tenant_id = $1;
DELETE FROM cg_runs       WHERE tenant_id = $1;
```

Every table carries a `tenant_id` index for exactly this.
