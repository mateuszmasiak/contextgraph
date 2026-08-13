# Extending contextgraph

contextgraph owns the graph — its schema, resolution rules, temporal semantics
and write gate. It deliberately owns none of the following, because every host
application already has opinions about them:

- which embedding model you pay for
- which LLM you call, and with whose key
- how (or whether) you bill for that
- what your canonical entities are

Each is a `Protocol`, satisfied **structurally**. No base class to inherit, no
registry to configure — if your object has the methods, it works.

## Embedder

```python
class Embedder(Protocol):
    dimensions: int
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def tokens_used(self) -> int: ...
```

Two hard requirements:

**`dimensions` must match the vector column the migration created.** The
mismatch is caught at construction, not 400 rows into a backfill.

**On failure, raise.** This is the single most consequential rule in the whole
extension surface. Returning a zero or random vector is undetectable downstream:
a fabricated vector is indistinguishable from a real one once stored, and it
silently corrupts every similarity decision made against it thereafter —
including merges, which are irreversible. Failing is recoverable; the raw source
is retained and structuring re-runs.

`embed_batch` must return exactly one vector per input, **in order**, and must
verify the count rather than zipping blindly — a short response otherwise pairs
every vector after the gap with the wrong text, invisibly.

### Example: a local sentence-transformers embedder

```python
import asyncio
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    dimensions = 384

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model)
        self._tokens = 0

    @property
    def tokens_used(self) -> int:
        return self._tokens  # local inference is free; report 0 spend

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Encoding is CPU-bound and blocking; keep it off the event loop.
        vectors = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True
        )
        out = [v.tolist() for v in vectors]
        if len(out) != len(texts):
            raise RuntimeError(f"got {len(out)} vectors for {len(texts)} texts")
        return out
```

A 384-dimension model needs a matching column. Change
`EMBEDDING_DIM` in a **new** migration and re-embed everything — distances
between vectors from different models are meaningless even at equal width.

```python
graph = ContextGraph(
    sessions,
    embedder=LocalEmbedder(),
    config=Config(embedding_dimensions=384),
)
```

## StructuredLLM

```python
class StructuredLLM(Protocol):
    async def complete(
        self, *, system: list[str], user: str, schema: type,
        max_tokens: int = 4096, temperature: float = 0.0,
    ) -> tuple[Any | None, dict[str, Any] | None]: ...
```

Returns `(parsed, usage)`.

**`system` is a list, and the order matters.** Block 0 is the stable prefix
(ontology, instructions); later blocks are per-call content (the roster).
Provider prompt caching is a prefix match, so reversing this invalidates the
cache on every call. Send them as separate system blocks rather than joining
them.

**Return `usage`.** Many structured-output wrappers discard it by default, which
makes every call unmeterable. Both bundled adapters pass `include_raw=True`
precisely to keep it.

**`parsed` may be `None`** when the model returned something unparseable. The
caller treats that as an empty result. Transport and auth failures should
propagate — swallowing them silently produces an empty graph.

Bundled: `contextgraph.llm.AnthropicLLM`, `contextgraph.llm.OpenAILLM`.

## Meter

```python
class Meter(Protocol):
    async def record(
        self, *, tenant_id: str, operation: str, model: str,
        input_tokens: int, output_tokens: int,
        embed_tokens: int, embed_model: str,
    ) -> None: ...
```

Called once per run, after the work. Two rules:

**It must not raise.** The tokens are already spent, so a metering failure must
never roll back real work. Exceptions are caught and logged.

**It is called for failed runs too**, because the provider was still paid for
whatever was consumed before the failure. Reporting spend only on success
under-reports exactly when things are going worst.

```python
class BillingMeter:
    async def record(self, *, tenant_id, operation, model,
                     input_tokens, output_tokens, embed_tokens, embed_model):
        await usage_table.insert(
            tenant_id=tenant_id, operation=operation,
            model=model, input_tokens=input_tokens,
            output_tokens=output_tokens,
            embed_tokens=embed_tokens, embed_model=embed_model,
        )
```

## CanonicalResolver

The important one.

```python
class CanonicalResolver(Protocol):
    async def resolve(
        self, *, graph_id: str, node_type: str, title: str
    ) -> dict[str, Any] | None: ...
```

This is the seam that keeps contextgraph honest about ownership. **If your
application already has a `screens` table, that table — not the graph — is the
authority on what screens exist.** Implement this and an extracted "checkout
page" resolves to your canonical "Checkout Screen" instead of becoming a second,
competing record of the same thing.

Return `None` when nothing matches (the common case). Otherwise return a dict
with at least `title`; include `node_id` if a graph node already represents this
entity, and `ref` for any host identifier to carry on the node as
`properties.source_entity`.

> **Return a human-readable `title` or `None` — never an opaque id or slug.**
> Whatever you return becomes the node's title and is then offered back to the
> extractor as canonical vocabulary, so an id returned here propagates
> permanently.

```python
class ScreenResolver:
    def __init__(self, session_factory):
        self._sessions = session_factory

    async def resolve(self, *, graph_id, node_type, title):
        if node_type != "entity/screen":
            return None
        async with self._sessions() as s:
            row = (await s.execute(
                text("""
                    SELECT id::text, name, cg_node_id::text
                    FROM screens
                    WHERE project_id = :g
                      AND lower(name) % lower(:t)     -- pg_trgm similarity
                    ORDER BY similarity(lower(name), lower(:t)) DESC
                    LIMIT 1
                """),
                {"g": graph_id, "t": title},
            )).first()
        if row is None:
            return None
        return {
            "title": row[1],                  # human-readable, never row[0]
            "node_id": row[2],                # if a node already exists
            "ref": {"screens_id": row[0]},    # your id, carried on the node
        }
```

How resolution uses it, in `_resolve_canonical`:

- Exact normalised title equality → merge, no LLM call.
- A fuzzy match with an existing `node_id` → adjudicated. If the verdict is
  "different", the canonical title is **not** adopted — the adjudicator just
  ruled it belongs to something else.
- A canonical entity with no node yet → create one under its name, keeping the
  extracted phrasing as an alias so the next run's wording matches.

Only types in `Ontology.canonical_types` are offered to the resolver.

## The Ontology

The vocabulary is data, not schema. Node kinds, node types and edge types are
plain strings in the database, so the taxonomy evolves without a migration.

```python
from contextgraph import Ontology, ContextGraph

legal = Ontology(
    node_types={
        "knowledge": (
            "knowledge/obligation",
            "knowledge/risk",
            "knowledge/precedent",
            "knowledge/open_question",
        ),
        "entity": (
            "entity/party",
            "entity/contract",
            "entity/clause",
            "entity/jurisdiction",
        ),
    },
    edge_types=(
        "binds", "amends", "references", "supersedes", "contradicts",
    ),
    open_question_type="knowledge/open_question",
    canonical_types=frozenset({"entity/party", "entity/contract"}),
    strict=False,
)

graph = ContextGraph(sessions, embedder=..., llm=..., ontology=legal)
```

Extraction, resolution and validation all follow it.

Notes:

- **Types are `"<kind>/<type>"`.** `kind` is derived from the prefix when the
  model omits it, which it routinely does however clearly you ask.
- **`open_question_type` must exist in `node_types`.** Contradictions become a
  node of this type, so a conflict is a thing you can look at, link to and
  resolve — not a notification.
- **`strict=False` is advisory**: an unknown type is logged, not rejected. Set
  `True` and a model that invents a type loses the whole node.
- **`canonical_types` empty means "the graph owns every type"**.

One rule is not negotiable whatever ontology you supply: **type equality gates
merging.** Two nodes of different types are never the same thing, however
similar their text.

## Tuning

Every default in `config.py` was measured. The comments record the measurement
because the numbers are only defensible alongside them.

| Setting | Default | Change it when |
| --- | --- | --- |
| `resolution.merge_threshold` | 0.90 | Almost never. Measured: fired **zero** times on a real corpus; the highest same-type pair was 0.804. Lowering it does not help — true duplicates measured 0.70–0.80 and genuinely distinct siblings occupy the same band, with ~0.035 separating the highest true negative from the lowest true positive. Cosine ranks candidates here; it cannot decide them. |
| `resolution.related_low` | 0.74 | You are paying for too many adjudication calls (raise), or missing duplicates (lower, carefully). |
| `resolution.candidate_limit` | 5 | Emphatically not 1 — a top-1 lookup lets a cross-type neighbour mask the true same-type match. |
| `resolution.fail_closed` | `False` | You would rather a provider outage produce gated ops than duplicates. Open is safer unattended: duplicates are recoverable, merges are not. |
| `roster.limit` | 200 | Measured at ~9 tokens/entry for entity titles and ~23 for full-sentence claims, so 200 lands between ~1.1k and ~4.5k tokens. |
| `roster.enabled` | `True` | Effectively never. This is the highest-leverage component in the pipeline. |
| `chunking.chunk_size` | 1200 | Your sources have very different structure. |
| `extraction.max_source_chars` | 200_000 | Rarely — beyond this, sources are batched, not truncated. |

**If you change a threshold, re-measure.** Do not reason about it from first
principles; the first-principles answer is wrong in an instructive way, which is
why the note is there.
