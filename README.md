# contextgraph

[![CI](https://github.com/mateuszmasiak/contextgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/mateuszmasiak/contextgraph/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/contextgraph.svg)](https://pypi.org/project/contextgraph/)
[![Python](https://img.shields.io/pypi/pyversions/contextgraph.svg)](https://pypi.org/project/contextgraph/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A governed, bi-temporal knowledge graph for agent memory. Postgres + pgvector — no graph database, no second datastore.

Most agent memory is a pile of embedded chunks. That works until the same fact
gets written three times in slightly different words, until an agent quietly
overwrites something a human decided, or until you need to know *why* the
system believes something. contextgraph is the layer that handles those three
problems, and not much else.

```python
from contextgraph import ContextGraph
from contextgraph.embeddings import OpenAIEmbedder
from contextgraph.llm import AnthropicLLM

graph = ContextGraph(
    session_factory,
    embedder=OpenAIEmbedder(api_key=...),
    llm=AnthropicLLM(api_key=...),
)

await graph.add("acme", "webapp", "We chose Postgres over DynamoDB for the event store.")
await graph.add("acme", "webapp", "Postgres is confirmed for event storage.")

ctx = await graph.recall("acme", "webapp", "what database did we pick?")
print(ctx.to_text())
```

The second `add` does not create a second node. It merges into the first and
keeps both phrasings.

---

## What it actually does

**De-duplicates properly.** Before extracting, the model is shown what the graph
already contains, so it reuses existing names instead of inventing variants.
This is the whole ballgame and it is not obvious — see [Why the roster
matters](#why-the-roster-matters).

**Gates writes that matter.** Additive operations apply immediately. Operations
that *change or retire* existing knowledge are queued for human review when an
agent proposes them, and applied-with-audit when a human does. Comparable memory
systems write unilaterally at ingest.

**Never deletes.** A contradiction produces an open question holding both
claims, not a winner. A superseded claim keeps its row and its edges. You can
ask what was believed and when.

**Cites everything.** Every claim points at the source span it came from, so an
extraction error is recoverable instead of permanent.

## Install

```bash
pip install contextgraph[openai,anthropic]
```

Requires Postgres with the `vector` extension. Apply the schema:

```bash
export DATABASE_URL=postgresql://user:pass@localhost/yourdb
alembic -c alembic.ini upgrade head
```

Six tables, all prefixed `cg_`. The library is a guest in your database: it
uses its own Alembic version table, touches nothing else, and drops cleanly.

## The API

Six questions, plus a governance pair.

| Method | What it answers |
|---|---|
| `add(tenant, graph, text)` | Record this |
| `recall(tenant, graph, query)` | What do we know about this, and how does it connect |
| `search(tenant, graph, query)` | What did the source material actually say |
| `neighbors(tenant, graph, node_id)` | What does this touch (impact analysis) |
| `evidence(tenant, graph, node_id)` | Why do we believe this |
| `conflicts(tenant, graph)` | What do we believe that can't all be true |
| `pending` / `review` | What needs a human, and act on it |

Plus `restructure()` to re-extract a stored source, and `health()`, which
reports the things that fail silently: nodes with no embedding (invisible to
search and de-duplication, forever), text captured while the embedder was down,
sources stuck pending, changesets nobody reviewed.

`tenant_id` and `graph_id` are required on every call and never inferred from
ambient state — an API that lets you forget the scope eventually lets you cross
it. Both are filtered in every scoped statement, not just `graph_id`: a
`graph_id` is an opaque string you choose, so two tenants can pick the same one,
and a read scoped by graph alone returns another tenant's rows with no error.
The write path is worse — resolution decides what a claim *merges into*, so an
unscoped candidate lookup would join two customers' graphs with a write nothing
downstream can tell from a legitimate merge. See
`tests/test_integration.py::TestTenantIsolation`.

## Why the roster matters

The default assumption is that de-duplication is a similarity-threshold
problem. Measured on a real corpus, it is not.

- The 0.90 auto-merge threshold fired **zero times**. The highest same-type
  pair in the corpus was 0.804.
- Two nodes with **byte-identical titles** scored **0.798**, because the
  embedded text includes the summary and differing summaries pull identical
  titles apart.
- Lowering the threshold does not help: true duplicates measured 0.70–0.80, and
  genuinely distinct siblings (`Login screen` vs `Dashboard screen`, 0.679) sit
  in the same band. About 0.035 separated the highest true negative from the
  lowest true positive.

Cosine ranks candidates here. It cannot decide them.

What actually worked was fixing the *cause*: 10 of 11 near-duplicate pairs were
cross-run, because the extractor had never been told what already existed and
re-invented a surface form every time. Showing it the existing titles turns
de-duplication from detection into prevention — a reused title matches on exact
equality and needs no threshold at all.

A/B over two sources, where the second restated the first in drifted wording:

| | nodes after | source 2 created | merged |
|---|---|---|---|
| roster on | 5 | **0** | 5 |
| roster off | 6–7 | 1 | 4 |

Roster-on captured `Task Status Tracking UI` as an *alias* of `Task Status
Tracking`. Roster-off invented a separate `Postgres Event Store` node alongside
the decision that already said so.

This has a cost, stated plainly: the model can force-fit a genuinely new thing
onto a listed title. The prompt biases toward "treat it as NEW when unsure",
and the wording the source actually used is recorded as `surface_phrase`, so a
force-fit leaves a trail instead of silently attaching a claim to the wrong
entity.

## Design decisions you may disagree with

**Merging requires type equality.** However similar the text, a `flow` never
merges into a `screen`. In measured data the single highest-scoring pair in the
corpus was exactly that — a screen and a flow sharing a name. An over-merge
destroys a distinct fact and cannot be spotted by inspection afterwards; a
duplicate is obvious and reversible. The asymmetry drives every rule here.

**Traversal stops at two hops.** Graph structure pays for itself on genuinely
multi-hop compositional questions and is otherwise a slower way to do vector
search. The dominant real access pattern is one hop. Deep traversal is a
deliberate non-feature.

**No ivfflat index by default.** Centroids trained on an empty table are
meaningless, and with `lists=100 / probes=1` plus a selective `graph_id`
post-filter, recall collapses *silently* — the query returns fewer rows than
asked for and reports success. At moderate scale an exact scan over the
filtered set is sub-millisecond at 100% recall. The migration ships commented
HNSW statements for when a graph outgrows that.

**Embedding failures raise.** Never a zero vector, never a random one. A
fabricated vector is indistinguishable from a real one once stored and silently
corrupts every similarity decision made against it thereafter. Failing is
recoverable — the raw source is kept and structuring re-runs.

**`add()` commits twice.** The raw text lands before any remote call is made.
That costs atomicity across the whole call and buys the library's one durability
guarantee: past the first commit, every later failure costs derived data that
can be rebuilt, and none of them can cost the source. An embedder outage then
leaves stored-but-unindexed text, which `health()` counts rather than hides.

**Bi-temporal, but half of it may stay inert.** Edges carry valid time and
system time. If your facts become true when you record them, valid time equals
system time and that is a legitimate resting state, not an unfinished one.

## Bring your own everything

Four protocols, satisfied structurally — no base class, no registry:

- `Embedder` — any model; `OpenAIEmbedder` included
- `StructuredLLM` — any provider; Anthropic and OpenAI included
- `Meter` — optional; report token spend to your billing system
- `CanonicalResolver` — **the important one.** If your app already has a
  `screens` table, that table is the authority on what screens exist.
  Implement this and an extracted "checkout page" resolves to your canonical
  "Checkout Screen" instead of becoming a competing record of the same thing.

The ontology is data, not schema. The default vocabulary is
product/specification shaped; pass your own `Ontology` and extraction,
resolution and validation all follow it.

## MCP server

```bash
pip install contextgraph[mcp]
export DATABASE_URL=... OPENAI_API_KEY=... ANTHROPIC_API_KEY=...
export CONTEXTGRAPH_TENANT=acme CONTEXTGRAPH_GRAPH=webapp
contextgraph-mcp
```

Exposes `recall`, `search_sources`, `remember`, `neighbors`, `evidence` and
`conflicts`.

Tenant and graph are **server configuration, never tool arguments** — if a model
could pass them, a prompt injection could redirect a read or a write to another
tenant's graph. Any scope keys appearing in tool arguments are discarded.

`remember` reports honestly when something was merged rather than added, and
reports gated operations as *awaiting review* rather than done. An agent that
believes it wrote something it did not will confidently tell the user so.

## Documentation

- **[How it works](docs/how-it-works.md)** — the illustrated walkthrough: the
  pipeline, the resolution ladder, and why cosine similarity cannot decide a
  merge. Start here.
- [Architecture](docs/architecture.md) — the pipeline, the six tables, and where
  each decision is made
- [Extending](docs/extending.md) — the four Protocols, custom ontologies, and
  what every threshold means
- [MCP server](docs/mcp.md) — running the graph as agent tools
- [Operations](docs/operations.md) — migrations, what to alert on, cost, scaling
- [Security](SECURITY.md) — threat model and reporting
- [Contributing](CONTRIBUTING.md) — the five invariants a patch must not break

## Status

`0.1.0`. The design is proven in production in a commercial product; this
extraction of it is new. The public API may shift before `1.0`.

Not yet included, in rough priority order: hybrid lexical+vector retrieval with
RRF fusion (the highest-value addition — users search for literal names, which
is where dense retrieval is weakest), reranking, community summarisation, and a
conflict-resolution operation to close open questions.

Contributions are welcome — [CONTRIBUTING.md](CONTRIBUTING.md) starts with the
five invariants, because that is where a well-meaning patch is most likely to go
wrong.

## License

MIT
