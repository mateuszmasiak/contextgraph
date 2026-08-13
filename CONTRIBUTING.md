# Contributing to contextgraph

Thanks for looking. This document is mostly about *what this project refuses to
do*, because that is where a well-meaning patch is most likely to go wrong.

## The five invariants

These are not style preferences. A change that violates one of them will be
asked to change, however good the rest of it is — each corresponds to a failure
that is silent, permanent, or both.

**1. An over-merge is worse than a duplicate.**
Merging two distinct claims destroys a fact and cannot be spotted by inspection
afterwards. A duplicate is obvious and reversible. Every ambiguous case must
resolve to "keep both". This is why merging requires type equality, why the
adjudicator's prompt says "when in doubt answer distinct", and why lexical
overlap nominates rather than decides.

**2. Never fabricate a vector.**
On failure, an `Embedder` raises. It does not return a zero vector, a random
vector, or a cached one. A fabricated vector is indistinguishable from a real
one once stored and silently corrupts every similarity decision made against it
thereafter — including merges, which are irreversible. Failing is recoverable:
the raw source is retained and structuring re-runs.

**3. Nothing is deleted.**
Contradictions produce an open question holding both claims, not a winner.
Superseded nodes keep their rows and their edges. `DISCONNECT` sets
`invalid_at`; it does not `DELETE`. An audit trail you can delete from is not an
audit trail.

**4. Every scoped statement filters `tenant_id` *and* `graph_id`.**
`graph_id` alone is not an isolation boundary — nothing stops two tenants
choosing the same graph name, and something will, since the MCP server's own
default is the literal string `"default"`. A read scoped by graph alone returns
another tenant's rows with no error. Worse, an unscoped *resolution* query folds
one tenant's claim into another's node, which no later query can distinguish
from a legitimate merge. If you add a query, add both predicates. See
`tests/test_integration.py::TestTenantIsolation`.

**5. The gate computes risk; the caller only proposes it.**
`ResolvedOp.risk` is a suggestion. `GraphService.apply` recomputes it from the
op kind and the actor via `classify()`. If the gate honoured the incoming field,
anything able to construct an op could label a supersede "additive" and retire a
claim nobody reviewed — which is the entire thing the gate exists to prevent.

## Setup

Requires Python 3.11+ and Docker (for the integration suite).

```bash
git clone https://github.com/mateuszmasiak/contextgraph
cd contextgraph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pre-commit install
```

Bring up Postgres with pgvector and apply the schema:

```bash
docker run -d --name cg-pg -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=contextgraph_test -p 5432:5432 pgvector/pgvector:pg16
```

```bash
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/contextgraph_test
alembic -c alembic.ini upgrade head
```

## Running the tests

A bare `pytest` runs the unit suite only — integration tests are deselected in
`pyproject.toml` because they need a live database.

```bash
pytest
```

```bash
pytest -m integration
```

```bash
ruff check . && mypy src
```

Install the `[all]` extra before trusting a local `mypy` run. The optional
provider SDKs are under `ignore_missing_imports`, which suppresses errors about
a package being *absent* — not errors against one that is installed. Without the
extras your local check is strictly weaker than CI's, which is exactly how an
incompatible `mcp` major version once reached `main`.

No test in this repository calls a real model provider. The suite uses a
deterministic `FakeEmbedder` whose similarity is *controllable* (see
`tests/conftest.py`), which is the only way to assert on threshold behaviour
without paying for it and accepting flakiness.

## What a good patch looks like

**Integration tests over mocked SQL.** The interesting failures in this package
are SQL failures, and a mocked SQL test asserts that a string equals a string.
If your change touches a query, test it against the real database.

**Comments explain *why*, not *what*.** The existing comments are unusually
dense because most of them record a decision that looks wrong until you know the
measurement behind it. Match that. A comment that restates the code is noise; a
comment that says "the instinct is to lower this threshold — do not, and here is
the data" is the reason this codebase is maintainable.

**Thresholds come with measurements.** Every number in `config.py` was measured
against a real corpus and the measurement is recorded next to it. If you change
one, re-measure and update the note. Do not reason about it from first
principles — the first-principles answer is wrong in an instructive way, which
is exactly why the note is there.

**Docstrings on the seams.** `protocols.py` is the public extension surface.
Changes there need to say what an implementer must guarantee, not just what the
signature is.

## Things that are deliberate non-features

Please open an issue to discuss before implementing these; each was considered
and declined for a stated reason.

- **Traversal beyond two hops.** Graph structure pays for itself on genuinely
  multi-hop compositional questions and is otherwise a slower way to do vector
  search. The dominant real access pattern is one hop.
- **An `ivfflat` index in the migration.** Centroids trained on an empty table
  are meaningless and are never retrained; with a selective post-filter, recall
  collapses *silently*. See the long comment at the end of
  `migrations/versions/0001_initial.py`. HNSW is the answer past ~50k nodes.
- **Auto-resolving contradictions.** Picking a winner automatically is how a
  graph quietly becomes wrong.
- **Inferring tenant or graph from ambient state.** Both are required on every
  call. An API that lets you forget the scope will eventually let you cross it.

## Wanted

Roughly in order of value:

1. **Hybrid retrieval** — BM25 or `tsvector` fused with the dense results via
   RRF. Users and agents search for literal names, which is exactly where dense
   retrieval is weakest. This is the highest-value addition to the library.
2. **Reranking** over the candidate set before it reaches the context window.
3. **A conflict-resolution operation** to close open questions with an audit
   record, rather than leaving them open forever.
4. **Community summarisation** over dense subgraphs.
5. **More `CanonicalResolver` examples** — this is the seam most people will
   need and the least obvious to implement well.

### Good first issues

**Provider adapter tests.** `llm/anthropic.py`, `llm/openai.py` and
`embeddings/openai.py` are the least-covered modules in the package, because
testing them means faking a provider client rather than a database. They have
real behaviour worth pinning: the OpenAI embedder orders vectors by the
provider's `index` field rather than response position, refuses a short batch,
rejects a degenerate (zero-norm) vector, and L2-normalises a `dimensions`-reduced
response. Both LLM adapters send one system block per prompt section so that
provider prompt caching sees a stable prefix. None of that is currently asserted.
Inject a fake client — `OpenAIEmbedder(client=...)` already takes one.

**A worked `CanonicalResolver` example** against a real table, as an integration
test. See `docs/extending.md`.

## Pull requests

- Branch from `main`, one logical change per PR.
- CI must be green: `ruff`, `mypy`, unit tests and integration tests on 3.11,
  3.12 and 3.13.
- Add an entry to `CHANGELOG.md` under "Unreleased".
- If you changed behaviour, say which invariant above it interacts with and why
  it does not violate it.

## Reporting security issues

Do not open a public issue. See [SECURITY.md](SECURITY.md).

## Licence

By contributing you agree that your contributions are licensed under the MIT
Licence, the same terms that cover the project.
