# Security Policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/mateuszmasiak/contextgraph/security/advisories/new).

Do not open a public issue for anything that could be exploited before a fix
ships.

Please include the affected version, a description of the impact, and the
smallest reproduction you can manage. You should get an acknowledgement within
72 hours and an assessment within a week. Anyone who reports a valid issue will
be credited in the advisory and the changelog unless they ask not to be.

## Supported versions

Pre-1.0, only the latest release receives fixes.

| Version | Supported |
| --- | --- |
| 0.1.x | ✅ |

## Threat model

contextgraph is a library that runs inside your application, against your
database, with credentials you supply. It has no network listener of its own.
The MCP server is the one exception and is covered separately below.

### What the library is responsible for

**Tenant and graph isolation.** `tenant_id` and `graph_id` scope every row.
Every statement that reads or writes scoped data filters on **both**, and node
ids supplied by a caller are re-checked in SQL at the point of write
(`_assert_nodes_in_scope`) rather than trusted from the route that produced
them.

`graph_id` alone is deliberately not treated as sufficient. It is an opaque
string the host chooses, so two tenants can pick the same one — and a read
scoped by graph alone would return another tenant's rows with no error at all.
The corresponding write path is worse: resolution decides what a new claim
*merges into*, so an unscoped candidate lookup would fold one tenant's claim
into another tenant's node, joining two customers' graphs with a write that no
later query can distinguish from a legitimate merge.

The roster carries the same weight for a different reason: it is the one read
whose rows are placed verbatim into a prompt, together with an instruction to
reuse them character-for-character. A leak there does not stay a read — the
model writes the other tenant's vocabulary into this tenant's graph.

Regression coverage for all of this lives in
`tests/test_integration.py::TestTenantIsolation`.

**Governed writes.** Operations that change or retire existing knowledge are
gated for review when an agent proposes them. The risk of an operation is
computed server-side from the operation kind and the actor; the `risk` field
travelling on an incoming operation is treated as a proposal and never as an
authority. Otherwise anything able to construct an operation could mark a
supersede "additive" and retire a claim with no review.

**SQL injection.** Every statement uses bound parameters. The one place a value
is interpolated into SQL text is the `SET` clause list in `_update_node`, which
is assembled from a fixed allowlist of column names (`title`, `type`, `status`)
and never from caller input.

### What the library is *not* responsible for

- **Authenticating your users, or deciding which tenant a request belongs to.**
  contextgraph enforces the boundary you pass it. It cannot tell you that you
  passed the wrong one.
- **Database credentials, TLS to Postgres, or at-rest encryption.** You supply
  the session factory.
- **Provider API keys.** They are yours; the `Meter` protocol exists so you can
  attribute spend per tenant, and `api_key=` on each provider adapter exists so
  you can use per-tenant keys rather than one process-wide bill.
- **The content of what an LLM extracts.** Extraction is grounded in the source
  text and every claim cites the span it came from, which makes an error
  *recoverable* — it does not make it impossible.

### MCP server

The MCP server (`contextgraph-mcp`) has a materially different exposure,
because a model chooses its arguments and a model can be manipulated by the
content it reads.

**Scope is server configuration, never a tool argument.** `tenant_id` and
`graph_id` come from environment variables. They are not in any tool's input
schema, and any keys by those names appearing in tool arguments are discarded
before dispatch. If a model could pass them, a prompt injection carried in
ingested text could redirect a read or a write to another tenant's graph.

**Run one server per tenant.** The scope is fixed at process start. Do not
front multiple tenants with one server process.

**Treat ingested content as untrusted.** `remember` stores text that may later
be returned by `recall` to another agent. contextgraph preserves provenance so
you can trace a claim to its source, but it does not sanitise instructions
embedded in source text.

**Set `CONTEXTGRAPH_TENANT` and `CONTEXTGRAPH_GRAPH` explicitly.** Both default
to the literal string `"default"`, which is exactly the collision the isolation
rules above are written against.

## Known limitations

- Vector similarity is computed over data supplied by the host. A tenant who
  can write arbitrary text into their own graph can influence their own
  retrieval results. They cannot influence another tenant's.
- The review gate protects *existing* knowledge. Additive operations apply
  immediately by design, so an agent with write access can add claims without
  review. Use `pending()`/`review()` and the `cg_runs` provenance table to audit
  what was added.
