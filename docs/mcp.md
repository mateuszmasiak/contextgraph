# MCP server

Exposes one graph as tools an agent can call. It is a thin adapter over the
library — about 300 lines — and everything it does is available directly from
`ContextGraph` if you are writing your own agent loop.

## Install and run

```bash
pip install "contextgraph[mcp,openai,anthropic]"
```

```bash
export DATABASE_URL=postgresql://user:pass@localhost/yourdb
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export CONTEXTGRAPH_TENANT=acme
export CONTEXTGRAPH_GRAPH=webapp
contextgraph-mcp
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | yes | Postgres with the `vector` extension. `postgresql://` is rewritten to `postgresql+asyncpg://`. |
| `OPENAI_API_KEY` | yes | Embeddings. |
| `ANTHROPIC_API_KEY` | no | Extraction and adjudication. Without it, `remember` stores text but does not structure it. |
| `CONTEXTGRAPH_TENANT` | **set it** | Defaults to `"default"`. |
| `CONTEXTGRAPH_GRAPH` | **set it** | Defaults to `"default"`. |

> Set the last two explicitly. Both default to the literal string `"default"`,
> which is exactly the collision the isolation rules are written against — two
> deployments that both leave them unset land in the same logical graph.

## Claude Desktop / Claude Code

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "contextgraph": {
      "command": "contextgraph-mcp",
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/yourdb",
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "CONTEXTGRAPH_TENANT": "acme",
        "CONTEXTGRAPH_GRAPH": "webapp"
      }
    }
  }
}
```

Or without installing, via `uvx`:

```json
{
  "mcpServers": {
    "contextgraph": {
      "command": "uvx",
      "args": ["--from", "contextgraph[mcp,openai,anthropic]", "contextgraph-mcp"],
      "env": { "DATABASE_URL": "postgresql://...", "CONTEXTGRAPH_TENANT": "acme" }
    }
  }
}
```

## Tools

| Tool | Answers |
| --- | --- |
| `recall(query, hops=1)` | What do we know about this, and how does it connect |
| `search_sources(query, top_k=5)` | What did the source material actually say |
| `remember(content, origin="agent")` | Record this |
| `neighbors(node_id, hops=1)` | What does this touch (impact analysis) |
| `evidence(node_id)` | Why do we believe this |
| `conflicts()` | What do we believe that cannot all be true |

Read tools return **node ids**. A read surface that returns titles alone forces
the model into a second, wider call just to obtain something it can act on —
the opposite of compression.

## Security

**Scope is server configuration, never a tool argument.** `tenant_id` and
`graph_id` are not in any tool's input schema, and any keys by those names
appearing in tool arguments are discarded before dispatch. If a model could pass
them, a prompt injection carried in ingested text could redirect a read or a
write to another tenant's graph.

**Run one server process per tenant.** Scope is fixed at process start.

**Ingested content is untrusted.** `remember` stores text that `recall` may
later return to another agent. contextgraph preserves provenance so you can
trace any claim back to its source, but it does not sanitise instructions
embedded in source text.

See [SECURITY.md](../SECURITY.md) for the full threat model.

## `remember` is honest about what it did

This matters more than it sounds. An agent that believes it wrote something it
did not will confidently tell the user so.

```
Recorded: 1 new fact(s), 2 merged into existing knowledge, 3 relationship(s)
```

```
Recorded: 2 new fact(s), 1 change(s) queued for human review (NOT yet applied)
```

Merges are reported as merges, not as additions. Gated operations are reported
as *awaiting review*, not as done. A failure says so rather than returning
silence that reads like success.

## What is not exposed

`pending()` and `review()` — the governance surface — are deliberately absent
from the tool list. The gate exists to put a human in the loop on operations
that change or retire existing knowledge; an agent that could approve its own
gated changes would make the gate ceremony. Drive those from your own
application UI against the library directly.

`health()` is likewise absent; it is an operational signal, not agent context.

## Writing your own agent integration

Nothing above is privileged. If the six tools do not fit your loop:

```python
from contextgraph import ContextGraph

graph = ContextGraph(sessions, embedder=..., llm=...)

ctx = await graph.recall(TENANT, GRAPH, user_question)
prompt = f"{ctx.to_text()}\n\nQuestion: {user_question}"
```

`GraphContext.to_text()` renders a connected subgraph with node ids, relationship
lines, and an explicit note when content was omitted for length — silent
truncation reads as "this is everything".

Keep `TENANT` and `GRAPH` server-side constants, for the same reason the MCP
server does.
