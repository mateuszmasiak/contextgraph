"""MCP server — the graph as tools an agent can call.

Design notes that matter more than the code:

**Scope is never a tool argument.** ``tenant_id`` and ``graph_id`` come from
server configuration, not from the model. If an agent could pass them, a prompt
injection could redirect a read or a write to another tenant's graph — and the
graph_id filter is the entire isolation boundary. Every handler reads them from
the server, and any values that appear in tool arguments are discarded.

**Write tools are honest about the gate.** ``remember`` says when something was
merged rather than added, and gated operations are reported as awaiting review
rather than as done. An agent that believes it wrote something it did not will
confidently tell the user so.

**Read tools return ids.** Returning titles alone forces a second, wider call
just to obtain something actionable.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any

from contextgraph.client import ContextGraph

logger = logging.getLogger(__name__)


TOOLS: list[dict[str, Any]] = [
    {
        "name": "recall",
        "description": (
            "Search everything known about a topic and return a connected "
            "subgraph — the relevant claims, the entities they concern, and how "
            "they relate. Use this before answering questions about prior "
            "decisions, constraints, or what exists."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you want to know"},
                "hops": {
                    "type": "integer", "default": 1, "minimum": 0, "maximum": 2,
                    "description": "Relationship hops to expand (1 is usually right)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_sources",
        "description": (
            "Search the original source text verbatim, with provenance. Use when "
            "you need exact wording or want to quote, rather than the graph's "
            "distilled claims."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5, "maximum": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember",
        "description": (
            "Record new information in the graph. Pass the raw statement; it is "
            "extracted, de-duplicated against what is already known, and linked. "
            "Facts that restate something already stored are merged, not "
            "duplicated. Changes that would alter or retire existing knowledge "
            "are queued for human review rather than applied."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember, in plain text",
                },
                "origin": {
                    "type": "string", "default": "agent",
                    "description": "Where this came from (meeting, doc, chat, ...)",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "neighbors",
        "description": (
            "Everything connected to a specific node, for impact analysis: what "
            "depends on it, what it affects, what motivated it. Takes a node id "
            "from a previous recall result."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "hops": {"type": "integer", "default": 1, "maximum": 2},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "evidence",
        "description": (
            "The original source spans a claim rests on. Use to verify a fact "
            "before relying on it, or to cite where something came from."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "conflicts",
        "description": (
            "Contradictions the graph is holding open — where two recorded "
            "claims cannot both be true. These are surfaced, never auto-resolved."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class GraphMCPServer:
    """Serves one graph. Scope is fixed at construction, never per-call."""

    def __init__(self, graph: ContextGraph, tenant_id: str, graph_id: str) -> None:
        self.graph = graph
        self.tenant_id = tenant_id
        self.graph_id = graph_id

    async def call(self, name: str, args: dict[str, Any]) -> str:
        # Any scope keys the model supplied are dropped, not honoured.
        args = {k: v for k, v in (args or {}).items()
                if k not in ("tenant_id", "graph_id")}
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            return f"Unknown tool: {name}"
        try:
            return str(await handler(args))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s failed", name)
            return f"Tool '{name}' failed: {exc}"

    async def _recall(self, args: dict[str, Any]) -> str:
        ctx = await self.graph.recall(
            self.tenant_id, self.graph_id, args["query"],
            hops=int(args.get("hops", 1)),
        )
        return ctx.to_text()

    async def _search_sources(self, args: dict[str, Any]) -> str:
        hits = await self.graph.search(
            self.tenant_id, self.graph_id, args["query"],
            top_k=int(args.get("top_k", 5)),
        )
        if not hits:
            return "No matching source material."
        return "\n\n".join(
            f"[{h.origin or 'source'} {h.source_id} · score {h.score:.2f}]\n{h.text}"
            for h in hits
        )

    async def _remember(self, args: dict[str, Any]) -> str:
        result = await self.graph.add(
            self.tenant_id, self.graph_id, args["content"],
            origin=args.get("origin", "agent"),
        )
        if result.status == "failed":
            return f"Could not record that: {result.error}"
        if result.status == "stored":
            return (
                f"Stored the text (source {result.source_id}) but did not "
                f"structure it: {result.error or 'no LLM configured'}."
            )
        parts = []
        if result.nodes_created:
            parts.append(f"{result.nodes_created} new fact(s)")
        if result.nodes_merged:
            parts.append(f"{result.nodes_merged} merged into existing knowledge")
        if result.edges_created:
            parts.append(f"{result.edges_created} relationship(s)")
        if result.gated_ops:
            parts.append(
                f"{result.gated_ops} change(s) queued for human review "
                f"(NOT yet applied)"
            )
        return "Recorded: " + (", ".join(parts) if parts else "nothing new to add")

    async def _neighbors(self, args: dict[str, Any]) -> str:
        ctx = await self.graph.neighbors(
            self.tenant_id, self.graph_id, args["node_id"],
            hops=int(args.get("hops", 1)),
        )
        return ctx.to_text()

    async def _evidence(self, args: dict[str, Any]) -> str:
        items = await self.graph.evidence(
            self.tenant_id, self.graph_id, args["node_id"]
        )
        if not items:
            return "No recorded evidence for that node."
        return "\n\n".join(
            f"[{i['origin']} · {i['source_id']}]\n{i['excerpt']}" for i in items
        )

    async def _conflicts(self, _: dict[str, Any]) -> str:
        items = await self.graph.conflicts(self.tenant_id, self.graph_id)
        if not items:
            return "No open contradictions."
        return "\n".join(
            f"- {i['title']} (open question {i['open_question_id']}, "
            f"nodes: {', '.join(i['conflict_node_ids'])})"
            for i in items
        )


def build_default_graph() -> tuple[ContextGraph, str, str]:
    """Wire a graph from environment variables, for the CLI entry point.

    DATABASE_URL, CONTEXTGRAPH_TENANT, CONTEXTGRAPH_GRAPH,
    OPENAI_API_KEY (embeddings), ANTHROPIC_API_KEY (extraction).
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from contextgraph.embeddings import OpenAIEmbedder

    url = os.environ["DATABASE_URL"].replace(
        "postgresql://", "postgresql+asyncpg://", 1
    )
    engine = create_async_engine(url, pool_pre_ping=True)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    llm = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from contextgraph.llm import AnthropicLLM

        llm = AnthropicLLM(api_key=os.environ["ANTHROPIC_API_KEY"])

    graph = ContextGraph(
        sessions,
        embedder=OpenAIEmbedder(api_key=os.environ.get("OPENAI_API_KEY")),
        llm=llm,
    )
    return (
        graph,
        os.environ.get("CONTEXTGRAPH_TENANT", "default"),
        os.environ.get("CONTEXTGRAPH_GRAPH", "default"),
    )


async def serve_stdio() -> None:
    """Run over stdio using the official MCP SDK."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    graph, tenant, graph_id = build_default_graph()
    backend = GraphMCPServer(graph, tenant, graph_id)
    server: Server = Server("contextgraph")

    @server.list_tools()
    async def _list() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def _call(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        return [TextContent(type="text", text=await backend.call(name, arguments))]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    parser = argparse.ArgumentParser(description="contextgraph MCP server")
    parser.add_argument(
        "--transport", default="stdio", choices=["stdio"],
        help="Transport (stdio only for now)",
    )
    parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve_stdio())


if __name__ == "__main__":
    main()
