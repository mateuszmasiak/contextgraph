"""Reading the graph back out.

Two shapes, for two different questions:

``search_segments`` answers "what did the source material say" — verbatim spans
with provenance. ``retrieve_subgraph`` answers "what do we know about this, and
how does it connect" — a locate-then-expand walk that returns a connected
neighbourhood rather than a ranked list of chunks.

Expansion is capped at 1-2 hops on purpose. Graph structure pays for itself on
genuinely multi-hop compositional questions and is otherwise a slower way to do
vector search; the access pattern that dominates in practice ("what screens are
in this flow") is one hop. Deep traversal is a deliberate non-feature, not a
missing one.

Retrieval is vector-only today. If you have a lexical index available, fusing
BM25 with these results (RRF over ranks) is the single highest-value addition —
users and agents search for literal names, which is exactly where dense
retrieval is weakest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from contextgraph.protocols import Embedder

logger = logging.getLogger(__name__)


def summary_of(props: dict[str, Any] | None) -> str | None:
    """A node's summary, including phrasings folded in by merges.

    A merge appends the losing node's text to ``properties.summaries`` rather
    than overwriting ``properties.summary`` — that is what makes it non-lossy.
    Readers that select only the singular key therefore lose the merged text
    entirely: preserved on disk, absent from every rendered surface. Reading
    both is what makes "non-lossy" true for consumers and not just for storage.
    """
    props = props or {}
    parts: list[str] = []
    for value in [props.get("summary"), *(props.get("summaries") or [])]:
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    return " · ".join(parts) or None


@dataclass
class SegmentHit:
    segment_id: str
    source_id: str
    text: str
    score: float
    locator: dict[str, Any]
    origin: str | None


@dataclass
class GraphContext:
    """A connected subgraph, ready to hand to a model."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def to_text(self, max_chars: int = 6000) -> str:
        """Render as grounded context.

        Node ids are included. A read surface that returns titles the model
        cannot act on forces it to make a second, wider call just to obtain an
        identifier — the opposite of compression.
        """
        if not self.nodes:
            return "No relevant context found in the graph."
        lines = ["Relevant context (knowledge graph):", ""]
        dropped = 0
        for n in self.nodes:
            summary = f" — {n['summary']}" if n.get("summary") else ""
            line = f"- [{n['type']}] {n['title']} {{id: {n['id']}}}{summary}"
            if sum(len(x) + 1 for x in lines) + len(line) > max_chars - 200:
                dropped += 1
                continue
            lines.append(line)
        if self.edges:
            title_by_id = {n["id"]: n["title"] for n in self.nodes}
            lines.extend(["", "Relationships:"])
            for e in self.edges:
                s = title_by_id.get(e["source_id"], "?")
                t = title_by_id.get(e["target_id"], "?")
                line = f"  {s} -[{e['type']}]-> {t}"
                if sum(len(x) + 1 for x in lines) + len(line) > max_chars:
                    dropped += 1
                    continue
                lines.append(line)
        if dropped:
            # Silent truncation reads as "this is everything". Say so instead.
            lines.append(f"\n[... {dropped} more item(s) omitted for length]")
        return "\n".join(lines)


class RetrievalService:
    def __init__(self, session: AsyncSession, embedder: Embedder) -> None:
        self.session = session
        self.embedder = embedder

    async def search_segments(
        self, *, graph_id: str, query: str, top_k: int = 10,
        min_score: float | None = None,
    ) -> list[SegmentHit]:
        """Vector search over source spans.

        ``min_score`` matters more than it looks: without a floor an off-topic
        query still returns ``top_k`` rows, and whatever consumes them presents
        arbitrary content as relevant context.
        """
        embedding = await self.embedder.embed(query)
        emb = "[" + ",".join(str(v) for v in embedding) + "]"
        rows = (
            await self.session.execute(
                text("""
                    SELECT s.id::text, s.source_id::text, s.text, s.locator,
                           1 - (s.embedding <=> CAST(:emb AS vector)) AS score,
                           src.origin
                    FROM cg_segments s
                    JOIN cg_sources src ON src.id = s.source_id
                    WHERE s.graph_id = :graph AND s.embedding IS NOT NULL
                    ORDER BY s.embedding <=> CAST(:emb AS vector)
                    LIMIT :k
                """),
                {"emb": emb, "graph": graph_id, "k": top_k},
            )
        ).fetchall()
        hits = [
            SegmentHit(
                segment_id=r[0], source_id=r[1], text=r[2],
                locator=r[3] or {}, score=float(r[4]), origin=r[5],
            )
            for r in rows
        ]
        if min_score is not None:
            hits = [h for h in hits if h.score >= min_score]
        return hits

    async def retrieve_subgraph(
        self, *, graph_id: str, query: str, top_k_nodes: int = 8,
        hops: int = 1, edge_types: list[str] | None = None, max_nodes: int = 30,
        min_score: float | None = None,
    ) -> GraphContext:
        """Locate entry nodes by similarity, then expand into a neighbourhood."""
        embedding = await self.embedder.embed(query)
        emb = "[" + ",".join(str(v) for v in embedding) + "]"
        rows = (
            await self.session.execute(
                text("""
                    SELECT id::text, kind, type, title, properties,
                           1 - (embedding <=> CAST(:emb AS vector)) AS score
                    FROM cg_nodes
                    WHERE graph_id = :graph AND status = 'active'
                      AND deleted_at IS NULL AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:emb AS vector)
                    LIMIT :k
                """),
                {"emb": emb, "graph": graph_id, "k": top_k_nodes},
            )
        ).fetchall()
        seeds = {
            r[0]: {
                "id": r[0], "kind": r[1], "type": r[2], "title": r[3],
                "summary": summary_of(r[4]), "score": float(r[5]),
            }
            for r in rows
            if min_score is None or float(r[5]) >= min_score
        }
        if not seeds:
            return GraphContext(nodes=[], edges=[])
        return await self._grow(graph_id, seeds, hops, edge_types, max_nodes)

    async def neighborhood(
        self, *, graph_id: str, node_id: str, hops: int = 1,
        edge_types: list[str] | None = None, max_nodes: int = 30,
    ) -> GraphContext:
        details = await self._node_details(graph_id, [node_id])
        if not details:
            return GraphContext(nodes=[], edges=[])
        return await self._grow(
            graph_id, {d["id"]: {**d, "score": 1.0} for d in details},
            hops, edge_types, max_nodes,
        )

    async def evidence_for(
        self, *, graph_id: str, node_id: str
    ) -> list[dict[str, Any]]:
        """The source spans a node's claim rests on."""
        row = (
            await self.session.execute(
                text("""
                    SELECT properties->'citations' FROM cg_nodes
                    WHERE id = CAST(:id AS uuid) AND graph_id = :graph
                """),
                {"id": node_id, "graph": graph_id},
            )
        ).first()
        citations = (row[0] if row else None) or []
        source_ids = [c.get("source_id") for c in citations if c.get("source_id")]
        if not source_ids:
            return []
        rows = (
            await self.session.execute(
                text("""
                    SELECT id::text, origin, left(coalesce(raw_content,''), 2000),
                           source_metadata, created_at
                    FROM cg_sources
                    WHERE id = ANY(CAST(:ids AS uuid[])) AND graph_id = :graph
                """),
                {"ids": source_ids, "graph": graph_id},
            )
        ).fetchall()
        return [
            {
                "source_id": r[0], "origin": r[1], "excerpt": r[2],
                "metadata": r[3] or {}, "created_at": r[4],
            }
            for r in rows
        ]

    async def conflicts(
        self, *, graph_id: str, open_question_type: str
    ) -> list[dict[str, Any]]:
        rows = (
            await self.session.execute(
                text("""
                    SELECT id::text, title, properties->'conflict_node_ids',
                           coalesce((properties->>'resolved')::boolean, false),
                           created_at
                    FROM cg_nodes
                    WHERE graph_id = :graph AND type = :oq
                      AND status = 'active' AND deleted_at IS NULL
                    ORDER BY created_at DESC
                """),
                {"graph": graph_id, "oq": open_question_type},
            )
        ).fetchall()
        return [
            {
                "open_question_id": r[0], "title": r[1],
                "conflict_node_ids": r[2] or [], "resolved": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    # -- expansion ----------------------------------------------------------- #

    async def _grow(
        self, graph_id: str, seeds: dict[str, dict[str, Any]], hops: int,
        edge_types: list[str] | None, max_nodes: int,
    ) -> GraphContext:
        collected: dict[str, dict[str, Any] | None] = dict(seeds)
        frontier = list(seeds.keys())
        edges_out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()

        for _ in range(max(0, hops)):
            if not frontier or len(collected) >= max_nodes:
                break
            rows = await self._edges_touching(graph_id, frontier, edge_types)
            new_frontier: list[str] = []
            for src, tgt, etype in rows:
                key = (src, tgt, etype)
                if key not in seen:
                    seen.add(key)
                    edges_out.append(
                        {"source_id": src, "target_id": tgt, "type": etype}
                    )
                for nid in (src, tgt):
                    if nid not in collected and len(collected) < max_nodes:
                        collected[nid] = None
                        new_frontier.append(nid)
            frontier = new_frontier

        missing = [nid for nid, v in collected.items() if v is None]
        if missing:
            for d in await self._node_details(graph_id, missing):
                collected[d["id"]] = d

        nodes = [v for v in collected.values() if v]
        node_ids = {n["id"] for n in nodes}
        edges_out = [
            e for e in edges_out
            if e["source_id"] in node_ids and e["target_id"] in node_ids
        ]
        # Seeds carry a similarity score; expanded nodes are ordered by how
        # close they were reached, which is what the hop order already encodes.
        nodes.sort(key=lambda n: n.get("score") or 0.0, reverse=True)
        return GraphContext(nodes=nodes, edges=edges_out)

    async def _edges_touching(
        self, graph_id: str, node_ids: list[str], edge_types: list[str] | None
    ) -> list[tuple[str, str, str]]:
        sql = """
            SELECT source_node_id::text, target_node_id::text, type
            FROM cg_edges
            WHERE graph_id = :graph
              AND invalid_at IS NULL
              AND (source_node_id = ANY(CAST(:ids AS uuid[]))
                   OR target_node_id = ANY(CAST(:ids AS uuid[])))
        """
        params: dict[str, Any] = {"graph": graph_id, "ids": node_ids}
        if edge_types:
            sql += " AND type = ANY(:etypes)"
            params["etypes"] = edge_types
        rows = (await self.session.execute(text(sql), params)).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    async def _node_details(
        self, graph_id: str, node_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        rows = (
            await self.session.execute(
                text("""
                    SELECT id::text, kind, type, title, properties, status, confidence
                    FROM cg_nodes
                    WHERE id = ANY(CAST(:ids AS uuid[])) AND graph_id = :graph
                      AND deleted_at IS NULL AND status = 'active'
                """),
                {"ids": node_ids, "graph": graph_id},
            )
        ).fetchall()
        return [
            {
                "id": r[0], "kind": r[1], "type": r[2], "title": r[3],
                "summary": summary_of(r[4]), "status": r[5], "confidence": r[6],
            }
            for r in rows
        ]
