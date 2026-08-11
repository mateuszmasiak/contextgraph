"""The public surface.

Everything below this line is assembly. If you only read one file, read this
one — the rest of the package exists to make these seven methods honest.

    graph = ContextGraph(session_factory, embedder=..., llm=...)

    await graph.add(tenant, project, "we chose Postgres over DynamoDB")
    await graph.recall(tenant, project, "what database did we pick?")
    await graph.pending(tenant, project)      # what needs a human
    await graph.review(tenant, project, id, approve=True)

``tenant_id`` and ``graph_id`` are required on every call and are never inferred
from ambient state. That is deliberate: the single most consequential bug class
in a multi-tenant memory system is a read or write that silently crosses a
boundary, and an API that lets you forget the scope will eventually let you
cross it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from contextgraph.config import DEFAULT_CONFIG, Config
from contextgraph.errors import ConcurrentModificationError, SchemaMismatchError
from contextgraph.models import Changeset
from contextgraph.ontology import DEFAULT_ONTOLOGY, Ontology
from contextgraph.protocols import (
    CanonicalResolver,
    Embedder,
    Meter,
    StructuredLLM,
)
from contextgraph.services.graph import Actor, GraphService
from contextgraph.services.ingestion import IngestionService
from contextgraph.services.retrieval import GraphContext, RetrievalService, SegmentHit
from contextgraph.services.structuring import StructureResult, structure_source

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


class ContextGraph:
    """A governed knowledge graph over Postgres + pgvector."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        embedder: Embedder,
        llm: StructuredLLM | None = None,
        ontology: Ontology = DEFAULT_ONTOLOGY,
        config: Config = DEFAULT_CONFIG,
        meter: Meter | None = None,
        canonical: CanonicalResolver | None = None,
    ) -> None:
        if embedder.dimensions != config.embedding_dimensions:
            # Caught here rather than as a pgvector error 400 rows into a
            # backfill.
            raise SchemaMismatchError(
                f"Embedder produces {embedder.dimensions}-d vectors but the "
                f"schema expects {config.embedding_dimensions}-d. Change the "
                f"embedder, or migrate the vector column and re-embed."
            )
        self._sessions = session_factory
        self.embedder = embedder
        self.llm = llm
        self.ontology = ontology
        self.config = config
        self.meter = meter
        self.canonical = canonical

    # -- write --------------------------------------------------------------- #

    async def add(
        self,
        tenant_id: str,
        graph_id: str,
        content: str,
        *,
        origin: str = "text",
        metadata: dict[str, Any] | None = None,
        structure: bool = True,
        actor: Actor | None = None,
    ) -> StructureResult:
        """Capture text and fold what it says into the graph.

        Capture and structuring are separable on purpose. ``structure=False``
        stores and indexes the text without an LLM call, so ingestion stays
        available when your provider is not — and structuring can be re-run
        later against a better model or a corrected ontology.
        """
        async with self._sessions() as session:
            source, _ = await IngestionService(
                session, self.embedder, self.config.chunking
            ).ingest(
                tenant_id=tenant_id,
                graph_id=graph_id,
                origin=origin,
                content=content,
                metadata=metadata,
            )
            source_id = source.id
            if not structure:
                await session.commit()
                return StructureResult(status="stored", source_id=str(source_id))

            if self.llm is None:
                await session.commit()
                return StructureResult(
                    status="stored",
                    source_id=str(source_id),
                    error="No LLM configured; source stored but not structured.",
                )

            result = await structure_source(
                session,
                source_id,
                embedder=self.embedder,
                llm=self.llm,
                ontology=self.ontology,
                config=self.config,
                meter=self.meter,
                canonical=self.canonical,
                actor=actor,
            )
            await session.commit()
            return result

    async def restructure(
        self, tenant_id: str, graph_id: str, source_id: UUID
    ) -> StructureResult:
        """Re-run extraction for a stored source.

        Safe to repeat: resolution folds re-extracted claims into the nodes
        they already produced rather than duplicating them.
        """
        if self.llm is None:
            raise ValueError("restructure() requires an LLM")
        async with self._sessions() as session:
            result = await structure_source(
                session, source_id,
                embedder=self.embedder, llm=self.llm, ontology=self.ontology,
                config=self.config, meter=self.meter, canonical=self.canonical,
            )
            await session.commit()
            return result

    # -- read ---------------------------------------------------------------- #

    async def recall(
        self, tenant_id: str, graph_id: str, query: str, *,
        hops: int = 1, top_k: int = 8, max_nodes: int = 30,
        min_score: float | None = None,
    ) -> GraphContext:
        """What the graph knows about a query, as a connected subgraph."""
        async with self._sessions() as session:
            return await RetrievalService(session, self.embedder).retrieve_subgraph(
                graph_id=graph_id, query=query, hops=hops,
                top_k_nodes=top_k, max_nodes=max_nodes, min_score=min_score,
            )

    async def search(
        self, tenant_id: str, graph_id: str, query: str, *,
        top_k: int = 10, min_score: float | None = None,
    ) -> list[SegmentHit]:
        """Verbatim source spans matching a query, with provenance."""
        async with self._sessions() as session:
            return await RetrievalService(session, self.embedder).search_segments(
                graph_id=graph_id, query=query, top_k=top_k, min_score=min_score,
            )

    async def neighbors(
        self, tenant_id: str, graph_id: str, node_id: str, *, hops: int = 1
    ) -> GraphContext:
        """Everything connected to a node — impact analysis."""
        async with self._sessions() as session:
            return await RetrievalService(session, self.embedder).neighborhood(
                graph_id=graph_id, node_id=node_id, hops=hops
            )

    async def evidence(
        self, tenant_id: str, graph_id: str, node_id: str
    ) -> list[dict[str, Any]]:
        """The source spans a claim rests on."""
        async with self._sessions() as session:
            return await RetrievalService(session, self.embedder).evidence_for(
                graph_id=graph_id, node_id=node_id
            )

    async def conflicts(self, tenant_id: str, graph_id: str) -> list[dict[str, Any]]:
        """Contradictions the graph is holding open."""
        async with self._sessions() as session:
            return await RetrievalService(session, self.embedder).conflicts(
                graph_id=graph_id, open_question_type=self.ontology.open_question_type
            )

    # -- govern -------------------------------------------------------------- #

    async def pending(
        self, tenant_id: str, graph_id: str
    ) -> list[dict[str, Any]]:
        """Changesets awaiting review."""
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    text("""
                        SELECT id::text, status, ops, origin, actor_kind,
                               summary, created_at
                        FROM cg_changesets
                        WHERE tenant_id = :tenant AND graph_id = :graph
                          AND status IN ('pending_review', 'partial')
                        ORDER BY created_at DESC
                    """),
                    {"tenant": tenant_id, "graph": graph_id},
                )
            ).fetchall()
            return [
                {
                    "id": r[0], "status": r[1],
                    "ops": [o for o in (r[2] or []) if o.get("risk") == "high"],
                    "origin": r[3], "actor_kind": r[4],
                    "summary": r[5], "created_at": r[6],
                }
                for r in rows
            ]

    async def review(
        self,
        tenant_id: str,
        graph_id: str,
        changeset_id: str,
        *,
        approve: bool,
        reviewer_id: str | None = None,
    ) -> dict[str, Any]:
        """Approve or reject the gated operations in a changeset."""
        async with self._sessions() as session:
            # Row lock: two reviewers hitting approve concurrently would
            # otherwise both apply the same operations.
            row = (
                await session.execute(
                    text("""
                        SELECT status FROM cg_changesets
                        WHERE id = CAST(:id AS uuid)
                          AND tenant_id = :tenant AND graph_id = :graph
                        FOR UPDATE
                    """),
                    {"id": changeset_id, "tenant": tenant_id, "graph": graph_id},
                )
            ).first()
            if row is None:
                raise ValueError(f"Changeset {changeset_id} not found")
            if row[0] not in ("pending_review", "partial"):
                raise ConcurrentModificationError(
                    f"Changeset {changeset_id} is already {row[0]}"
                )

            changeset = await session.get(Changeset, UUID(changeset_id))
            assert changeset is not None
            gated = [o for o in (changeset.ops or []) if o.get("risk") == "high"]

            applied_counts: dict[str, int] = {}
            if approve and gated:
                result = await GraphService(session, self.ontology).apply_gated(
                    gated, tenant_id=tenant_id, graph_id=graph_id,
                    run_id=changeset.run_id, embedder=self.embedder,
                )
                applied_counts = {
                    "created_nodes": len(result.created_node_ids),
                    "merged_nodes": len(result.merged_node_ids),
                    "updated_nodes": len(result.updated_node_ids),
                    "created_edges": len(result.created_edge_ids),
                }

            changeset.status = "applied" if approve else "rejected"
            changeset.reviewed_by = reviewer_id
            from datetime import UTC, datetime

            changeset.reviewed_at = datetime.now(UTC)
            await session.commit()
            return {
                "changeset_id": changeset_id,
                "status": changeset.status,
                "gated_ops": len(gated),
                **applied_counts,
            }

    async def health(self, tenant_id: str, graph_id: str) -> dict[str, Any]:
        """Signals that a write path is degrading.

        Nodes with no embedding are the leading indicator: they are invisible
        to de-duplication and to retrieval, permanently, and they accumulate
        silently. Sources stuck pending mean structuring is not running at all.
        """
        async with self._sessions() as session:
            row = (
                await session.execute(
                    text("""
                        SELECT
                          (SELECT count(*) FROM cg_nodes
                             WHERE graph_id = :g AND deleted_at IS NULL),
                          (SELECT count(*) FROM cg_nodes
                             WHERE graph_id = :g AND deleted_at IS NULL
                               AND embedding IS NULL),
                          (SELECT count(*) FROM cg_edges
                             WHERE graph_id = :g AND invalid_at IS NULL),
                          (SELECT count(*) FROM cg_sources
                             WHERE graph_id = :g AND extraction_status = 'pending'),
                          (SELECT count(*) FROM cg_sources
                             WHERE graph_id = :g AND extraction_status = 'failed'),
                          (SELECT count(*) FROM cg_changesets
                             WHERE graph_id = :g
                               AND status IN ('pending_review','partial'))
                    """),
                    {"g": graph_id},
                )
            ).first()
            if row is None:  # scalar subqueries always return a row; be explicit
                return {}
            keys = (
                "nodes", "nodes_without_embedding", "active_edges",
                "sources_pending", "sources_failed", "changesets_awaiting_review",
            )
            return {k: int(row[i] or 0) for i, k in enumerate(keys)}
