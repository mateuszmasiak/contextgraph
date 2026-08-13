"""The orchestrator: extract, resolve, apply — once, for one source.

Two details here are easy to get wrong and expensive to discover later.

**Transaction ownership.** This function does not own the session. It must not
call ``rollback()`` on it: on the paths where ingest and structuring share a
transaction, a rollback here erases the source that was just captured, and the
subsequent "mark failed" write then has nothing to mark. Failures are isolated
with a SAVEPOINT so only the structuring work is undone.

**Metering runs in ``finally``.** The provider was already paid for whatever was
consumed before a failure, so a failed run is still a billed run. Reporting
spend only on success under-reports exactly when things are going worst.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from contextgraph.config import Config
from contextgraph.errors import ScopeViolationError
from contextgraph.models import Changeset, GraphRun, GraphSource
from contextgraph.ontology import Ontology
from contextgraph.protocols import (
    CanonicalResolver,
    Embedder,
    Meter,
    StructuredLLM,
)
from contextgraph.services.extraction import ExtractionService
from contextgraph.services.graph import Actor, GraphService
from contextgraph.services.resolution import ResolutionService
from contextgraph.services.roster import build_roster_block

logger = logging.getLogger(__name__)


@dataclass
class Spend:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    embed_tokens: int = 0

    def add(self, model: str, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        self.model = model or self.model
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
        self.llm_calls += 1


@dataclass
class StructureResult:
    status: str
    source_id: str
    run_id: str | None = None
    changeset_id: str | None = None
    nodes_created: int = 0
    nodes_merged: int = 0
    edges_created: int = 0
    gated_ops: int = 0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


async def structure_source(
    session: AsyncSession,
    source_id: UUID,
    *,
    tenant_id: str,
    graph_id: str,
    embedder: Embedder,
    llm: StructuredLLM,
    ontology: Ontology,
    config: Config,
    meter: Meter | None = None,
    canonical: CanonicalResolver | None = None,
    actor: Actor | None = None,
) -> StructureResult:
    source = await session.get(GraphSource, source_id)
    if source is None:
        return StructureResult(status="not_found", source_id=str(source_id))

    # A source is addressed by a bare UUID, which carries no scope of its own.
    # Callers that hand one in from outside (``restructure``) would otherwise be
    # able to re-extract any tenant's source and write the results into the
    # caller's own graph — a read leak and a write, from one guessed id.
    if source.tenant_id != tenant_id or source.graph_id != graph_id:
        raise ScopeViolationError(
            f"Source {source_id} belongs to another tenant or graph"
        )

    actor = actor or Actor(kind="agent")
    source.extraction_status = "running"
    await session.flush()

    run = GraphRun(
        tenant_id=source.tenant_id,
        graph_id=source.graph_id,
        actor_id=actor.id,
        status="running",
        input={"source_id": str(source_id)},
        metrics={},
    )
    session.add(run)
    await session.flush()

    spend = Spend()
    embed_before = embedder.tokens_used

    try:
        # SAVEPOINT: a structuring failure must not take the caller's
        # transaction — or the source row — down with it.
        async with session.begin_nested():
            roster = await build_roster_block(
                session, source.tenant_id, source.graph_id, config.roster
            )

            extractor = ExtractionService(
                llm, ontology=ontology, config=config.extraction
            )
            extracted = await extractor.extract(
                source.raw_content or "", origin=source.origin, roster=roster
            )
            for usage in extracted.usage:
                spend.add(extracted.model, usage)

            resolver = ResolutionService(
                session, embedder, tenant_id=source.tenant_id, llm=llm,
                ontology=ontology, config=config.resolution, canonical=canonical,
            )
            ops = await resolver.resolve(
                graph_id=source.graph_id,
                run_id=str(run.id),
                source_id=str(source.id),
                nodes=extracted.nodes,
                edges=extracted.edges,
            )
            for usage in resolver.usage:
                spend.add(resolver.model, usage)

            applied = await GraphService(session, ontology).apply(
                ops,
                tenant_id=source.tenant_id,
                graph_id=source.graph_id,
                actor=actor,
                run_id=run.id,
            )

            changeset = Changeset(
                tenant_id=source.tenant_id,
                graph_id=source.graph_id,
                run_id=run.id,
                status="partial" if applied.gated_ops else "applied",
                # The server's classification, not the resolver's proposal.
                # Reviewers act on this row, so it has to be the same verdict
                # the gate acted on.
                ops=applied.classified_ops,
                origin="structuring",
                actor_kind=actor.kind,
                actor_id=actor.id,
            )
            session.add(changeset)

            spend.embed_tokens = embedder.tokens_used - embed_before
            metrics = {
                "model": spend.model,
                "llm_calls": spend.llm_calls,
                "input_tokens": spend.input_tokens,
                "output_tokens": spend.output_tokens,
                "embed_tokens": spend.embed_tokens,
                "nodes_created": len(applied.created_node_ids),
                "nodes_merged": len(applied.merged_node_ids),
                "edges_created": len(applied.created_edge_ids),
                "gated_ops": len(applied.gated_ops),
            }
            run.status = "completed"
            run.metrics = metrics
            source.extraction_status = "done"
            source.extraction_error = None
            await session.flush()

        return StructureResult(
            status="done",
            source_id=str(source_id),
            run_id=str(run.id),
            changeset_id=str(changeset.id),
            nodes_created=len(applied.created_node_ids),
            nodes_merged=len(applied.merged_node_ids),
            edges_created=len(applied.created_edge_ids),
            gated_ops=len(applied.gated_ops),
            metrics=metrics,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception("Structuring failed for source %s", source_id)
        # The savepoint rolled back the graph writes; the source row survives,
        # so the failure is recorded rather than lost.
        source.extraction_status = "failed"
        source.extraction_error = str(exc)[:2000]
        run.status = "failed"
        await session.flush()
        return StructureResult(
            status="failed", source_id=str(source_id),
            run_id=str(run.id), error=str(exc),
        )

    finally:
        if meter is not None:
            spend.embed_tokens = spend.embed_tokens or (
                embedder.tokens_used - embed_before
            )
            try:
                await meter.record(
                    tenant_id=source.tenant_id,
                    operation="structure",
                    model=spend.model,
                    input_tokens=spend.input_tokens,
                    output_tokens=spend.output_tokens,
                    embed_tokens=spend.embed_tokens,
                    embed_model=getattr(embedder, "model", "unknown"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Metering failed (ignored): %s", exc)
