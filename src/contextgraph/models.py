"""Schema. Six tables, one idea each.

Two design decisions are load-bearing and easy to undo by accident:

**Segments are not nodes.** ``source_segments`` is a retrieval index — search
plumbing. ``graph_nodes`` holds meaning. Collapsing them turns the graph into a
chunk store with extra steps and loses the thing that makes it worth having:
typed, queryable relationships between claims.

**Nothing is deleted.** Edges carry valid time (``valid_at``/``invalid_at``)
and system time (``created_at``/``expired_at``); a contradiction invalidates
rather than overwrites, and a superseded node keeps its row. The graph is an
account of what was believed and when, not a snapshot of what is believed now.
An audit trail you can delete from is not an audit trail.

Tenancy is two opaque strings — ``tenant_id`` (your organization/account) and
``graph_id`` (your project/workspace). They are ``String`` rather than UUID and
carry no foreign key to host tables on purpose: the library must never assume
your id scheme, and must be droppable without touching your schema.

The ``embedding vector(N)`` columns are created by the migration, not mapped
here, and are read via raw SQL — the same approach pgvector users converge on
because the ORM has nothing useful to add to a distance operator.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """contextgraph's own declarative base — never the host's."""


class _Row:
    """id + timestamps, shared by every table."""

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class _Scoped(_Row):
    """Rows that belong to a tenant and a graph."""

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    graph_id: Mapped[str] = mapped_column(String(64), nullable=False)


class GraphRun(Base, _Scoped):
    """One structuring pass. The provenance spine.

    Every node and edge points back to the run that created it, so the graph is
    fully attributable. ``metrics`` records what the run actually cost and
    produced — model, tokens, nodes created/merged, ops gated. A provenance
    table that records nothing about the run is just a foreign key.
    """

    __tablename__ = "cg_runs"

    actor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="running"
    )  # running | completed | failed
    input: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (Index("ix_cg_runs_graph", "graph_id"),)


class GraphSource(Base, _Scoped):
    """Raw evidence, retained verbatim.

    The atomic unit of provenance: every derived claim cites a span in here.
    This is what makes extraction errors recoverable rather than permanent, and
    it is the least contested idea in the whole design space — keep the
    original, always.
    """

    __tablename__ = "cg_sources"

    origin: Mapped[str] = mapped_column(String(50), nullable=False)
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_ref: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    source_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    extraction_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )  # pending | running | done | failed
    extraction_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("cg_runs.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_cg_sources_graph", "graph_id"),
        Index("ix_cg_sources_status", "extraction_status"),
    )


class SourceSegment(Base, _Scoped):
    """An embedded chunk of a source. Search plumbing, not meaning."""

    __tablename__ = "cg_segments"

    source_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("cg_sources.id", ondelete="CASCADE"),
        nullable=False,
    )
    locator: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("ix_cg_segments_source", "source_id"),
        Index("ix_cg_segments_graph", "graph_id"),
    )


class GraphNode(Base, _Scoped):
    """A claim or an entity.

    ``aliases`` is not decoration. Every surface form that has ever resolved to
    this node is kept here, and the roster offers them back to the extractor —
    which is how "Task Status Tracking UI" stops becoming a second node next
    time someone phrases it that way.

    ``properties`` carries ``summary``, ``summaries`` (phrasings folded in by
    merges), ``citations``, ``merges`` (the decision trail), and any host data.
    """

    __tablename__ = "cg_nodes"

    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    type: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    aliases: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    properties: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active"
    )  # active | superseded | rejected
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_by_run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("cg_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_cg_nodes_graph_kind_type", "graph_id", "kind", "type"),
        Index("ix_cg_nodes_status", "status"),
    )


class GraphEdge(Base, _Scoped):
    """A typed, bi-temporal relationship.

    ``valid_at``/``invalid_at`` is when the fact was true; ``created_at``/
    ``expired_at`` is when the row existed. Most applications only ever need
    the first pair — if your facts become true at the moment you record them,
    valid time equals system time and that is fine. The second pair is there
    for hosts that backfill history, and is inert otherwise. Leaving it inert
    is a legitimate choice, not an unfinished one.
    """

    __tablename__ = "cg_edges"

    source_node_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("cg_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_node_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("cg_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    properties: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    valid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    invalid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_by_run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("cg_runs.id", ondelete="SET NULL"),
        nullable=True,
    )

    __table_args__ = (
        Index("ix_cg_edges_source", "source_node_id", "type"),
        Index("ix_cg_edges_target", "target_node_id", "type"),
    )


class Changeset(Base, _Scoped):
    """Proposed mutations, and the gate they wait at.

    The part of this design with no equivalent in comparable systems, which
    write memory unilaterally at ingest. Additive operations auto-apply;
    operations that change or retire existing truth are gated for review when
    an agent proposes them, and applied-with-audit when a human does. If your
    graph is a cache, you do not need this. If your graph is a deliverable —
    something users read, edit and ship from — you do.
    """

    __tablename__ = "cg_changesets"

    run_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("cg_runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending_review"
    )  # pending_review | applied | rejected | partial | superseded
    ops: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    origin: Mapped[str | None] = mapped_column(String(30), nullable=True)
    actor_kind: Mapped[str | None] = mapped_column(String(10), nullable=True)
    actor_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (Index("ix_cg_changesets_graph_status", "graph_id", "status"),)


__all__ = [
    "Base",
    "Changeset",
    "GraphEdge",
    "GraphNode",
    "GraphRun",
    "GraphSource",
    "SourceSegment",
]
