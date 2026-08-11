"""Initial contextgraph schema: six tables, pgvector, no vector index.

Revision ID: 0001
Revises:
Create Date: initial

Mirrors ``contextgraph.models`` exactly. Two things live here that the ORM
does not express:

- the ``embedding vector(N)`` columns, which are read through raw SQL because
  pgvector's distance operators have no ORM equivalent worth the mapping;
- the index set, including a partial index the hot read path depends on.

Column defaults for ``status``/``extraction_status``/``ops`` are deliberately
Python-side only, as in the models. A server default here would let a direct
INSERT that bypasses the library land a row the library never validated, and
the two defaults would then drift independently.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# Must equal Config.embedding_dimensions (1536, OpenAI text-embedding-3-small).
# Changing it is a new migration plus a full re-embed, not an edit to this one:
# pgvector fixes the width at column creation and existing rows cannot be
# reinterpreted at another dimension.
EMBEDDING_DIM = 1536


def _index(name: str, table: str, columns: str, where: str | None = None) -> None:
    # Raw DDL rather than op.create_index: IF NOT EXISTS is available on every
    # Alembic version this way, which matters when a half-applied migration is
    # re-run against a database that was already partially built.
    clause = f" WHERE {where}" if where else ""
    op.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({columns}){clause}")


def _drop_index(name: str) -> None:
    op.execute(f"DROP INDEX IF EXISTS {name}")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "cg_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("actor_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("input", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_cg_runs"),
    )

    op.create_table(
        "cg_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("origin", sa.String(length=50), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=True),
        sa.Column(
            "external_ref", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "source_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("extraction_status", sa.String(length=20), nullable=False),
        sa.Column("extraction_error", sa.Text(), nullable=True),
        sa.Column("created_by_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["created_by_run_id"],
            ["cg_runs.id"],
            name="fk_cg_sources_created_by_run_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cg_sources"),
    )

    # Segments cascade from their source: a segment is a derived slice of that
    # text and has no meaning once the text is gone. Nodes do not cascade —
    # they are claims that outlive the evidence that first produced them.
    op.create_table(
        "cg_segments",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("locator", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["cg_sources.id"],
            name="fk_cg_segments_source_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cg_segments"),
    )

    op.create_table(
        "cg_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("type", sa.String(length=100), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("aliases", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("properties", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_by_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["created_by_run_id"],
            ["cg_runs.id"],
            name="fk_cg_nodes_created_by_run_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cg_nodes"),
    )

    op.create_table(
        "cg_edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("source_node_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("target_node_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("properties", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("valid_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("invalid_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["source_node_id"],
            ["cg_nodes.id"],
            name="fk_cg_edges_source_node_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_node_id"],
            ["cg_nodes.id"],
            name="fk_cg_edges_target_node_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_run_id"],
            ["cg_runs.id"],
            name="fk_cg_edges_created_by_run_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cg_edges"),
    )

    op.create_table(
        "cg_changesets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(length=64), nullable=False),
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("ops", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("origin", sa.String(length=30), nullable=True),
        sa.Column("actor_kind", sa.String(length=10), nullable=True),
        sa.Column("actor_id", sa.String(length=64), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("reviewed_by", sa.String(length=64), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["cg_runs.id"],
            name="fk_cg_changesets_run_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_cg_changesets"),
    )

    # Not ORM-mapped: pgvector columns are written and searched through raw SQL,
    # so mapping them would add a type dependency to the model layer and buy
    # nothing. Nullable because a row is inserted before it is embedded — an
    # embedding failure must leave a retryable row, not lose the text.
    op.execute(
        f"ALTER TABLE cg_nodes ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM})"
    )
    op.execute(
        "ALTER TABLE cg_segments ADD COLUMN IF NOT EXISTS "
        f"embedding vector({EMBEDDING_DIM})"
    )

    # --- Indexes declared in models.__table_args__ -------------------------
    _index("ix_cg_runs_graph", "cg_runs", "graph_id")
    _index("ix_cg_sources_graph", "cg_sources", "graph_id")
    _index("ix_cg_sources_status", "cg_sources", "extraction_status")
    _index("ix_cg_segments_source", "cg_segments", "source_id")
    _index("ix_cg_segments_graph", "cg_segments", "graph_id")
    _index("ix_cg_nodes_graph_kind_type", "cg_nodes", "graph_id, kind, type")
    _index("ix_cg_nodes_status", "cg_nodes", "status")
    _index("ix_cg_edges_source", "cg_edges", "source_node_id, type")
    _index("ix_cg_edges_target", "cg_edges", "target_node_id, type")
    _index("ix_cg_changesets_graph_status", "cg_changesets", "graph_id, status")

    # --- Tenancy ------------------------------------------------------------
    # Every scoped read filters tenant_id even when it also filters graph_id;
    # deletion and export run tenant-wide, where graph_id is not available to
    # narrow the scan.
    _index("ix_cg_runs_tenant", "cg_runs", "tenant_id")
    _index("ix_cg_sources_tenant", "cg_sources", "tenant_id")
    _index("ix_cg_segments_tenant", "cg_segments", "tenant_id")
    _index("ix_cg_nodes_tenant", "cg_nodes", "tenant_id")
    _index("ix_cg_edges_tenant", "cg_edges", "tenant_id")
    _index("ix_cg_changesets_tenant", "cg_changesets", "tenant_id")

    # --- Hot read path ------------------------------------------------------
    # Retrieval asks one question of this table: the currently-valid edges of a
    # graph. Because nothing is deleted, invalidated edges accumulate without
    # bound while the live set stays roughly constant, so a plain index on
    # graph_id degrades as history grows. The partial index indexes only rows
    # the predicate can ever return, and matches the query's WHERE clause
    # exactly — Postgres will only use it if the clause is spelled
    # `invalid_at IS NULL`, not `invalid_at IS NOT DISTINCT FROM NULL`.
    _index(
        "ix_cg_edges_graph_live", "cg_edges", "graph_id", where="invalid_at IS NULL"
    )

    # --- Vector indexes: deliberately absent --------------------------------
    #
    # There is no ivfflat index here, and adding one at install time would make
    # retrieval quietly worse.
    #
    # ivfflat trains its centroids on the table's contents at CREATE INDEX
    # time. Run on an empty table — which is the only state a fresh migration
    # can guarantee — the centroids are meaningless, and they are never
    # retrained; the index has to be rebuilt by hand once data exists, and
    # nothing in Postgres will tell you that it wasn't.
    #
    # The failure that follows is silent. With lists=100 / probes=1 the scan
    # visits a single cell, and the graph_id filter is applied *after* the
    # index returns its candidates: the surviving rows can be fewer than the
    # LIMIT asked for. The query succeeds, returns a short result, and the
    # caller reads it as "the graph contains nothing more relevant" rather than
    # "recall collapsed".
    #
    # Exact search has none of these properties. At small-to-moderate scale
    # (thousands of nodes per graph) a sequential scan over the graph_id-
    # filtered set is sub-millisecond at 100% recall, which is both faster and
    # more correct than a mistrained approximate index.
    #
    # Past roughly 50k nodes in a single graph, add HNSW — not ivfflat. It
    # builds incrementally, needs no training pass, and degrades gracefully:
    #
    #   CREATE INDEX CONCURRENTLY ix_cg_nodes_embedding_hnsw
    #       ON cg_nodes USING hnsw (embedding vector_cosine_ops)
    #       WITH (m = 16, ef_construction = 128);
    #
    #   CREATE INDEX CONCURRENTLY ix_cg_segments_embedding_hnsw
    #       ON cg_segments USING hnsw (embedding vector_cosine_ops)
    #       WITH (m = 16, ef_construction = 128);
    #
    # Then, for the filtered queries this schema actually issues (pgvector
    # >= 0.8), let the scan keep going until enough rows survive the filter:
    #
    #   SET hnsw.iterative_scan = relaxed_order;
    #
    # Match the operator class to the distance operator in the query —
    # vector_cosine_ops serves `<=>`; an index built for `<->` will simply not
    # be used, with no error.


def downgrade() -> None:
    _drop_index("ix_cg_edges_graph_live")

    for name in (
        "ix_cg_runs_tenant",
        "ix_cg_sources_tenant",
        "ix_cg_segments_tenant",
        "ix_cg_nodes_tenant",
        "ix_cg_edges_tenant",
        "ix_cg_changesets_tenant",
        "ix_cg_runs_graph",
        "ix_cg_sources_graph",
        "ix_cg_sources_status",
        "ix_cg_segments_source",
        "ix_cg_segments_graph",
        "ix_cg_nodes_graph_kind_type",
        "ix_cg_nodes_status",
        "ix_cg_edges_source",
        "ix_cg_edges_target",
        "ix_cg_changesets_graph_status",
    ):
        _drop_index(name)

    # Reverse dependency order; the vector columns go with their tables.
    op.drop_table("cg_changesets")
    op.drop_table("cg_edges")
    op.drop_table("cg_nodes")
    op.drop_table("cg_segments")
    op.drop_table("cg_sources")
    op.drop_table("cg_runs")

    # The vector extension is not dropped. It is database-wide and may be in
    # use by the host's own tables; uninstalling someone else's dependency is
    # not this migration's business.
