"""Capturing raw evidence and indexing it for search.

Deliberately cheap and synchronous: capture must not depend on an LLM being
available, because losing the source is the one unrecoverable failure in the
pipeline. Everything derived from it — nodes, edges, conflicts — can be rebuilt
by re-running structuring. The raw text cannot be rebuilt from anything.

So this does exactly two things, and they are **two methods rather than one**:
``capture`` stores the text, ``index`` embeds its segments. The split is the
whole point. Embedding calls a paid remote service that fails routinely, and
while both steps shared one uncommitted transaction, an embedding outage rolled
the source row back with it — the pipeline losing the one artefact it exists to
never lose, at exactly the moment it was under stress. The caller commits
between the two, so a failure past that point costs an embedding call and
leaves a row that can be indexed again.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from contextgraph.config import ChunkConfig
from contextgraph.embeddings.chunker import chunk_text
from contextgraph.models import GraphSource, SourceSegment
from contextgraph.protocols import Embedder

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self, session: AsyncSession, embedder: Embedder, config: ChunkConfig
    ) -> None:
        self.session = session
        self.embedder = embedder
        self.config = config

    async def capture(
        self,
        *,
        tenant_id: str,
        graph_id: str,
        origin: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        external_ref: dict[str, Any] | None = None,
        run_id: UUID | None = None,
    ) -> GraphSource:
        """Store the raw text. Touches nothing remote, so it cannot fail on a
        provider outage — the caller should commit before doing anything that
        can."""
        source = GraphSource(
            tenant_id=tenant_id,
            graph_id=graph_id,
            origin=origin,
            raw_content=content,
            source_metadata=metadata,
            external_ref=external_ref,
            extraction_status="pending",
            created_by_run_id=run_id,
        )
        self.session.add(source)
        await self.session.flush()
        return source

    async def index(
        self,
        *,
        tenant_id: str,
        graph_id: str,
        source_id: UUID,
        content: str,
    ) -> int:
        """Chunk and embed a captured source. Raises on embedder failure.

        Raising is correct and the source is already safe: re-running this is
        cheap, whereas a source indexed with a fabricated vector is corrupt in
        a way nothing downstream can detect.

        Takes plain values rather than the ``GraphSource`` on purpose. The
        caller commits between ``capture`` and this call, and under the default
        ``expire_on_commit=True`` every attribute of that instance is then
        expired — so reading one back would emit IO from what looks like a
        plain attribute access, which under asyncio is a ``MissingGreenlet``
        crash rather than a lazy load.
        """
        return await self._segment_and_embed(
            tenant_id, graph_id, source_id, content
        )

    async def _segment_and_embed(
        self, tenant_id: str, graph_id: str, source_id: UUID, content: str
    ) -> int:
        chunks = chunk_text(content, self.config)
        if not chunks:
            return 0

        vectors = await self.embedder.embed_batch([c.text for c in chunks])
        # strict=True: a short batch would otherwise pair chunks with the wrong
        # vectors and silently drop the tail while reporting full indexing.
        for chunk, vector in zip(chunks, vectors, strict=True):
            segment = SourceSegment(
                tenant_id=tenant_id,
                graph_id=graph_id,
                source_id=source_id,
                text=chunk.text,
                locator=chunk.locator,
            )
            self.session.add(segment)
            await self.session.flush()
            await self._write_embedding(segment.id, vector)
        return len(chunks)

    async def _write_embedding(self, segment_id: UUID, vector: list[float]) -> None:
        from sqlalchemy import text as sql

        await self.session.execute(
            sql(
                "UPDATE cg_segments SET embedding = CAST(:v AS vector) "
                "WHERE id = CAST(:id AS uuid)"
            ),
            {
                "v": "[" + ",".join(str(x) for x in vector) + "]",
                "id": str(segment_id),
            },
        )
