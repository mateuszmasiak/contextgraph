"""Turning raw text into candidate nodes and edges.

Extraction quality is the ceiling on graph quality — nothing downstream can
recover a claim that was never extracted, or un-invent one that was
hallucinated. Three things matter more than prompt wording:

**Bounded input.** Unbounded text against a bounded output budget fails as a
truncated object, and a truncated object is indistinguishable from "this source
was empty" — permanently, with no signal. Long sources are segmented and
extracted in batches instead.

**Loud failure.** A provider error must not be reported as an empty result. The
caller marks the source `failed` and can retry; silently marking it `done`
loses the content forever while looking successful.

**The roster.** Passed as a separate system block, after the static prompt, so
the stable prefix stays cacheable while the volatile part changes per graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from contextgraph.config import ExtractionConfig
from contextgraph.embeddings.chunker import chunk_text
from contextgraph.errors import ExtractionError
from contextgraph.ontology import Ontology
from contextgraph.protocols import StructuredLLM
from contextgraph.services.resolution import ExtractedEdge, ExtractedNode

logger = logging.getLogger(__name__)


class _Node(BaseModel):
    """Wire schema. Tolerant on input, strict on output."""

    temp_id: str = Field(..., description="Local id (e.g. 'n1') so edges can refer to it")
    type: str = Field(..., description="A '<kind>/<type>' label from the vocabulary")
    title: str = Field(..., description="Short canonical statement of the claim or thing")
    kind: str = Field(default="", description="Derived from `type` when omitted")
    summary: str | None = Field(default=None, description="Optional 1-2 sentence detail")
    surface_phrase: str | None = Field(
        default=None,
        description=(
            "If `title` reuses an existing entity's title, the wording THIS "
            "source used. Null for new entities."
        ),
    )
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _derive_kind(self) -> _Node:
        # Models routinely omit `kind` however clearly it is requested.
        if not self.kind:
            self.kind = self.type.split("/", 1)[0]
        return self


class _Edge(BaseModel):
    source_temp_id: str
    target_temp_id: str
    type: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    nodes: list[_Node] = Field(default_factory=list)
    edges: list[_Edge] = Field(default_factory=list)

    @field_validator("nodes", "edges", mode="before")
    @classmethod
    def _tolerate_stringified(cls, v: Any) -> Any:
        # Some providers return nested arrays as a JSON string.
        if isinstance(v, str):
            import json

            try:
                return json.loads(v)
            except (ValueError, TypeError):
                return []
        return v or []


@dataclass
class ExtractionOutput:
    nodes: list[ExtractedNode] = field(default_factory=list)
    edges: list[ExtractedEdge] = field(default_factory=list)
    usage: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""


def build_system_prompt(ontology: Ontology) -> str:
    return f"""You extract structured knowledge from raw text into a graph.

{ontology.describe()}

Rules:
- ATOMIC: one claim per knowledge node. Split compound statements.
- ONE NODE PER THING, NOT PER MENTION: a thing discussed three times is one
  node. Do not emit a node that only restates something you already emitted
  under a different type.
- GROUNDED: use only what the text supports. Do not invent facts, names, or
  relationships.
- SELECTIVE: skip greetings, chit-chat, and process talk. Extract only
  decision-, constraint-, problem-, request- or entity-bearing content.
- Each node needs a short `title` and a `confidence` (0-1).
- Reference nodes in edges by `temp_id`.
- If the text has nothing substantive, return empty lists.

Return ONLY the structured result."""


class ExtractionService:
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        ontology: Ontology,
        config: ExtractionConfig,
    ) -> None:
        self.llm = llm
        self.ontology = ontology
        self.config = config

    async def extract(
        self, content: str, *, origin: str = "text", roster: str = ""
    ) -> ExtractionOutput:
        text_in = (content or "").strip()
        if not text_in:
            return ExtractionOutput()

        # Long sources are batched rather than truncated. Truncation here is
        # invisible downstream and unrecoverable.
        batches = self._batch(text_in)
        out = ExtractionOutput()
        offset = 0
        for batch in batches:
            nodes, edges, usage, model = await self._extract_one(
                batch, origin=origin, roster=roster, temp_offset=offset
            )
            out.nodes.extend(nodes)
            out.edges.extend(edges)
            if usage:
                out.usage.append(usage)
            out.model = out.model or model
            offset += len(nodes)
        return out

    def _batch(self, content: str) -> list[str]:
        if len(content) <= self.config.max_source_chars:
            return [content]
        from contextgraph.config import ChunkConfig

        chunks = chunk_text(
            content,
            ChunkConfig(
                chunk_size=self.config.max_source_chars,
                chunk_overlap=min(2000, self.config.max_source_chars // 20),
            ),
        )
        return [c.text for c in chunks]

    async def _extract_one(
        self, content: str, *, origin: str, roster: str, temp_offset: int
    ) -> tuple[list[ExtractedNode], list[ExtractedEdge], dict[str, Any] | None, str]:
        # Stable prefix first, volatile roster second: reversing this
        # invalidates the provider's cached prefix on every call.
        system = [build_system_prompt(self.ontology)]
        if roster:
            system.append(roster)

        try:
            parsed, usage = await self.llm.complete(
                system=system,
                user=f"SOURCE (origin={origin}):\n\n{content}",
                schema=ExtractionResult,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        except Exception as exc:
            # Deliberately NOT swallowed. A provider failure reported as an
            # empty result marks the source done and loses it forever.
            raise ExtractionError(f"Extraction call failed: {exc}") from exc

        model = str((usage or {}).get("model", ""))
        if parsed is None:
            logger.warning("Extraction returned no parsable result")
            return [], [], usage, model

        nodes: list[ExtractedNode] = []
        for i, n in enumerate(parsed.nodes):
            if self.ontology.strict and not self.ontology.is_known_node_type(n.type):
                logger.info("Dropping node with unknown type %r (strict)", n.type)
                continue
            if not self.ontology.is_known_node_type(n.type):
                logger.debug("Unknown node type %r (advisory)", n.type)
            nodes.append(
                ExtractedNode(
                    temp_id=f"{temp_offset}:{n.temp_id or i}",
                    type=n.type,
                    title=n.title,
                    kind=n.kind or (self.ontology.kind_of(n.type) or "knowledge"),
                    summary=n.summary,
                    confidence=n.confidence,
                    surface_phrase=n.surface_phrase,
                )
            )

        valid = {n.temp_id for n in nodes}
        edges = [
            ExtractedEdge(
                source_temp_id=f"{temp_offset}:{e.source_temp_id}",
                target_temp_id=f"{temp_offset}:{e.target_temp_id}",
                type=e.type,
                confidence=e.confidence,
            )
            for e in parsed.edges
        ]
        edges = [
            e for e in edges
            if e.source_temp_id in valid and e.target_temp_id in valid
        ]
        return nodes, edges, usage, model
