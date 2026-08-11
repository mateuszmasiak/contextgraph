"""Pipeline stages. Most callers should use ``contextgraph.ContextGraph`` instead."""

from contextgraph.services.graph import Actor, ApplyResult, GraphService, ResolvedOp
from contextgraph.services.ingestion import IngestionService
from contextgraph.services.resolution import (
    Candidate,
    ExtractedEdge,
    ExtractedNode,
    ResolutionService,
)
from contextgraph.services.retrieval import (
    GraphContext,
    RetrievalService,
    SegmentHit,
    summary_of,
)
from contextgraph.services.roster import build_roster, build_roster_block, render_roster
from contextgraph.services.structuring import StructureResult, structure_source

__all__ = [
    "Actor",
    "ApplyResult",
    "Candidate",
    "ExtractedEdge",
    "ExtractedNode",
    "GraphContext",
    "GraphService",
    "IngestionService",
    "ResolutionService",
    "ResolvedOp",
    "RetrievalService",
    "SegmentHit",
    "StructureResult",
    "build_roster",
    "build_roster_block",
    "render_roster",
    "structure_source",
    "summary_of",
]
