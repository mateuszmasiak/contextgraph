"""contextgraph — a governed, bi-temporal knowledge graph for agent memory.

Postgres + pgvector. No graph database, no second datastore, no vendor lock.

    from contextgraph import ContextGraph
    from contextgraph.embeddings import OpenAIEmbedder
    from contextgraph.llm import AnthropicLLM

    graph = ContextGraph(
        session_factory,
        embedder=OpenAIEmbedder(api_key=...),
        llm=AnthropicLLM(api_key=...),
    )

    await graph.add("acme", "webapp", "We chose Postgres over DynamoDB.")
    ctx = await graph.recall("acme", "webapp", "what database did we pick?")
    print(ctx.to_text())

What distinguishes it from a vector store with extra steps:

- **De-duplication that works.** Extraction is shown what the graph already
  contains, so the same fact phrased differently resolves to the same node
  instead of a near-twin. Merging requires type equality and, below a high
  similarity bar, an explicit adjudication.
- **Governed writes.** Additive operations apply; operations that change or
  retire existing truth gate for review when an agent proposes them. Most
  memory systems write unilaterally at ingest.
- **Nothing is deleted.** Contradictions are represented rather than resolved;
  superseded claims keep their rows and their edges.
- **Provenance everywhere.** Every claim cites the source span it came from, so
  an extraction error is recoverable rather than permanent.
"""

from contextgraph.client import ContextGraph
from contextgraph.config import (
    ChunkConfig,
    Config,
    ExtractionConfig,
    ResolutionConfig,
    RosterConfig,
)
from contextgraph.errors import (
    ConcurrentModificationError,
    ContextGraphError,
    EmbeddingUnavailableError,
    ExtractionError,
    SchemaMismatchError,
    ScopeViolationError,
)
from contextgraph.models import (
    Base,
    Changeset,
    GraphEdge,
    GraphNode,
    GraphRun,
    GraphSource,
    SourceSegment,
)
from contextgraph.ontology import DEFAULT_ONTOLOGY, Ontology
from contextgraph.protocols import (
    CanonicalResolver,
    Embedder,
    Meter,
    NullCanonicalResolver,
    NullMeter,
    StructuredLLM,
)
from contextgraph.services.graph import Actor
from contextgraph.services.retrieval import GraphContext, SegmentHit
from contextgraph.services.structuring import StructureResult

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_ONTOLOGY",
    "Actor",
    "Base",
    "CanonicalResolver",
    "Changeset",
    "ChunkConfig",
    "ConcurrentModificationError",
    "Config",
    "ContextGraph",
    "ContextGraphError",
    "Embedder",
    "EmbeddingUnavailableError",
    "ExtractionConfig",
    "ExtractionError",
    "GraphContext",
    "GraphEdge",
    "GraphNode",
    "GraphRun",
    "GraphSource",
    "Meter",
    "NullCanonicalResolver",
    "NullMeter",
    "Ontology",
    "ResolutionConfig",
    "RosterConfig",
    "SchemaMismatchError",
    "ScopeViolationError",
    "SegmentHit",
    "SourceSegment",
    "StructureResult",
    "StructuredLLM",
    "__version__",
]
