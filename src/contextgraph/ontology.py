"""The vocabulary — data, not schema.

Node kinds, node types and edge types are plain strings in the database, and
the taxonomy lives here so it can evolve without a migration. Validation is
advisory by default: an unknown type is flagged, not rejected, which keeps the
graph flexible without letting it become a junk drawer.

The default ontology below is product/specification shaped because that is the
domain it was proven in. It is a *default*, not a commitment — pass your own
``Ontology`` to ``ContextGraph`` and extraction, resolution and validation all
follow it.

One rule is not negotiable, whatever ontology you supply: **type equality
gates merging**. Two nodes of different types are never "the same thing", no
matter how similar their text. That single constraint is what stops a
"Checkout flow" being folded into a "Checkout screen" — an over-merge destroys
a distinct fact irreversibly, where a duplicate merely annoys.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KIND_SOURCE = "source"
KIND_KNOWLEDGE = "knowledge"
KIND_ENTITY = "entity"

# Edge vocabulary. The graph alone owns these — they are the connective tissue
# that a pile of embedded chunks cannot express.
EDGE_MOTIVATES = "motivates"
EDGE_AFFECTS = "affects"
EDGE_CONTAINS = "contains"
EDGE_DEPENDS_ON = "depends_on"
EDGE_IMPLEMENTS = "implements"
EDGE_DERIVED_FROM = "derived_from"
EDGE_REFERENCES = "references"
EDGE_CONTRADICTS = "contradicts"
EDGE_SUPERSEDES = "supersedes"

DEFAULT_EDGE_TYPES: tuple[str, ...] = (
    EDGE_MOTIVATES,
    EDGE_AFFECTS,
    EDGE_CONTAINS,
    EDGE_DEPENDS_ON,
    EDGE_IMPLEMENTS,
    EDGE_DERIVED_FROM,
    EDGE_REFERENCES,
    EDGE_CONTRADICTS,
    EDGE_SUPERSEDES,
)

# Conflict/temporal edges. These are never overwritten — they are invalidated
# bi-temporally, so the graph can answer "what did we believe, and when".
TEMPORAL_EDGE_TYPES: frozenset[str] = frozenset({EDGE_CONTRADICTS, EDGE_SUPERSEDES})

DEFAULT_NODE_TYPES: dict[str, tuple[str, ...]] = {
    KIND_KNOWLEDGE: (
        "knowledge/decision",
        "knowledge/constraint",
        "knowledge/problem",
        "knowledge/insight",
        "knowledge/open_question",
        "knowledge/feedback",
        "knowledge/request",
    ),
    KIND_ENTITY: (
        "entity/feature",
        "entity/component",
        "entity/screen",
        "entity/flow",
        "entity/api",
        "entity/data_model",
        "entity/actor",
    ),
}

# Surfaced contradictions become a node of this type rather than a
# notification, so a conflict is a thing you can look at, link to and resolve.
TYPE_OPEN_QUESTION = "knowledge/open_question"


@dataclass(frozen=True)
class Ontology:
    """A node/edge vocabulary. Pass your own to re-target the whole pipeline."""

    node_types: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_NODE_TYPES)
    )
    edge_types: tuple[str, ...] = DEFAULT_EDGE_TYPES
    open_question_type: str = TYPE_OPEN_QUESTION
    # Types resolved against the host's system of record via CanonicalResolver.
    # Empty means "every type is owned by the graph".
    canonical_types: frozenset[str] = frozenset(DEFAULT_NODE_TYPES[KIND_ENTITY])
    # Advisory by default. Set True to reject unknown types at extraction time —
    # tighter, but a model that invents a type then loses the whole node.
    strict: bool = False

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self.node_types.keys())

    @property
    def all_node_types(self) -> frozenset[str]:
        return frozenset(t for types in self.node_types.values() for t in types)

    def kind_of(self, node_type: str) -> str | None:
        """Derive kind from a ``"<kind>/<type>"`` label.

        Models routinely omit ``kind`` even when asked for it, so it is derived
        from the type prefix rather than trusted.
        """
        head = node_type.split("/", 1)[0]
        return head if head in self.node_types else None

    def is_known_node_type(self, node_type: str) -> bool:
        return node_type in self.all_node_types

    def is_known_edge_type(self, edge_type: str) -> bool:
        return edge_type in self.edge_types

    def describe(self) -> str:
        """The vocabulary, rendered for an extraction prompt."""
        lines = []
        for kind, types in self.node_types.items():
            lines.append(f"{kind.upper()} nodes — types: {', '.join(types)}")
        lines.append(f"EDGE types: {', '.join(self.edge_types)}")
        return "\n".join(lines)


DEFAULT_ONTOLOGY = Ontology()
