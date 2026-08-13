"""Deciding whether a new claim is something you already know.

The make-or-break step. Everything upstream is plumbing; everything downstream
depends on this being right, and "right" is asymmetric:

    a duplicate is annoying and reversible
    an over-merge destroys a distinct fact and is not

So every rule here is biased toward keeping two nodes when unsure.

The order of decisions, and why:

1. **Same run.** Ops apply only after the whole extraction resolves, so two
   candidates naming the same thing cannot see each other in the database.
   Without an in-run memo the pipeline manufactures duplicates inside the
   de-duplication pass itself.
2. **High cosine, same type.** Cheap and unambiguous — when it fires.
3. **Canonical entity.** If the host owns this entity, adopt its name.
4. **Adjudication.** The band where cosine is suggestive but not decisive.
5. **New node.**

Type equality gates every merge. Cosine cannot see the type system, and in
measured data the single highest-scoring pair in a corpus was a screen and a
flow that shared a name — exactly the merge that must never happen.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from contextgraph.config import ResolutionConfig
from contextgraph.ontology import Ontology
from contextgraph.protocols import CanonicalResolver, Embedder, StructuredLLM
from contextgraph.services.graph import (
    OP_CONTRADICT,
    OP_MERGE_NODE,
    OP_SUPERSEDE,
    RISK_ADDITIVE,
    RISK_HIGH,
    ResolvedOp,
)

logger = logging.getLogger(__name__)

# Tokens that carry no identity — stripped before comparing titles so
# "Checkout screen" and "Checkout" compare equal.
#
# Keep these ONLY when type equality is separately enforced. Without a type
# filter, stripping "screen"/"flow" makes a screen and a feature collapse to
# the same key and merge — a real, silent, cross-type over-merge.
_STOPWORDS = frozenset(
    {"a", "an", "the", "and", "or", "screen", "screens", "flow", "flows",
     "feature", "features", "page", "pages", "view", "views"}
)


def normalize_title(value: str | None) -> str:
    text_value = re.sub(r"[^a-z0-9]+", " ", (value or "").strip().lower())
    return " ".join(t for t in text_value.split() if t and t not in _STOPWORDS)


def titles_match(left: str | None, right: str | None) -> bool:
    """Exact-after-normalisation, or >=0.8 token overlap.

    The overlap arm is deliberately fuzzy and therefore NOT strong enough to
    merge on alone — "Task Status Tracking" vs "Task Status Tracking UI" is
    0.75 and a real duplicate, while plenty of 0.8 pairs are not. Callers
    treat exact equality as decisive and overlap as a nomination for
    adjudication.
    """
    a, b = normalize_title(left), normalize_title(right)
    if not a or not b:
        return False
    if a == b:
        return True
    at, bt = set(a.split()), set(b.split())
    return bool(at and bt) and len(at & bt) / max(len(at), len(bt)) >= 0.8


@dataclass
class Candidate:
    """An existing node considered as a match."""

    id: str
    score: float
    title: str
    type: str
    summary: str | None = None
    aliases: list[str] = field(default_factory=list)


class RelationshipVerdict(BaseModel):
    relationship: str = Field(
        ..., description="one of: duplicate | supersedes | contradicts | distinct"
    )
    reason: str = Field(default="")


@dataclass
class ExtractedNode:
    """A candidate emitted by extraction."""

    temp_id: str
    type: str
    title: str
    kind: str = ""
    summary: str | None = None
    confidence: float = 0.7
    surface_phrase: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedEdge:
    source_temp_id: str
    target_temp_id: str
    type: str
    confidence: float = 0.7


class ResolutionService:
    """Turns extraction candidates into apply-ready operations."""

    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        *,
        tenant_id: str,
        llm: StructuredLLM | None = None,
        ontology: Ontology,
        config: ResolutionConfig,
        canonical: CanonicalResolver | None = None,
    ) -> None:
        self.session = session
        self.embedder = embedder
        # Required, and bound for the life of the service. Candidate generation
        # decides what a new claim MERGES INTO, so an unscoped lookup here is
        # worse than a leaked read: it would fold this tenant's claim into
        # another tenant's node, joining two graphs by a write that no later
        # query can distinguish from a legitimate merge.
        self.tenant_id = tenant_id
        self.llm = llm
        self.ontology = ontology
        self.config = config
        self.canonical = canonical
        self.usage: list[dict[str, Any]] = []
        self.model = ""

    async def resolve(
        self,
        *,
        graph_id: str,
        run_id: str,
        source_id: str,
        nodes: list[ExtractedNode],
        edges: list[ExtractedEdge],
    ) -> list[ResolvedOp]:
        ops: list[ResolvedOp] = []
        temp_to_real: dict[str, str | None] = {}
        created_this_run: dict[tuple[str, str], str] = {}

        embeddings = (
            await self.embedder.embed_batch(
                [f"{n.title}\n{n.summary or ''}".strip() for n in nodes]
            )
            if nodes
            else []
        )
        citation = {"source_id": source_id, "run_id": run_id}

        # strict=True: a short embedding batch would otherwise pair each vector
        # with the wrong node and silently drop the tail, corrupting every
        # decision in the run rather than failing it.
        for node, embedding in zip(nodes, embeddings, strict=True):
            node_ops, real_id = await self._resolve_one(
                node, embedding, graph_id, citation, created_this_run
            )
            temp_to_real[node.temp_id] = real_id
            ops.extend(node_ops)

        for edge in edges:
            src = temp_to_real.get(edge.source_temp_id)
            tgt = temp_to_real.get(edge.target_temp_id)
            if not src or not tgt or src == tgt:
                continue
            ops.append(
                ResolvedOp(
                    op="ADD_EDGE",
                    risk=RISK_ADDITIVE,
                    payload={
                        "source_node_id": src,
                        "target_node_id": tgt,
                        "type": edge.type,
                        "confidence": edge.confidence,
                    },
                )
            )
        return ops

    async def _resolve_one(
        self,
        node: ExtractedNode,
        embedding: list[float],
        graph_id: str,
        citation: dict,
        memo: dict[tuple[str, str], str],
    ) -> tuple[list[ResolvedOp], str | None]:
        key = (normalize_title(node.title), node.type)

        prior = memo.get(key)
        if prior is not None:
            return ([self._merge_op(prior, node, citation, branch="same_run")], prior)

        candidates = await self._block_candidates(graph_id, node.kind, embedding)
        same_type = [c for c in candidates if c.type == node.type]
        best_same_type = same_type[0] if same_type else None
        best_any = candidates[0] if candidates else None

        if (
            best_same_type is not None
            and best_same_type.score >= self.config.merge_threshold
        ):
            return (
                [
                    self._merge_op(
                        best_same_type.id, node, citation,
                        branch="vector_threshold", score=best_same_type.score,
                    )
                ],
                best_same_type.id,
            )

        # Exact title match inside the graph, always — never gated behind an
        # optional resolver.
        #
        # This is the single most common duplicate and the vector path does NOT
        # catch it: the embedded text is "title\nsummary", so two nodes with
        # byte-identical titles and differently-worded summaries measured 0.798
        # on real data, well under any safe merge bar. Exact equality after
        # normalisation is decisive on its own and needs no threshold.
        exact, lexical_nominee = await self._match_by_title(graph_id, node)
        if exact is not None:
            return (
                [self._merge_op(exact, node, citation, branch="exact_title")],
                exact,
            )

        # Host-owned entity?
        if self.canonical is not None and node.type in self.ontology.canonical_types:
            match = await self.canonical.resolve(
                graph_id=graph_id, node_type=node.type, title=node.title
            )
            if match is not None:
                decided = await self._resolve_canonical(
                    node, embedding, citation, memo, match, key
                )
                if decided is not None:
                    return decided

        target = (
            best_same_type
            if best_same_type is not None
            and best_same_type.score >= self.config.related_low
            else (
                best_any
                if best_any is not None and best_any.score >= self.config.related_low
                else None
            )
        )
        # Nothing close enough by vector, but a same-type title overlaps enough
        # to be worth asking about. Adjudicated, never merged directly.
        if target is None:
            target = lexical_nominee
        if target is not None:
            return await self._adjudicate(
                node, embedding, citation, memo, key, target
            )

        new_id = str(uuid4())
        memo[key] = new_id
        return ([self._add_op(new_id, node, embedding, citation)], new_id)

    async def _resolve_canonical(
        self,
        node: ExtractedNode,
        embedding: list[float],
        citation: dict,
        memo: dict[tuple[str, str], str],
        match: dict[str, Any],
        key: tuple[str, str],
    ) -> tuple[list[ResolvedOp], str | None] | None:
        """Fold into, or create under, a host-owned canonical entity."""
        existing_id = match.get("node_id")
        canonical_title = match.get("title") or node.title

        if existing_id:
            # Exact normalised equality is decisive; a fuzzy nomination is not.
            if normalize_title(node.title) == normalize_title(match.get("title")):
                return (
                    [
                        self._merge_op(
                            existing_id, node, citation, branch="exact_title"
                        )
                    ],
                    existing_id,
                )
            verdict = await self._classify(
                node,
                Candidate(
                    id=existing_id, score=0.0,
                    title=match.get("title") or "", type=node.type,
                ),
            )
            if verdict == "duplicate":
                return (
                    [
                        self._merge_op(
                            existing_id, node, citation, branch="canonical_adjudicated"
                        )
                    ],
                    existing_id,
                )
            # Adjudicated as different. Fall through to normal handling — and
            # emphatically do NOT adopt the canonical title, which the
            # adjudicator just ruled belongs to something else.
            return None

        # Canonical entity with no node yet: create one under its name, keeping
        # the extracted phrasing as an alias so the next run's wording matches.
        #
        # An earlier version dropped the node entirely here, which also silently
        # discarded every edge referencing it — the relationship the source
        # established was lost with no log line.
        prior = memo.get((normalize_title(canonical_title), node.type))
        if prior is not None:
            return (
                [self._merge_op(prior, node, citation, branch="same_run_canonical")],
                prior,
            )
        new_id = str(uuid4())
        memo[(normalize_title(canonical_title), node.type)] = new_id
        memo[key] = new_id
        return (
            [
                self._add_op(
                    new_id, node, embedding, citation,
                    canonical_title=canonical_title,
                    source_entity=match.get("ref") or {"title": canonical_title},
                    resolution={"decision": "adopted_canonical_entity"},
                )
            ],
            new_id,
        )

    async def _adjudicate(
        self,
        node: ExtractedNode,
        embedding: list[float],
        citation: dict,
        memo: dict[tuple[str, str], str],
        key: tuple[str, str],
        target: Candidate,
    ) -> tuple[list[ResolvedOp], str | None]:
        rel = await self._classify(node, target)

        # The adjudicator reads text, not types. A cross-type "duplicate" is
        # refused in code rather than trusted to the prompt.
        if rel == "duplicate" and target.type != node.type:
            logger.info(
                "Refusing cross-type merge of %r (%s) into %r (%s)",
                node.title, node.type, target.title, target.type,
            )
            rel = "distinct"

        new_id = str(uuid4())
        add_op = self._add_op(
            new_id, node, embedding, citation,
            resolution={
                "decision": rel,
                "score": round(target.score, 4),
                "candidate_id": target.id,
            },
        )

        if rel == "duplicate":
            return (
                [
                    self._merge_op(
                        target.id, node, citation, branch="adjudicated",
                        score=target.score, candidate_id=target.id,
                    )
                ],
                target.id,
            )

        memo[key] = new_id
        if rel == "supersedes":
            return (
                [
                    add_op,
                    ResolvedOp(
                        op=OP_SUPERSEDE,
                        risk=RISK_HIGH,  # gated: retires an existing claim
                        payload={
                            "new_id": new_id, "old_id": target.id,
                            "new_title": node.title, "new_type": node.type,
                            "old_title": target.title, "old_type": target.type,
                        },
                    ),
                ],
                new_id,
            )
        if rel == "contradicts":
            return (
                [
                    add_op,
                    ResolvedOp(
                        op=OP_CONTRADICT,
                        risk=RISK_ADDITIVE,  # represent it, never overwrite
                        payload={
                            "new_id": new_id, "old_id": target.id,
                            "new_title": node.title, "old_title": target.title,
                        },
                    ),
                ],
                new_id,
            )
        return ([add_op], new_id)

    # -- candidate generation ------------------------------------------------ #

    async def _match_by_title(
        self, graph_id: str, node: ExtractedNode
    ) -> tuple[str | None, Candidate | None]:
        """Lexical lookup over same-type nodes: ``(exact_id, nominee)``.

        Two results from one scan, because the two are decided differently.

        *Exact after normalisation* is decisive and merges on its own. An alias
        counts as much as a title, which is what lets a phrasing recorded by an
        earlier merge resolve the next occurrence of that phrasing.

        *Token overlap* is only a nomination. "Task Status Tracking" against
        "Task Status Tracking Detail" is 0.75 overlap and a real duplicate,
        while plenty of pairs at the same overlap are siblings that must stay
        apart — so the nominee goes to the adjudicator, which is biased toward
        "distinct", and never merges on the lexical signal alone.

        The nomination exists because the vector path cannot be relied on to
        raise these: the embedded text is "title\\nsummary", so a near-identical
        title with a differently-worded summary can score below ``related_low``
        and never become a candidate at all. This scan sees every same-type
        node, not just the vector top-k.

        Compared in Python rather than SQL so the normalisation rules stay in
        one place. Scanning same-type rows is acceptable because the set is
        small by construction; if a graph grows a type with tens of thousands
        of nodes, add a generated normalised-title column and index it.
        """
        target = normalize_title(node.title)
        if not target:
            return None, None
        rows = (
            await self.session.execute(
                text("""
                    SELECT id::text, title, aliases,
                           properties->>'summary'
                    FROM cg_nodes
                    WHERE tenant_id = :tenant AND graph_id = :graph
                      AND type = :type
                      AND status = 'active' AND deleted_at IS NULL
                    ORDER BY created_at ASC
                """),
                {"tenant": self.tenant_id, "graph": graph_id, "type": node.type},
            )
        ).fetchall()

        nominee: Candidate | None = None
        for row in rows:
            surfaces = [row[1], *(row[2] or [])]
            if any(normalize_title(s) == target for s in surfaces):
                return str(row[0]), None
            if nominee is None and any(
                titles_match(node.title, s) for s in surfaces
            ):
                nominee = Candidate(
                    id=str(row[0]), score=0.0, title=row[1], type=node.type,
                    summary=row[3], aliases=list(row[2] or []),
                )
        return None, nominee

    async def _block_candidates(
        self, graph_id: str, kind: str, embedding: list[float]
    ) -> list[Candidate]:
        """Nearest same-kind nodes, best first.

        A list, not a single row. Top-1 lets a cross-type neighbour mask the
        true same-type match, and since merging requires type equality the
        candidate set must be deep enough to contain a same-type row.

        The projected summary includes phrasings folded in by earlier merges,
        so the adjudicator judges a node as it actually is rather than only its
        original wording.
        """
        emb = "[" + ",".join(str(v) for v in embedding) + "]"
        rows = (
            await self.session.execute(
                text("""
                    SELECT id::text,
                           1 - (embedding <=> CAST(:emb AS vector)) AS score,
                           title, type,
                           trim(both E'\n' from
                                coalesce(properties->>'summary', '')
                                || coalesce(E'\n' || (
                                     SELECT string_agg(value #>> '{}', E'\n')
                                     FROM jsonb_array_elements(
                                       coalesce(properties->'summaries','[]'::jsonb))
                                   ), '')
                           ) AS summary,
                           aliases
                    FROM cg_nodes
                    WHERE tenant_id = :tenant AND graph_id = :graph
                      AND kind = :kind
                      AND status = 'active'
                      AND deleted_at IS NULL
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:emb AS vector)
                    LIMIT :k
                """),
                {
                    "emb": emb, "tenant": self.tenant_id, "graph": graph_id,
                    "kind": kind, "k": self.config.candidate_limit,
                },
            )
        ).fetchall()
        return [
            Candidate(
                id=r[0], score=float(r[1]), title=r[2], type=r[3],
                summary=r[4] or None, aliases=list(r[5] or []),
            )
            for r in rows
        ]

    async def _classify(self, node: ExtractedNode, target: Candidate) -> str:
        """Ask the model how a new claim relates to a similar existing one.

        Sends types, summaries and aliases — not two bare titles. Titles alone
        are frequently insufficient in both directions: two differently-worded
        statements of one fact look distinct, and two similarly-worded
        statements about different things look identical.
        """
        if self.llm is None:
            return "distinct"
        aka = (
            f"\nEXISTING also known as: {', '.join(target.aliases[:5])}"
            if target.aliases else ""
        )
        system = [
            "Classify how a NEW claim relates to an EXISTING one. Answer with "
            "exactly one relationship:\n"
            "- duplicate: the same thing or claim, restated. Only when merging "
            "them would lose nothing.\n"
            "- supersedes: NEW updates or replaces EXISTING.\n"
            "- contradicts: NEW conflicts with EXISTING; both cannot be true.\n"
            "- distinct: unrelated, independently true, or RELATED BUT NOT THE "
            "SAME — a part of it, a variant of it, or a different stage of the "
            "same workflow.\n"
            "When in doubt answer distinct. Keeping two nodes is recoverable; "
            "merging two different things is not."
        ]
        user = (
            f"EXISTING [{target.type}]: {target.title}\n"
            f"EXISTING detail: {target.summary or '(none)'}{aka}\n\n"
            f"NEW [{node.type}]: {node.title}\n"
            f"NEW detail: {node.summary or '(none)'}"
        )
        try:
            parsed, usage = await self.llm.complete(
                system=system, user=user, schema=RelationshipVerdict,
                max_tokens=256, temperature=0.0,
            )
            if usage:
                self.usage.append(usage)
                self.model = self.model or str(usage.get("model", ""))
            if parsed is None:
                return self._on_classify_failure()
            rel = (parsed.relationship or "distinct").strip().lower()
            return (
                rel
                if rel in {"duplicate", "supersedes", "contradicts", "distinct"}
                else "distinct"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Adjudication failed: %s", exc)
            return self._on_classify_failure()

    def _on_classify_failure(self) -> str:
        """Fail open (duplicate) or closed (gated) per config.

        Open is the default: a provider outage then degrades into extra nodes,
        which a human can merge, rather than into merges nobody reviewed.
        """
        return "gate" if self.config.fail_closed else "distinct"

    # -- op builders --------------------------------------------------------- #

    @staticmethod
    def _merge_op(
        existing_id: str,
        node: ExtractedNode,
        citation: dict,
        *,
        branch: str = "unknown",
        score: float | None = None,
        candidate_id: str | None = None,
    ) -> ResolvedOp:
        return ResolvedOp(
            op=OP_MERGE_NODE,
            risk=RISK_ADDITIVE,
            payload={
                "existing_id": existing_id,
                "alias": node.title,
                "aliases": [node.surface_phrase] if node.surface_phrase else [],
                "summary": node.summary,
                "resolution": {
                    "decision": "merge",
                    "branch": branch,
                    **({"score": round(score, 4)} if score is not None else {}),
                    **({"candidate_id": candidate_id} if candidate_id else {}),
                },
                "citation": citation,
            },
        )

    @staticmethod
    def _add_op(
        new_id: str,
        node: ExtractedNode,
        embedding: list[float],
        citation: dict,
        *,
        canonical_title: str | None = None,
        source_entity: dict[str, Any] | None = None,
        resolution: dict[str, Any] | None = None,
    ) -> ResolvedOp:
        title = canonical_title or node.title
        aliases = [
            a for a in [
                node.title if canonical_title and node.title != title else None,
                node.surface_phrase,
            ] if a and a != title
        ]
        payload: dict[str, Any] = {
            "id": new_id,
            "kind": node.kind,
            "type": node.type,
            "title": title,
            "aliases": aliases,
            "summary": node.summary,
            "confidence": node.confidence,
            "citations": [citation],
        }
        if node.attributes:
            payload["attributes"] = node.attributes
        if source_entity:
            payload["source_entity"] = source_entity
        if resolution:
            payload["resolution"] = resolution
        return ResolvedOp(
            op="ADD_NODE", risk=RISK_ADDITIVE, payload=payload, embedding=embedding
        )
