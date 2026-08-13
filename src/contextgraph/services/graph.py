"""Applying mutations — and deciding which ones need a human first.

Every write to ``cg_nodes``/``cg_edges`` funnels through ``GraphService``. That
is not tidiness; it is what makes the gate meaningful. A second write path is a
hole in the gate.

Two axes decide what happens to an operation:

  op kind    — presentation / additive / truth
  actor kind — human / agent

Presentation never gates. Additive always applies. Truth operations gate when
an agent proposes them and apply-with-audit when a human does. Crucially the
classification is computed server-side from the op itself; a caller's claimed
risk is ignored, so a client cannot smuggle a title change inside a layout
update.

Writes use raw SQL rather than the ORM because the ``embedding vector(N)``
column needs an explicit cast and the ORM has nothing to add to that.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from contextgraph.errors import ScopeViolationError
from contextgraph.ontology import Ontology

logger = logging.getLogger(__name__)

# -- operation vocabulary ---------------------------------------------------- #

OP_ADD_NODE = "ADD_NODE"
OP_MERGE_NODE = "MERGE_NODE"
OP_ADD_EDGE = "ADD_EDGE"
OP_ADD_CITATION = "ADD_CITATION"
OP_SUPERSEDE = "SUPERSEDE"
OP_CONTRADICT = "CONTRADICT"
OP_UPDATE_NODE = "UPDATE_NODE"
OP_DISCONNECT = "DISCONNECT"
OP_REMOVE_NODE = "REMOVE_NODE"
OP_SET_LAYOUT = "SET_LAYOUT"

RISK_ADDITIVE = "additive"
RISK_HIGH = "high"

PRESENTATION_OPS = frozenset({OP_SET_LAYOUT})
ADDITIVE_OPS = frozenset(
    {OP_ADD_NODE, OP_ADD_EDGE, OP_ADD_CITATION, OP_MERGE_NODE, OP_CONTRADICT}
)
TRUTH_OPS = frozenset({OP_UPDATE_NODE, OP_DISCONNECT, OP_REMOVE_NODE, OP_SUPERSEDE})

# Keys a SET_LAYOUT may carry. Anything else is a truth change wearing a
# presentation costume.
_LAYOUT_ALLOWED_KEYS = frozenset({"node_id", "view", "layout"})


@dataclass
class ResolvedOp:
    """One mutation, ready to apply or gate."""

    op: str
    risk: str
    payload: dict[str, Any]
    embedding: list[float] | None = None


@dataclass
class ApplyResult:
    created_node_ids: list[str] = field(default_factory=list)
    merged_node_ids: list[str] = field(default_factory=list)
    created_edge_ids: list[str] = field(default_factory=list)
    updated_node_ids: list[str] = field(default_factory=list)
    gated_ops: list[dict[str, Any]] = field(default_factory=list)
    # Every op, carrying the risk the SERVER computed rather than the one the
    # caller proposed. The changeset is written from this list so that the
    # record a reviewer later reads and the decision the gate actually made
    # cannot disagree — if they were derived separately, a changeset could show
    # "pending review" for an op that already applied, or hide one that didn't.
    classified_ops: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Actor:
    """Who is proposing. Drives the truth-op gating policy."""

    kind: str = "agent"  # "agent" | "human"
    id: str | None = None
    run_id: UUID | None = None


def compact_op(op: ResolvedOp, *, risk: str | None = None) -> dict[str, Any]:
    """JSON-safe summary for storage in a changeset.

    Note the embedding is stripped: a 1536-float vector per op would bloat the
    changeset row for no benefit. ``apply_gated`` re-embeds on approval rather
    than persisting the vector here — a node approved without an embedding is
    invisible to de-duplication forever, which is far worse than one extra
    embedding call.

    ``risk`` overrides the value on the op. Callers inside ``apply`` pass the
    server-computed classification; the op's own field is only a proposal.
    """
    return {"op": op.op, "risk": risk or op.risk, "payload": op.payload}


def classify(op_kind: str, actor: Actor) -> str:
    """Server-side risk for an op. The caller's claim is never consulted."""
    if op_kind in PRESENTATION_OPS:
        return RISK_ADDITIVE
    if op_kind in ADDITIVE_OPS:
        return RISK_ADDITIVE
    if op_kind in TRUTH_OPS:
        return RISK_ADDITIVE if actor.kind == "human" else RISK_HIGH
    return RISK_HIGH  # unknown op: gate it


def _sanitize_layout(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip everything a layout op is not allowed to carry."""
    return {k: v for k, v in payload.items() if k in _LAYOUT_ALLOWED_KEYS}


class GraphService:
    """Applies resolved mutations inside the caller's transaction."""

    def __init__(self, session: AsyncSession, ontology: Ontology | None = None) -> None:
        self.session = session
        self.ontology = ontology

    async def apply(
        self,
        ops: list[ResolvedOp],
        *,
        tenant_id: str,
        graph_id: str,
        actor: Actor | None = None,
        run_id: UUID | None = None,
    ) -> ApplyResult:
        """Apply what may apply; gate what may not.

        The risk of each op is recomputed here from the op kind and the actor.
        ``ResolvedOp.risk`` is treated as a proposal and never as an authority:
        if the gate honoured the field on the incoming op, anything able to
        construct an op could mark a supersede "additive" and retire a claim
        with no review — which is the entire thing the gate exists to prevent.

        Defaulting ``actor`` to an agent is the strict reading: unattended
        callers get the gated policy, and the permissive human path has to be
        asked for explicitly.
        """
        actor = actor or Actor(kind="agent")
        result = ApplyResult()
        for op in ops:
            risk = classify(op.op, actor)
            record = compact_op(op, risk=risk)
            result.classified_ops.append(record)
            if risk == RISK_HIGH:
                result.gated_ops.append(record)
                continue
            await self._dispatch(op, tenant_id, graph_id, run_id, result)
        return result

    async def apply_gated(
        self,
        ops: list[dict[str, Any]],
        *,
        tenant_id: str,
        graph_id: str,
        run_id: UUID | None = None,
        embedder: Any | None = None,
    ) -> ApplyResult:
        """Apply previously-gated ops after human approval.

        Goes through the same dispatcher as ``apply`` with the risk check
        bypassed. An earlier design had a separate, partial implementation here
        that silently skipped op kinds it did not handle while still marking the
        changeset applied — the reviewer approved changes that never happened.
        Unknown ops now raise.
        """
        result = ApplyResult()
        for raw in ops:
            resolved = ResolvedOp(
                op=raw.get("op", ""),
                risk=RISK_ADDITIVE,
                payload=raw.get("payload", {}),
            )
            # Re-embed: compact_op dropped the vector on the way in.
            if resolved.op == OP_ADD_NODE and embedder is not None:
                title = resolved.payload.get("title", "")
                summary = resolved.payload.get("summary") or ""
                resolved.embedding = await embedder.embed(f"{title}\n{summary}".strip())
            await self._dispatch(resolved, tenant_id, graph_id, run_id, result)
        return result

    async def _dispatch(
        self,
        op: ResolvedOp,
        tenant_id: str,
        graph_id: str,
        run_id: UUID | None,
        result: ApplyResult,
    ) -> None:
        if op.op == OP_ADD_NODE:
            result.created_node_ids.append(
                await self._add_node(op, tenant_id, graph_id, run_id)
            )
        elif op.op == OP_MERGE_NODE:
            result.merged_node_ids.append(
                await self._merge_node(op, tenant_id, graph_id)
            )
        elif op.op in (OP_ADD_EDGE, OP_ADD_CITATION):
            edge_id = await self._add_edge(op, tenant_id, graph_id, run_id)
            if edge_id:
                result.created_edge_ids.append(edge_id)
        elif op.op == OP_SUPERSEDE:
            result.updated_node_ids.append(
                await self._supersede(op, tenant_id, graph_id, run_id)
            )
        elif op.op == OP_CONTRADICT:
            await self._contradict(op, tenant_id, graph_id, run_id, result)
        elif op.op == OP_UPDATE_NODE:
            result.updated_node_ids.append(
                await self._update_node(op, tenant_id, graph_id)
            )
        elif op.op == OP_DISCONNECT:
            await self._disconnect(op, tenant_id, graph_id)
        elif op.op == OP_REMOVE_NODE:
            result.updated_node_ids.append(
                await self._remove_node(op, tenant_id, graph_id)
            )
        elif op.op == OP_SET_LAYOUT:
            await self._set_layout(op, tenant_id, graph_id)
        else:
            raise ValueError(f"Unknown op: {op.op!r}")

    # -- scope guards -------------------------------------------------------- #

    async def _assert_nodes_in_scope(
        self, node_ids: list[str], tenant_id: str, graph_id: str
    ) -> None:
        """Defence in depth: every id a caller supplies is re-checked in SQL.

        The route guard is the first line, but a single missing predicate in
        hand-written SQL is a cross-tenant leak, so scope is asserted again at
        the point of write.
        """
        ids = [i for i in node_ids if i]
        if not ids:
            return
        found = (
            await self.session.execute(
                text("""
                    SELECT count(*) FROM cg_nodes
                    WHERE id = ANY(CAST(:ids AS uuid[]))
                      AND tenant_id = :tenant AND graph_id = :graph
                """),
                {"ids": ids, "tenant": tenant_id, "graph": graph_id},
            )
        ).scalar_one()
        if int(found) != len(set(ids)):
            raise ScopeViolationError(
                f"{len(set(ids)) - int(found)} node(s) not in {tenant_id}/{graph_id}"
            )

    # -- writes -------------------------------------------------------------- #

    async def _add_node(
        self, op: ResolvedOp, tenant_id: str, graph_id: str, run_id: UUID | None
    ) -> str:
        p = op.payload
        node_id = p.get("id") or str(uuid4())
        embedding = op.embedding or []
        embedding_str = (
            "[" + ",".join(str(v) for v in embedding) + "]" if embedding else None
        )

        properties: dict[str, Any] = {
            "summary": p.get("summary"),
            "citations": p.get("citations", []),
        }
        for key in ("layout", "source_entity", "resolution", "attributes"):
            if p.get(key):
                properties[key] = p[key]

        # Aliases are honoured on insert. When resolution adopts a canonical
        # title it carries the extracted phrasing across as an alias, and that
        # alias is what lets the NEXT extraction's wording match this node
        # instead of creating a twin.
        aliases = [
            a for a in (p.get("aliases") or []) if a and a != p["title"]
        ]
        await self.session.execute(
            text("""
                INSERT INTO cg_nodes
                    (id, tenant_id, graph_id, kind, type, title, aliases, properties,
                     status, confidence, created_by_run_id, embedding,
                     created_at, updated_at)
                VALUES
                    (CAST(:id AS uuid), :tenant, :graph, :kind, :type, :title,
                     CAST(:aliases AS text[]), CAST(:props AS jsonb),
                     'active', :confidence, CAST(:run_id AS uuid),
                     CAST(:embedding AS vector), NOW(), NOW())
            """),
            {
                "id": node_id,
                "tenant": tenant_id,
                "graph": graph_id,
                "kind": p["kind"],
                "type": p["type"],
                "title": p["title"],
                "aliases": aliases,
                "props": json.dumps(properties),
                "confidence": p.get("confidence"),
                "run_id": str(run_id) if run_id else None,
                "embedding": embedding_str,
            },
        )
        return str(node_id)

    async def _merge_node(
        self, op: ResolvedOp, tenant_id: str, graph_id: str
    ) -> str:
        """Fold a candidate into an existing node without losing anything.

        A merge that keeps only the alias throws away the incoming claim's own
        wording — which is precisely the text that distinguished the two nodes,
        and often the more informative of the two. Distinct phrasings are kept
        under ``properties.summaries``; ``properties.merges`` records which rule
        decided, because a merge is the irreversible direction and needs at
        least the diagnosability an insert already gets.
        """
        p = op.payload
        existing_id = p["existing_id"]
        await self._assert_nodes_in_scope([existing_id], tenant_id, graph_id)

        new_aliases = [a for a in [p.get("alias"), *(p.get("aliases") or [])] if a]
        summary = (p.get("summary") or "").strip() or None
        citation = p.get("citation")

        await self.session.execute(
            text("""
                UPDATE cg_nodes
                SET aliases = (
                        SELECT ARRAY(
                            SELECT DISTINCT e
                            FROM unnest(
                                coalesce(aliases, '{}'::text[])
                                || CAST(:aliases AS text[])
                            ) AS e
                            WHERE e IS NOT NULL AND e <> title
                        )
                    ),
                    properties = jsonb_set(
                        jsonb_set(
                            jsonb_set(
                                coalesce(properties, '{}'::jsonb),
                                '{citations}',
                                coalesce(properties->'citations', '[]'::jsonb)
                                    || CAST(:citation AS jsonb)
                            ),
                            '{summaries}',
                            CASE
                                WHEN CAST(:summary AS text) IS NULL
                                  OR properties->>'summary' = CAST(:summary AS text)
                                  OR coalesce(properties->'summaries', '[]'::jsonb)
                                         @> CAST(:summary_json AS jsonb)
                                THEN coalesce(properties->'summaries', '[]'::jsonb)
                                ELSE coalesce(properties->'summaries', '[]'::jsonb)
                                         || CAST(:summary_json AS jsonb)
                            END
                        ),
                        '{merges}',
                        coalesce(properties->'merges', '[]'::jsonb)
                            || CAST(:merge_log AS jsonb)
                    ),
                    updated_at = NOW()
                WHERE id = CAST(:id AS uuid)
                  AND tenant_id = :tenant AND graph_id = :graph
            """),
            {
                "id": existing_id,
                "tenant": tenant_id,
                "graph": graph_id,
                "aliases": new_aliases,
                "citation": json.dumps([citation] if citation else []),
                "summary": summary,
                "summary_json": json.dumps([summary] if summary else []),
                "merge_log": json.dumps(
                    [{"alias": p.get("alias"), **(p.get("resolution") or {})}]
                ),
            },
        )
        return str(existing_id)

    async def _add_edge(
        self, op: ResolvedOp, tenant_id: str, graph_id: str, run_id: UUID | None
    ) -> str | None:
        p = op.payload
        src, tgt = p.get("source_node_id"), p.get("target_node_id")
        if not src or not tgt or src == tgt:
            return None
        await self._assert_nodes_in_scope([src, tgt], tenant_id, graph_id)
        edge_id = str(uuid4())
        # WHERE NOT EXISTS rather than a unique constraint: the edge vocabulary
        # is open, and a uniqueness violation mid-run would roll back an entire
        # structuring pass over a duplicate relationship nobody cares about.
        inserted = (
            await self.session.execute(
                text("""
                    INSERT INTO cg_edges
                        (id, tenant_id, graph_id, source_node_id, target_node_id,
                         type, properties, status, confidence, valid_at,
                         created_by_run_id, created_at, updated_at)
                    -- Every projected parameter carries an explicit cast.
                    -- INSERT...SELECT gives asyncpg two conflicting inferences
                    -- for any param used both in the projection and in the
                    -- WHERE clause (varchar from the column, text from the
                    -- comparison) and it fails with AmbiguousParameterError.
                    SELECT CAST(:id AS uuid),
                           CAST(:tenant AS varchar), CAST(:graph AS varchar),
                           CAST(:src AS uuid), CAST(:tgt AS uuid),
                           CAST(:type AS varchar),
                           CAST(:props AS jsonb), 'active',
                           CAST(:confidence AS double precision), NOW(),
                           CAST(:run_id AS uuid), NOW(), NOW()
                    WHERE NOT EXISTS (
                        SELECT 1 FROM cg_edges
                        WHERE source_node_id = CAST(:src AS uuid)
                          AND target_node_id = CAST(:tgt AS uuid)
                          AND type = CAST(:type AS varchar)
                          AND invalid_at IS NULL
                    )
                    RETURNING id::text
                """),
                {
                    "id": edge_id,
                    "tenant": tenant_id,
                    "graph": graph_id,
                    "src": src,
                    "tgt": tgt,
                    "type": p["type"],
                    "props": json.dumps(p.get("properties") or {}),
                    "confidence": p.get("confidence"),
                    "run_id": str(run_id) if run_id else None,
                },
            )
        ).first()
        return inserted[0] if inserted else None

    async def _supersede(
        self, op: ResolvedOp, tenant_id: str, graph_id: str, run_id: UUID | None
    ) -> str:
        """Retire a claim without erasing it."""
        p = op.payload
        old_id, new_id = p["old_id"], p["new_id"]
        await self._assert_nodes_in_scope([old_id, new_id], tenant_id, graph_id)
        await self.session.execute(
            text("""
                UPDATE cg_nodes SET status = 'superseded', updated_at = NOW()
                WHERE id = CAST(:id AS uuid)
                  AND tenant_id = :tenant AND graph_id = :graph
            """),
            {"id": old_id, "tenant": tenant_id, "graph": graph_id},
        )
        # Invalidate the old node's still-active edges, except supersedes edges
        # themselves — the chain of what replaced what must stay traversable.
        await self.session.execute(
            text("""
                UPDATE cg_edges SET invalid_at = NOW(), updated_at = NOW()
                WHERE tenant_id = :tenant AND graph_id = :graph
                  AND invalid_at IS NULL
                  AND type <> 'supersedes'
                  AND (source_node_id = CAST(:id AS uuid)
                       OR target_node_id = CAST(:id AS uuid))
            """),
            {"id": old_id, "tenant": tenant_id, "graph": graph_id},
        )
        await self._add_edge(
            ResolvedOp(
                op=OP_ADD_EDGE,
                risk=RISK_ADDITIVE,
                payload={
                    "source_node_id": new_id,
                    "target_node_id": old_id,
                    "type": "supersedes",
                },
            ),
            tenant_id,
            graph_id,
            run_id,
        )
        return str(old_id)

    async def _contradict(
        self,
        op: ResolvedOp,
        tenant_id: str,
        graph_id: str,
        run_id: UUID | None,
        result: ApplyResult,
    ) -> None:
        """Represent a conflict; never resolve it by fiat.

        Both claims survive. A ``contradicts`` edge links them and an
        open-question node makes the conflict a thing a human can find and
        settle. Picking a winner automatically is how a graph quietly becomes
        wrong.
        """
        p = op.payload
        new_id, old_id = p["new_id"], p["old_id"]
        await self._assert_nodes_in_scope([new_id, old_id], tenant_id, graph_id)

        edge_id = await self._add_edge(
            ResolvedOp(
                op=OP_ADD_EDGE,
                risk=RISK_ADDITIVE,
                payload={
                    "source_node_id": new_id,
                    "target_node_id": old_id,
                    "type": "contradicts",
                },
            ),
            tenant_id,
            graph_id,
            run_id,
        )
        if edge_id:
            result.created_edge_ids.append(edge_id)

        oq_type = (self.ontology.open_question_type if self.ontology
                   else "knowledge/open_question")
        question_id = str(uuid4())
        await self.session.execute(
            text("""
                INSERT INTO cg_nodes
                    (id, tenant_id, graph_id, kind, type, title, properties,
                     status, created_by_run_id, created_at, updated_at)
                VALUES (CAST(:id AS uuid), :tenant, :graph, :kind, :type, :title,
                        CAST(:props AS jsonb), 'active', CAST(:run_id AS uuid),
                        NOW(), NOW())
            """),
            {
                "id": question_id,
                "tenant": tenant_id,
                "graph": graph_id,
                "kind": oq_type.split("/", 1)[0],
                "type": oq_type,
                "title": (
                    f"Conflict: {p.get('new_title', 'new claim')} vs "
                    f"{p.get('old_title', 'existing claim')}"
                ),
                "props": json.dumps(
                    {
                        "conflict_node_ids": [new_id, old_id],
                        "resolved": False,
                        "detected_at": datetime.now(UTC).isoformat(),
                    }
                ),
                "run_id": str(run_id) if run_id else None,
            },
        )
        result.created_node_ids.append(question_id)

    async def _update_node(
        self, op: ResolvedOp, tenant_id: str, graph_id: str
    ) -> str:
        p = op.payload
        node_id = p["node_id"]
        await self._assert_nodes_in_scope([node_id], tenant_id, graph_id)
        sets, params = [], {"id": node_id, "tenant": tenant_id, "graph": graph_id}
        for column in ("title", "type", "status"):
            if p.get(column) is not None:
                sets.append(f"{column} = :{column}")
                params[column] = p[column]
        if p.get("summary") is not None:
            sets.append(
                "properties = jsonb_set(coalesce(properties,'{}'::jsonb),"
                "'{summary}', to_jsonb(CAST(:summary AS text)))"
            )
            params["summary"] = p["summary"]
        if not sets:
            return str(node_id)
        await self.session.execute(
            text(f"""
                UPDATE cg_nodes SET {", ".join(sets)}, updated_at = NOW()
                WHERE id = CAST(:id AS uuid)
                  AND tenant_id = :tenant AND graph_id = :graph
            """),
            params,
        )
        return str(node_id)

    async def _disconnect(
        self, op: ResolvedOp, tenant_id: str, graph_id: str
    ) -> None:
        """Invalidate an edge bi-temporally. Never DELETE."""
        await self.session.execute(
            text("""
                UPDATE cg_edges SET invalid_at = NOW(), updated_at = NOW()
                WHERE id = CAST(:id AS uuid) AND invalid_at IS NULL
                  AND tenant_id = :tenant AND graph_id = :graph
            """),
            {
                "id": op.payload["edge_id"],
                "tenant": tenant_id,
                "graph": graph_id,
            },
        )

    async def _remove_node(
        self, op: ResolvedOp, tenant_id: str, graph_id: str
    ) -> str:
        node_id = op.payload["node_id"]
        await self._assert_nodes_in_scope([node_id], tenant_id, graph_id)
        await self.session.execute(
            text("""
                UPDATE cg_nodes
                SET deleted_at = NOW(), status = 'rejected', updated_at = NOW()
                WHERE id = CAST(:id AS uuid)
                  AND tenant_id = :tenant AND graph_id = :graph
            """),
            {"id": node_id, "tenant": tenant_id, "graph": graph_id},
        )
        await self.session.execute(
            text("""
                UPDATE cg_edges SET invalid_at = NOW(), updated_at = NOW()
                WHERE tenant_id = :tenant AND graph_id = :graph
                  AND invalid_at IS NULL
                  AND (source_node_id = CAST(:id AS uuid)
                       OR target_node_id = CAST(:id AS uuid))
            """),
            {"id": node_id, "tenant": tenant_id, "graph": graph_id},
        )
        return str(node_id)

    async def _set_layout(
        self, op: ResolvedOp, tenant_id: str, graph_id: str
    ) -> None:
        p = _sanitize_layout(op.payload)
        node_id = p.get("node_id")
        if not node_id:
            return
        await self.session.execute(
            text("""
                UPDATE cg_nodes
                SET properties = jsonb_set(coalesce(properties,'{}'::jsonb),
                                           '{layout}', CAST(:layout AS jsonb)),
                    updated_at = NOW()
                WHERE id = CAST(:id AS uuid)
                  AND tenant_id = :tenant AND graph_id = :graph
            """),
            {
                "id": node_id,
                "tenant": tenant_id,
                "graph": graph_id,
                "layout": json.dumps(p.get("layout") or {}),
            },
        )
