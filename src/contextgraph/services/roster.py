"""Telling the extractor what you already know.

The highest-leverage component in the pipeline, and the least obvious one.

Measured on a real corpus: 10 of 11 near-duplicate pairs were *cross-run*. The
extractor had never been shown the graph, so each pass re-invented a surface
form for something already stored — "Task Status Tracking" became "Task Status
Tracking UI", "Task Assignment Notifications" became "Assignment Notifications"
— and resolution was left to recover identity from cosine, which measurably
cannot separate those from genuine siblings.

Showing the extractor the existing titles moves de-duplication from *detection*
to *prevention*. A reused title matches on exact equality, which needs no
threshold at all. In an A/B where a second source restated the first in drifted
wording, roster-on produced 0 new nodes and 5 merges; roster-off invented
spurious extra nodes in both runs.

The cost is real and worth naming: the model can force-fit a genuinely new
thing onto a listed title. That is why the prompt biases explicitly toward
"treat it as NEW when unsure", and why the phrasing the source actually used is
recorded as ``surface_phrase`` — so a force-fit leaves a trail instead of
silently attaching a claim to the wrong entity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from contextgraph.config import RosterConfig

logger = logging.getLogger(__name__)


@dataclass
class RosterEntry:
    type: str
    title: str
    aliases: list[str]


async def build_roster(
    session: AsyncSession, tenant_id: str, graph_id: str, config: RosterConfig
) -> list[RosterEntry]:
    """Existing titles for a graph, most reusable first.

    Entities lead: they are named again every time they are discussed, so they
    duplicate far more than one-off claims do.

    Scoped by tenant as well as graph. The roster is the one read whose output
    goes verbatim into a prompt, so an unscoped query here would not merely
    leak another tenant's titles — it would invite the extractor to reuse them,
    writing the leak into this tenant's graph as node titles of its own.
    """
    rows = (
        await session.execute(
            text("""
                SELECT type, title, aliases
                FROM cg_nodes
                WHERE tenant_id = :tenant AND graph_id = :graph
                  AND deleted_at IS NULL
                  AND status = 'active'
                ORDER BY (kind = 'entity') DESC, updated_at DESC
                LIMIT :limit
            """),
            {"tenant": tenant_id, "graph": graph_id, "limit": config.limit},
        )
    ).fetchall()
    return [
        RosterEntry(
            type=r[0],
            title=r[1],
            aliases=[a for a in (r[2] or []) if a][: config.max_aliases_per_entry],
        )
        for r in rows
    ]


def render_roster(entries: list[RosterEntry]) -> str:
    """Render as a prompt block. Empty string when there is nothing to show."""
    if not entries:
        return ""
    by_type: dict[str, list[str]] = {}
    for e in entries:
        label = f"{e.title} (aka: {', '.join(e.aliases)})" if e.aliases else e.title
        by_type.setdefault(e.type, []).append(label)

    lines = ["ENTITIES AND CLAIMS ALREADY IN THIS GRAPH:"]
    for node_type in sorted(by_type):
        lines.append(f"- {node_type}: " + " | ".join(by_type[node_type]))
    lines.append(
        "\nWhen something you find IS one of the above, reuse its title "
        "CHARACTER-FOR-CHARACTER as `title`, and put the wording this source "
        "actually used in `surface_phrase`. Do not invent a variant spelling, a "
        "longer form, or a suffix for a thing already listed.\n"
        "Only invent a new title for something genuinely absent from the list. "
        "If you cannot tell whether it is the same thing, treat it as NEW — a "
        "duplicate is easy to merge later, a wrongly reused title silently "
        "attaches your claim to the wrong entity."
    )
    return "\n".join(lines)


async def build_roster_block(
    session: AsyncSession, tenant_id: str, graph_id: str, config: RosterConfig
) -> str:
    """Best-effort roster. Never raises — extraction proceeds without it."""
    if not config.enabled:
        return ""
    try:
        return render_roster(
            await build_roster(session, tenant_id, graph_id, config)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Roster build failed for graph %s: %s", graph_id, exc)
        return ""
