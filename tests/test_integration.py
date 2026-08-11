"""End-to-end against a real Postgres. Skipped without DATABASE_URL.

These exist because the interesting failures in this package are SQL failures,
and a mocked SQL test asserts that a string equals a string. Everything here
runs the real statements against real pgvector.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from contextgraph import ContextGraph
from contextgraph.services.extraction import ExtractionResult, _Edge, _Node

pytestmark = pytest.mark.integration


def _extraction(*nodes: tuple[str, str, str], edges=()) -> ExtractionResult:
    """(temp_id, type, title) triples -> a wire-shaped extraction result."""
    return ExtractionResult(
        nodes=[
            _Node(
                temp_id=t, type=ty, title=ti,
                kind=ty.split("/")[0], summary=f"about {ti}",
            )
            for t, ty, ti in nodes
        ],
        edges=[
            _Edge(source_temp_id=s, target_temp_id=d, type=k) for s, d, k in edges
        ],
    )


@pytest.fixture
def graph(session_factory, embedder, llm, meter):
    return ContextGraph(
        session_factory, embedder=embedder, llm=llm, meter=meter
    )


class TestPipeline:
    @pytest.mark.asyncio
    async def test_add_creates_nodes_edges_and_provenance(
        self, graph, llm, clean_graph, session_factory
    ):
        llm.responses = [
            _extraction(
                ("n1", "entity/screen", "Task Dashboard"),
                ("n2", "knowledge/decision", "Use Postgres for the event store"),
                edges=[("n2", "n1", "affects")],
            )
        ]
        result = await graph.add("t1", clean_graph, "some meeting notes")

        assert result.status == "done"
        assert result.nodes_created == 2
        assert result.edges_created == 1

        async with session_factory() as s:
            nodes = (
                await s.execute(
                    text("SELECT title, kind, type FROM cg_nodes "
                         "WHERE graph_id = :g ORDER BY title"),
                    {"g": clean_graph},
                )
            ).fetchall()
            assert [n[0] for n in nodes] == [
                "Task Dashboard", "Use Postgres for the event store"
            ]
            # Provenance: the run recorded what it actually did.
            run = (
                await s.execute(
                    text("SELECT metrics FROM cg_runs WHERE graph_id = :g"),
                    {"g": clean_graph},
                )
            ).scalar_one()
            assert run["nodes_created"] == 2
            assert run["llm_calls"] >= 1

    @pytest.mark.asyncio
    async def test_identical_fact_twice_merges_instead_of_duplicating(
        self, graph, llm, clean_graph, session_factory
    ):
        """The headline behaviour."""
        llm.responses = [
            _extraction(("n1", "entity/feature", "Notifications")),
            _extraction(("n1", "entity/feature", "Notifications")),
        ]
        first = await graph.add("t1", clean_graph, "we need notifications")
        second = await graph.add("t1", clean_graph, "notifications are needed")

        assert first.nodes_created == 1
        assert second.nodes_created == 0, "a restated fact must not create a node"
        assert second.nodes_merged == 1

        async with session_factory() as s:
            count = (
                await s.execute(
                    text("SELECT count(*) FROM cg_nodes WHERE graph_id = :g "
                         "AND deleted_at IS NULL"),
                    {"g": clean_graph},
                )
            ).scalar_one()
            assert count == 1

    @pytest.mark.asyncio
    async def test_merge_keeps_both_phrasings_and_a_trail(
        self, graph, llm, clean_graph, session_factory
    ):
        llm.responses = [
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/feature", title="Notifications",
                kind="entity", summary="System alerts",
            )]),
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/feature", title="Notifications",
                kind="entity", summary="Keeps teams in sync",
                surface_phrase="Alerts",
            )]),
        ]
        await graph.add("t1", clean_graph, "a")
        await graph.add("t1", clean_graph, "b")

        async with session_factory() as s:
            row = (
                await s.execute(
                    text("SELECT aliases, properties FROM cg_nodes "
                         "WHERE graph_id = :g AND deleted_at IS NULL"),
                    {"g": clean_graph},
                )
            ).first()
            aliases, props = list(row[0] or []), row[1]
            assert "Alerts" in aliases
            assert "Notifications" not in aliases, "own title must not be an alias"
            assert "Keeps teams in sync" in (props.get("summaries") or [])
            assert props["merges"], "a merge must leave a decision trail"
            assert props["merges"][0]["branch"]

    @pytest.mark.asyncio
    async def test_identical_title_different_summary_still_merges(
        self, graph, llm, clean_graph, session_factory
    ):
        """The case cosine cannot catch.

        The embedded text is "title\\nsummary", so byte-identical titles with
        differently-worded summaries land well below any safe merge bar —
        measured at 0.798 on real data. Exact title equality has to catch it,
        and must not depend on a CanonicalResolver being configured.
        """
        llm.responses = [
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/feature", title="Notifications",
                kind="entity", summary="System alerts for the team",
            )]),
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/feature", title="Notifications",
                kind="entity", summary="Something else entirely different",
            )]),
        ]
        await graph.add("t1", clean_graph, "a")
        second = await graph.add("t1", clean_graph, "b")

        assert second.nodes_created == 0
        assert second.nodes_merged == 1
        async with session_factory() as s:
            count = (
                await s.execute(
                    text("SELECT count(*) FROM cg_nodes WHERE graph_id = :g "
                         "AND deleted_at IS NULL"),
                    {"g": clean_graph},
                )
            ).scalar_one()
            assert count == 1

    @pytest.mark.asyncio
    async def test_alias_recorded_by_a_merge_resolves_the_next_occurrence(
        self, graph, llm, clean_graph, session_factory
    ):
        """The roster/alias loop closing: a phrasing learned once is known."""
        llm.responses = [
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/feature", title="Notifications",
                kind="entity", summary="alerts", surface_phrase="Push Alerts",
            )]),
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/feature", title="Push Alerts",
                kind="entity", summary="totally different wording",
            )]),
        ]
        await graph.add("t1", clean_graph, "a")
        second = await graph.add("t1", clean_graph, "b")
        assert second.nodes_merged == 1, "an alias must resolve a later title"

    @pytest.mark.asyncio
    async def test_different_type_same_name_stays_two_nodes(
        self, graph, llm, clean_graph, session_factory
    ):
        llm.responses = [
            _extraction(
                ("n1", "entity/screen", "Checkout"),
                ("n2", "entity/flow", "Checkout"),
            )
        ]
        await graph.add("t1", clean_graph, "checkout stuff")
        async with session_factory() as s:
            count = (
                await s.execute(
                    text("SELECT count(*) FROM cg_nodes WHERE graph_id = :g"),
                    {"g": clean_graph},
                )
            ).scalar_one()
            assert count == 2, "a screen and a flow are not the same thing"


class TestRetrieval:
    @pytest.mark.asyncio
    async def test_recall_returns_connected_subgraph_with_ids(
        self, graph, llm, clean_graph
    ):
        llm.responses = [
            _extraction(
                ("n1", "entity/screen", "Task Dashboard"),
                ("n2", "knowledge/constraint", "Dashboard must load in 2s"),
                edges=[("n2", "n1", "affects")],
            )
        ]
        await graph.add("t1", clean_graph, "notes")
        ctx = await graph.recall("t1", clean_graph, "Task Dashboard", hops=1)

        assert ctx.nodes
        rendered = ctx.to_text()
        assert "Task Dashboard" in rendered
        assert "{id:" in rendered, "the model needs ids it can act on"

    @pytest.mark.asyncio
    async def test_search_returns_verbatim_spans(self, graph, llm, clean_graph):
        llm.responses = [_extraction(("n1", "entity/feature", "X"))]
        await graph.add("t1", clean_graph, "The quick brown fox jumps over the dog.")
        hits = await graph.search("t1", clean_graph, "quick brown fox")
        assert hits
        assert "quick brown fox" in hits[0].text

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, graph, llm, clean_graph):
        llm.responses = [_extraction(("n1", "entity/feature", "Secret Feature"))]
        await graph.add("t1", clean_graph, "notes")
        # Same tenant, a different graph, must see nothing.
        ctx = await graph.recall("t1", clean_graph + "-other", "Secret Feature")
        assert ctx.nodes == []


class TestGovernance:
    @pytest.mark.asyncio
    async def test_health_reports_the_silent_failures(
        self, graph, llm, clean_graph
    ):
        llm.responses = [_extraction(("n1", "entity/feature", "A"))]
        await graph.add("t1", clean_graph, "notes")
        health = await graph.health("t1", clean_graph)
        assert health["nodes"] == 1
        assert health["nodes_without_embedding"] == 0
        assert health["sources_pending"] == 0

    @pytest.mark.asyncio
    async def test_failed_extraction_marks_the_source_and_keeps_it(
        self, graph, llm, clean_graph, session_factory
    ):
        """The source must survive a structuring failure."""

        async def boom(**_):
            raise RuntimeError("provider exploded")

        llm.complete = boom
        result = await graph.add("t1", clean_graph, "some text")
        assert result.status == "failed"

        async with session_factory() as s:
            row = (
                await s.execute(
                    text("SELECT extraction_status, extraction_error, raw_content "
                         "FROM cg_sources WHERE graph_id = :g"),
                    {"g": clean_graph},
                )
            ).first()
            assert row is not None, "the source must not be rolled back"
            assert row[0] == "failed"
            assert "provider exploded" in row[1]
            assert row[2] == "some text"

    @pytest.mark.asyncio
    async def test_meter_is_called_even_when_the_run_fails(
        self, graph, llm, meter, clean_graph
    ):
        async def boom(**_):
            raise RuntimeError("nope")

        llm.complete = boom
        await graph.add("t1", clean_graph, "text")
        assert meter.records, "spend before a failure is still spend"


class TestStoreWithoutLLM:
    @pytest.mark.asyncio
    async def test_add_without_structuring_still_indexes_for_search(
        self, session_factory, embedder, clean_graph
    ):
        graph = ContextGraph(session_factory, embedder=embedder, llm=None)
        result = await graph.add("t1", clean_graph, "some searchable text here")
        assert result.status == "stored"
        hits = await graph.search("t1", clean_graph, "searchable text")
        assert hits, "capture must not depend on an LLM"
