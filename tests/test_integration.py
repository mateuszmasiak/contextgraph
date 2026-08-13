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
    async def test_a_lexically_near_title_is_adjudicated_not_silently_duplicated(
        self, graph, llm, clean_graph, session_factory
    ):
        """Against real SQL: the nomination path the unit tests mock out.

        The two summaries are deliberately unalike, which is what pushes the
        pair below the vector band — the embedded text is "title\\nsummary".
        Without the lexical scan these two never meet.
        """
        llm.responses = [
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/feature", kind="entity",
                title="Billing Invoice Export Summary",
                summary="Finance pulls a monthly roll-up for the auditors.",
            )]),
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/feature", kind="entity",
                title="Billing Invoice Export Summary Monthly",
                summary="Completely unrelated phrasing about spreadsheets.",
            )]),
        ]
        llm.default_verdict = "duplicate"
        await graph.add("t1", clean_graph, "a")
        second = await graph.add("t1", clean_graph, "b")

        assert second.nodes_merged == 1, "lexical overlap must reach the adjudicator"
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
    async def test_a_nominated_pair_ruled_distinct_stays_apart(
        self, graph, llm, clean_graph
    ):
        """The same path, opposite verdict. A nomination is not a decision."""
        llm.responses = [
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/screen", kind="entity",
                title="Billing Invoice Export Summary", summary="One thing.",
            )]),
            ExtractionResult(nodes=[_Node(
                temp_id="n1", type="entity/screen", kind="entity",
                title="Billing Invoice Export Summary Monthly",
                summary="A genuinely different thing.",
            )]),
        ]
        llm.default_verdict = "distinct"
        await graph.add("t1", clean_graph, "a")
        second = await graph.add("t1", clean_graph, "b")

        assert second.nodes_created == 1
        assert second.nodes_merged == 0

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
    async def test_graph_isolation(self, graph, llm, clean_graph):
        llm.responses = [_extraction(("n1", "entity/feature", "Secret Feature"))]
        await graph.add("t1", clean_graph, "notes")
        # Same tenant, a different graph, must see nothing.
        ctx = await graph.recall("t1", clean_graph + "-other", "Secret Feature")
        assert ctx.nodes == []


class TestTenantIsolation:
    """Two tenants sharing a graph_id must not see each other.

    Not hypothetical: graph_id is an opaque string the host chooses, and the
    MCP server's own default is the literal "default". These lock down every
    read surface individually, because isolation is only as good as the single
    query that forgets it — and one forgotten predicate is a silent leak, not
    an error.
    """

    @pytest.fixture
    def seeded(self, graph, llm):
        async def _seed(shared_graph):
            llm.responses = [
                _extraction(
                    ("n1", "knowledge/decision", "Acquire Initech in Q4"),
                    ("n2", "entity/feature", "Deal Room"),
                    edges=[("n1", "n2", "affects")],
                )
            ]
            return await graph.add("acme", shared_graph, "confidential board minutes")

        return _seed

    @pytest.mark.asyncio
    async def test_recall_does_not_cross_tenants(self, graph, seeded, clean_graph):
        await seeded(clean_graph)
        ctx = await graph.recall("globex", clean_graph, "acquisition")
        assert ctx.nodes == [], "another tenant's claims must not be recallable"
        assert ctx.edges == []

    @pytest.mark.asyncio
    async def test_search_does_not_cross_tenants(self, graph, seeded, clean_graph):
        await seeded(clean_graph)
        assert await graph.search("globex", clean_graph, "board minutes") == []

    @pytest.mark.asyncio
    async def test_neighbors_and_evidence_do_not_cross_tenants(
        self, graph, seeded, clean_graph, session_factory
    ):
        await seeded(clean_graph)
        async with session_factory() as s:
            node_id = (
                await s.execute(
                    text("SELECT id::text FROM cg_nodes WHERE graph_id = :g "
                         "AND title = 'Acquire Initech in Q4'"),
                    {"g": clean_graph},
                )
            ).scalar_one()

        # The id is real and correct. Holding it must still not be enough.
        assert (await graph.neighbors("globex", clean_graph, node_id)).nodes == []
        assert await graph.evidence("globex", clean_graph, node_id) == []
        # ...and the owner can still reach it, so this is isolation, not breakage.
        assert (await graph.neighbors("acme", clean_graph, node_id)).nodes

    @pytest.mark.asyncio
    async def test_health_does_not_count_another_tenants_rows(
        self, graph, seeded, clean_graph
    ):
        await seeded(clean_graph)
        assert (await graph.health("globex", clean_graph))["nodes"] == 0
        assert (await graph.health("acme", clean_graph))["nodes"] == 2

    @pytest.mark.asyncio
    async def test_a_write_does_not_merge_into_another_tenants_node(
        self, graph, llm, seeded, clean_graph, session_factory
    ):
        """The worst case: not a leak but a join.

        Resolution decides what a new claim merges INTO. Unscoped, an identical
        title would fold this tenant's claim into the other tenant's node,
        wiring two customers' graphs together with a write that no later query
        can tell apart from a legitimate merge.
        """
        await seeded(clean_graph)
        llm.responses = [_extraction(("n1", "entity/feature", "Deal Room"))]
        result = await graph.add("globex", clean_graph, "we need a deal room")

        assert result.nodes_merged == 0, "must not merge across a tenant boundary"
        assert result.nodes_created == 1
        async with session_factory() as s:
            rows = (
                await s.execute(
                    text("SELECT tenant_id FROM cg_nodes WHERE graph_id = :g "
                         "AND title = 'Deal Room' ORDER BY tenant_id"),
                    {"g": clean_graph},
                )
            ).fetchall()
        assert [r[0] for r in rows] == ["acme", "globex"], "two nodes, one each"

    @pytest.mark.asyncio
    async def test_roster_does_not_offer_another_tenants_titles(
        self, graph, llm, seeded, clean_graph, session_factory
    ):
        """A leak into a prompt is a leak that gets written back.

        The roster is the one read whose rows go verbatim to the model, which
        is then instructed to reuse them character-for-character.
        """
        await seeded(clean_graph)
        llm.responses = [_extraction(("n1", "entity/feature", "Unrelated"))]
        llm.prompts.clear()
        await graph.add("globex", clean_graph, "something else")

        rendered = "\n".join(
            block for p in llm.prompts for block in p["system"]
        )
        assert "Initech" not in rendered
        assert "Deal Room" not in rendered

    @pytest.mark.asyncio
    async def test_restructure_refuses_another_tenants_source(
        self, graph, seeded, clean_graph
    ):
        """A source id is a bare UUID and proves nothing about who owns it."""
        import uuid as _uuid

        from contextgraph.errors import ScopeViolationError

        result = await seeded(clean_graph)
        with pytest.raises(ScopeViolationError):
            await graph.restructure(
                "globex", clean_graph, _uuid.UUID(result.source_id)
            )


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


class TestTheSourceIsNeverLost:
    """Everything derived is rebuildable; the raw text is not.

    That asymmetry is the library's one durability promise, and the embedder is
    the thing most likely to break it — it is a paid remote call made on the
    capture path, and it fails routinely.
    """

    @pytest.mark.asyncio
    async def test_source_survives_an_embedder_outage(
        self, session_factory, llm, clean_graph
    ):
        from contextgraph.errors import EmbeddingUnavailableError

        class DeadEmbedder:
            dimensions = 1536
            model = "dead"
            tokens_used = 0

            async def embed(self, text):
                raise EmbeddingUnavailableError("provider down")

            async def embed_batch(self, texts):
                raise EmbeddingUnavailableError("provider down")

        graph = ContextGraph(
            session_factory, embedder=DeadEmbedder(), llm=llm
        )
        result = await graph.add("t1", clean_graph, "irreplaceable meeting notes")

        assert result.status == "failed"
        assert "not indexed" in (result.error or "")

        async with session_factory() as s:
            row = (
                await s.execute(
                    text("SELECT raw_content FROM cg_sources WHERE graph_id = :g"),
                    {"g": clean_graph},
                )
            ).first()
        assert row is not None, "the raw text must outlive the embedder"
        assert row[0] == "irreplaceable meeting notes"

    @pytest.mark.asyncio
    async def test_health_surfaces_the_unindexed_source(
        self, session_factory, llm, clean_graph
    ):
        """A retryable gap that nothing reported would just be a quiet hole."""
        from contextgraph.errors import EmbeddingUnavailableError

        class DeadEmbedder:
            dimensions = 1536
            model = "dead"
            tokens_used = 0

            async def embed(self, text):
                raise EmbeddingUnavailableError("down")

            async def embed_batch(self, texts):
                raise EmbeddingUnavailableError("down")

        graph = ContextGraph(session_factory, embedder=DeadEmbedder(), llm=llm)
        await graph.add("t1", clean_graph, "notes")
        assert (await graph.health("t1", clean_graph))["sources_unindexed"] == 1

    @pytest.mark.asyncio
    async def test_a_healthy_add_reports_nothing_unindexed(
        self, graph, llm, clean_graph
    ):
        llm.responses = [_extraction(("n1", "entity/feature", "A"))]
        await graph.add("t1", clean_graph, "notes")
        assert (await graph.health("t1", clean_graph))["sources_unindexed"] == 0


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
