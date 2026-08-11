"""The de-duplication decision logic.

These lock down the behaviours that were expensive to learn, in the direction
that matters: an over-merge destroys a distinct fact and cannot be spotted
afterwards, so every ambiguous case must resolve to "keep both".
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from contextgraph.config import ResolutionConfig
from contextgraph.ontology import DEFAULT_ONTOLOGY
from contextgraph.services.graph import OP_MERGE_NODE, OP_SUPERSEDE
from contextgraph.services.resolution import (
    Candidate,
    ExtractedNode,
    ResolutionService,
    normalize_title,
    titles_match,
)

CITATION = {"source_id": "s1", "run_id": "r1"}
EMB = [0.0] * 1536


def _svc(
    candidates, *, canonical=None, verdict="distinct", config=None, title_match=None
):
    svc = ResolutionService(
        MagicMock(),
        MagicMock(),
        llm=MagicMock(),
        ontology=DEFAULT_ONTOLOGY,
        config=config or ResolutionConfig(),
        canonical=canonical,
    )
    svc._block_candidates = AsyncMock(return_value=candidates)
    svc._classify = AsyncMock(return_value=verdict)
    # The in-graph exact-title lookup is a DB read; these tests exercise the
    # decision logic above it. Covered against real SQL in test_integration.
    svc._match_by_title = AsyncMock(return_value=title_match)
    return svc


def _node(title="Notifications", type_="entity/feature", **kw):
    return ExtractedNode(
        temp_id="n1", type=type_, title=title, kind=type_.split("/")[0], **kw
    )


class TestTitleNormalisation:
    def test_strips_noise_words(self):
        assert normalize_title("The Checkout Screen") == "checkout"
        assert normalize_title("Checkout") == "checkout"

    def test_exact_after_normalisation(self):
        assert titles_match("Checkout screen", "checkout")

    def test_partial_overlap_at_or_above_the_bar_matches(self):
        # 4 of 5 tokens shared = 0.8, exactly the threshold.
        assert titles_match(
            "billing invoice export summary",
            "billing invoice export summary monthly",
        )

    def test_one_extra_token_falls_below_the_bar(self):
        """Documented limitation, asserted so it cannot regress silently.

        3 of 4 tokens = 0.75. This IS a real duplicate that lexical matching
        misses, which is precisely why a fuzzy nomination is adjudicated rather
        than merged on directly.
        """
        assert not titles_match("Task Status Tracking", "Task Status Tracking Detail")

    def test_unrelated_do_not_match(self):
        assert not titles_match("Login", "Dashboard")

    def test_empty_never_matches(self):
        assert not titles_match("", "anything")
        assert not titles_match(None, "anything")


class TestCrossTypeMergeIsRefused:
    """The failure mode worth the most defensive code in the package."""

    @pytest.mark.asyncio
    async def test_high_similarity_cross_type_creates_a_new_node(self):
        # 0.95, far above the merge bar — but a flow is not a screen.
        svc = _svc([Candidate(id="x", score=0.95, title="Landing Page",
                              type="entity/flow")])
        ops, real_id = await svc._resolve_one(
            _node(title="Landing Page", type_="entity/screen"),
            EMB, "g", CITATION, {},
        )
        assert [o.op for o in ops] == ["ADD_NODE"]
        assert real_id != "x"

    @pytest.mark.asyncio
    async def test_duplicate_verdict_across_types_is_downgraded(self):
        svc = _svc(
            [Candidate(id="x", score=0.80, title="Landing Page Journey",
                       type="entity/flow")],
            verdict="duplicate",
        )
        ops, _ = await svc._resolve_one(
            _node(title="Landing Page", type_="entity/screen"),
            EMB, "g", CITATION, {},
        )
        assert [o.op for o in ops] == ["ADD_NODE"]

    @pytest.mark.asyncio
    async def test_cross_type_neighbour_does_not_mask_same_type_match(self):
        """Why candidate_limit is not 1."""
        svc = _svc([
            Candidate(id="flow", score=0.99, title="Login Journey",
                      type="entity/flow"),
            Candidate(id="screen", score=0.93, title="Login", type="entity/screen"),
        ])
        ops, real_id = await svc._resolve_one(
            _node(title="Login screen", type_="entity/screen"),
            EMB, "g", CITATION, {},
        )
        assert [o.op for o in ops] == [OP_MERGE_NODE]
        assert real_id == "screen"


class TestThresholds:
    @pytest.mark.asyncio
    async def test_same_type_above_threshold_merges_without_an_llm_call(self):
        svc = _svc([Candidate(id="x", score=0.93, title="Notifications",
                              type="entity/feature")])
        ops, real_id = await svc._resolve_one(_node(), EMB, "g", CITATION, {})
        assert [o.op for o in ops] == [OP_MERGE_NODE]
        assert real_id == "x"
        svc._classify.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_in_band_is_adjudicated(self):
        svc = _svc(
            [Candidate(id="x", score=0.80, title="Status Tracking",
                       type="entity/feature")],
            verdict="duplicate",
        )
        ops, real_id = await svc._resolve_one(
            _node(title="Status Tracking UI"), EMB, "g", CITATION, {}
        )
        assert [o.op for o in ops] == [OP_MERGE_NODE]
        svc._classify.assert_awaited()

    @pytest.mark.asyncio
    async def test_below_band_is_never_adjudicated(self):
        svc = _svc([Candidate(id="x", score=0.50, title="Dashboard",
                              type="entity/screen")])
        ops, _ = await svc._resolve_one(
            _node(title="Login", type_="entity/screen"), EMB, "g", CITATION, {}
        )
        assert [o.op for o in ops] == ["ADD_NODE"]
        svc._classify.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_supersede_is_gated(self):
        svc = _svc(
            [Candidate(id="x", score=0.80, title="Old", type="knowledge/decision")],
            verdict="supersedes",
        )
        ops, _ = await svc._resolve_one(
            _node(title="New", type_="knowledge/decision"), EMB, "g", CITATION, {}
        )
        assert [o.op for o in ops] == ["ADD_NODE", OP_SUPERSEDE]
        assert ops[1].risk == "high", "retiring a claim must not auto-apply"


class TestWithinRunDuplicates:
    """Ops apply after the whole run, so the DB cannot show a same-run twin."""

    @pytest.mark.asyncio
    async def test_second_candidate_folds_into_the_first(self):
        svc = _svc([])
        memo: dict = {}
        ops1, id1 = await svc._resolve_one(
            _node(title="Bulk export"), EMB, "g", CITATION, memo
        )
        ops2, id2 = await svc._resolve_one(
            _node(title="bulk  export"), EMB, "g", CITATION, memo
        )
        assert [o.op for o in ops1] == ["ADD_NODE"]
        assert [o.op for o in ops2] == [OP_MERGE_NODE]
        assert id2 == id1

    @pytest.mark.asyncio
    async def test_same_title_different_type_is_not_folded(self):
        svc = _svc([])
        memo: dict = {}
        _, id1 = await svc._resolve_one(
            _node(title="Settings", type_="entity/screen"), EMB, "g", CITATION, memo
        )
        ops2, id2 = await svc._resolve_one(
            _node(title="Settings", type_="entity/feature"), EMB, "g", CITATION, memo
        )
        assert [o.op for o in ops2] == ["ADD_NODE"]
        assert id2 != id1


class TestCanonicalResolver:
    @pytest.mark.asyncio
    async def test_exact_match_merges_without_adjudication(self):
        canonical = MagicMock()
        canonical.resolve = AsyncMock(
            return_value={"node_id": "g1", "title": "Checkout Screen"}
        )
        svc = _svc([], canonical=canonical)
        ops, real_id = await svc._resolve_one(
            _node(title="checkout screen", type_="entity/screen"),
            EMB, "g", CITATION, {},
        )
        assert [o.op for o in ops] == [OP_MERGE_NODE]
        assert real_id == "g1"
        assert ops[0].payload["resolution"]["branch"] == "exact_title"

    @pytest.mark.asyncio
    async def test_fuzzy_match_must_be_adjudicated_and_keeps_its_own_title(self):
        canonical = MagicMock()
        canonical.resolve = AsyncMock(
            return_value={"node_id": "g1", "title": "Status Tracking"}
        )
        svc = _svc([], canonical=canonical, verdict="distinct")
        ops, _ = await svc._resolve_one(
            _node(title="Status Tracking Detail", type_="entity/screen"),
            EMB, "g", CITATION, {},
        )
        assert [o.op for o in ops] == ["ADD_NODE"]
        assert ops[0].payload["title"] == "Status Tracking Detail"

    @pytest.mark.asyncio
    async def test_unknown_entity_is_created_under_the_canonical_name(self):
        """And must return a real id, or every edge referencing it is dropped."""
        canonical = MagicMock()
        canonical.resolve = AsyncMock(return_value={"title": "Checkout Screen"})
        svc = _svc([], canonical=canonical)
        ops, real_id = await svc._resolve_one(
            _node(title="checkout page", type_="entity/screen"),
            EMB, "g", CITATION, {},
        )
        assert [o.op for o in ops] == ["ADD_NODE"]
        assert real_id is not None
        assert ops[0].payload["title"] == "Checkout Screen"
        assert "checkout page" in ops[0].payload["aliases"]


class TestMergeIsNonLossy:
    def test_merge_carries_summary_surface_phrase_and_trail(self):
        node = _node(summary="System alerts", surface_phrase="Alerts")
        op = ResolutionService._merge_op(
            "x", node, CITATION, branch="vector_threshold", score=0.9312
        )
        assert op.payload["summary"] == "System alerts"
        assert "Alerts" in op.payload["aliases"]
        assert op.payload["resolution"]["branch"] == "vector_threshold"
        assert op.payload["resolution"]["score"] == 0.9312

    def test_add_never_aliases_its_own_title(self):
        op = ResolutionService._add_op(
            "n", _node(surface_phrase="Notifications"), EMB, CITATION
        )
        assert op.payload["aliases"] == []
