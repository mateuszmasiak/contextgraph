"""The write gate.

The gate is the claim that distinguishes this from a memory system that writes
unilaterally at ingest, so the property worth testing is not "supersede gets
gated" — it is that the gate reaches its verdict *without consulting the
caller*. An op arrives as a proposal. If the risk field travelling on it were
honoured, anything that can build an op could label a supersede "additive" and
retire a claim nobody reviewed, and every other test here would still pass.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from contextgraph.services.graph import (
    OP_ADD_EDGE,
    OP_REMOVE_NODE,
    OP_SET_LAYOUT,
    OP_SUPERSEDE,
    OP_UPDATE_NODE,
    RISK_ADDITIVE,
    RISK_HIGH,
    Actor,
    GraphService,
    ResolvedOp,
    classify,
)

AGENT = Actor(kind="agent")
HUMAN = Actor(kind="human", id="reviewer-1")


def _service() -> GraphService:
    """A GraphService whose dispatcher is recorded rather than executed."""
    svc = GraphService(MagicMock(), MagicMock())
    svc._dispatch = AsyncMock()
    return svc


class TestClassification:
    def test_truth_ops_gate_for_agents(self):
        for op in (OP_SUPERSEDE, OP_UPDATE_NODE, OP_REMOVE_NODE):
            assert classify(op, AGENT) == RISK_HIGH

    def test_truth_ops_apply_with_audit_for_humans(self):
        for op in (OP_SUPERSEDE, OP_UPDATE_NODE, OP_REMOVE_NODE):
            assert classify(op, HUMAN) == RISK_ADDITIVE

    def test_additive_ops_never_gate(self):
        assert classify(OP_ADD_EDGE, AGENT) == RISK_ADDITIVE

    def test_presentation_ops_never_gate(self):
        assert classify(OP_SET_LAYOUT, AGENT) == RISK_ADDITIVE

    def test_an_unrecognised_op_is_gated_not_waved_through(self):
        """Fail closed. An op nobody classified is one nobody reasoned about."""
        assert classify("DROP_EVERYTHING", AGENT) == RISK_HIGH
        assert classify("DROP_EVERYTHING", HUMAN) == RISK_HIGH


class TestTheGateIgnoresTheCaller:
    @pytest.mark.asyncio
    async def test_a_supersede_labelled_additive_is_still_gated(self):
        """The privilege-escalation case.

        A caller that could set risk on the op it submits would be setting its
        own permissions.
        """
        svc = _service()
        result = await svc.apply(
            [ResolvedOp(op=OP_SUPERSEDE, risk=RISK_ADDITIVE, payload={})],
            tenant_id="t", graph_id="g", actor=AGENT,
        )
        assert len(result.gated_ops) == 1
        assert result.gated_ops[0]["risk"] == RISK_HIGH
        svc._dispatch.assert_not_awaited(), "a gated op must not touch the graph"

    @pytest.mark.asyncio
    async def test_an_additive_op_labelled_high_still_applies(self):
        """The inverse. The op's field is ignored in both directions."""
        svc = _service()
        result = await svc.apply(
            [ResolvedOp(op=OP_ADD_EDGE, risk=RISK_HIGH, payload={})],
            tenant_id="t", graph_id="g", actor=AGENT,
        )
        assert result.gated_ops == []
        svc._dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_omitting_the_actor_gets_the_strict_policy(self):
        """An unattended caller must not receive the permissive default."""
        svc = _service()
        result = await svc.apply(
            [ResolvedOp(op=OP_SUPERSEDE, risk=RISK_ADDITIVE, payload={})],
            tenant_id="t", graph_id="g",
        )
        assert len(result.gated_ops) == 1

    @pytest.mark.asyncio
    async def test_a_human_supersede_applies_immediately(self):
        svc = _service()
        result = await svc.apply(
            [ResolvedOp(op=OP_SUPERSEDE, risk=RISK_HIGH, payload={})],
            tenant_id="t", graph_id="g", actor=HUMAN,
        )
        assert result.gated_ops == []
        svc._dispatch.assert_awaited_once()


class TestTheChangesetMatchesTheVerdict:
    @pytest.mark.asyncio
    async def test_classified_ops_carry_the_server_verdict_for_every_op(self):
        """What the reviewer reads must be what the gate decided.

        Derived separately, the changeset could show "awaiting review" for an
        op that already applied — the exact failure the audit trail exists to
        make impossible.
        """
        svc = _service()
        result = await svc.apply(
            [
                ResolvedOp(op=OP_ADD_EDGE, risk=RISK_HIGH, payload={}),
                ResolvedOp(op=OP_SUPERSEDE, risk=RISK_ADDITIVE, payload={}),
            ],
            tenant_id="t", graph_id="g", actor=AGENT,
        )
        assert [o["risk"] for o in result.classified_ops] == [
            RISK_ADDITIVE, RISK_HIGH
        ]
        # The gated list is drawn from the same records, not recomputed.
        assert result.gated_ops == [result.classified_ops[1]]

    @pytest.mark.asyncio
    async def test_the_embedding_is_stripped_from_the_stored_record(self):
        """A vector per op would bloat the changeset row for no benefit."""
        svc = _service()
        result = await svc.apply(
            [ResolvedOp(
                op=OP_ADD_EDGE, risk=RISK_ADDITIVE, payload={},
                embedding=[0.1] * 1536,
            )],
            tenant_id="t", graph_id="g", actor=AGENT,
        )
        assert "embedding" not in result.classified_ops[0]
