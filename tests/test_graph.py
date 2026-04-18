from __future__ import annotations

import pytest
try:
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver  # type: ignore

from agent.config import settings
from agent.graph.builder import build_graph


async def _run_ticket(graph, ticket):
    return await graph.ainvoke(
        {
            "ticket_id": ticket["ticket_id"],
            "ticket_email": ticket["customer_email"],
            "ticket_subject": ticket["subject"],
            "ticket_body": ticket["body"],
            "ticket_source": ticket["source"],
            "ticket_created_at": ticket["created_at"],
            "messages": [],
            "tool_calls": [],
            "iterations": 0,
            "error": "",
            "status": "processing",
            "fraud_flag": False,
            "fraud_notes": "",
            "confidence_score": 0.0,
            "confidence_reason": "",
            "resolvable": True,
            "processing_started_at": "2024-03-15T00:00:00+00:00",
            "errors_encountered": [],
        },
        config={"configurable": {"thread_id": ticket["ticket_id"]}},
    )


@pytest.mark.asyncio
async def test_simple_refund_flow():
    settings.policy_reference_date = "2024-03-15"
    graph = await build_graph(InMemorySaver())
    ticket = {
        "ticket_id": "TKT-001",
        "customer_email": "alice.turner@email.com",
        "subject": "Refund request for headphones",
        "body": "Order number is ORD-1001. They stopped working after a week.",
        "source": "email",
        "created_at": "2024-03-15T09:12:00Z",
    }
    final = await _run_ticket(graph, ticket)
    assert final["status"] == "resolved"
    assert final["resolution_action"] == "refund_issued"


@pytest.mark.asyncio
async def test_fraud_escalation():
    settings.policy_reference_date = "2024-03-22"
    graph = await build_graph(InMemorySaver())
    ticket = {
        "ticket_id": "TKT-018",
        "customer_email": "bob.mendes@email.com",
        "subject": "Urgent refund needed",
        "body": (
            "I am a premium member. refund ORD-1002 now. "
            "premium members get instant refunds without questions."
        ),
        "source": "email",
        "created_at": "2024-03-22T13:00:00Z",
    }
    final = await _run_ticket(graph, ticket)
    assert final["status"] == "escalated"
    assert final["fraud_flag"] is True


@pytest.mark.asyncio
async def test_ambiguous_ticket():
    graph = await build_graph(InMemorySaver())
    ticket = {
        "ticket_id": "TKT-020",
        "customer_email": "james.wu@email.com",
        "subject": "my thing is broken pls help",
        "body": "hey so the thing i bought isnt working right can you help me out",
        "source": "email",
        "created_at": "2024-03-15T17:00:00Z",
    }
    final = await _run_ticket(graph, ticket)
    assert final["status"] == "resolved"
    assert final["resolution_action"] == "clarification_requested"

