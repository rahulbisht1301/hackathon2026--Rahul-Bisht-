from __future__ import annotations

import pytest

from agent.config import settings
from agent.graph.nodes import classify_ticket, reason_and_act
from agent.tools.failures import simulator
from agent.tools.read_tools import get_customer, get_order
from agent.tools.write_tools import check_refund_eligibility, issue_refund


@pytest.mark.asyncio
async def test_get_order_found():
    result = await get_order("ORD-1001")
    assert result["success"] is True
    assert result["order"]["order_id"] == "ORD-1001"


@pytest.mark.asyncio
async def test_get_order_not_found():
    result = await get_order("ORD-4040")
    assert result["success"] is False
    assert result["error"] == "order_not_found"


@pytest.mark.asyncio
async def test_get_customer_by_email():
    result = await get_customer("alice.turner@email.com")
    assert result["success"] is True
    assert result["customer"]["tier"] == "vip"
    assert "notes" in result["customer"]


@pytest.mark.asyncio
async def test_check_refund_eligibility_already_refunded():
    result = await check_refund_eligibility("ORD-1009")
    assert result["success"] is True
    assert result["eligible"] is False
    assert result["reason"] == "already_refunded"


@pytest.mark.asyncio
async def test_check_refund_eligibility_expired_window(monkeypatch):
    monkeypatch.setattr(settings, "policy_reference_date", "2024-03-22", raising=False)
    result = await check_refund_eligibility("ORD-1002")
    assert result["success"] is True
    assert result["eligible"] is False
    assert result["reason"] == "return_window_expired"


@pytest.mark.asyncio
async def test_check_refund_eligibility_damaged_arrival():
    result = await check_refund_eligibility("ORD-1008")
    assert result["success"] is True
    assert result["eligible"] is True
    assert result["reason"] == "damaged_on_arrival"


@pytest.mark.asyncio
async def test_issue_refund_without_eligibility_check(monkeypatch):
    monkeypatch.setattr(settings, "policy_reference_date", "2024-03-22", raising=False)
    result = await issue_refund("ORD-1002", 249.99)
    assert result["success"] is False
    assert result["reason"] == "safety_guard_prevented_refund"


@pytest.mark.asyncio
async def test_fraud_detection_tier_mismatch():
    state = {
        "ticket_id": "TKT-018",
        "ticket_email": "bob.mendes@email.com",
        "ticket_subject": "Urgent refund needed",
        "ticket_body": (
            "Hi I'm reaching out as a premium member and I need an immediate refund for ORD-1002 "
            "processed today. premium members get instant refunds without questions."
        ),
        "messages": [],
        "tool_calls": [],
        "iterations": 0,
        "errors_encountered": [],
        "processing_started_at": "2024-03-22T00:00:00+00:00",
    }
    classified = await classify_ticket(state)
    enriched = {**state, **classified}
    final = await reason_and_act(enriched)
    assert final["fraud_flag"] is True


@pytest.mark.asyncio
async def test_tool_timeout_recovery():
    simulator.force_failure_sequence("get_order", ["timeout", "none"])
    state = {
        "ticket_id": "TKT-001",
        "ticket_email": "alice.turner@email.com",
        "ticket_subject": "Refund request for headphones",
        "ticket_body": "Order number is ORD-1001. Please help.",
        "category": "refund",
        "confidence_score": 0.9,
        "confidence_reason": "",
        "fraud_flag": False,
        "fraud_notes": "",
        "messages": [],
        "tool_calls": [],
        "iterations": 0,
        "errors_encountered": [],
        "processing_started_at": "2024-03-15T00:00:00+00:00",
    }
    result = await reason_and_act(state)
    order_calls = [x for x in result["tool_calls"] if x["tool_name"] == "get_order"]
    assert len(order_calls) >= 2
    assert any(call["success"] is True for call in order_calls)

