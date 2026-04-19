from __future__ import annotations

import pytest

from agent.config import settings
from agent.tools.failures import simulator
from agent.tools.lc_tools import (
    check_refund_eligibility,
    escalate,
    get_customer,
    get_order,
    get_product,
    issue_refund,
    search_knowledge_base,
    send_reply,
)


@pytest.mark.asyncio
async def test_get_order_found():
    result = await get_order.ainvoke({"order_id": "ORD-1001"})
    assert result["found"] is True
    assert result["order_id"] == "ORD-1001"
    assert result["status"] == "delivered"


@pytest.mark.asyncio
async def test_get_order_not_found():
    result = await get_order.ainvoke({"order_id": "ORD-9999"})
    assert result["found"] is False
    assert result["error"] == "order_not_found"


@pytest.mark.asyncio
async def test_get_customer_by_email():
    result = await get_customer.ainvoke({"email": "alice.turner@email.com"})
    assert result["success"] is True
    assert result["tier"] == "vip"


@pytest.mark.asyncio
async def test_get_product_ergolift_60_days():
    result = await get_product.ainvoke({"product_id": "P004"})
    assert result["success"] is True
    assert result["return_window_days"] == 60


@pytest.mark.asyncio
async def test_search_knowledge_base_returns_results():
    result = await search_knowledge_base.ainvoke({"query": "refund processing time"})
    assert result["success"] is True
    assert len(result["results"]) >= 1


@pytest.mark.asyncio
async def test_check_refund_eligibility_already_refunded():
    result = await check_refund_eligibility.ainvoke({"order_id": "ORD-1009"})
    assert result["success"] is True
    assert result["eligible"] is False
    assert "already_refunded" in result["reasons"]


@pytest.mark.asyncio
async def test_check_refund_eligibility_dual_reason_tkt013():
    result = await check_refund_eligibility.ainvoke({"order_id": "ORD-1013"})
    assert result["success"] is True
    assert result["eligible"] is False
    assert "device_registered_online" in result["reasons"]
    assert "return_window_expired" in result["reasons"]


@pytest.mark.asyncio
async def test_check_refund_eligibility_60_day_window_tkt007():
    result = await check_refund_eligibility.ainvoke({"order_id": "ORD-1007"})
    assert result["success"] is True
    assert result["eligible"] is True


@pytest.mark.asyncio
async def test_issue_refund_safety_guard():
    result = await issue_refund.ainvoke({"order_id": "ORD-1009", "amount": 129.99})
    assert result["success"] is False
    assert "safety_guard" in result["reason"] or "already_refunded" in result["reason"]


@pytest.mark.asyncio
async def test_send_reply_success():
    result = await send_reply.ainvoke(
        {"ticket_id": "TKT-001", "message": "Hi Alice, your request is processed."}
    )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_send_reply_rejects_empty_message():
    result = await send_reply.ainvoke({"ticket_id": "TKT-001", "message": "   "})
    assert result["success"] is False
    assert result["error"] == "empty_message"


@pytest.mark.asyncio
async def test_escalate_priority_validation():
    result = await escalate.ainvoke(
        {"ticket_id": "TKT-017", "summary": "Order not found + legal pressure", "priority": "urgent"}
    )
    assert result["success"] is True
    assert result["priority"] == "urgent"


@pytest.mark.asyncio
async def test_escalate_invalid_priority():
    result = await escalate.ainvoke(
        {"ticket_id": "TKT-001", "summary": "test", "priority": "critical"}
    )
    assert result["success"] is False
    assert result["error"] == "invalid_priority"


@pytest.mark.asyncio
async def test_failure_injection_timeout_path():
    simulator.force_failure_sequence("get_order", ["timeout"] * max(1, settings.tool_max_retries))
    result = await get_order.ainvoke({"order_id": "ORD-1001"})
    assert result["success"] is False
    assert "timed out" in result["error"].lower()


@pytest.mark.asyncio
async def test_failure_injection_malformed_path():
    simulator.force_failure_sequence("get_product", ["malformed"])
    result = await get_product.ainvoke({"product_id": "P006"})
    assert result["success"] is True
    assert result.get("warning") == "malformed_response"


@pytest.mark.asyncio
async def test_retry_recovers_transient_failure():
    simulator.force_failure_sequence("get_order", ["timeout", "none"])
    result = await get_order.ainvoke({"order_id": "ORD-1001"})
    assert result["success"] is True
    assert result["found"] is True

