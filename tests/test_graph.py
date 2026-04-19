from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
try:
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver  # type: ignore

import agent.graph.nodes as graph_nodes
from agent.graph.builder import build_graph
from agent.graph.edges import route_resolution


def _fake_classification(ticket_id: str) -> dict[str, Any]:
    if ticket_id == "TKT-018":
        return {
            "category": "fraud_suspected",
            "urgency": "urgent",
            "resolvable": False,
            "confidence_score": 0.2,
            "confidence_reason": "Tier mismatch and fabricated policy claim.",
            "fraud_flag": True,
            "fraud_notes": "Customer claims premium privileges while account is standard.",
            "classification_notes": "Escalation expected.",
            "status": "processing",
        }
    if ticket_id == "TKT-020":
        return {
            "category": "ambiguous",
            "urgency": "medium",
            "resolvable": False,
            "confidence_score": 0.35,
            "confidence_reason": "Missing order and product context.",
            "fraud_flag": False,
            "fraud_notes": "",
            "classification_notes": "Clarification required.",
            "status": "processing",
        }
    if ticket_id == "TKT-003":
        return {
            "category": "warranty",
            "urgency": "high",
            "resolvable": False,
            "confidence_score": 0.85,
            "confidence_reason": "Warranty claim requires specialist handling.",
            "fraud_flag": False,
            "fraud_notes": "",
            "classification_notes": "Warranty route expected.",
            "status": "processing",
        }
    if ticket_id == "TKT-099":
        return {
            "category": "shipping",
            "urgency": "medium",
            "resolvable": True,
            "confidence_score": 0.7,
            "confidence_reason": "Requires manual review despite complete context.",
            "fraud_flag": False,
            "fraud_notes": "",
            "classification_notes": "Planner should force escalation.",
            "status": "processing",
        }
    return {
        "category": "refund",
        "urgency": "high",
        "resolvable": True,
        "confidence_score": 0.9,
        "confidence_reason": "Clear defect + order context present.",
        "fraud_flag": False,
        "fraud_notes": "",
        "classification_notes": "Likely autonomous resolution.",
        "status": "processing",
    }


def _make_scripted_agent(message_sequence: list[Any]) -> MagicMock:
    mock_agent = MagicMock()

    async def _fake_ainvoke(inputs: dict[str, Any], config: dict[str, Any] | None = None):
        messages = list(inputs.get("messages", []))
        messages.extend(message_sequence)
        return {"messages": messages}

    mock_agent.ainvoke = AsyncMock(side_effect=_fake_ainvoke)
    return mock_agent


def _tkt001_messages() -> list[Any]:
    return [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_customer", "args": {"email": "alice.turner@email.com"}}]),
        ToolMessage(content='{"success": true, "tier": "vip"}', tool_call_id="c1", name="get_customer"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "get_order", "args": {"order_id": "ORD-1001"}}]),
        ToolMessage(content='{"success": true, "found": true, "order_id": "ORD-1001"}', tool_call_id="c2", name="get_order"),
        AIMessage(content="", tool_calls=[{"id": "c3", "name": "check_refund_eligibility", "args": {"order_id": "ORD-1001"}}]),
        ToolMessage(content='{"success": true, "eligible": true, "amount": 129.99}', tool_call_id="c3", name="check_refund_eligibility"),
        AIMessage(content="", tool_calls=[{"id": "c4", "name": "issue_refund", "args": {"order_id": "ORD-1001", "amount": 129.99}}]),
        ToolMessage(content='{"success": true, "refund_id": "RFD-001", "amount": 129.99}', tool_call_id="c4", name="issue_refund"),
        AIMessage(content="", tool_calls=[{"id": "c5", "name": "send_reply", "args": {"ticket_id": "TKT-001", "message": "Hi Alice, your refund has been issued."}}]),
        ToolMessage(content='{"success": true}', tool_call_id="c5", name="send_reply"),
        AIMessage(content="Refund issued successfully for ORD-1001."),
    ]


def _tkt018_messages() -> list[Any]:
    return [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_customer", "args": {"email": "bob.mendes@email.com"}}]),
        ToolMessage(content='{"success": true, "tier": "standard"}', tool_call_id="c1", name="get_customer"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "search_knowledge_base", "args": {"query": "premium instant refund policy"}}]),
        ToolMessage(content='{"success": true, "results": [{"section": "Refund Policy", "text": "No instant refund policy exists."}]}', tool_call_id="c2", name="search_knowledge_base"),
        AIMessage(content="", tool_calls=[{"id": "c3", "name": "send_reply", "args": {"ticket_id": "TKT-018", "message": "Hi Bob, we are reviewing your request."}}]),
        ToolMessage(content='{"success": true}', tool_call_id="c3", name="send_reply"),
        AIMessage(content="", tool_calls=[{"id": "c4", "name": "escalate", "args": {"ticket_id": "TKT-018", "summary": "Tier mismatch + fabricated policy claim.", "priority": "urgent"}}]),
        ToolMessage(content='{"success": true, "priority": "urgent"}', tool_call_id="c4", name="escalate"),
        AIMessage(content="Escalated due to social-engineering indicators."),
    ]


def _tkt020_messages() -> list[Any]:
    return [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_customer", "args": {"email": "james.wu@email.com"}}]),
        ToolMessage(content='{"success": true, "tier": "standard"}', tool_call_id="c1", name="get_customer"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "search_knowledge_base", "args": {"query": "clarification requirements for broken item report"}}]),
        ToolMessage(content='{"success": true, "results": [{"section": "FAQs", "text": "Ask for order ID and issue details."}]}', tool_call_id="c2", name="search_knowledge_base"),
        AIMessage(content="", tool_calls=[{"id": "c3", "name": "send_reply", "args": {"ticket_id": "TKT-020", "message": "Hi James, could you please share your order number and a short description of what is broken?"}}]),
        ToolMessage(content='{"success": true}', tool_call_id="c3", name="send_reply"),
        AIMessage(content="Hi James, could you please share your order number and a short description of what is broken?"),
    ]


def _single_tool_call_messages() -> list[Any]:
    return [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_order", "args": {"order_id": "ORD-1006"}}]),
        ToolMessage(content='{"success": true, "found": true, "order_id": "ORD-1006"}', tool_call_id="c1", name="get_order"),
        AIMessage(content="Done."),
    ]


def _draft_reply_without_send_tool_messages() -> list[Any]:
    return [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_customer", "args": {"email": "james.wu@email.com"}}]),
        ToolMessage(content='{"success": true, "tier": "standard"}', tool_call_id="c1", name="get_customer"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "search_knowledge_base", "args": {"query": "clarification requirements"}}]),
        ToolMessage(content='{"success": true, "results": [{"section": "FAQs", "text": "Ask for order and issue details first."}]}', tool_call_id="c2", name="search_knowledge_base"),
        AIMessage(content="", tool_calls=[{"id": "c3", "name": "get_order", "args": {"order_id": "ORD-1010"}}]),
        ToolMessage(content='{"success": true, "found": true, "order_id": "ORD-1010"}', tool_call_id="c3", name="get_order"),
        AIMessage(content="Hi James, could you please share your order ID and a short description of the issue?"),
    ]


def _refund_ineligible_with_reply_messages() -> list[Any]:
    return [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_customer", "args": {"email": "irene.castillo@email.com"}}]),
        ToolMessage(content='{"success": true, "tier": "standard"}', tool_call_id="c1", name="get_customer"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "get_order", "args": {"order_id": "ORD-1009"}}]),
        ToolMessage(content='{"success": true, "found": true, "order_id": "ORD-1009"}', tool_call_id="c2", name="get_order"),
        AIMessage(content="", tool_calls=[{"id": "c3", "name": "check_refund_eligibility", "args": {"order_id": "ORD-1009"}}]),
        ToolMessage(content='{"success": true, "eligible": false, "reasons": ["already_refunded"], "reason": "already_refunded"}', tool_call_id="c3", name="check_refund_eligibility"),
        AIMessage(content="", tool_calls=[{"id": "c4", "name": "send_reply", "args": {"ticket_id": "TKT-009", "message": "Your refund was already processed and should settle in 5-7 business days."}}]),
        ToolMessage(content='{"success": true}', tool_call_id="c4", name="send_reply"),
        AIMessage(content="Your refund is already processed and should settle in 5-7 business days."),
    ]


def _tkt003_warranty_messages() -> list[Any]:
    return [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_customer", "args": {"email": "carol.nguyen@email.com"}}]),
        ToolMessage(content='{"success": true, "tier": "premium"}', tool_call_id="c1", name="get_customer"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "search_knowledge_base", "args": {"query": "warranty defective items handling"}}]),
        ToolMessage(content='{"success": true, "results": [{"section": "Warranty Policy", "text": "Warranty claims require specialist review."}]}', tool_call_id="c2", name="search_knowledge_base"),
        AIMessage(content="", tool_calls=[{"id": "c3", "name": "send_reply", "args": {"ticket_id": "TKT-003", "message": "Your warranty case is being reviewed."}}]),
        ToolMessage(content='{"success": true}', tool_call_id="c3", name="send_reply"),
        AIMessage(content="Your warranty case has been escalated to our specialist team."),
    ]


def _tkt099_planned_escalation_messages() -> list[Any]:
    return [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_customer", "args": {"email": "liam.ross@email.com"}}]),
        ToolMessage(content='{"success": true, "tier": "standard"}', tool_call_id="c1", name="get_customer"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "search_knowledge_base", "args": {"query": "manual review policy for shipping dispute"}}]),
        ToolMessage(content='{"success": true, "results": [{"section": "Shipping Policy", "text": "Escalate disputed shipments for manual review."}]}', tool_call_id="c2", name="search_knowledge_base"),
        AIMessage(content="", tool_calls=[{"id": "c3", "name": "send_reply", "args": {"ticket_id": "TKT-099", "message": "We are escalating your case for specialist review."}}]),
        ToolMessage(content='{"success": true}', tool_call_id="c3", name="send_reply"),
        AIMessage(content="Escalating for manual review."),
    ]


async def _fake_classify_ticket(state: dict[str, Any]) -> dict[str, Any]:
    payload = _fake_classification(state["ticket_id"])
    payload.update(
        {
            "processing_started_at": state.get("processing_started_at", "2024-03-15T00:00:00+00:00"),
            "errors_encountered": state.get("errors_encountered", []),
            "tool_calls": state.get("tool_calls", []),
        }
    )
    return payload


async def _fake_plan_ticket(state: dict[str, Any]) -> dict[str, Any]:
    ticket_id = state["ticket_id"]
    if ticket_id == "TKT-018":
        return {
            "planned_target_action": "escalated",
            "planned_required_tools": ["get_customer", "search_knowledge_base", "escalate", "send_reply"],
            "planned_must_escalate": True,
            "planned_rationale": "Fraud scenario should escalate.",
            "planned_expected_outcome": "escalate urgently",
            "planned_escalation_priority": "urgent",
            "planned_kb_evidence": [],
            "tool_calls": state.get("tool_calls", []),
            "errors_encountered": state.get("errors_encountered", []),
            "status": "processing",
        }
    if ticket_id == "TKT-003":
        return {
            "planned_target_action": "escalated",
            "planned_required_tools": ["get_customer", "search_knowledge_base", "escalate", "send_reply"],
            "planned_must_escalate": True,
            "planned_rationale": "Warranty scenario should escalate.",
            "planned_expected_outcome": "warranty escalation",
            "planned_escalation_priority": "high",
            "planned_kb_evidence": [],
            "tool_calls": state.get("tool_calls", []),
            "errors_encountered": state.get("errors_encountered", []),
            "status": "processing",
        }
    if ticket_id == "TKT-020":
        return {
            "planned_target_action": "clarification_requested",
            "planned_required_tools": ["get_customer", "search_knowledge_base", "send_reply"],
            "planned_must_escalate": False,
            "planned_rationale": "Missing details require clarification.",
            "planned_expected_outcome": "ask for order details",
            "planned_escalation_priority": "medium",
            "planned_kb_evidence": [],
            "tool_calls": state.get("tool_calls", []),
            "errors_encountered": state.get("errors_encountered", []),
            "status": "processing",
        }
    if ticket_id == "TKT-099":
        return {
            "planned_target_action": "escalated",
            "planned_required_tools": ["get_customer", "search_knowledge_base", "send_reply", "escalate"],
            "planned_must_escalate": True,
            "planned_rationale": "Shipping dispute requires specialist manual review.",
            "planned_expected_outcome": "manual escalation",
            "planned_escalation_priority": "high",
            "planned_kb_evidence": [],
            "tool_calls": state.get("tool_calls", []),
            "errors_encountered": state.get("errors_encountered", []),
            "status": "processing",
        }
    return {
        "planned_target_action": "refund_issued",
        "planned_required_tools": [
            "get_customer",
            "get_order",
            "check_refund_eligibility",
            "issue_refund",
            "send_reply",
        ],
        "planned_must_escalate": False,
        "planned_rationale": "Refund scenario should complete autonomously.",
        "planned_expected_outcome": "issue refund",
        "planned_escalation_priority": "medium",
        "planned_kb_evidence": [],
        "tool_calls": state.get("tool_calls", []),
        "errors_encountered": state.get("errors_encountered", []),
        "status": "processing",
    }


BASE_STATE = {
    "messages": [],
    "tool_calls": [],
    "iterations": 0,
    "error": "",
    "status": "processing",
    "fraud_flag": False,
    "fraud_notes": "",
    "confidence_score": 0.0,
    "confidence_reason": "",
    "escalation_reason_code": "",
    "expected_action": "",
    "planned_target_action": "",
    "planned_required_tools": [],
    "planned_must_escalate": False,
    "planned_rationale": "",
    "planned_expected_outcome": "",
    "planned_escalation_priority": "medium",
    "planned_kb_evidence": [],
    "resolvable": True,
    "processing_started_at": "2024-03-15T00:00:00+00:00",
    "errors_encountered": [],
}


@pytest.mark.asyncio
async def test_tkt001_refund_flow_with_mocked_llm():
    mock_agent = _make_scripted_agent(_tkt001_messages())
    with (
        patch("agent.graph.builder.classify_ticket", new=_fake_classify_ticket),
        patch("agent.graph.builder.plan_ticket", new=_fake_plan_ticket),
        patch("agent.graph.nodes.get_react_agent", return_value=mock_agent),
    ):
        graph = await build_graph(InMemorySaver())
        result = await graph.ainvoke(
            {
                **BASE_STATE,
                "ticket_id": "TKT-001",
                "ticket_email": "alice.turner@email.com",
                "ticket_subject": "Refund request for headphones",
                "ticket_body": "My headphones stopped working. Order ORD-1001.",
                "ticket_source": "email",
                "ticket_created_at": "2024-03-15T09:12:00Z",
            },
            config={"configurable": {"thread_id": "test-001"}},
        )

    assert result["status"] == "resolved"
    tool_names = [_tool_name(tc) for tc in result["tool_calls"]]
    assert "get_customer" in tool_names
    assert "check_refund_eligibility" in tool_names
    assert "issue_refund" in tool_names
    assert tool_names.index("check_refund_eligibility") < tool_names.index("issue_refund")
    assert len(tool_names) >= 3


@pytest.mark.asyncio
async def test_tkt018_fraud_escalation_with_mocked_llm():
    mock_agent = _make_scripted_agent(_tkt018_messages())
    with (
        patch("agent.graph.builder.classify_ticket", new=_fake_classify_ticket),
        patch("agent.graph.builder.plan_ticket", new=_fake_plan_ticket),
        patch("agent.graph.nodes.get_react_agent", return_value=mock_agent),
    ):
        graph = await build_graph(InMemorySaver())
        result = await graph.ainvoke(
            {
                **BASE_STATE,
                "ticket_id": "TKT-018",
                "ticket_email": "bob.mendes@email.com",
                "ticket_subject": "Urgent refund needed",
                "ticket_body": "I am a premium member and need an instant refund for ORD-1002.",
                "ticket_source": "email",
                "ticket_created_at": "2024-03-22T13:00:00Z",
            },
            config={"configurable": {"thread_id": "test-018"}},
        )

    assert result["status"] == "escalated"
    tool_names = [_tool_name(tc) for tc in result["tool_calls"]]
    assert "issue_refund" not in tool_names
    assert "escalate" in tool_names
    assert "get_customer" in tool_names
    assert result.get("escalation_reason_code") == "fraud_flag"


@pytest.mark.asyncio
async def test_tkt020_ambiguous_requests_clarification_with_mocked_llm():
    mock_agent = _make_scripted_agent(_tkt020_messages())
    with (
        patch("agent.graph.builder.classify_ticket", new=_fake_classify_ticket),
        patch("agent.graph.builder.plan_ticket", new=_fake_plan_ticket),
        patch("agent.graph.nodes.get_react_agent", return_value=mock_agent),
    ):
        graph = await build_graph(InMemorySaver())
        result = await graph.ainvoke(
            {
                **BASE_STATE,
                "ticket_id": "TKT-020",
                "ticket_email": "james.wu@email.com",
                "ticket_subject": "my thing is broken pls help",
                "ticket_body": "hey so the thing i bought isnt working right",
                "ticket_source": "email",
                "ticket_created_at": "2024-03-15T17:00:00Z",
            },
            config={"configurable": {"thread_id": "test-020"}},
        )

    assert result["status"] == "resolved"
    tool_names = [_tool_name(tc) for tc in result["tool_calls"]]
    assert "issue_refund" not in tool_names
    assert "send_reply" in tool_names
    assert len(tool_names) >= 3


@pytest.mark.asyncio
async def test_reason_and_act_enforces_minimum_tool_calls():
    mock_agent = _make_scripted_agent(_single_tool_call_messages())
    with (
        patch("agent.graph.builder.classify_ticket", new=_fake_classify_ticket),
        patch("agent.graph.builder.plan_ticket", new=_fake_plan_ticket),
        patch("agent.graph.nodes.get_react_agent", return_value=mock_agent),
    ):
        graph = await build_graph(InMemorySaver())
        result = await graph.ainvoke(
            {
                **BASE_STATE,
                "ticket_id": "TKT-006",
                "ticket_email": "frank.rivera@email.com",
                "ticket_subject": "Need help with order",
                "ticket_body": "Please check order ORD-1006.",
                "ticket_source": "email",
                "ticket_created_at": "2024-03-15T10:00:00Z",
            },
            config={"configurable": {"thread_id": "test-min-tool-calls"}},
        )
    assert len(result["tool_calls"]) >= 3


@pytest.mark.asyncio
async def test_reason_and_act_resolves_when_reply_is_drafted_but_not_sent():
    mock_agent = _make_scripted_agent(_draft_reply_without_send_tool_messages())
    with (
        patch("agent.graph.builder.classify_ticket", new=_fake_classify_ticket),
        patch("agent.graph.builder.plan_ticket", new=_fake_plan_ticket),
        patch("agent.graph.nodes.get_react_agent", return_value=mock_agent),
    ):
        graph = await build_graph(InMemorySaver())
        result = await graph.ainvoke(
            {
                **BASE_STATE,
                "ticket_id": "TKT-020",
                "ticket_email": "james.wu@email.com",
                "ticket_subject": "my thing is broken pls help",
                "ticket_body": "hey so the thing i bought isnt working right",
                "ticket_source": "email",
                "ticket_created_at": "2024-03-15T17:00:00Z",
            },
            config={"configurable": {"thread_id": "test-draft-reply-no-send-tool"}},
        )
    assert result["status"] == "resolved"
    assert result.get("resolution_action") == "clarification_requested"
    assert result.get("escalation_reason_code", "") == ""


@pytest.mark.asyncio
async def test_reason_and_act_handles_refund_ineligible_without_escalating():
    mock_agent = _make_scripted_agent(_refund_ineligible_with_reply_messages())
    with (
        patch("agent.graph.builder.classify_ticket", new=_fake_classify_ticket),
        patch("agent.graph.builder.plan_ticket", new=_fake_plan_ticket),
        patch("agent.graph.nodes.get_react_agent", return_value=mock_agent),
    ):
        graph = await build_graph(InMemorySaver())
        result = await graph.ainvoke(
            {
                **BASE_STATE,
                "ticket_id": "TKT-009",
                "ticket_email": "irene.castillo@email.com",
                "ticket_subject": "Refund already done?",
                "ticket_body": "Can you confirm if refund ORD-1009 is already processed?",
                "ticket_source": "ticket_queue",
                "ticket_created_at": "2024-03-15T11:00:00Z",
            },
            config={"configurable": {"thread_id": "test-refund-ineligible-no-escalation"}},
        )
    assert result["status"] == "resolved"
    assert result.get("resolution_action") == "info_provided"
    assert result.get("escalation_reason_code", "") == ""


@pytest.mark.asyncio
async def test_warranty_category_forces_escalation():
    mock_agent = _make_scripted_agent(_tkt003_warranty_messages())
    with (
        patch("agent.graph.builder.classify_ticket", new=_fake_classify_ticket),
        patch("agent.graph.builder.plan_ticket", new=_fake_plan_ticket),
        patch("agent.graph.nodes.get_react_agent", return_value=mock_agent),
    ):
        graph = await build_graph(InMemorySaver())
        result = await graph.ainvoke(
            {
                **BASE_STATE,
                "ticket_id": "TKT-003",
                "ticket_email": "carol.nguyen@email.com",
                "ticket_subject": "Coffee maker stopped heating",
                "ticket_body": "Need replacement under warranty.",
                "ticket_source": "email",
                "ticket_created_at": "2024-03-15T11:00:00Z",
            },
            config={"configurable": {"thread_id": "test-warranty-route"}},
        )
    assert result["status"] == "escalated"
    assert result.get("escalation_reason_code") == "warranty_or_replacement"


@pytest.mark.asyncio
async def test_planned_escalation_sets_reason_code():
    mock_agent = _make_scripted_agent(_tkt099_planned_escalation_messages())
    with (
        patch("agent.graph.builder.classify_ticket", new=_fake_classify_ticket),
        patch("agent.graph.builder.plan_ticket", new=_fake_plan_ticket),
        patch("agent.graph.nodes.get_react_agent", return_value=mock_agent),
    ):
        graph = await build_graph(InMemorySaver())
        result = await graph.ainvoke(
            {
                **BASE_STATE,
                "ticket_id": "TKT-099",
                "ticket_email": "liam.ross@email.com",
                "ticket_subject": "Shipping dispute requiring manual review",
                "ticket_body": "My package was marked delivered but never arrived.",
                "ticket_source": "email",
                "ticket_created_at": "2024-03-15T12:00:00Z",
            },
            config={"configurable": {"thread_id": "test-planned-escalation"}},
        )
    assert result["status"] == "escalated"
    assert result.get("escalation_reason_code") == "planned_escalation"


def _tool_name(call: dict[str, Any]) -> str:
    return str(call.get("tool") or call.get("tool_name") or "")


def test_route_resolution_prefers_explicit_escalation_action():
    decision = route_resolution(
        {
            "fraud_flag": False,
            "status": "processing",
            "resolution_action": "escalated",
            "confidence_score": 0.99,
        }
    )
    assert decision == "escalate"


def test_derive_plan_avoids_refund_target_for_status_confirmations():
    plan = graph_nodes._derive_plan_from_expected_action(
        "refund already processed — confirm status, advise 5-7 business days"
    )
    assert plan["target"] == "info_provided"
    assert plan["must_escalate"] is False
    assert "issue_refund" not in plan["required_tools"]


def test_derive_plan_avoids_refund_target_for_exchange_or_refund_guidance():
    plan = graph_nodes._derive_plan_from_expected_action(
        "wrong item delivered — within return window, initiate exchange or refund"
    )
    assert plan["target"] == "info_provided"
    assert "issue_refund" not in plan["required_tools"]
    assert "check_refund_eligibility" in plan["required_tools"]


@pytest.mark.asyncio
async def test_plan_ticket_uses_expected_action_when_strict(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(graph_nodes.settings, "planner_strict_expected_action", True, raising=False)

    async def _fake_call_tool_with_retries(**kwargs: Any) -> dict[str, Any]:
        kwargs["tool_calls"].append(
            {
                "tool_name": kwargs["tool_name"],
                "input_params": kwargs["params"],
                "output": {"success": True},
                "success": True,
                "timestamp": "2024-03-15T00:00:00+00:00",
            }
        )
        return {
            "success": True,
            "results": [
                {"section": "Refund Policy", "score": 0.93, "text": "Refunds require eligibility checks."}
            ],
        }

    monkeypatch.setattr(graph_nodes, "_call_tool_with_retries", _fake_call_tool_with_retries)
    result = await graph_nodes.plan_ticket(
        {
            "ticket_id": "TKT-050",
            "ticket_subject": "Refund for damaged item",
            "ticket_body": "Please issue refund for order ORD-1050.",
            "category": "refund",
            "expected_action": "Issue refund for order ORD-1050 after eligibility checks.",
            "fraud_flag": False,
            "tool_calls": [],
            "errors_encountered": [],
        }
    )

    assert result["planned_target_action"] == "refund_issued"
    assert result["planned_expected_outcome"] == "Issue refund for order ORD-1050 after eligibility checks."
    assert result["planned_must_escalate"] is False
    assert result["planned_escalation_priority"] == "medium"
    assert set(result["planned_required_tools"]) >= {
        "get_customer",
        "get_order",
        "get_product",
        "search_knowledge_base",
        "check_refund_eligibility",
        "issue_refund",
        "send_reply",
    }
    assert result["planned_kb_evidence"]


@pytest.mark.asyncio
async def test_plan_ticket_uses_category_when_strict_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(graph_nodes.settings, "planner_strict_expected_action", False, raising=False)

    async def _fake_call_tool_with_retries(**kwargs: Any) -> dict[str, Any]:
        kwargs["tool_calls"].append(
            {
                "tool_name": kwargs["tool_name"],
                "input_params": kwargs["params"],
                "output": {"success": True},
                "success": True,
                "timestamp": "2024-03-15T00:00:00+00:00",
            }
        )
        return {"success": True, "results": []}

    monkeypatch.setattr(graph_nodes, "_call_tool_with_retries", _fake_call_tool_with_retries)
    result = await graph_nodes.plan_ticket(
        {
            "ticket_id": "TKT-051",
            "ticket_subject": "Replacement request",
            "ticket_body": "Need a replacement for a broken item.",
            "category": "warranty",
            "expected_action": "Issue refund for order ORD-1051.",
            "fraud_flag": False,
            "tool_calls": [],
            "errors_encountered": [],
        }
    )

    assert result["planned_target_action"] == "escalated"
    assert result["planned_must_escalate"] is True
    assert result["planned_escalation_priority"] == "high"
    assert "escalate" in result["planned_required_tools"]
    assert result["planned_expected_outcome"] == "Issue refund for order ORD-1051."

