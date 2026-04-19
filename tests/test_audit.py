from __future__ import annotations

import json
from pathlib import Path

import pytest

REQUIRED_PER_TICKET = {
    "ticket_id",
    "customer_email",
    "subject",
    "expected_action",
    "category",
    "urgency",
    "resolvable",
    "fraud_flag",
    "fraud_notes",
    "tool_calls",
    "llm_reasoning_summary",
    "confidence_reason",
    "status",
    "resolution_action",
    "resolution_detail",
    "customer_reply",
    "escalation_summary",
    "escalation_priority",
    "escalation_reason_code",
    "planned_target_action",
    "planned_required_tools",
    "planned_must_escalate",
    "planned_rationale",
    "confidence_score",
    "processing_started_at",
    "processing_completed_at",
    "total_duration_ms",
    "iterations",
    "errors_encountered",
}
REQUIRED_PER_TOOL_CALL = {"output", "success", "timestamp"}
VALID_STATUSES = {"resolved", "escalated", "failed"}
VALID_ESCALATION_REASON_CODES = {
    "agent_requested_escalation",
    "fraud_flag",
    "low_confidence",
    "max_iterations_reached",
    "min_tool_calls_not_met",
    "no_safe_resolution_path",
    "planned_escalation",
    "planned_refund_unmet",
    "planned_tools_missing",
    "react_recursion_limit",
    "tool_failures_exhausted",
    "unsafe_refund_sequence",
    "warranty_or_replacement",
}
FRAUD_TICKETS = {"TKT-017", "TKT-018"}
NO_REFUND_TICKETS = {"TKT-017", "TKT-018", "TKT-009"}


def _resolve_audit_path() -> Path | None:
    candidates = [Path("audit_log.json"), Path("output/audit_log.json")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@pytest.fixture(scope="module")
def audit() -> dict:
    audit_path = _resolve_audit_path()
    if audit_path is None:
        pytest.skip("audit_log.json not found. Run the pipeline first.")
    return json.loads(audit_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tickets(audit: dict) -> list[dict]:
    return audit["tickets"]


def _tool_name(call: dict) -> str:
    return str(call.get("tool") or call.get("tool_name") or "")


def _tool_input(call: dict) -> dict:
    payload = call.get("input") or call.get("input_params") or {}
    return payload if isinstance(payload, dict) else {}


def test_all_20_tickets_present(tickets: list[dict]):
    assert len(tickets) == 20


def test_all_ticket_ids_unique(tickets: list[dict]):
    ids = [t["ticket_id"] for t in tickets]
    assert len(ids) == len(set(ids))


def test_every_ticket_has_required_fields(tickets: list[dict]):
    for t in tickets:
        missing = REQUIRED_PER_TICKET - set(t.keys())
        assert not missing, f"{t['ticket_id']}: missing fields {missing}"


def test_every_ticket_has_valid_status(tickets: list[dict]):
    for t in tickets:
        assert t["status"] in VALID_STATUSES


def test_every_ticket_has_positive_duration(tickets: list[dict]):
    for t in tickets:
        assert t["total_duration_ms"] > 0


def test_every_ticket_has_valid_confidence_score(tickets: list[dict]):
    for t in tickets:
        score = float(t["confidence_score"])
        assert 0.0 <= score <= 1.0


def test_every_ticket_has_at_least_one_tool_call(tickets: list[dict]):
    for t in tickets:
        assert len(t.get("tool_calls", [])) >= 1, f"{t['ticket_id']}: no tool calls"


def test_every_ticket_meets_minimum_3_tool_calls(tickets: list[dict]):
    for t in tickets:
        calls = t.get("tool_calls", [])
        assert len(calls) >= 3, f"{t['ticket_id']}: only {len(calls)} tool calls"


def test_every_tool_call_has_required_fields(tickets: list[dict]):
    for t in tickets:
        for i, tc in enumerate(t.get("tool_calls", [])):
            missing = REQUIRED_PER_TOOL_CALL - set(tc.keys())
            assert not missing, f"{t['ticket_id']} tool_call[{i}] missing {missing}"


def test_get_customer_is_first_tool_call_when_available(tickets: list[dict]):
    for t in tickets:
        calls = t.get("tool_calls", [])
        if not calls:
            continue
        assert _tool_name(calls[0]) == "get_customer", (
            f"{t['ticket_id']}: first tool was {_tool_name(calls[0])}"
        )


def test_no_refund_without_eligibility_check(tickets: list[dict]):
    for t in tickets:
        tool_names = [_tool_name(tc) for tc in t.get("tool_calls", [])]
        if "issue_refund" not in tool_names:
            continue
        assert "check_refund_eligibility" in tool_names
        assert tool_names.index("check_refund_eligibility") < tool_names.index("issue_refund")


def test_no_refund_on_protected_tickets(tickets: list[dict]):
    by_id = {t["ticket_id"]: t for t in tickets}
    for tid in NO_REFUND_TICKETS:
        if tid not in by_id:
            continue
        tool_names = [_tool_name(tc) for tc in by_id[tid].get("tool_calls", [])]
        assert "issue_refund" not in tool_names


def test_fraud_tickets_are_escalated(tickets: list[dict]):
    by_id = {t["ticket_id"]: t for t in tickets}
    for tid in FRAUD_TICKETS:
        if tid not in by_id:
            continue
        assert by_id[tid]["status"] == "escalated"
        tool_names = [_tool_name(tc) for tc in by_id[tid].get("tool_calls", [])]
        assert "escalate" in tool_names


def test_escalated_tickets_have_reason_code(tickets: list[dict]):
    for ticket in tickets:
        if ticket.get("status") != "escalated":
            continue
        reason_code = str(ticket.get("escalation_reason_code", "")).strip()
        assert reason_code, f"{ticket['ticket_id']}: missing escalation_reason_code"
        assert reason_code in VALID_ESCALATION_REASON_CODES, (
            f"{ticket['ticket_id']}: unsupported escalation_reason_code={reason_code}"
        )


def test_run_metadata_present(audit: dict):
    meta = audit.get("run_metadata", {})
    required = {
        "run_id",
        "started_at",
        "completed_at",
        "total_tickets",
        "resolved_autonomously",
        "escalated",
        "failed",
        "total_tool_calls",
        "tool_failures_injected",
        "tool_failures_recovered",
        "average_confidence_score",
        "model_used",
    }
    missing = required - set(meta.keys())
    assert not missing, f"run_metadata missing: {missing}"


def test_run_metadata_counts_match_tickets(audit: dict, tickets: list[dict]):
    meta = audit["run_metadata"]
    resolved = sum(1 for t in tickets if t["status"] == "resolved")
    escalated = sum(1 for t in tickets if t["status"] == "escalated")
    failed = sum(1 for t in tickets if t["status"] == "failed")
    assert meta["total_tickets"] == len(tickets)
    assert meta["resolved_autonomously"] == resolved
    assert meta["escalated"] == escalated
    assert meta["failed"] == failed
    assert meta["total_tool_calls"] == sum(len(t.get("tool_calls", [])) for t in tickets)
    assert 0.0 <= float(meta["average_confidence_score"]) <= 1.0


def test_tkt017_and_tkt013_behavior(tickets: list[dict]):
    by_id = {t["ticket_id"]: t for t in tickets}
    if "TKT-017" in by_id:
        calls = by_id["TKT-017"].get("tool_calls", [])
        send_reply_inputs = [_tool_input(c) for c in calls if _tool_name(c) == "send_reply"]
        assert send_reply_inputs, "TKT-017 should call send_reply with correction request"
    if "TKT-013" in by_id:
        calls = by_id["TKT-013"].get("tool_calls", [])
        elig_outputs = [c.get("output") for c in calls if _tool_name(c) == "check_refund_eligibility"]
        if elig_outputs:
            serialized = json.dumps(elig_outputs)
            assert "device_registered_online" in serialized
            assert "return_window_expired" in serialized

