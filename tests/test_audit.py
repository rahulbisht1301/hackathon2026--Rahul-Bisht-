from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.audit.audit_log import AuditWriter


def _sample_state(ticket_id: str, tool_calls: list[dict], duration: float = 10.0):
    return {
        "ticket_id": ticket_id,
        "ticket_email": "alice.turner@email.com",
        "ticket_subject": "subject",
        "category": "refund",
        "urgency": "medium",
        "resolvable": True,
        "fraud_flag": False,
        "fraud_notes": "",
        "tool_calls": tool_calls,
        "llm_reasoning_summary": "summary",
        "iterations": 1,
        "status": "resolved",
        "resolution_action": "refund_issued",
        "resolution_detail": "done",
        "customer_reply": "reply",
        "escalation_summary": "",
        "escalation_priority": "",
        "confidence_score": 0.9,
        "confidence_reason": "good",
        "processing_started_at": "2024-03-15T00:00:00+00:00",
        "processing_completed_at": "2024-03-15T00:00:01+00:00",
        "total_duration_ms": duration,
        "errors_encountered": [],
    }


def test_audit_entry_completeness(tmp_path):
    writer = AuditWriter(str(tmp_path / "audit_log.json"))
    state = _sample_state("TKT-001", [])
    writer.write([state], datetime.now(timezone.utc), datetime.now(timezone.utc))
    payload = json.loads((tmp_path / "audit_log.json").read_text(encoding="utf-8"))
    ticket = payload["tickets"][0]
    required = {
        "ticket_id",
        "customer_email",
        "subject",
        "category",
        "urgency",
        "resolvable",
        "fraud_flag",
        "tool_calls",
        "status",
        "resolution_action",
        "confidence_score",
        "total_duration_ms",
    }
    assert required.issubset(ticket.keys())


def test_audit_tool_calls_recorded(tmp_path):
    writer = AuditWriter(str(tmp_path / "audit_log.json"))
    tool_calls = [
        {
            "tool_name": "get_order",
            "input_params": {"order_id": "ORD-1001"},
            "output": {"success": True},
            "success": True,
            "error": None,
            "duration_ms": 12.0,
            "attempt_number": 1,
            "timestamp": "2024-03-15T00:00:00+00:00",
        }
    ]
    writer.write(
        [_sample_state("TKT-001", tool_calls)],
        datetime.now(timezone.utc),
        datetime.now(timezone.utc),
    )
    payload = json.loads((tmp_path / "audit_log.json").read_text(encoding="utf-8"))
    assert payload["tickets"][0]["tool_calls"][0]["tool_name"] == "get_order"


def test_audit_timing_present(tmp_path):
    writer = AuditWriter(str(tmp_path / "audit_log.json"))
    states = [
        _sample_state("TKT-001", [], duration=11.0),
        _sample_state("TKT-002", [], duration=15.0),
    ]
    writer.write(states, datetime.now(timezone.utc), datetime.now(timezone.utc))
    payload = json.loads((tmp_path / "audit_log.json").read_text(encoding="utf-8"))
    assert all(t["total_duration_ms"] > 0 for t in payload["tickets"])

