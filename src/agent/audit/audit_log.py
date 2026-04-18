from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.config import settings
from agent.tools.failures import simulator


@dataclass
class ToolCallRecord:
    tool_name: str
    input_params: dict[str, Any]
    output: dict[str, Any] | None
    success: bool
    error: str | None
    duration_ms: float
    attempt_number: int
    timestamp: str


@dataclass
class AuditEntry:
    ticket_id: str
    customer_email: str
    subject: str
    category: str
    urgency: str
    resolvable: bool
    fraud_flag: bool
    fraud_notes: str
    tool_calls: list[dict[str, Any]]
    llm_reasoning_summary: str
    iterations: int
    status: str
    resolution_action: str
    resolution_detail: str
    customer_reply: str
    escalation_summary: str
    escalation_priority: str
    confidence_score: float
    confidence_reason: str
    processing_started_at: str
    processing_completed_at: str
    total_duration_ms: float
    errors_encountered: list[str]


def state_to_audit_entry(state: dict[str, Any]) -> AuditEntry:
    return AuditEntry(
        ticket_id=state.get("ticket_id", ""),
        customer_email=state.get("ticket_email", ""),
        subject=state.get("ticket_subject", ""),
        category=state.get("category", ""),
        urgency=state.get("urgency", ""),
        resolvable=bool(state.get("resolvable", False)),
        fraud_flag=bool(state.get("fraud_flag", False)),
        fraud_notes=state.get("fraud_notes", ""),
        tool_calls=state.get("tool_calls", []),
        llm_reasoning_summary=state.get("llm_reasoning_summary", ""),
        iterations=int(state.get("iterations", 0)),
        status=state.get("status", "failed"),
        resolution_action=state.get("resolution_action", ""),
        resolution_detail=state.get("resolution_detail", ""),
        customer_reply=state.get("customer_reply", ""),
        escalation_summary=state.get("escalation_summary", ""),
        escalation_priority=state.get("escalation_priority", ""),
        confidence_score=float(state.get("confidence_score", 0.0)),
        confidence_reason=state.get("confidence_reason", ""),
        processing_started_at=state.get("processing_started_at", ""),
        processing_completed_at=state.get("processing_completed_at", ""),
        total_duration_ms=float(state.get("total_duration_ms", 0.0)),
        errors_encountered=state.get("errors_encountered", []),
    )


class AuditWriter:
    def __init__(self, path: str):
        self.path = Path(path)

    def write(
        self,
        results: list[dict[str, Any]],
        run_started: datetime,
        run_completed: datetime,
    ) -> None:
        entries = [state_to_audit_entry(r) for r in results]
        resolved = sum(1 for r in entries if r.status == "resolved")
        escalated = sum(1 for r in entries if r.status == "escalated")
        failed = sum(1 for r in entries if r.status == "failed")
        tool_calls = sum(len(r.tool_calls) for r in entries)
        average_confidence = (
            sum(r.confidence_score for r in entries) / len(entries) if entries else 0.0
        )
        stats = simulator.get_stats()

        payload = {
            "run_metadata": {
                "run_id": str(uuid.uuid4()),
                "started_at": run_started.astimezone(timezone.utc).isoformat(),
                "completed_at": run_completed.astimezone(timezone.utc).isoformat(),
                "total_tickets": len(results),
                "resolved_autonomously": resolved,
                "escalated": escalated,
                "failed": failed,
                "total_tool_calls": tool_calls,
                "tool_failures_injected": stats["injected_failures"],
                "tool_failures_recovered": stats["recovered_failures"],
                "average_confidence_score": round(average_confidence, 4),
                "model_used": settings.llm_model,
            },
            "tickets": [asdict(entry) for entry in entries],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

