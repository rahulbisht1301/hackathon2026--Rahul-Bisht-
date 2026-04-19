from typing import Any, Annotated, TypedDict

from langgraph.graph.message import add_messages


class TicketState(TypedDict, total=False):
    ticket_id: str
    ticket_email: str
    ticket_subject: str
    ticket_body: str
    ticket_source: str
    ticket_created_at: str
    ticket_tier: int
    expected_action: str

    category: str
    urgency: str
    resolvable: bool
    classification_notes: str

    messages: Annotated[list, add_messages]
    tool_calls: list[dict[str, Any]]

    resolution_action: str
    resolution_detail: str
    customer_reply: str
    escalation_summary: str
    escalation_priority: str
    escalation_reason_code: str
    planned_target_action: str
    planned_required_tools: list[str]
    planned_must_escalate: bool
    planned_rationale: str
    planned_expected_outcome: str
    planned_escalation_priority: str
    planned_kb_evidence: list[dict[str, Any]]

    confidence_score: float
    confidence_reason: str
    fraud_flag: bool
    fraud_notes: str

    llm_reasoning_summary: str
    iterations: int
    error: str
    status: str
    errors_encountered: list[str]

    processing_started_at: str
    processing_completed_at: str
    total_duration_ms: float

