from typing import Any, Annotated, TypedDict

from langgraph.graph.message import add_messages


class TicketState(TypedDict, total=False):
    ticket_id: str
    ticket_email: str
    ticket_subject: str
    ticket_body: str
    ticket_source: str
    ticket_created_at: str

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

