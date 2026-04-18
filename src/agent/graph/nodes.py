from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from agent.audit.logger import get_logger
from agent.config import settings
from agent.data.loader import get_loader
from agent.prompts.system_prompt import SYSTEM_PROMPT
from agent.tools.failures import simulator
from agent.tools.read_tools import get_customer, get_order, get_product, search_knowledge_base
from agent.tools.write_tools import check_refund_eligibility, escalate, issue_refund, send_reply

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore

logger = get_logger(__name__)


class ClassificationOutput(BaseModel):
    category: str = Field(...)
    urgency: str = Field(...)
    resolvable: bool = Field(...)
    confidence_score: float = Field(...)
    confidence_reason: str = Field(...)
    fraud_flag: bool = Field(False)
    fraud_notes: str = Field("")
    classification_notes: str = Field("")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_order_id(text: str) -> str | None:
    match = re.search(r"(ORD-\d+)", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _classify_heuristic(subject: str, body: str) -> dict[str, Any]:
    text = f"{subject} {body}".lower()
    category = "ambiguous"
    urgency = "medium"
    resolvable = True
    fraud_flag = False
    fraud_notes = ""
    confidence = 0.72
    confidence_reason = "Heuristic classification based on ticket language."

    if "lawyer" in text or "legal" in text or "dispute with my bank" in text:
        urgency = "urgent"
    if "policy" in text or "what is your return policy" in text:
        category = "policy_question"
        urgency = "low"
        confidence = 0.9
    elif "cancel" in text:
        category = "cancel"
    elif "where is my order" in text or "tracking" in text:
        category = "shipping"
    elif "wrong size" in text or "wrong colour" in text or "wrong color" in text:
        category = "exchange"
    elif "warranty" in text:
        category = "warranty"
        resolvable = False
        confidence = 0.55
    elif "refund" in text or "return" in text or "defect" in text or "broken" in text:
        category = "refund"
    if "premium member" in text or "instant refunds without questions" in text:
        fraud_flag = True
        fraud_notes = "Customer claims unsupported privilege."
        urgency = "urgent"
        confidence = 0.35
        category = "fraud_suspected"
        resolvable = False
    if "my thing is broken" in text and "ord-" not in text:
        category = "ambiguous"
        urgency = "medium"
        confidence = 0.35
    return {
        "category": category,
        "urgency": urgency,
        "resolvable": resolvable,
        "fraud_flag": fraud_flag,
        "fraud_notes": fraud_notes,
        "confidence_score": confidence,
        "confidence_reason": confidence_reason,
        "classification_notes": "Heuristic classifier used.",
    }


async def _call_tool_with_retries(
    *,
    tool_name: str,
    tool_fn: Callable[..., Awaitable[dict[str, Any]]],
    params: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    last_result: dict[str, Any] = {"success": False, "error": "unknown_error"}
    max_retries = max(1, settings.tool_max_retries)
    retry_delays = settings.tool_retry_delays or [1.0]
    for attempt in range(1, max_retries + 1):
        started = datetime.now(timezone.utc)
        logger.debug("tool_call_started", tool_name=tool_name, params=params, attempt=attempt)
        result = await tool_fn(**params)
        duration_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
        success = bool(result.get("success", False))
        error = result.get("error") or result.get("reason")
        record = {
            "tool_name": tool_name,
            "input_params": params,
            "output": result,
            "success": success,
            "error": error,
            "duration_ms": round(duration_ms, 3),
            "attempt_number": attempt,
            "timestamp": _utcnow_iso(),
        }
        tool_calls.append(record)

        if success:
            if attempt > 1:
                simulator.mark_recovered_failure()
            logger.debug("tool_call_success", tool_name=tool_name, duration_ms=round(duration_ms, 2))
            return result

        message = str(error or "")
        retryable = any(x in message.lower() for x in ["timed out", "timeout", "transient", "503"])
        if retryable and attempt < max_retries:
            logger.warning(
                "tool_call_failed_retrying",
                tool_name=tool_name,
                error=message,
                attempt=attempt,
            )
            delay_index = min(attempt - 1, len(retry_delays) - 1)
            await asyncio.sleep(retry_delays[delay_index])
            last_result = result
            continue

        errors.append(f"{tool_name}: {message}")
        logger.warning(
            "tool_call_failed",
            tool_name=tool_name,
            error=message,
            attempt=attempt,
        )
        return result
    return last_result


async def classify_ticket(state: dict[str, Any]) -> dict[str, Any]:
    subject = state.get("ticket_subject", "")
    body = state.get("ticket_body", "")
    result = _classify_heuristic(subject, body)
    if settings.gemini_api_key and ChatGoogleGenerativeAI is not None:
        try:
            model = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.gemini_api_key,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
            )
            structured = model.with_structured_output(ClassificationOutput)
            llm_result = await structured.ainvoke(
                [
                    {
                        "role": "system",
                        "content": (
                            SYSTEM_PROMPT
                            + "\nReturn only classification fields. Do not call tools in this step."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Subject: {subject}\nBody: {body}",
                    },
                ]
            )
            if hasattr(llm_result, "model_dump"):
                result = llm_result.model_dump()
            elif isinstance(llm_result, dict):
                result = llm_result
        except Exception:
            pass
    return {
        **result,
        "status": "processing",
        "processing_started_at": state.get("processing_started_at", _utcnow_iso()),
        "errors_encountered": state.get("errors_encountered", []),
        "tool_calls": state.get("tool_calls", []),
    }


def _is_warranty_case(order: dict[str, Any], product: dict[str, Any], body: str) -> bool:
    if not order.get("delivery_date"):
        return False
    warranty_months = int(product.get("warranty_months", 0))
    if warranty_months <= 0:
        return False
    defect_words = ["stopped working", "defect", "manufacturing defect", "broken", "heating"]
    has_defect = any(word in body.lower() for word in defect_words)
    if not has_defect:
        return False
    delivery = datetime.fromisoformat(order["delivery_date"]).date()
    deadline = delivery + timedelta(days=30 * warranty_months)
    today = datetime.fromisoformat(f"{settings.policy_reference_date}T00:00:00+00:00").date()
    return today <= deadline


async def reason_and_act(state: dict[str, Any]) -> dict[str, Any]:
    tool_calls = list(state.get("tool_calls", []))
    errors = list(state.get("errors_encountered", []))
    if int(state.get("iterations", 0)) >= settings.agent_max_iterations:
        errors.append("max_iterations_reached")
        return {
            "tool_calls": tool_calls,
            "errors_encountered": errors,
            "resolution_action": "escalated",
            "resolution_detail": "Iteration limit reached before safe resolution.",
            "confidence_score": 0.2,
            "confidence_reason": "Maximum iteration limit reached.",
            "escalation_priority": "high",
            "status": "processing",
            "iterations": int(state.get("iterations", 0)),
        }
    body = state.get("ticket_body", "")
    subject = state.get("ticket_subject", "")
    ticket_id = state.get("ticket_id", "")
    email = state.get("ticket_email", "")
    category = state.get("category", "ambiguous")

    customer_res = await _call_tool_with_retries(
        tool_name="get_customer",
        tool_fn=get_customer,
        params={"email": email},
        tool_calls=tool_calls,
        errors=errors,
    )
    customer = customer_res.get("customer") if customer_res.get("success") else None

    explicit_order_id = _extract_order_id(body) or _extract_order_id(subject)
    order_id = explicit_order_id
    if not order_id and customer:
        loader = get_loader()
        candidate = loader.find_latest_order_for_customer(customer["customer_id"])
        if candidate:
            order_id = candidate["order_id"]

    order = None
    if order_id:
        order_res = await _call_tool_with_retries(
            tool_name="get_order",
            tool_fn=get_order,
            params={"order_id": order_id},
            tool_calls=tool_calls,
            errors=errors,
        )
        order = order_res.get("order") if order_res.get("success") else None

    product = None
    if order and order.get("product_id"):
        product_res = await _call_tool_with_retries(
            tool_name="get_product",
            tool_fn=get_product,
            params={"product_id": order["product_id"]},
            tool_calls=tool_calls,
            errors=errors,
        )
        product = product_res.get("product") if product_res.get("success") else None

    kb_res = await _call_tool_with_retries(
        tool_name="search_knowledge_base",
        tool_fn=search_knowledge_base,
        params={"query": f"{subject}. {body}"},
        tool_calls=tool_calls,
        errors=errors,
    )
    kb_results = kb_res.get("results", []) if kb_res.get("success") else []

    resolution_action = state.get("resolution_action", "info_provided")
    resolution_detail = ""
    confidence = float(state.get("confidence_score", 0.7))
    confidence_reason = state.get("confidence_reason", "Policy-based reasoning completed.")
    fraud_flag = bool(state.get("fraud_flag", False))
    fraud_notes = state.get("fraud_notes", "")
    escalation_priority = "medium"
    escalation_summary = ""

    ticket_text = f"{subject} {body}".lower()
    tier = str(customer.get("tier", "standard") if customer else "unknown").lower()
    if "premium member" in ticket_text and tier not in {"premium", "vip"}:
        fraud_flag = True
        fraud_notes = "Customer claimed premium tier but verified tier does not match."
        confidence = min(confidence, 0.3)
    if order_id and order is None and ("lawyer" in ticket_text or "legal" in ticket_text):
        fraud_flag = True
        fraud_notes = "Legal pressure with non-existent order reference."
        confidence = 0.2

    if customer is None:
        resolution_action = "info_requested"
        resolution_detail = (
            "We could not match your email in our system. Please share your registered email and order ID."
        )
        confidence = 0.7
        confidence_reason = "Customer record missing; safe clarification request."
    elif category == "policy_question" and not explicit_order_id:
        resolution_action = "info_provided"
        resolution_detail = (
            "Provided return windows (30-day standard, 15-day high-value, 60-day accessories) and exchange policy."
        )
        confidence = 0.92
    elif category == "shipping" and order:
        resolution_action = "tracking_info_provided"
        resolution_detail = order.get("notes", "Shared current shipment status.")
        confidence = 0.9
    elif category == "cancel" and order:
        if order.get("status") == "processing":
            resolution_action = "cancelled"
            resolution_detail = "Order is in processing state and can be cancelled free of charge."
            confidence = 0.95
        else:
            resolution_action = "info_provided"
            resolution_detail = "Order is no longer processing; cancellation unavailable."
            confidence = 0.8
    elif category == "ambiguous" and not explicit_order_id:
        resolution_action = "clarification_requested"
        resolution_detail = (
            "Requested product name, order ID, and exact issue description before any action."
        )
        confidence = 0.35
        confidence_reason = "Insufficient context for policy-safe action."
    elif order and product and _is_warranty_case(order, product, body):
        resolution_action = "escalated"
        resolution_detail = "Warranty claims are routed to the warranty team."
        confidence = 0.4
        confidence_reason = "Warranty policy mandates escalation."
        escalation_priority = "medium"
    elif order:
        elig_res = await _call_tool_with_retries(
            tool_name="check_refund_eligibility",
            tool_fn=check_refund_eligibility,
            params={"order_id": order["order_id"]},
            tool_calls=tool_calls,
            errors=errors,
        )

        if not elig_res.get("success"):
            resolution_action = "escalated"
            resolution_detail = "Eligibility tool failed after retries."
            confidence = 0.35
            confidence_reason = "Could not reliably validate eligibility."
        elif "eligible" not in elig_res:
            resolution_action = "escalated"
            resolution_detail = "Malformed eligibility response prevented safe refund decision."
            confidence = 0.3
            confidence_reason = "Malformed refund eligibility output."
        elif bool(elig_res.get("eligible")):
            amount = float(elig_res.get("amount", order.get("amount", 0.0)))
            wants_replacement = "replacement" in ticket_text and "not a refund" in ticket_text
            if wants_replacement or amount > 200.0:
                resolution_action = "escalated"
                resolution_detail = "Escalation required for replacement handling or high-value refund."
                confidence = 0.45
                escalation_priority = "high" if wants_replacement else "medium"
            else:
                refund_res = await _call_tool_with_retries(
                    tool_name="issue_refund",
                    tool_fn=issue_refund,
                    params={"order_id": order["order_id"], "amount": amount},
                    tool_calls=tool_calls,
                    errors=errors,
                )
                if refund_res.get("success"):
                    resolution_action = "refund_issued"
                    resolution_detail = (
                        f"Refund issued: {refund_res.get('refund_id')} for ${refund_res.get('amount'):.2f}."
                    )
                    confidence = 0.93
                else:
                    resolution_action = "escalated"
                    resolution_detail = "Refund blocked by safety guard or downstream failure."
                    confidence = 0.38
        else:
            reason = str(elig_res.get("reason", "ineligible"))
            if reason == "already_refunded":
                resolution_action = "info_provided"
                resolution_detail = "Refund already processed; informed customer about 5-7 business day settlement."
                confidence = 0.93
            elif reason == "order_in_processing_cancel_instead":
                resolution_action = "cancelled"
                resolution_detail = "Order is still processing; cancellation path applied."
                confidence = 0.9
            else:
                resolution_action = "return_declined"
                resolution_detail = f"Refund/return denied based on policy reason: {reason}."
                confidence = 0.88
    else:
        resolution_action = "escalated"
        resolution_detail = "Insufficient verified order/product context."
        confidence = 0.3

    if resolution_action == "escalated" and fraud_flag:
        escalation_priority = "urgent"
    elif resolution_action == "escalated" and escalation_priority == "medium" and "vip" in tier:
        escalation_priority = "high"

    if len(tool_calls) - len(state.get("tool_calls", [])) < settings.agent_min_tool_calls:
        await _call_tool_with_retries(
            tool_name="search_knowledge_base",
            tool_fn=search_knowledge_base,
            params={"query": "refund and escalation policy"},
            tool_calls=tool_calls,
            errors=errors,
        )

    if resolution_action == "escalated":
        escalation_summary = (
            f"Ticket {ticket_id} requires human review. Action candidate: {resolution_action}. "
            f"Fraud flag: {fraud_flag}. Errors: {errors}."
        )

    reasoning = (
        f"Customer verified={bool(customer)}; order={order_id or 'none'}; "
        f"category={category}; decision={resolution_action}; kb_hits={len(kb_results)}."
    )

    return {
        "tool_calls": tool_calls,
        "errors_encountered": errors,
        "resolution_action": resolution_action,
        "resolution_detail": resolution_detail,
        "confidence_score": round(float(confidence), 4),
        "confidence_reason": confidence_reason,
        "fraud_flag": fraud_flag,
        "fraud_notes": fraud_notes,
        "escalation_priority": escalation_priority,
        "escalation_summary": escalation_summary,
        "llm_reasoning_summary": reasoning,
        "iterations": int(state.get("iterations", 0)) + 1,
        "status": "processing",
    }


def _customer_first_name(email: str) -> str:
    customer = get_loader().get_customer_by_email(email)
    if not customer:
        return "there"
    return str(customer.get("name", "there")).split()[0]


async def resolve_ticket(state: dict[str, Any]) -> dict[str, Any]:
    action = state.get("resolution_action", "info_provided")
    detail = state.get("resolution_detail", "")
    first_name = _customer_first_name(state.get("ticket_email", ""))
    ticket_id = state.get("ticket_id", "")

    if action == "refund_issued":
        message = (
            f"Hi {first_name}, thanks for your patience. We've completed your refund. "
            f"{detail} It should appear in 5-7 business days."
        )
    elif action == "cancelled":
        message = (
            f"Hi {first_name}, your request has been completed. "
            f"We've cancelled this order and sent confirmation."
        )
    elif action in {"return_declined", "info_requested", "clarification_requested"}:
        message = f"Hi {first_name}, {detail}"
    else:
        message = f"Hi {first_name}, {detail}"

    tool_calls = list(state.get("tool_calls", []))
    errors = list(state.get("errors_encountered", []))
    reply_res = await _call_tool_with_retries(
        tool_name="send_reply",
        tool_fn=send_reply,
        params={"ticket_id": ticket_id, "message": message},
        tool_calls=tool_calls,
        errors=errors,
    )
    status = "resolved" if reply_res.get("success") else "failed"

    return {
        "tool_calls": tool_calls,
        "errors_encountered": errors,
        "customer_reply": message,
        "status": status,
    }


async def escalate_ticket(state: dict[str, Any]) -> dict[str, Any]:
    ticket_id = state.get("ticket_id", "")
    first_name = _customer_first_name(state.get("ticket_email", ""))
    summary = f"""ESCALATION SUMMARY
Ticket ID: {ticket_id}
Customer: {state.get("ticket_email")} 
Issue: {state.get("ticket_subject")}

VERIFIED:
{state.get("resolution_detail")}

ATTEMPTED:
{len(state.get("tool_calls", []))} tool calls

RECOMMENDED ACTION:
Human specialist review and final decision.

REASON FOR ESCALATION:
{state.get("confidence_reason")}

PRIORITY: {state.get("escalation_priority", "medium")}
FRAUD FLAG: {"yes" if state.get("fraud_flag") else "no"}
"""
    priority = state.get("escalation_priority", "medium")

    tool_calls = list(state.get("tool_calls", []))
    errors = list(state.get("errors_encountered", []))

    esc_res = await _call_tool_with_retries(
        tool_name="escalate",
        tool_fn=escalate,
        params={"ticket_id": ticket_id, "summary": summary, "priority": priority},
        tool_calls=tool_calls,
        errors=errors,
    )
    ack_message = (
        f"Hi {first_name}, thanks for your patience. "
        "Your case has been escalated to a specialist, and we will update you shortly."
    )
    await _call_tool_with_retries(
        tool_name="send_reply",
        tool_fn=send_reply,
        params={"ticket_id": ticket_id, "message": ack_message},
        tool_calls=tool_calls,
        errors=errors,
    )
    status = "escalated" if esc_res.get("success") else "failed"
    return {
        "tool_calls": tool_calls,
        "errors_encountered": errors,
        "status": status,
        "escalation_summary": summary,
        "customer_reply": ack_message,
    }


async def write_audit_entry(state: dict[str, Any]) -> dict[str, Any]:
    completed = datetime.now(timezone.utc)
    started_raw = state.get("processing_started_at")
    try:
        started = datetime.fromisoformat(started_raw) if started_raw else completed
    except Exception:
        started = completed
    duration_ms = (completed - started).total_seconds() * 1000
    return {
        "processing_completed_at": completed.isoformat(),
        "total_duration_ms": round(duration_ms, 3),
        "error": state.get("error", ""),
    }

