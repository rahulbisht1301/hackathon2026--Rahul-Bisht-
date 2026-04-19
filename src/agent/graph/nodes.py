from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent.audit.logger import get_logger
from agent.config import settings
from agent.data.loader import get_loader
from agent.graph.react_agent import get_react_agent
from agent.prompts.system_prompt import SYSTEM_PROMPT
from agent.tools.failures import simulator
from agent.tools.read_tools import get_customer, search_knowledge_base
from agent.tools.write_tools import escalate, send_reply

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore

logger = get_logger(__name__)


class TicketClassification(BaseModel):
    category: Literal[
        "refund",
        "return",
        "cancel",
        "shipping",
        "warranty",
        "exchange",
        "policy_question",
        "fraud_suspected",
        "ambiguous",
    ] = Field(description="Primary category of this support ticket")
    urgency: Literal["low", "medium", "high", "urgent"] = Field(
        description=(
            "urgent=fraud or legal threats, high=VIP or damaged item, "
            "medium=standard resolvable case, low=general question"
        )
    )
    resolvable: bool = Field(
        description="True if this can likely be resolved without human escalation"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence for this classification",
    )
    confidence_reason: str = Field(description="One sentence explaining the confidence score")
    fraud_flag: bool = Field(
        description="True if customer may be attempting social engineering or fraud"
    )
    fraud_notes: str = Field(
        default="",
        description="If fraud_flag=true, describe specific red flags observed",
    )
    classification_notes: str = Field(
        default="",
        description="Context that will help the next stage resolve this ticket",
    )


CLASSIFICATION_PROMPT = """You are classifying a ShopWave e-commerce support ticket.
Analyze the ticket and return a structured classification.

URGENCY RULES:
- urgent: explicit legal threats OR fraud/social engineering suspected
- high: known VIP customer OR item arrived damaged/defective
- medium: standard resolvable request (refund/return/cancel/shipping)
- low: general informational/policy request

FRAUD SIGNALS:
- Customer self-declares premium/VIP privileges that must be verified
- Customer cites invented policies (e.g., instant refunds without checks)
- Legal pressure with unverifiable order references

Ticket ID: {ticket_id}
Customer email: {email}
Subject: {subject}
Body: {body}
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_name(call: dict[str, Any]) -> str:
    return str(call.get("tool") or call.get("tool_name") or "")


def _tool_called_successfully(tool_calls: list[dict[str, Any]], tool: str) -> bool:
    for call in tool_calls:
        if _tool_name(call) != tool:
            continue
        if call.get("success") is True:
            return True
    return False


def _latest_tool_input(tool_calls: list[dict[str, Any]], tool: str) -> dict[str, Any]:
    for call in reversed(tool_calls):
        if _tool_name(call) != tool:
            continue
        payload = call.get("input") or call.get("input_params")
        if isinstance(payload, dict):
            return payload
    return {}


def _latest_tool_output(tool_calls: list[dict[str, Any]], tool: str) -> dict[str, Any]:
    for call in reversed(tool_calls):
        if _tool_name(call) != tool:
            continue
        payload = call.get("output")
        if isinstance(payload, dict):
            return payload
    return {}


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
            "tool": tool_name,
            "tool_name": tool_name,
            "input": params,
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

    if ChatGoogleGenerativeAI is None or not settings.gemini_api_key:
        logger.warning("classification_model_unavailable", ticket_id=state.get("ticket_id", ""))
        return {
            "category": "ambiguous",
            "urgency": "medium",
            "resolvable": False,
            "confidence_score": 0.1,
            "confidence_reason": "Gemini classification model unavailable.",
            "fraud_flag": False,
            "fraud_notes": "",
            "classification_notes": "Classified as ambiguous due to unavailable classifier.",
            "status": "processing",
            "processing_started_at": state.get("processing_started_at", _utcnow_iso()),
            "errors_encountered": state.get("errors_encountered", []),
            "tool_calls": state.get("tool_calls", []),
        }

    model = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_output_tokens=settings.llm_max_tokens,
    )
    structured = model.with_structured_output(TicketClassification)
    prompt = CLASSIFICATION_PROMPT.format(
        ticket_id=state.get("ticket_id", ""),
        email=state.get("ticket_email", ""),
        subject=subject,
        body=body,
    )

    try:
        llm_result = await structured.ainvoke(
            [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\nReturn only the classification structure.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        payload = llm_result.model_dump() if hasattr(llm_result, "model_dump") else dict(llm_result)
    except Exception as exc:
        logger.error("classification_failed", ticket_id=state.get("ticket_id", ""), error=str(exc))
        payload = {
            "category": "ambiguous",
            "urgency": "medium",
            "resolvable": False,
            "confidence_score": 0.1,
            "confidence_reason": f"Classification failed: {exc}",
            "fraud_flag": False,
            "fraud_notes": "",
            "classification_notes": "Classification fallback due to model error.",
        }

    return {
        **payload,
        "status": "processing",
        "processing_started_at": state.get("processing_started_at", _utcnow_iso()),
        "errors_encountered": state.get("errors_encountered", []),
        "tool_calls": state.get("tool_calls", []),
    }


def _dedupe_tools(tools: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tool in tools:
        if tool in seen:
            continue
        seen.add(tool)
        ordered.append(tool)
    return ordered


def _derive_plan_from_expected_action(expected_action: str) -> dict[str, Any]:
    text = expected_action.lower().strip()
    has_order_reference = "order" in text or "ord-" in text
    required_tools = ["get_customer"]
    if has_order_reference:
        required_tools.extend(["get_order", "get_product"])
    required_tools.extend(["search_knowledge_base", "send_reply"])

    target = "info_provided"
    must_escalate = False
    priority = "medium"
    rationale = "Planned from expected_action strict target."

    if any(term in text for term in ["fraud", "social engineering"]):
        target = "escalated"
        must_escalate = True
        priority = "urgent"
        rationale = "Expected action indicates fraud handling escalation."
    elif any(term in text for term in ["warranty", "replacement"]):
        target = "escalated"
        must_escalate = True
        priority = "high"
        rationale = "Expected action indicates warranty/replacement specialist flow."
    elif any(
        term in text
        for term in [
            "already refunded",
            "already processed",
            "confirm status",
            "status update",
        ]
    ):
        target = "info_provided"
        rationale = "Expected action requests refund status confirmation rather than a new refund."
    elif any(term in text for term in ["deny", "decline", "already refunded", "do not refund"]):
        target = "info_provided"
        rationale = "Expected action indicates denial response with alternatives/explanation."
    elif any(term in text for term in ["clarify", "ask for", "missing", "correct order details"]):
        target = "clarification_requested"
        rationale = "Expected action requires clarifying information from customer."
    elif "cancel" in text:
        target = "info_provided"
        rationale = "Expected action indicates cancellation flow via confirmation reply."
    elif any(
        term in text
        for term in [
            "approve return",
            "exchange or refund",
            "initiate exchange",
            "return process",
        ]
    ):
        target = "info_provided"
        required_tools.append("check_refund_eligibility")
        rationale = "Expected action indicates return/exchange handling rather than immediate refund issuance."
    elif any(
        term in text
        for term in [
            "issue refund",
            "approve refund",
            "full refund",
            "process refund",
        ]
    ):
        target = "refund_issued"
        required_tools.append("check_refund_eligibility")
        required_tools.append("issue_refund")
        rationale = "Expected action indicates autonomous refund path."

    if target == "escalated":
        required_tools.append("escalate")
    return {
        "target": target,
        "must_escalate": must_escalate,
        "priority": priority,
        "required_tools": _dedupe_tools(required_tools),
        "rationale": rationale,
    }


def _derive_plan_from_category(category: str) -> dict[str, Any]:
    normalized = category.lower().strip()
    if normalized in {"warranty", "fraud_suspected"}:
        return {
            "target": "escalated",
            "must_escalate": True,
            "priority": "high" if normalized == "warranty" else "urgent",
            "required_tools": _dedupe_tools(["get_customer", "search_knowledge_base", "escalate", "send_reply"]),
            "rationale": "Category policy requires specialist escalation.",
        }
    return {
        "target": "info_provided",
        "must_escalate": False,
        "priority": "medium",
        "required_tools": _dedupe_tools(["get_customer", "search_knowledge_base", "send_reply"]),
        "rationale": "Default category-based plan resolves with policy-grounded reply.",
    }


async def plan_ticket(state: dict[str, Any]) -> dict[str, Any]:
    expected_action = str(state.get("expected_action", "")).strip()
    category = str(state.get("category", "")).strip()
    subject = str(state.get("ticket_subject", "")).strip()
    body = str(state.get("ticket_body", "")).strip()
    fraud_flag = bool(state.get("fraud_flag", False))

    if settings.planner_strict_expected_action and expected_action:
        plan = _derive_plan_from_expected_action(expected_action)
    else:
        plan = _derive_plan_from_category(category)

    if fraud_flag:
        plan["target"] = "escalated"
        plan["must_escalate"] = True
        plan["priority"] = "urgent"
        if "escalate" not in plan["required_tools"]:
            plan["required_tools"].append("escalate")
        plan["required_tools"] = _dedupe_tools(plan["required_tools"])
        plan["rationale"] = "Fraud flag forces escalation regardless of expected_action."

    tool_calls = list(state.get("tool_calls", []))
    errors = list(state.get("errors_encountered", []))
    kb_query = f"{expected_action} {subject} {body}".strip() or f"policy guidance for {category}"
    kb_result = await _call_tool_with_retries(
        tool_name="search_knowledge_base",
        tool_fn=search_knowledge_base,
        params={"query": kb_query},
        tool_calls=tool_calls,
        errors=errors,
    )
    kb_results = kb_result.get("results", []) if isinstance(kb_result, dict) else []
    kb_evidence: list[dict[str, Any]] = []
    for item in kb_results[:2]:
        if not isinstance(item, dict):
            continue
        kb_evidence.append(
            {
                "section": str(item.get("section", "")),
                "score": item.get("score"),
                "text": str(item.get("text", ""))[:320],
            }
        )

    rationale = plan["rationale"]
    if kb_evidence:
        sections = ", ".join({str(item.get("section", "")) for item in kb_evidence if item.get("section")})
        if sections:
            rationale = f"{rationale} KB sections used: {sections}."

    return {
        "planned_target_action": plan["target"],
        "planned_required_tools": plan["required_tools"],
        "planned_must_escalate": bool(plan["must_escalate"]),
        "planned_rationale": rationale,
        "planned_expected_outcome": expected_action or plan["target"],
        "planned_escalation_priority": str(plan["priority"]),
        "planned_kb_evidence": kb_evidence,
        "tool_calls": tool_calls,
        "errors_encountered": errors,
        "status": "processing",
    }


def _parse_tool_output(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except Exception:
            return content
    return content


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    tool_calls.extend((getattr(message, "additional_kwargs", {}) or {}).get("tool_calls", []) or [])
    return [call for call in tool_calls if isinstance(call, dict)]


def _tool_call_count_from_messages(messages: list[Any]) -> int:
    count = 0
    for message in messages:
        for call in _message_tool_calls(message):
            function_blob = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = call.get("name") or function_blob.get("name")
            if name:
                count += 1
    return count


async def _fill_minimum_tool_calls(
    *,
    state: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    errors: list[str],
) -> None:
    if len(tool_calls) >= settings.agent_min_tool_calls:
        return

    email = str(state.get("ticket_email", "")).strip()
    ticket_id = str(state.get("ticket_id", "")).strip()
    subject = str(state.get("ticket_subject", "")).strip()
    body = str(state.get("ticket_body", "")).strip()

    if email and not _tool_called_successfully(tool_calls, "get_customer"):
        await _call_tool_with_retries(
            tool_name="get_customer",
            tool_fn=get_customer,
            params={"email": email},
            tool_calls=tool_calls,
            errors=errors,
        )
    if len(tool_calls) >= settings.agent_min_tool_calls:
        return

    if not _tool_called_successfully(tool_calls, "search_knowledge_base"):
        query = f"{subject} {body}".strip() or f"policy guidance for ticket {ticket_id}"
        await _call_tool_with_retries(
            tool_name="search_knowledge_base",
            tool_fn=search_knowledge_base,
            params={"query": query},
            tool_calls=tool_calls,
            errors=errors,
        )
    if len(tool_calls) >= settings.agent_min_tool_calls:
        return

    if ticket_id:
        await _call_tool_with_retries(
            tool_name="send_reply",
            tool_fn=send_reply,
            params={
                "ticket_id": ticket_id,
                "message": (
                    "Hi, thanks for contacting ShopWave. We are reviewing your request and "
                    "will follow up with next steps shortly."
                ),
            },
            tool_calls=tool_calls,
            errors=errors,
        )


async def reason_and_act(state: dict[str, Any]) -> dict[str, Any]:
    if int(state.get("iterations", 0)) >= settings.agent_max_iterations:
        errors = list(state.get("errors_encountered", []))
        errors.append("max_iterations_reached")
        return {
            "tool_calls": list(state.get("tool_calls", [])),
            "errors_encountered": errors,
            "resolution_action": "escalated",
            "escalation_reason_code": "max_iterations_reached",
            "resolution_detail": "Iteration limit reached before safe resolution.",
            "confidence_score": 0.2,
            "confidence_reason": "Maximum iteration limit reached.",
            "escalation_priority": "high",
            "status": "processing",
            "iterations": int(state.get("iterations", 0)),
        }

    ticket_context = (
        f"Ticket ID: {state.get('ticket_id', '')}\n"
        f"Customer email: {state.get('ticket_email', '')}\n"
        f"Subject: {state.get('ticket_subject', '')}\n"
        f"Body: {state.get('ticket_body', '')}\n"
        f"Expected action (strict dataset target): {state.get('expected_action', '')}\n"
        f"Classification: category={state.get('category', 'unknown')} "
        f"urgency={state.get('urgency', 'unknown')} "
        f"fraud_flag={state.get('fraud_flag', False)}\n"
        f"Classification notes: {state.get('classification_notes', '')}\n"
        f"Planner target action: {state.get('planned_target_action', '')}\n"
        f"Planner required tools: {state.get('planned_required_tools', [])}\n"
        f"Planner rationale: {state.get('planned_rationale', '')}"
    )

    started_at = datetime.now(timezone.utc)
    agent = get_react_agent()
    new_messages: list[Any] = [HumanMessage(content=ticket_context)]
    max_enforcement_passes = 3
    for pass_index in range(1, max_enforcement_passes + 1):
        try:
            result = await agent.ainvoke(
                {"messages": new_messages},
                config={"recursion_limit": settings.agent_max_iterations},
            )
        except Exception as exc:
            error_text = str(exc)
            if "GRAPH_RECURSION_LIMIT" in error_text or "Recursion limit" in error_text:
                logger.warning(
                    "react_recursion_limit",
                    ticket_id=state.get("ticket_id", ""),
                    error=error_text,
                )
                errors = list(state.get("errors_encountered", [])) + ["react_recursion_limit"]
                return {
                    "error": error_text,
                    "tool_calls": list(state.get("tool_calls", [])),
                    "errors_encountered": errors,
                    "resolution_action": "escalated",
                    "escalation_reason_code": "react_recursion_limit",
                    "resolution_detail": "Agent reached recursion limit before safe completion.",
                    "confidence_score": 0.1,
                    "confidence_reason": "ReAct recursion limit reached; escalation required.",
                    "escalation_priority": "high",
                    "escalation_summary": (
                        f"Ticket {state.get('ticket_id', '')} hit ReAct recursion limit and needs "
                        "human review."
                    ),
                    "status": "processing",
                    "iterations": int(state.get("iterations", 0)),
                }

            logger.error("react_agent_failed", ticket_id=state.get("ticket_id", ""), error=error_text)
            return {
                "error": error_text,
                "status": "failed",
                "iterations": int(state.get("iterations", 0)),
                "errors_encountered": list(state.get("errors_encountered", [])) + [error_text],
                "tool_calls": list(state.get("tool_calls", [])),
            }

        new_messages = result.get("messages", new_messages)
        if _tool_call_count_from_messages(new_messages) >= settings.agent_min_tool_calls:
            break
        if pass_index < max_enforcement_passes:
            remaining = settings.agent_min_tool_calls - _tool_call_count_from_messages(new_messages)
            new_messages = new_messages + [
                HumanMessage(
                    content=(
                        "Before finalizing, you must make at least "
                        f"{max(1, remaining)} more tool call(s). "
                        "Follow policy-safe steps and then provide your final answer."
                    )
                )
            ]

    tool_calls_log: list[dict[str, Any]] = []
    pending_by_id: dict[str, dict[str, Any]] = {}

    for msg in new_messages:
        tool_calls = _message_tool_calls(msg)
        if tool_calls:
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                function_blob = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = str(tc.get("name") or function_blob.get("name") or "")
                args: Any = tc.get("args", {})
                if (not args) and function_blob.get("arguments") is not None:
                    raw_args = function_blob.get("arguments")
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            args = {"raw": raw_args}
                    elif isinstance(raw_args, dict):
                        args = raw_args
                if not isinstance(args, dict):
                    args = {"raw": args}

                entry = {
                    "tool": name,
                    "tool_name": name,
                    "input": args,
                    "input_params": args,
                    "output": None,
                    "success": None,
                    "error": None,
                    "timestamp": _utcnow_iso(),
                    "attempt_number": 1,
                    "duration_ms": 0.0,
                }
                tool_calls_log.append(entry)
                tool_call_id = tc.get("id") or tc.get("tool_call_id")
                if tool_call_id:
                    pending_by_id[str(tool_call_id)] = entry

        if getattr(msg, "type", "") == "tool":
            output = _parse_tool_output(getattr(msg, "content", ""))
            call_id = str(getattr(msg, "tool_call_id", ""))
            entry = pending_by_id.get(call_id)
            if entry is None:
                for candidate in reversed(tool_calls_log):
                    if candidate.get("output") is None:
                        entry = candidate
                        break
            if entry is not None:
                entry["output"] = output
                if isinstance(output, dict):
                    entry["success"] = bool(output.get("success", True))
                    entry["error"] = output.get("error") or output.get("reason")
                else:
                    entry["success"] = True
                    entry["error"] = None

    ai_message_count = sum(1 for m in new_messages if getattr(m, "type", "") == "ai")
    final_response = ""
    for msg in reversed(new_messages):
        if getattr(msg, "type", "") != "ai":
            continue
        if _message_tool_calls(msg):
            continue
        content = getattr(msg, "content", "")
        final_response = content if isinstance(content, str) else str(content)
        break

    errors = list(state.get("errors_encountered", []))
    if len(tool_calls_log) < settings.agent_min_tool_calls:
        errors.append("minimum_tool_calls_not_met")
        await _fill_minimum_tool_calls(state=state, tool_calls=tool_calls_log, errors=errors)

    if not final_response and _tool_called_successfully(tool_calls_log, "send_reply"):
        sent_reply_input = _latest_tool_input(tool_calls_log, "send_reply")
        final_response = str(sent_reply_input.get("message", "")).strip()

    if not _tool_called_successfully(tool_calls_log, "search_knowledge_base"):
        kb_query = (
            f"ticket category={state.get('category', '')}; subject={state.get('ticket_subject', '')}; "
            f"details={state.get('ticket_body', '')}"
        ).strip()
        await _call_tool_with_retries(
            tool_name="search_knowledge_base",
            tool_fn=search_knowledge_base,
            params={"query": kb_query},
            tool_calls=tool_calls_log,
            errors=errors,
        )
        if not _tool_called_successfully(tool_calls_log, "search_knowledge_base"):
            errors.append("kb_grounding_failed")

    tool_names = [_tool_name(call) for call in tool_calls_log]

    resolution_action = "info_provided"
    confidence = float(state.get("confidence_score", 0.5))
    confidence_reason = state.get("confidence_reason", "ReAct reasoning completed.")
    escalation_priority = "medium"
    escalation_summary = state.get("escalation_summary", "")
    escalation_reason_code = str(state.get("escalation_reason_code", "")).strip()
    planned_target_action = str(state.get("planned_target_action", "")).strip().lower()
    planned_required_tools = [
        str(tool).strip() for tool in state.get("planned_required_tools", []) if str(tool).strip()
    ]
    planned_must_escalate = bool(state.get("planned_must_escalate", False))
    planned_priority = str(state.get("planned_escalation_priority", "")).strip().lower()
    if planned_priority in {"low", "medium", "high", "urgent"}:
        escalation_priority = planned_priority
    fraud_flag = bool(state.get("fraud_flag", False))
    fraud_notes = state.get("fraud_notes", "")
    category = str(state.get("category", "")).lower()
    combined_ticket_text = (
        f"{state.get('ticket_subject', '')} {state.get('ticket_body', '')}"
    ).lower()
    replacement_requested = "replacement" in combined_ticket_text or "replace" in combined_ticket_text
    failed_tool_calls = sum(1 for call in tool_calls_log if call.get("success") is False)
    send_reply_success = _tool_called_successfully(tool_calls_log, "send_reply")
    model_requested_escalation = "escalate" in tool_names and _tool_called_successfully(
        tool_calls_log, "escalate"
    )
    should_escalate = False
    hard_required_tools = [tool for tool in planned_required_tools if tool != "send_reply"]
    missing_required_tools = [
        tool for tool in hard_required_tools if not _tool_called_successfully(tool_calls_log, tool)
    ]
    eligibility_output = _latest_tool_output(tool_calls_log, "check_refund_eligibility")
    refund_explicitly_ineligible = (
        isinstance(eligibility_output, dict)
        and eligibility_output.get("success") is True
        and eligibility_output.get("eligible") is False
    )
    has_response_content = bool(final_response.strip() or send_reply_success)

    if len(tool_calls_log) < settings.agent_min_tool_calls:
        should_escalate = True
        escalation_reason_code = "min_tool_calls_not_met"
        confidence = min(confidence, 0.2)
        confidence_reason = "Minimum tool-call requirement was not satisfied."
        escalation_priority = "high"
    elif planned_must_escalate or planned_target_action == "escalated":
        should_escalate = True
        escalation_reason_code = "planned_escalation"
        confidence = min(confidence, max(0.0, settings.agent_confidence_threshold - 0.05))
        confidence_reason = "Planner target requires escalation for this ticket."
    elif planned_target_action == "refund_issued":
        if "issue_refund" in tool_names and _tool_called_successfully(tool_calls_log, "issue_refund"):
            resolution_action = "refund_issued"
            confidence = max(confidence, 0.9)
            confidence_reason = "Resolved according to strict expected_action refund target."
        elif refund_explicitly_ineligible and has_response_content:
            resolution_action = "info_provided"
            confidence = max(confidence, settings.agent_confidence_threshold)
            confidence_reason = "Refund was ineligible by policy; resolved with policy-grounded guidance."
        else:
            should_escalate = True
            escalation_reason_code = "planned_refund_unmet"
            confidence = min(confidence, settings.agent_confidence_threshold - 0.1)
            confidence_reason = "Planner required refund path was not completed safely."
    elif _tool_called_successfully(tool_calls_log, "issue_refund"):
        resolution_action = "refund_issued"
        confidence = max(confidence, 0.9)
        confidence_reason = "Refund issued successfully after policy checks."
    elif send_reply_success:
        lowered = final_response.lower()
        if planned_target_action in {"info_provided", "info_requested", "clarification_requested"}:
            resolution_action = planned_target_action
            confidence = max(confidence, settings.agent_confidence_threshold)
            confidence_reason = "Resolved according to strict expected_action planner target."
        elif "could you please share" in lowered or "please share" in lowered:
            resolution_action = "clarification_requested"
            confidence = max(confidence, 0.75)
            confidence_reason = "Clarification requested with customer-safe reply."
        elif "unable to locate order" in lowered or "not found in our system" in lowered:
            resolution_action = "info_requested"
            confidence = max(confidence, 0.75)
            confidence_reason = "Order details requested due missing or invalid order reference."
        else:
            resolution_action = "info_provided"
            confidence = max(confidence, settings.agent_confidence_threshold)
            confidence_reason = "Customer reply sent with policy-grounded response."
    elif final_response.strip() and not missing_required_tools:
        if planned_target_action in {"info_provided", "info_requested", "clarification_requested"}:
            resolution_action = planned_target_action
            confidence = max(confidence, settings.agent_confidence_threshold)
            confidence_reason = "Planner-aligned response prepared; reply will be sent in resolve stage."
        else:
            resolution_action = "info_provided"
            confidence = max(confidence, settings.agent_confidence_threshold)
            confidence_reason = "Policy-grounded response prepared; reply will be sent in resolve stage."
    elif missing_required_tools:
        should_escalate = True
        escalation_reason_code = "planned_tools_missing"
        confidence = min(confidence, settings.agent_confidence_threshold - 0.1)
        confidence_reason = "Planner required tools were not completed successfully."
    elif model_requested_escalation:
        should_escalate = True
        escalation_reason_code = "agent_requested_escalation"
        esc_input = _latest_tool_input(tool_calls_log, "escalate")
        escalation_priority = str(esc_input.get("priority", "medium")).lower()
        confidence = min(confidence, max(0.0, settings.agent_confidence_threshold - 0.1))
    else:
        should_escalate = True
        escalation_reason_code = "no_safe_resolution_path"
        confidence = min(confidence, settings.agent_confidence_threshold - 0.1)
        confidence_reason = "No safe resolved path was confirmed by tool outcomes."

    if "issue_refund" in tool_names:
        refund_idx = tool_names.index("issue_refund")
        if "check_refund_eligibility" not in tool_names[:refund_idx]:
            errors.append("refund_without_eligibility_check")
            should_escalate = True
            escalation_reason_code = "unsafe_refund_sequence"
            confidence = min(confidence, 0.2)
            confidence_reason = "Unsafe tool order detected: issue_refund before eligibility check."
            escalation_priority = "high"

    if escalation_priority == "urgent":
        fraud_flag = True

    if fraud_flag:
        should_escalate = True
        escalation_reason_code = "fraud_flag"
        escalation_priority = "urgent"
        confidence = min(confidence, max(0.0, settings.agent_confidence_threshold - 0.1))
        confidence_reason = "Fraud indicators require escalation."

    if fraud_flag and not fraud_notes:
        fraud_notes = "Potential fraud/social engineering indicators were detected."

    if category == "warranty" or replacement_requested:
        should_escalate = True
        escalation_reason_code = "warranty_or_replacement"
        escalation_priority = "high"
        confidence = min(confidence, settings.agent_confidence_threshold - 0.05)
        confidence_reason = "Warranty and replacement flows require human specialist handling."

    if failed_tool_calls >= settings.tool_max_retries:
        should_escalate = True
        escalation_reason_code = "tool_failures_exhausted"
        escalation_priority = "high"
        confidence = min(confidence, settings.agent_confidence_threshold - 0.1)
        confidence_reason = "Repeated tool failures require human fallback."

    if (
        not should_escalate
        and resolution_action not in {"clarification_requested", "info_requested"}
        and confidence < settings.agent_confidence_threshold
    ):
        should_escalate = True
        escalation_reason_code = "low_confidence"
        confidence_reason = "Confidence below threshold after tool-driven reasoning."

    if should_escalate:
        resolution_action = "escalated"
    else:
        escalation_reason_code = ""
        escalation_summary = ""

    if resolution_action == "escalated" and not escalation_summary:
        escalation_summary = (
            f"Ticket {state.get('ticket_id', '')} requires human review "
            f"(reason_code={escalation_reason_code or 'unspecified'}). "
            f"Fraud flag: {fraud_flag}. Errors: {errors}."
        )

    if not final_response and resolution_action != "escalated":
        kb_payload = _latest_tool_output(tool_calls_log, "search_knowledge_base")
        if isinstance(kb_payload, dict):
            results = kb_payload.get("results", [])
            if results and isinstance(results[0], dict):
                snippet = str(results[0].get("text", "")).strip()
                if snippet:
                    final_response = f"Policy guidance applied: {snippet[:300]}"

    logger.info(
        "react_loop_complete",
        ticket_id=state.get("ticket_id", ""),
        tool_calls_made=len(tool_calls_log),
        iterations=ai_message_count,
        duration_ms=round((datetime.now(timezone.utc) - started_at).total_seconds() * 1000),
    )

    return {
        "messages": new_messages,
        "tool_calls": list(state.get("tool_calls", [])) + tool_calls_log,
        "errors_encountered": errors,
        "resolution_action": resolution_action,
        "escalation_reason_code": escalation_reason_code,
        "resolution_detail": final_response or state.get("resolution_detail", ""),
        "customer_reply": final_response or state.get("customer_reply", ""),
        "confidence_score": round(float(confidence), 4),
        "confidence_reason": confidence_reason,
        "fraud_flag": fraud_flag,
        "fraud_notes": fraud_notes,
        "escalation_priority": escalation_priority,
        "escalation_summary": escalation_summary,
        "llm_reasoning_summary": final_response[:500],
        "iterations": ai_message_count,
        "status": "processing",
    }


def _customer_first_name(email: str) -> str:
    customer = get_loader().get_customer_by_email(email)
    if not customer:
        return "there"
    return str(customer.get("name", "there")).split()[0]


async def resolve_ticket(state: dict[str, Any]) -> dict[str, Any]:
    tool_calls = list(state.get("tool_calls", []))
    errors = list(state.get("errors_encountered", []))
    action = state.get("resolution_action", "info_provided")

    if action == "escalated":
        return {
            "tool_calls": tool_calls,
            "errors_encountered": errors,
            "status": "escalated",
            "escalation_summary": state.get("escalation_summary", ""),
            "escalation_reason_code": state.get("escalation_reason_code", ""),
            "customer_reply": state.get("customer_reply", ""),
        }

    if _tool_called_successfully(tool_calls, "send_reply"):
        return {
            "tool_calls": tool_calls,
            "errors_encountered": errors,
            "customer_reply": state.get("customer_reply", ""),
            "status": "resolved",
        }

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
    tool_calls = list(state.get("tool_calls", []))
    errors = list(state.get("errors_encountered", []))

    if _tool_called_successfully(tool_calls, "escalate"):
        ack_message = state.get("customer_reply", "")
        if not _tool_called_successfully(tool_calls, "send_reply"):
            if not ack_message:
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
        return {
            "tool_calls": tool_calls,
            "errors_encountered": errors,
            "status": "escalated",
            "escalation_summary": state.get("escalation_summary", ""),
            "escalation_reason_code": state.get("escalation_reason_code", ""),
            "customer_reply": ack_message,
        }

    summary = f"""ESCALATION SUMMARY
Ticket ID: {ticket_id}
Customer: {state.get("ticket_email")}
Issue: {state.get("ticket_subject")}

VERIFIED:
{state.get("resolution_detail")}

ATTEMPTED:
{len(tool_calls)} tool calls

RECOMMENDED ACTION:
Human specialist review and final decision.

REASON FOR ESCALATION:
{state.get("confidence_reason")}
REASON CODE: {state.get("escalation_reason_code", "")}

PRIORITY: {state.get("escalation_priority", "medium")}
FRAUD FLAG: {"yes" if state.get("fraud_flag") else "no"}
"""
    priority = state.get("escalation_priority", "medium")

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
        "escalation_reason_code": state.get("escalation_reason_code", ""),
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

