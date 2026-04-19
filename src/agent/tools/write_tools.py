from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Any

from agent.audit.logger import get_logger
from agent.config import settings
from agent.data.loader import get_loader
from agent.tools.failures import simulator

logger = get_logger(__name__)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    normalized = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except Exception:
        return None


def _today() -> date:
    return datetime.fromisoformat(f"{settings.policy_reference_date}T00:00:00+00:00").date()


def _has_any(text: str, *phrases: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in phrases)


async def _check_refund_eligibility_impl(order_id: str, *, _skip_fail: bool = False) -> dict[str, Any]:
    """Core refund-eligibility implementation shared by MCP and LangChain tools."""
    try:
        failure = "none"
        if not _skip_fail:
            failure = await simulator.maybe_fail("check_refund_eligibility")

        loader = get_loader()
        order = loader.get_order(order_id)
        if order is None:
            return {
                "success": True,
                "eligible": False,
                "reasons": ["order_not_found"],
                "reason": "order_not_found",
                "amount": 0.0,
                "notes": f"Order {order_id} was not found in the ShopWave system.",
            }

        customer = loader.get_customer_by_id(order["customer_id"])
        if customer is None:
            return {
                "success": True,
                "eligible": False,
                "reasons": ["customer_not_found"],
                "reason": "customer_not_found",
                "amount": 0.0,
                "notes": "Customer record linked to the order was not found.",
            }

        product = loader.get_product(order["product_id"])
        if product is None:
            return {
                "success": True,
                "eligible": False,
                "reasons": ["product_not_found"],
                "reason": "product_not_found",
                "amount": 0.0,
                "notes": "Product metadata linked to the order was not found.",
            }

        notes = str(order.get("notes", "")).lower()
        product_notes = str(product.get("notes", "")).lower()

        if order.get("refund_status") == "refunded":
            result = {
                "success": True,
                "eligible": False,
                "reasons": ["already_refunded"],
                "reason": "already_refunded",
                "amount": 0.0,
                "notes": "This order has already been refunded.",
            }
            if failure == "malformed":
                malformed = dict(result)
                malformed.pop("eligible", None)
                return {**malformed, "warning": "malformed_response"}
            return result

        if order.get("status") == "processing":
            result = {
                "success": True,
                "eligible": False,
                "reasons": ["order_in_processing_cancel_instead"],
                "reason": "order_in_processing_cancel_instead",
                "amount": 0.0,
                "notes": "Order is still processing and should be cancelled instead of refunded.",
            }
            if failure == "malformed":
                malformed = dict(result)
                malformed.pop("eligible", None)
                return {**malformed, "warning": "malformed_response"}
            return result

        ineligibility_reasons: list[str] = []
        eligibility_overrides: list[str] = []
        notes_parts: list[str] = []

        if _has_any(notes, "registered online") or _has_any(product_notes, "registered online"):
            ineligibility_reasons.append("device_registered_online")
            notes_parts.append(
                "This device was registered online after purchase, making it non-returnable per policy."
            )

        actual_deadline_str: str | None = None
        delivery_date_str = str(order.get("delivery_date") or "")
        delivery_date = _parse_date(delivery_date_str)
        window_expired = False
        return_window_days = int(product.get("return_window_days", 30))
        if delivery_date is not None:
            actual_deadline = delivery_date + timedelta(days=return_window_days)
            actual_deadline_str = actual_deadline.isoformat()
            window_expired = _today() > actual_deadline
            logger.debug(
                "return_window_computed",
                order_id=order_id,
                product_id=order.get("product_id"),
                delivery_date=delivery_date_str,
                return_window_days=return_window_days,
                actual_deadline=actual_deadline_str,
                window_expired=window_expired,
            )

            if window_expired:
                tier = str(customer.get("tier", "standard")).lower()
                customer_notes = str(customer.get("notes", "")).lower()
                if tier == "vip" and ("pre-approved" in customer_notes or "exception" in customer_notes):
                    eligibility_overrides.append("vip_preapproved_exception")
                    notes_parts.append(
                        f"Return window ({return_window_days} days) expired on {actual_deadline_str}, "
                        "but VIP pre-approval exception applies."
                    )
                elif tier == "premium":
                    delta_days = (_today() - actual_deadline).days
                    if delta_days <= 3:
                        eligibility_overrides.append("premium_borderline_exception")
                        notes_parts.append(
                            f"Return window expired on {actual_deadline_str}. "
                            "Premium borderline exception (<=3 days) applied."
                        )
                    else:
                        ineligibility_reasons.append("return_window_expired")
                        notes_parts.append(
                            f"The {return_window_days}-day return window expired on {actual_deadline_str}."
                        )
                else:
                    ineligibility_reasons.append("return_window_expired")
                    notes_parts.append(
                        f"The {return_window_days}-day return window expired on {actual_deadline_str}."
                    )

        if _has_any(notes, "damaged on arrival", "arrived with cracked", "manufacturing defect") or _has_any(
            notes, "wrong colour", "wrong color", "wrong size", "wrong item"
        ):
            eligibility_overrides.append("damaged_or_wrong_item")
            notes_parts.append(
                "Damaged-on-arrival or wrong-item override applies regardless of return window."
            )
            ineligibility_reasons = [r for r in ineligibility_reasons if r != "return_window_expired"]

        has_registered_block = "device_registered_online" in ineligibility_reasons
        has_damage_override = "damaged_or_wrong_item" in eligibility_overrides

        if has_registered_block:
            result = {
                "success": True,
                "eligible": False,
                "reasons": ineligibility_reasons,
                "reason": ", ".join(ineligibility_reasons),
                "amount": 0.0,
                "actual_deadline": actual_deadline_str,
                "notes": " ".join(notes_parts),
            }
        elif has_damage_override:
            result = {
                "success": True,
                "eligible": True,
                "reasons": eligibility_overrides,
                "reason": "damaged_or_wrong_item_override",
                "amount": float(order.get("amount", 0.0)),
                "actual_deadline": actual_deadline_str,
                "notes": " ".join(notes_parts),
            }
        elif ineligibility_reasons:
            result = {
                "success": True,
                "eligible": False,
                "reasons": ineligibility_reasons,
                "reason": ", ".join(ineligibility_reasons),
                "amount": 0.0,
                "actual_deadline": actual_deadline_str,
                "notes": " ".join(notes_parts),
            }
        else:
            amount = float(order.get("amount", 0.0))
            restocking_note = ""
            if return_window_days == 15 and delivery_date is not None:
                days_since_delivery = (_today() - delivery_date).days
                if days_since_delivery > 7:
                    restocking_fee = round(amount * 0.10, 2)
                    amount = round(amount - restocking_fee, 2)
                    restocking_note = f" 10% restocking fee applied: -{restocking_fee}"
            reason_list = eligibility_overrides if eligibility_overrides else ["within_policy"]
            result = {
                "success": True,
                "eligible": True,
                "reasons": reason_list,
                "reason": reason_list[0],
                "amount": amount,
                "actual_deadline": actual_deadline_str,
                "notes": (" ".join(notes_parts) + restocking_note).strip(),
            }

        if failure == "malformed":
            malformed = dict(result)
            malformed.pop("eligible", None)
            return {**malformed, "warning": "malformed_response"}
        if failure == "partial":
            return {
                "success": True,
                "reason": result.get("reason"),
                "eligible": bool(result.get("eligible", False)),
                "warning": "partial_response",
            }
        return result
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def check_refund_eligibility(order_id: str, *, _skip_fail: bool = False) -> dict[str, Any]:
    """Check policy eligibility and return structured refund decision."""
    return await _check_refund_eligibility_impl(order_id, _skip_fail=_skip_fail)


async def _issue_refund_impl(order_id: str, amount: float) -> dict[str, Any]:
    """Core issue_refund implementation shared by MCP and LangChain tools."""
    try:
        await simulator.maybe_fail("issue_refund")
        eligibility = await _check_refund_eligibility_impl(order_id, _skip_fail=True)
        if not eligibility.get("success"):
            return {"success": False, "reason": "safety_guard_prevented_refund"}
        if not eligibility.get("eligible"):
            return {"success": False, "reason": "safety_guard_prevented_refund"}
        expected_amount = float(eligibility.get("amount", 0.0))
        if amount > expected_amount + 0.01:
            return {"success": False, "reason": "safety_guard_prevented_refund"}

        loader = get_loader()
        marked = await loader.mark_refunded(order_id, amount)
        if not marked:
            return {"success": False, "reason": "already_refunded_or_invalid_order"}
        return {
            "success": True,
            "refund_id": f"RFD-{uuid.uuid4().hex[:10].upper()}",
            "amount": float(amount),
            "estimated_days": settings.refund_estimated_days,
        }
    except Exception as exc:
        return {"success": False, "reason": str(exc)}


async def issue_refund(order_id: str, amount: float) -> dict[str, Any]:
    """Issue irreversible refund only after internal eligibility re-check."""
    return await _issue_refund_impl(order_id, amount)


async def _send_reply_impl(ticket_id: str, message: str) -> dict[str, Any]:
    """Core send_reply implementation shared by MCP and LangChain tools."""
    try:
        await simulator.maybe_fail("send_reply")
        if not message.strip():
            return {"success": False, "error": "empty_message"}
        loader = get_loader()
        ticket_ids = {t["ticket_id"] for t in loader.tickets}
        if ticket_id not in ticket_ids:
            return {"success": False, "error": "ticket_not_found"}
        return {
            "success": True,
            "ticket_id": ticket_id,
            "message_preview": message.strip()[:100],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def send_reply(ticket_id: str, message: str) -> dict[str, Any]:
    """Simulate customer email response."""
    return await _send_reply_impl(ticket_id, message)


async def _escalate_impl(ticket_id: str, summary: str, priority: str) -> dict[str, Any]:
    """Core escalate implementation shared by MCP and LangChain tools."""
    normalized_priority = priority.lower().strip()
    if normalized_priority not in {"low", "medium", "high", "urgent"}:
        return {"success": False, "error": "invalid_priority"}
    if not summary.strip():
        return {"success": False, "error": "empty_summary"}
    return {
        "success": True,
        "escalation_id": f"ESC-{uuid.uuid4().hex[:10].upper()}",
        "assigned_to": settings.escalation_assigned_to,
        "priority": normalized_priority,
    }


async def escalate(ticket_id: str, summary: str, priority: str) -> dict[str, Any]:
    """Escalate ticket to human queue. This safety net must always succeed."""
    return await _escalate_impl(ticket_id, summary, priority)

