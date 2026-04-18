from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from agent.config import settings
from agent.data.loader import get_loader
from agent.tools.failures import simulator


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _today() -> date:
    return datetime.fromisoformat(f"{settings.policy_reference_date}T00:00:00+00:00").date()


def _evaluate_refund(order: dict[str, Any], customer: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
    notes = str(order.get("notes", "")).lower()
    if order.get("refund_status") == "refunded":
        return {"eligible": False, "reason": "already_refunded", "amount": 0.0, "notes": ""}

    if order.get("status") == "processing":
        return {
            "eligible": False,
            "reason": "order_in_processing_cancel_instead",
            "amount": 0.0,
            "notes": "Order is still processing and should be cancelled instead of refunded.",
        }

    if "registered online" in notes:
        return {
            "eligible": False,
            "reason": "registered_online_non_returnable",
            "amount": 0.0,
            "notes": "Item was registered online after purchase and is non-returnable.",
        }

    if "damaged on arrival" in notes or "arrived with cracked" in notes:
        return {
            "eligible": True,
            "reason": "damaged_on_arrival",
            "amount": float(order["amount"]),
            "notes": "Damaged-on-arrival override applies regardless of return window.",
        }

    if "wrong colour" in notes or "wrong size" in notes or "wrong item" in notes:
        return {
            "eligible": True,
            "reason": "wrong_item_delivered",
            "amount": float(order["amount"]),
            "notes": "Wrong-item override applies regardless of return window.",
        }

    deadline = _parse_date(order.get("return_deadline"))
    today = _today()
    tier = str(customer.get("tier", "standard")).lower()
    if deadline and today > deadline:
        if tier == "vip" and "pre-approved extended return exception" in str(customer.get("notes", "")).lower():
            return {
                "eligible": True,
                "reason": "vip_exception",
                "amount": float(order["amount"]),
                "notes": "VIP pre-approved extended return exception applied.",
            }
        if tier == "premium":
            delta = (today - deadline).days
            if delta <= 3:
                return {
                    "eligible": True,
                    "reason": "premium_borderline_exception",
                    "amount": float(order["amount"]),
                    "notes": "Premium borderline exception (<=3 days outside window).",
                }
        return {
            "eligible": False,
            "reason": "return_window_expired",
            "amount": 0.0,
            "notes": "Return window is expired for this order.",
        }

    extra_note = ""
    if int(product.get("return_window_days", 30)) == 15:
        extra_note = "10% restocking fee may apply for high-value electronics outside 7 days."
    return {
        "eligible": True,
        "reason": "within_policy",
        "amount": float(order["amount"]),
        "notes": extra_note,
    }


async def check_refund_eligibility(order_id: str, *, _skip_fail: bool = False) -> dict[str, Any]:
    """Check policy eligibility and return structured refund decision."""
    try:
        failure = "none"
        if not _skip_fail:
            failure = await simulator.maybe_fail("check_refund_eligibility")

        loader = get_loader()
        order = loader.get_order(order_id)
        if order is None:
            return {"success": False, "error": "order_not_found"}

        customer = loader.get_customer_by_id(order["customer_id"])
        if customer is None:
            return {"success": False, "error": "customer_not_found"}

        product = loader.get_product(order["product_id"])
        if product is None:
            return {"success": False, "error": "product_not_found"}

        result = _evaluate_refund(order, customer, product)
        if failure == "malformed":
            malformed = dict(result)
            malformed.pop("eligible", None)
            return {"success": True, **malformed, "warning": "malformed_response"}
        return {"success": True, **result}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def issue_refund(order_id: str, amount: float) -> dict[str, Any]:
    """Issue irreversible refund only after internal eligibility re-check."""
    try:
        await simulator.maybe_fail("issue_refund")
        eligibility = await check_refund_eligibility(order_id, _skip_fail=True)
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


async def send_reply(ticket_id: str, message: str) -> dict[str, Any]:
    """Simulate customer email response."""
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


async def escalate(ticket_id: str, summary: str, priority: str) -> dict[str, Any]:
    """Escalate ticket to human queue. This safety net must always succeed."""
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

