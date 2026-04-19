from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from agent.config import settings
from agent.tools.read_tools import (
    _get_customer_impl,
    _get_order_impl,
    _get_product_impl,
    _search_knowledge_base_impl,
)
from agent.tools.write_tools import (
    _check_refund_eligibility_impl,
    _escalate_impl,
    _issue_refund_impl,
    _send_reply_impl,
)


def _is_retryable_error(result: dict) -> bool:
    message = str(result.get("error") or result.get("reason") or "").lower()
    return any(marker in message for marker in ("timed out", "timeout", "transient", "503"))


async def _invoke_with_retries(tool_name: str, operation):
    max_retries = max(1, settings.tool_max_retries)
    retry_delays = settings.tool_retry_delays or [1.0]
    last_result: dict = {"success": False, "error": f"{tool_name} failed"}

    for attempt in range(1, max_retries + 1):
        result = await operation()
        if not isinstance(result, dict):
            last_result = {"success": False, "error": f"{tool_name} returned non-dict response"}
            break

        last_result = result
        if bool(result.get("success", False)):
            return result

        if attempt < max_retries and _is_retryable_error(result):
            delay_index = min(attempt - 1, len(retry_delays) - 1)
            await asyncio.sleep(float(retry_delays[delay_index]))
            continue
        break

    return last_result


@tool
async def get_order(order_id: str) -> dict:
    """Look up a ShopWave order by order ID (ORD-XXXX). Returns found/order status and metadata."""
    return await _invoke_with_retries("get_order", lambda: _get_order_impl(order_id))


@tool
async def get_customer(email: str) -> dict:
    """Look up a customer by registered email. Tier from this tool is authoritative and must be trusted."""
    return await _invoke_with_retries("get_customer", lambda: _get_customer_impl(email))


@tool
async def get_product(product_id: str) -> dict:
    """Look up product policy metadata. Use return_window_days for deadline calculation, not stored return_deadline."""
    return await _invoke_with_retries("get_product", lambda: _get_product_impl(product_id))


@tool
async def search_knowledge_base(query: str) -> dict:
    """Semantic search over the policy knowledge base. Use before applying policy decisions."""
    return await _invoke_with_retries("search_knowledge_base", lambda: _search_knowledge_base_impl(query))


@tool
async def check_refund_eligibility(order_id: str) -> dict:
    """Validate refund eligibility and reasons list. Must run before issue_refund."""
    return await _invoke_with_retries(
        "check_refund_eligibility",
        lambda: _check_refund_eligibility_impl(order_id),
    )


@tool
async def issue_refund(order_id: str, amount: float) -> dict:
    """Issue irreversible refund only after eligibility confirmation."""
    return await _issue_refund_impl(order_id, amount)


@tool
async def send_reply(ticket_id: str, message: str) -> dict:
    """Send customer-facing reply for resolution, clarification, or escalation acknowledgement."""
    return await _send_reply_impl(ticket_id, message)


@tool
async def escalate(ticket_id: str, summary: str, priority: str) -> dict:
    """Escalate to human support. Priority must be low/medium/high/urgent."""
    return await _escalate_impl(ticket_id, summary, priority)

