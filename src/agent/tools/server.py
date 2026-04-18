from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP
except Exception:
    from fastmcp import FastMCP  # type: ignore

from agent.tools.read_tools import (
    get_customer as _get_customer,
    get_order as _get_order,
    get_product as _get_product,
    search_knowledge_base as _search_knowledge_base,
)
from agent.tools.write_tools import (
    check_refund_eligibility as _check_refund_eligibility,
    escalate as _escalate,
    issue_refund as _issue_refund,
    send_reply as _send_reply,
)

mcp = FastMCP("shopwave-tools")


@mcp.tool()
async def get_order(order_id: str):
    """Fetch complete order record by order id."""
    return await _get_order(order_id)


@mcp.tool()
async def get_customer(email: str):
    """Fetch complete customer profile by registered email."""
    return await _get_customer(email)


@mcp.tool()
async def get_product(product_id: str):
    """Fetch product policy metadata including warranty and return window."""
    return await _get_product(product_id)


@mcp.tool()
async def search_knowledge_base(query: str):
    """Search policy knowledge base semantically and return top relevant passages."""
    return await _search_knowledge_base(query)


@mcp.tool()
async def check_refund_eligibility(order_id: str):
    """Validate refund eligibility before any irreversible refund action."""
    return await _check_refund_eligibility(order_id)


@mcp.tool()
async def issue_refund(order_id: str, amount: float):
    """Issue refund after defense-in-depth internal eligibility guard."""
    return await _issue_refund(order_id, amount)


@mcp.tool()
async def send_reply(ticket_id: str, message: str):
    """Send customer-facing support reply."""
    return await _send_reply(ticket_id, message)


@mcp.tool()
async def escalate(ticket_id: str, summary: str, priority: str):
    """Escalate ticket to human support team with structured summary and priority."""
    return await _escalate(ticket_id, summary, priority)

