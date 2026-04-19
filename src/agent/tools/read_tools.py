from __future__ import annotations

from typing import Any

from agent.config import settings
from agent.data.loader import get_loader
from agent.data.vector_store import search_knowledge_base as kb_search
from agent.tools.failures import simulator


async def _get_order_impl(order_id: str) -> dict[str, Any]:
    """Core get_order implementation shared by MCP and LangChain tools."""
    try:
        failure = await simulator.maybe_fail("get_order")
        loader = get_loader()
        order = loader.get_order(order_id)
        if order is None:
            return {
                "success": False,
                "found": False,
                "order_id": order_id,
                "error": "order_not_found",
                "message": (
                    f"No order with ID '{order_id}' exists in the ShopWave system. "
                    "Please verify the order ID from the confirmation email."
                ),
            }
        if failure == "malformed":
            malformed = dict(order)
            malformed.pop("delivery_date", None)
            return {
                "success": True,
                "found": True,
                "order": malformed,
                **malformed,
                "warning": "malformed_response",
            }
        if failure == "partial":
            return {
                "success": True,
                "found": True,
                "order": {"order_id": order["order_id"], "status": order["status"]},
                "order_id": order["order_id"],
                "status": order["status"],
                "warning": "partial_response",
            }
        return {"success": True, "found": True, "order": order, **order}
    except Exception as exc:
        return {"success": False, "found": False, "order_id": order_id, "error": str(exc)}


async def get_order(order_id: str) -> dict[str, Any]:
    """Return full order record by order_id."""
    return await _get_order_impl(order_id)


async def _get_customer_impl(email: str) -> dict[str, Any]:
    """Core get_customer implementation shared by MCP and LangChain tools."""
    try:
        failure = await simulator.maybe_fail("get_customer")
        loader = get_loader()
        customer = loader.get_customer_by_email(email)
        if customer is None:
            return {"success": False, "found": False, "email": email, "error": "customer_not_found"}
        if failure == "partial":
            return {
                "success": True,
                "found": True,
                "customer": {"name": customer["name"], "tier": customer["tier"]},
                "name": customer["name"],
                "tier": customer["tier"],
                "warning": "partial_response",
            }
        return {"success": True, "found": True, "customer": customer, **customer}
    except Exception as exc:
        return {"success": False, "found": False, "email": email, "error": str(exc)}


async def get_customer(email: str) -> dict[str, Any]:
    """Return customer record by registered email. Tier here is authoritative."""
    return await _get_customer_impl(email)


async def _get_product_impl(product_id: str) -> dict[str, Any]:
    """Core get_product implementation shared by MCP and LangChain tools."""
    try:
        failure = await simulator.maybe_fail("get_product")
        loader = get_loader()
        product = loader.get_product(product_id)
        if product is None:
            return {"success": False, "found": False, "product_id": product_id, "error": "product_not_found"}
        if failure == "malformed":
            malformed = dict(product)
            malformed["return_window_days"] = str(malformed.get("return_window_days"))
            return {
                "success": True,
                "found": True,
                "product": malformed,
                **malformed,
                "warning": "malformed_response",
            }
        return {"success": True, "found": True, "product": product, **product}
    except Exception as exc:
        return {"success": False, "found": False, "product_id": product_id, "error": str(exc)}


async def get_product(product_id: str) -> dict[str, Any]:
    """Return product metadata used for return and warranty decisions."""
    return await _get_product_impl(product_id)


async def _search_knowledge_base_impl(query: str) -> dict[str, Any]:
    """Core knowledge-base search implementation shared by MCP and LangChain tools."""
    try:
        failure = await simulator.maybe_fail("search_knowledge_base")
        payload = await kb_search(query)
        if not payload.get("success"):
            return payload
        results = payload.get("results", [])
        top_k = max(1, settings.kb_top_k)
        if failure == "partial":
            results = results[:1]
            return {"success": True, "results": results, "warning": "partial_response"}
        return {"success": True, "results": results[:top_k]}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def search_knowledge_base(query: str) -> dict[str, Any]:
    """Semantic policy search over knowledge-base.md chunks (top 3 results)."""
    return await _search_knowledge_base_impl(query)

