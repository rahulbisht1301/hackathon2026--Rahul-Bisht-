from __future__ import annotations

from typing import Any

from agent.config import settings
from agent.data.loader import get_loader
from agent.data.vector_store import search_knowledge_base as kb_search
from agent.tools.failures import simulator


async def get_order(order_id: str) -> dict[str, Any]:
    """Return full order record by order_id."""
    try:
        failure = await simulator.maybe_fail("get_order")
        loader = get_loader()
        order = loader.get_order(order_id)
        if order is None:
            return {"success": False, "error": "order_not_found"}
        if failure == "malformed":
            malformed = dict(order)
            malformed.pop("delivery_date", None)
            return {"success": True, "order": malformed, "warning": "malformed_response"}
        if failure == "partial":
            return {
                "success": True,
                "order": {"order_id": order["order_id"], "status": order["status"]},
                "warning": "partial_response",
            }
        return {"success": True, "order": order}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def get_customer(email: str) -> dict[str, Any]:
    """Return customer record by registered email. Tier here is authoritative."""
    try:
        failure = await simulator.maybe_fail("get_customer")
        loader = get_loader()
        customer = loader.get_customer_by_email(email)
        if customer is None:
            return {"success": False, "error": "customer_not_found"}
        if failure == "partial":
            return {
                "success": True,
                "customer": {"name": customer["name"], "tier": customer["tier"]},
                "warning": "partial_response",
            }
        return {"success": True, "customer": customer}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def get_product(product_id: str) -> dict[str, Any]:
    """Return product metadata used for return and warranty decisions."""
    try:
        failure = await simulator.maybe_fail("get_product")
        loader = get_loader()
        product = loader.get_product(product_id)
        if product is None:
            return {"success": False, "error": "product_not_found"}
        if failure == "malformed":
            malformed = dict(product)
            malformed["return_window_days"] = str(malformed.get("return_window_days"))
            return {"success": True, "product": malformed, "warning": "malformed_response"}
        return {"success": True, "product": product}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def search_knowledge_base(query: str) -> dict[str, Any]:
    """Semantic policy search over knowledge-base.md chunks (top 3 results)."""
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

