import asyncio
import json
from pathlib import Path
from typing import Any


class DataLoader:
    """In-memory indexed data store for customers, orders, products, and tickets."""

    def __init__(self, data_dir: str):
        self.customers: dict[str, dict[str, Any]] = {}
        self.customers_by_email: dict[str, dict[str, Any]] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.products: dict[str, dict[str, Any]] = {}
        self.tickets: list[dict[str, Any]] = []
        self._refund_lock = asyncio.Lock()
        self._load(data_dir)

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load(self, data_dir: str) -> None:
        base = Path(data_dir)
        customers = self._load_json(base / "customers.json")
        orders = self._load_json(base / "orders.json")
        products = self._load_json(base / "products.json")
        tickets = self._load_json(base / "tickets.json")

        self.customers = {c["customer_id"]: c for c in customers}
        self.customers_by_email = {c["email"].lower(): c for c in customers}
        self.orders = {o["order_id"]: o for o in orders}
        self.products = {p["product_id"]: p for p in products}
        self.tickets = tickets

    def get_customer_by_email(self, email: str) -> dict[str, Any] | None:
        return self.customers_by_email.get(email.lower())

    def get_customer_by_id(self, customer_id: str) -> dict[str, Any] | None:
        return self.customers.get(customer_id)

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        return self.orders.get(order_id)

    def get_product(self, product_id: str) -> dict[str, Any] | None:
        return self.products.get(product_id)

    def find_latest_order_for_customer(self, customer_id: str) -> dict[str, Any] | None:
        customer_orders = [o for o in self.orders.values() if o["customer_id"] == customer_id]
        if not customer_orders:
            return None
        return sorted(customer_orders, key=lambda x: x.get("order_date", ""), reverse=True)[0]

    async def mark_refunded(self, order_id: str, amount: float) -> bool:
        async with self._refund_lock:
            order = self.orders.get(order_id)
            if order is None:
                return False
            if order.get("refund_status") == "refunded":
                return False
            if float(order.get("amount", 0.0)) + 0.01 < amount:
                return False
            order["refund_status"] = "refunded"
            return True


_loader: DataLoader | None = None


def get_loader() -> DataLoader:
    if _loader is None:
        raise RuntimeError("DataLoader not initialized. Call init_loader() first.")
    return _loader


def init_loader(data_dir: str) -> DataLoader:
    global _loader
    _loader = DataLoader(data_dir)
    return _loader

