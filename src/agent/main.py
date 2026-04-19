from __future__ import annotations

import asyncio
import signal
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
try:
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver  # type: ignore

from agent.audit.audit_log import AuditWriter
from agent.audit.run_report import write_run_report
from agent.audit.logger import configure_logging, get_logger
from agent.config import settings
from agent.data.loader import init_loader
from agent.data.vector_store import init_vector_store
from agent.graph.builder import build_graph

logger = get_logger(__name__)


async def process_ticket(graph, ticket: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    async with semaphore:
        thread_id = f"{ticket['ticket_id']}-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "ticket_id": ticket["ticket_id"],
            "ticket_email": ticket["customer_email"],
            "ticket_subject": ticket["subject"],
            "ticket_body": ticket["body"],
            "ticket_source": ticket["source"],
            "ticket_created_at": ticket["created_at"],
            "ticket_tier": int(ticket.get("tier", 0)),
            "expected_action": str(ticket.get("expected_action", "")),
            "messages": [],
            "tool_calls": [],
            "iterations": 0,
            "error": "",
            "status": "processing",
            "fraud_flag": False,
            "fraud_notes": "",
            "confidence_score": 0.0,
            "confidence_reason": "",
            "escalation_reason_code": "",
            "planned_target_action": "",
            "planned_required_tools": [],
            "planned_must_escalate": False,
            "planned_rationale": "",
            "planned_expected_outcome": "",
            "planned_escalation_priority": "medium",
            "planned_kb_evidence": [],
            "resolvable": True,
            "processing_started_at": datetime.now(timezone.utc).isoformat(),
            "errors_encountered": [],
        }
        try:
            started = datetime.now(timezone.utc)
            final_state = await graph.ainvoke(initial_state, config=config)
            completed = datetime.now(timezone.utc)
            duration_ms = (completed - started).total_seconds() * 1000
            logger.info(
                "ticket_processed",
                ticket_id=ticket["ticket_id"],
                status=final_state.get("status"),
                action=final_state.get("resolution_action"),
                duration_ms=round(duration_ms),
            )
            return final_state
        except Exception as exc:
            logger.error("ticket_failed", ticket_id=ticket["ticket_id"], error=str(exc))
            return {**initial_state, "status": "failed", "error": str(exc)}


async def _build_graph_with_checkpointer():
    try:
        conn = await psycopg.AsyncConnection.connect(
            settings.postgres_sync_dsn,
            autocommit=True,
        )
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()
        graph = await build_graph(checkpointer)
        return graph, conn
    except Exception as exc:
        logger.warning("postgres_unavailable_fallback_inmemory", error=str(exc))
        checkpointer = InMemorySaver()
        return await build_graph(checkpointer), None


async def main() -> int:
    configure_logging()
    stop_event = asyncio.Event()

    def _shutdown_handler(*_: object) -> None:
        logger.info("shutdown_signal_received")
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    logger.info(
        "shopwave_agent_starting",
        model=settings.llm_model,
        concurrency=settings.agent_concurrency_limit,
        failure_rate=settings.tool_failure_rate,
    )

    loader = init_loader(settings.data_dir)
    await init_vector_store()
    graph, pg_conn = await _build_graph_with_checkpointer()
    tickets = loader.tickets

    semaphore = asyncio.Semaphore(settings.agent_concurrency_limit)
    run_started = datetime.now(timezone.utc)
    tasks = [asyncio.create_task(process_ticket(graph, ticket, semaphore)) for ticket in tickets]
    if stop_event.is_set():
        for task in tasks:
            task.cancel()
        results: list[dict[str, Any]] = []
    else:
        results = await asyncio.gather(*tasks, return_exceptions=False)

    run_completed = datetime.now(timezone.utc)
    writer = AuditWriter(settings.audit_log_path)
    writer.write(results, run_started, run_completed)
    report = write_run_report(settings.run_report_path, results, run_started, run_completed)

    resolved = sum(1 for r in results if r.get("status") == "resolved")
    escalated = sum(1 for r in results if r.get("status") == "escalated")
    failed = sum(1 for r in results if r.get("status") == "failed")
    logger.info(
        "run_report_written",
        run_report_path=settings.run_report_path,
        total=report["total_tickets"],
        resolved=report["resolved"],
        escalated=report["escalated"],
        failed=report["failed"],
    )
    logger.info(
        "run_complete",
        total=len(results),
        resolved=resolved,
        escalated=escalated,
        failed=failed,
        audit_log_path=settings.audit_log_path,
        run_report_path=settings.run_report_path,
        duration_seconds=round((run_completed - run_started).total_seconds(), 1),
    )
    if pg_conn is not None:
        await pg_conn.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

