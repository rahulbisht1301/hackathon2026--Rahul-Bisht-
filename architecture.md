# ShopWave Agent Architecture

```mermaid
flowchart TD
    A[tickets.json] --> B[main.py]
    C[customers/orders/products] --> B
    D[knowledge-base.md] --> E[vector_store.py]
    E --> F[ChromaDB]

    B --> G[LangGraph StateGraph]
    G --> N1[classify_ticket]
    N1 --> N2[reason_and_act]
    N2 --> R{route_resolution}
    R -->|resolve| N3[resolve_ticket]
    R -->|escalate| N4[escalate_ticket]
    N3 --> N5[audit_and_end]
    N4 --> N5
    N5 --> H[audit_log.json]

    N2 --> T[FastMCP Tools]
    T --> T1[get_customer]
    T --> T2[get_order]
    T --> T3[get_product]
    T --> T4[search_knowledge_base]
    T --> T5[check_refund_eligibility]
    T --> T6[issue_refund]
    T --> T7[send_reply]
    T --> T8[escalate]

    B --> P[(PostgreSQL Checkpointer)]
    G <--> P
```

## Notes

- Ticket workers run concurrently with `asyncio.gather` + `Semaphore`.
- Each graph run uses a unique `thread_id` for checkpoint continuity.
- Every tool call attempt is audit-tracked with retries, duration, and error detail.

