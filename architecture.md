# ShopWave Agent Architecture (Detailed)

## 1. System/runtime architecture

```mermaid
flowchart LR
    subgraph Inputs["Input datasets (loaded at startup)"]
        I1["data/tickets.json"]
        I2["data/customers.json"]
        I3["data/orders.json"]
        I4["data/products.json"]
        I5["data/knowledge-base.md"]
    end

    subgraph Boot["Bootstrap in src/agent/main.py"]
        B1["configure_logging()"]
        B2["init_loader(DATA_DIR)"]
        B3["init_vector_store()"]
        B4{"Postgres reachable?"}
        B5["AsyncPostgresSaver\n(langgraph-checkpoint-postgres)"]
        B6["InMemorySaver fallback"]
        B7["build_graph(checkpointer)"]
        B8["Create ticket tasks\n(asyncio + Semaphore)"]
    end

    subgraph Runtime["Concurrent ticket runtime"]
        R1["process_ticket(ticket)\nthread_id=ticket_id+uuid"]
        R2["graph.ainvoke(initial_state)"]
        R3["collect per-ticket final state"]
    end

    subgraph Outputs["Run artifacts"]
        O1["audit_log.json\n(AuditWriter)"]
        O2["output/run_report.json\n(write_run_report)"]
        O3["structlog output\n(json/text)"]
    end

    I1 --> B2
    I2 --> B2
    I3 --> B2
    I4 --> B2
    I5 --> B3

    B1 --> B2 --> B3 --> B4
    B4 -->|yes| B5 --> B7
    B4 -->|no| B6 --> B7
    B7 --> B8 --> R1 --> R2 --> R3
    R3 --> O1
    R3 --> O2
    B1 --> O3
```

## 2. Per-ticket LangGraph flow

```mermaid
flowchart TD
    S["Initial TicketState\n(ticket fields + planner/audit fields)"] --> C["classify_ticket"]
    C --> P["plan_ticket"]
    P --> A["reason_and_act"]
    A --> G{"route_resolution()"}
    G -->|resolve| R["resolve_ticket"]
    G -->|escalate| E["escalate_ticket"]
    R --> W["write_audit_entry"]
    E --> W
    W --> End["END"]

    subgraph ReAct["reason_and_act internals"]
        RA1["Invoke LangGraph ReAct agent\n(ChatGoogleGenerativeAI + tools)"]
        RA2["Parse tool call/tool message pairs"]
        RA3["Enforce min tool calls\n(AGENT_MIN_TOOL_CALLS)"]
        RA4["Apply policy gates:\nfraud, warranty/replacement,\nmissing required tools,\nunsafe refund sequence,\nlow confidence,\nretry exhaustion"]
        RA5["Emit:\nresolution_action,\nconfidence,\nescalation_reason_code,\ncustomer_reply,\nfull tool log"]
        RA1 --> RA2 --> RA3 --> RA4 --> RA5
    end

    A -. executes .-> RA1
```

## 3. Tool and policy/data architecture

```mermaid
flowchart LR
    subgraph AgentTools["LangChain + MCP tool surface"]
        T1["get_customer"]
        T2["get_order"]
        T3["get_product"]
        T4["search_knowledge_base"]
        T5["check_refund_eligibility"]
        T6["issue_refund"]
        T7["send_reply"]
        T8["escalate"]
    end

    subgraph CoreData["In-memory data layer"]
        D1["DataLoader\n(customers/orders/products/tickets indexes)"]
        D2["mark_refunded() lock\n(asyncio.Lock)"]
    end

    subgraph KB["Policy retrieval layer"]
        K1["knowledge-base.md section splitter"]
        K2["Chroma PersistentClient\n(all-MiniLM-L6-v2 embeddings)"]
        K3["Lexical fallback search\n(if Chroma unavailable)"]
    end

    subgraph Reliability["Reliability controls"]
        F1["ToolFailureSimulator\n(timeout/transient/malformed/partial)"]
        F2["Retry loop\n(TOOL_MAX_RETRIES + TOOL_RETRY_DELAYS)"]
        F3["Refund safety guard:\nissue_refund re-checks eligibility"]
    end

    T1 --> D1
    T2 --> D1
    T3 --> D1
    T5 --> D1
    T6 --> D2
    T4 --> K1 --> K2
    T4 --> K3

    F1 --> T1
    F1 --> T2
    F1 --> T3
    F1 --> T4
    F1 --> T5
    F1 --> T6
    F1 --> T7
    F2 --> T1
    F2 --> T2
    F2 --> T3
    F2 --> T4
    F2 --> T5
    F3 --> T6
```

## 4. Key architecture behaviors captured by this design

- **Concurrency:** all tickets are processed concurrently with a semaphore cap (`AGENT_CONCURRENCY_LIMIT`).
- **Durability:** graph state is checkpointed in PostgreSQL when available, with in-memory fallback.
- **Policy-first planning:** `plan_ticket` derives expected tool chain and escalation intent before ReAct execution.
- **Defense-in-depth:** refund issuance is guarded both by planning rules and internal `issue_refund` eligibility re-check.
- **Traceability:** every tool call attempt (including retries/failures) is recorded and emitted to `audit_log.json`.

