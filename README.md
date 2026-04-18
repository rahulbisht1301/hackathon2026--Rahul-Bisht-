# ShopWave Autonomous Support Resolution Agent

Production-grade LangGraph agent that ingests 20 support tickets, triages them concurrently, executes multi-tool resolution paths, escalates unresolved/fraud/warranty cases, and writes a complete structured `audit_log.json` for every run.

## Architecture Overview

The system is a stateful LangGraph pipeline with five core stages: classify, reason-and-act, route, resolve/escalate, and audit. Tools are exposed through FastMCP and called from graph nodes. Persistence uses PostgreSQL checkpointers; policy retrieval uses ChromaDB over `knowledge-base.md`.  
See `architecture.md` (source) and `architecture.pdf` (rendered artifact).

## Prerequisites

- Docker + Docker Compose
- Python 3.11+ (for local run)
- Poetry
- Gemini API key (`GEMINI_API_KEY`)

## Quick Start

```bash
git clone <your-repo-url> && cd hackathon2026-shopwave
cp .env.example .env
docker compose up --build --abort-on-container-exit
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | required |
| `LANGSMITH_API_KEY` | LangSmith API key | empty |
| `LANGSMITH_TRACING_V2` | Enable LangSmith tracing | `true` |
| `LANGSMITH_PROJECT` | LangSmith project name | `shopwave-agent` |
| `POSTGRES_HOST` | Postgres host | `postgres` |
| `POSTGRES_PORT` | Postgres port | `5432` |
| `POSTGRES_DB` | Postgres database | `shopwave_agent` |
| `POSTGRES_USER` | Postgres user | `shopwave` |
| `POSTGRES_PASSWORD` | Postgres password | required |
| `CHROMA_PERSIST_DIR` | Chroma persistence directory | `./chroma_db` |
| `CHROMA_COLLECTION_NAME` | Chroma collection name | `shopwave_kb` |
| `LLM_MODEL` | Gemini model identifier | `models/gemini-2.5-flash` |
| `LLM_TEMPERATURE` | LLM temperature | `0.0` |
| `LLM_MAX_TOKENS` | Max completion tokens | `2048` |
| `AGENT_MAX_ITERATIONS` | Iteration ceiling per ticket | `15` |
| `AGENT_MIN_TOOL_CALLS` | Minimum tool calls enforced per ticket | `3` |
| `AGENT_CONFIDENCE_THRESHOLD` | Escalation confidence floor | `0.6` |
| `AGENT_CONCURRENCY_LIMIT` | Max parallel ticket workers | `5` |
| `TOOL_FAILURE_RATE` | Tool failure injection probability | `0.15` |
| `TOOL_TIMEOUT_SECONDS` | Simulated timeout threshold | `3.0` |
| `TOOL_FAILURE_SEED` | RNG seed for deterministic failures | `7` |
| `TOOL_MAX_RETRIES` | Max retries for retryable tool failures | `3` |
| `TOOL_RETRY_DELAYS` | Retry delays in seconds (CSV) | `1.0,2.0,4.0` |
| `POLICY_REFERENCE_DATE` | Policy decision date for deterministic tests | `2024-03-15` |
| `KB_TOP_K` | Number of KB chunks retrieved per query | `3` |
| `REFUND_ESTIMATED_DAYS` | Default refund settlement estimate | `7` |
| `ESCALATION_ASSIGNED_TO` | Escalation queue identifier | `human_support_queue` |
| `DATA_DIR` | Input data directory | `./data` |
| `AUDIT_LOG_PATH` | Output path for audit log | `./audit_log.json` |
| `LOG_LEVEL` | Structlog level | `INFO` |
| `LOG_FORMAT` | `json` or `text` | `json` |

## Running Locally (No Docker)

```bash
poetry install
cp .env.example .env
PYTHONPATH=src poetry run python -m agent.main
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Orchestration | LangGraph |
| Model | Google Gemini via `langchain-google-genai` |
| Tooling | FastMCP |
| Persistence | PostgreSQL checkpointer (`AsyncPostgresSaver`) |
| Retrieval | ChromaDB + sentence-transformers |
| Logging | structlog |
| Config | pydantic-settings |
| Packaging | Poetry |
| Testing | pytest + pytest-asyncio |
| Containers | Docker + Docker Compose |

## How the Agent Works

1. **Classify** ticket category, urgency, fraud, and confidence.
2. **Reason + Act** with enforced multi-tool chain and retries.
3. **Route** by fraud/confidence threshold.
4. **Resolve or Escalate** with customer-safe responses.
5. **Audit** every tool call, retries, outcomes, timing, and errors.

## Tool Inventory (8 Tools)

| Tool | Type | Purpose |
|---|---|---|
| `get_order` | Read | Fetch order data by `order_id` |
| `get_customer` | Read | Fetch customer data by email (authoritative tier) |
| `get_product` | Read | Fetch product return/warranty metadata |
| `search_knowledge_base` | Read | Semantic policy retrieval from Chroma |
| `check_refund_eligibility` | Write-safe | Enforce policy before refund |
| `issue_refund` | Write | Irreversible refund with internal safety guard |
| `send_reply` | Write | Customer email simulation |
| `escalate` | Write | Human escalation record creation |

## Failure Handling Strategy

- Tool failures are injected (`timeout`, `transient`, `malformed`, `partial`) via `failures.py`.
- Retry policy: 3 attempts with exponential backoff.
- Malformed/partial payloads are logged and handled conservatively.
- Exhausted retries route tickets to escalation with structured context.

## Audit Log Format

`audit_log.json` contains:

- `run_metadata`: run-level counts, timestamps, model, failure metrics.
- `tickets`: one complete `AuditEntry` per ticket, including full tool call trace.

## LangSmith Integration

1. Set `LANGSMITH_API_KEY`.
2. Set `LANGSMITH_TRACING_V2=true`.
3. Set `LANGSMITH_PROJECT` (default `shopwave-agent`).
4. Run the pipeline and inspect traces in LangSmith.

## Running Tests

```bash
poetry run pytest tests/ -v --asyncio-mode=auto
```

## Design Decisions and Tradeoffs

- **Deterministic policy reference date** for reproducible test outcomes.
- **Defense-in-depth**: `issue_refund` rechecks eligibility internally.
- **Graceful degradation**: retrieval falls back to lexical scoring if Chroma/model runtime is unavailable.
- **Concurrency-first architecture** with semaphore-bounded workers for stable throughput.
- **Centralized configuration**: operational knobs are controlled through `.env` + `src/agent/config.py`.

