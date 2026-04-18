import asyncio
import sys
from pathlib import Path

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.config import settings


async def main() -> None:
    async with await psycopg.AsyncConnection.connect(
        settings.postgres_sync_dsn,
        autocommit=True,
    ) as conn:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        print("PostgreSQL checkpoint tables initialized.")


if __name__ == "__main__":
    asyncio.run(main())

