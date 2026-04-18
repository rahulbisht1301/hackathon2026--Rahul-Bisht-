import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.data.loader import init_loader
from agent.data.vector_store import init_vector_store


async def main() -> None:
    init_loader("./data")
    await init_vector_store()
    print("Chroma vector store initialized.")


if __name__ == "__main__":
    asyncio.run(main())

