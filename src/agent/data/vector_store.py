from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from agent.config import settings

_sections: list[dict[str, Any]] = []
_chroma_collection: Any | None = None


def _split_markdown_sections(markdown_text: str) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    current_section = "General"
    current_lines: list[str] = []
    for line in markdown_text.splitlines():
        if line.startswith("## "):
            if current_lines:
                chunks.append(
                    {
                        "section": current_section,
                        "text": f"{current_section}\n" + "\n".join(current_lines).strip(),
                    }
                )
            current_section = line.replace("## ", "", 1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        chunks.append(
            {"section": current_section, "text": f"{current_section}\n" + "\n".join(current_lines).strip()}
        )
    return [c for c in chunks if c["text"].strip()]


async def init_vector_store() -> None:
    global _sections, _chroma_collection
    kb_path = Path(settings.data_dir) / "knowledge-base.md"
    markdown_text = kb_path.read_text(encoding="utf-8")
    _sections = _split_markdown_sections(markdown_text)

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            embedding_function=embedding_fn,
        )
        existing_count = collection.count()
        if existing_count == 0:
            collection.add(
                ids=[f"kb-{i}" for i in range(len(_sections))],
                documents=[s["text"] for s in _sections],
                metadatas=[{"section": s["section"], "source": "knowledge-base.md"} for s in _sections],
            )
        _chroma_collection = collection
    except Exception:
        _chroma_collection = None


def _tokenize(text: str) -> set[str]:
    clean = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {t for t in clean.split() if t}


def _lexical_search(query: str, k: int = 3) -> list[dict[str, Any]]:
    query_tokens = _tokenize(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    for section in _sections:
        section_tokens = _tokenize(section["text"])
        if not query_tokens or not section_tokens:
            score = 0.0
        else:
            overlap = len(query_tokens & section_tokens)
            score = overlap / max(math.sqrt(len(query_tokens) * len(section_tokens)), 1.0)
        scored.append((score, section))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]
    return [
        {"text": item["text"], "section": item["section"], "score": round(float(score), 4)}
        for score, item in top
    ]


async def search_knowledge_base(query: str) -> dict[str, Any]:
    top_k = max(1, settings.kb_top_k)
    if _chroma_collection is not None:
        try:
            results = _chroma_collection.query(query_texts=[query], n_results=top_k)
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            payload = []
            for i, text in enumerate(docs):
                metadata = metas[i] if i < len(metas) else {}
                distance = float(distances[i]) if i < len(distances) else 1.0
                payload.append(
                    {
                        "text": text,
                        "section": metadata.get("section", "Unknown"),
                        "score": round(1.0 / (1.0 + distance), 4),
                    }
                )
            return {"success": True, "results": payload}
        except Exception:
            pass

    return {"success": True, "results": _lexical_search(query, k=top_k)}

