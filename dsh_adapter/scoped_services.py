"""Session-scoped memory recall and profile-scoped capability retrieval for DSH."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

from retrieval.bm25 import BM25Retriever
from retrieval.factory import skills_to_corpus, tools_to_corpus

from .backend import open_backend, resolve_backend_mode


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(parsed, high))


async def recall_memory(
    *,
    memory_dir: str,
    operator_id: str,
    session_id: str,
    query: str,
    max_chars: int = 1200,
    recent_turns: int = 4,
) -> dict[str, Any]:
    """Recall memory without creating a missing store or crossing scope keys."""
    root = Path(memory_dir).expanduser().resolve()
    database = root / "memory.db"
    if not database.is_file():
        return {
            "available": False,
            "prompt_context": "",
            "results": [],
            "fact_count": 0,
            "chunk_count": 0,
            "reason": f"memory store not found: {database}",
        }

    from agent_memory import MemoryManager
    from agent_memory.retrieval.recall_orchestrator import recall

    manager = MemoryManager(data_dir=str(root), enable_user_model=True)
    try:
        result = await asyncio.to_thread(
            recall,
            manager,
            operator_id,
            str(query)[:4000],
            session_id,
            _bounded_int(max_chars, 1200, 200, 4000),
            _bounded_int(recent_turns, 4, 0, 10),
            cross_session=False,
        )
        return {"available": True, **asdict(result)}
    finally:
        manager.close()


async def search_capabilities(
    *,
    profile_id: str,
    query: str,
    top_k: int = 5,
    kinds: list[str] | None = None,
    allowed_tool_names: list[str] | None = None,
) -> dict[str, Any]:
    """Search active tools and skills with the legacy CJK-aware BM25 engine."""
    selected = set(kinds or ["tool", "skill"])
    if not selected <= {"tool", "skill"}:
        raise ValueError("capability kinds must contain only 'tool' and/or 'skill'")
    limit = _bounded_int(top_k, 5, 1, 20)
    corpus: list[dict[str, Any]] = []
    backend = await open_backend(profile_id)
    try:
        if "tool" in selected:
            allowed = set(allowed_tool_names) if allowed_tool_names is not None else None
            visible_metadata = {
                name: metadata for name, metadata in backend.metadata.items()
                if allowed is None or name in allowed
            }
            for item in tools_to_corpus(visible_metadata):
                item = dict(item)
                item["id"] = f"tool:{item['id']}"
                item["kind"] = "tool"
                item["source"] = backend.sources.get(item["id"].removeprefix("tool:"), "unknown")
                corpus.append(item)

        if "skill" in selected:
            from skills import SkillLoader

            backend_mode = resolve_backend_mode()
            definitions = SkillLoader(mode=backend_mode, profile=profile_id).skill_definitions()
            for item in skills_to_corpus(definitions):
                item = dict(item)
                item["id"] = f"skill:{item['id']}"
                item["kind"] = "skill"
                item["source"] = "netopyu-skill"
                corpus.append(item)

        retriever = BM25Retriever()
        retriever.index(corpus)
        # Keep a full top-k just like the legacy retrieval bench. A positive
        # cutoff silently shortened sparse/CJK result lists and created a DSH
        # parity regression even though both paths used the same BM25 index.
        result = retriever.retrieve(str(query)[:4000], top_k=limit, min_score=0.0)
        matches = [
            {
                "id": match.id.split(":", 1)[1],
                "kind": match.item["kind"],
                "score": round(match.score, 4),
                "description": match.item.get("description", ""),
                "tags": match.item.get("tags", []),
                "requires_approval": bool(match.item.get("hitl")),
                "source": match.item.get("source", "unknown"),
            }
            for match in result.matches
        ]
        return {
            "query": result.query,
            "backend": result.backend,
            "total_pool": result.total_pool,
            "matches": matches,
        }
    finally:
        await backend.close()
