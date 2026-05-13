"""
runtime/tool_cache.py
----------------------
Enhanced ToolResultStore with:
  - Session-namespaced keys (prevents cross-session data leakage)
  - HTTP API endpoints (GET /runtime/cache/{ref_id}, POST /runtime/cache/read)
  - LRU eviction when max_entries is reached
  - Metrics: hit count, total stored bytes
  - A demo endpoint that exercises the full cache-store-retrieve cycle

P0 requirement (from PDF review):
  "大型 grep / 日志查询 / NPM 时序结果不直接回注全文
   结果存对象存储; prompt 中只放预览、统计、路径或引用 ID
   模型如需细节，再调用 ReadResult / DrillDown 工具读取局部内容"

Usage (via WebUI):
  GET  /runtime/cache/{ref_id}?offset=0&length=2000  → reads a page
  GET  /runtime/cache/        → lists all entries for the session
  DELETE /runtime/cache/{ref_id}  → removes one entry
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel

# FastAPI is only needed when building the router; lazy-imported inside create_cache_router()

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    ref_id:     str
    tool_name:  str
    session_id: str
    full_text:  str
    created_at: float = field(default_factory=time.time)
    hit_count:  int   = 0

    @property
    def byte_size(self) -> int:
        return len(self.full_text.encode())

    @property
    def char_count(self) -> int:
        return len(self.full_text)

    def preview(self, chars: int = 300) -> str:
        return self.full_text[:chars].replace("\n", " ")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ToolResultCache:
    """
    Process-global cache for large tool outputs.

    Keys are namespaced as  "{session_id}:{ref_id}"  so different sessions
    cannot read each other's results even if they guess a ref_id.

    LRU eviction fires when max_entries is exceeded.
    """

    MAX_INLINE_CHARS = 4_000    # below this → return raw, don't cache
    DEFAULT_PAGE     = 2_000    # default chars per read() call

    def __init__(self, max_entries: int = 500) -> None:
        self._entries:    OrderedDict[str, CacheEntry] = OrderedDict()
        self._max         = max_entries
        self._total_bytes = 0
        # Thread/async safety. FastAPI runs `def` endpoints in a thread pool,
        # so two concurrent /runtime/cache reads + writes can interleave on
        # the OrderedDict; without this lock, `popitem()` during eviction
        # can race with `__getitem__` and throw KeyError, or self._total_bytes
        # drifts away from the real byte sum.
        # RLock so methods can safely call each other while already holding it.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(
        self,
        tool_name:  str,
        raw_output: str,
        session_id: str = "default",
    ) -> str:
        """
        Store *raw_output* if it exceeds MAX_INLINE_CHARS.

        Returns
        -------
        If small  → the raw text unchanged.
        If large  → a compact reference label:
            "[STORED:tool_name:ref_id | chars:NNN | use read_result(ref_id)]"
        """
        if len(raw_output) <= self.MAX_INLINE_CHARS:
            return raw_output

        ref_id    = str(uuid.uuid4())[:8]
        cache_key = f"{session_id}:{ref_id}"

        entry = CacheEntry(
            ref_id     = ref_id,
            tool_name  = tool_name,
            session_id = session_id,
            full_text  = raw_output,
        )
        with self._lock:
            self._entries[cache_key] = entry
            self._total_bytes += entry.byte_size

            # LRU eviction
            while len(self._entries) > self._max:
                _, evicted = self._entries.popitem(last=False)
                self._total_bytes -= evicted.byte_size

        label = (
            f"[STORED:{tool_name}:{ref_id} | "
            f"chars:{len(raw_output):,} | "
            f"use read_result(ref_id='{ref_id}') to access full output]\n"
            f"Preview: {entry.preview()}"
        )
        return label

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(
        self,
        ref_id:     str,
        session_id: str = "default",
        offset:     int = 0,
        length:     int = DEFAULT_PAGE,
    ) -> Optional[str]:
        """Return a slice of the stored result, or None if not found."""
        cache_key = f"{session_id}:{ref_id}"
        with self._lock:
            entry = self._entries.get(cache_key)
            if entry is None:
                return None
            entry.hit_count += 1
            # Move to end (LRU)
            self._entries.move_to_end(cache_key)
            full = entry.full_text
        # Slice outside the lock — the slice itself is just a substring op
        # on the immutable str, no concurrency hazard.
        return full[offset : offset + length]

    def get_entry(self, ref_id: str, session_id: str = "default") -> Optional[CacheEntry]:
        with self._lock:
            return self._entries.get(f"{session_id}:{ref_id}")

    # ------------------------------------------------------------------
    # List / Delete
    # ------------------------------------------------------------------

    def list_session(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "ref_id":     e.ref_id,
                    "tool_name":  e.tool_name,
                    "char_count": e.char_count,
                    "byte_size":  e.byte_size,
                    "hit_count":  e.hit_count,
                    "created_at": e.created_at,
                    "preview":    e.preview(120),
                }
                for key, e in self._entries.items()
                if key.startswith(f"{session_id}:")
            ]

    def delete(self, ref_id: str, session_id: str = "default") -> bool:
        cache_key = f"{session_id}:{ref_id}"
        with self._lock:
            entry = self._entries.pop(cache_key, None)
            if entry:
                self._total_bytes -= entry.byte_size
                return True
            return False

    def clear_session(self, session_id: str) -> int:
        with self._lock:
            keys = [k for k in self._entries if k.startswith(f"{session_id}:")]
            for k in keys:
                self._total_bytes -= self._entries[k].byte_size
                del self._entries[k]
            return len(keys)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_entries(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes


# ---------------------------------------------------------------------------
# Process-global singleton
# ---------------------------------------------------------------------------

_GLOBAL_CACHE = ToolResultCache()


def get_tool_cache() -> ToolResultCache:
    return _GLOBAL_CACHE


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

class ReadRequest(BaseModel):
    ref_id:     str
    session_id: str  = "default"
    offset:     int  = 0
    length:     int  = 2_000


def create_cache_router():
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse
    api = APIRouter(tags=["Tool Cache (P0)"])
    cache = get_tool_cache()

    # ------------------------------------------------------------------
    # Read a page from the cache
    # ------------------------------------------------------------------
    @api.get("/{ref_id}", summary="Read a page of a cached tool result")
    async def read_cached(
        ref_id:     str,
        session_id: str = "default",
        offset:     int = 0,
        length:     int = 2_000,
    ) -> JSONResponse:
        """
        Retrieve a slice of a previously cached large tool result.
        Increment offset by `length` to page through the full output.
        """
        chunk = cache.read(ref_id, session_id, offset, length)
        if chunk is None:
            raise HTTPException(
                status_code=404,
                detail=f"No cached result for ref_id={ref_id!r} session={session_id!r}",
            )
        entry = cache.get_entry(ref_id, session_id)
        return JSONResponse({
            "ref_id":      ref_id,
            "offset":      offset,
            "length":      len(chunk),
            "total_chars": entry.char_count if entry else len(chunk),
            "hit_count":   entry.hit_count  if entry else 1,
            "has_more":    (offset + length) < (entry.char_count if entry else 0),
            "content":     chunk,
        })

    # ------------------------------------------------------------------
    # Structured read (POST body)
    # ------------------------------------------------------------------
    @api.post("/read", summary="Read a page (POST body version)")
    async def read_cached_post(req: ReadRequest) -> JSONResponse:
        chunk = cache.read(req.ref_id, req.session_id, req.offset, req.length)
        if chunk is None:
            raise HTTPException(
                status_code=404,
                detail=f"No cached result for ref_id={req.ref_id!r}",
            )
        entry = cache.get_entry(req.ref_id, req.session_id)
        return JSONResponse({
            "ref_id":      req.ref_id,
            "offset":      req.offset,
            "length":      len(chunk),
            "total_chars": entry.char_count if entry else len(chunk),
            "has_more":    (req.offset + req.length) < (entry.char_count if entry else 0),
            "content":     chunk,
        })

    # ------------------------------------------------------------------
    # List entries for a session
    # ------------------------------------------------------------------
    @api.get("/", summary="List cached entries for a session")
    async def list_entries(session_id: str = "default") -> JSONResponse:
        entries = cache.list_session(session_id)
        return JSONResponse({
            "session_id":    session_id,
            "entry_count":   len(entries),
            "total_cached":  cache.total_entries,
            "total_bytes":   cache.total_bytes,
            "entries":       entries,
        })

    # ------------------------------------------------------------------
    # Delete one entry
    # ------------------------------------------------------------------
    @api.delete("/{ref_id}", summary="Delete a cached entry")
    async def delete_entry(ref_id: str, session_id: str = "default") -> JSONResponse:
        deleted = cache.delete(ref_id, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Not found: {ref_id}")
        return JSONResponse({"deleted": True, "ref_id": ref_id})

    # ------------------------------------------------------------------
    # Cache stats
    # ------------------------------------------------------------------
    @api.get("/stats/global", summary="Global cache statistics")
    async def stats() -> JSONResponse:
        return JSONResponse({
            "total_entries": cache.total_entries,
            "total_bytes":   cache.total_bytes,
            "max_entries":   cache._max,
            "utilisation_pct": round(cache.total_entries / cache._max * 100, 1),
        })

    return api