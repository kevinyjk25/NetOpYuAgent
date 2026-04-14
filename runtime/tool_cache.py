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
  POST /runtime/cache/demo   → stores a large mock tool result, returns ref_id
  GET  /runtime/cache/{ref_id}?offset=0&length=2000  → reads a page
  GET  /runtime/cache/        → lists all entries for the session
  DELETE /runtime/cache/{ref_id}  → removes one entry
"""
from __future__ import annotations

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
        entry = self._entries.get(cache_key)
        if entry is None:
            return None
        entry.hit_count += 1
        # Move to end (LRU)
        self._entries.move_to_end(cache_key)
        return entry.full_text[offset : offset + length]

    def get_entry(self, ref_id: str, session_id: str = "default") -> Optional[CacheEntry]:
        return self._entries.get(f"{session_id}:{ref_id}")

    # ------------------------------------------------------------------
    # List / Delete
    # ------------------------------------------------------------------

    def list_session(self, session_id: str) -> list[dict[str, Any]]:
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
        entry = self._entries.pop(cache_key, None)
        if entry:
            self._total_bytes -= entry.byte_size
            return True
        return False

    def clear_session(self, session_id: str) -> int:
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
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
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


class DemoRequest(BaseModel):
    session_id:  str = "demo-session"
    tool_name:   str = "syslog_search"
    size_chars:  int = 20_000   # how large a payload to generate


def create_cache_router():
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse
    api = APIRouter(tags=["Tool Cache (P0)"])
    cache = get_tool_cache()

    # ------------------------------------------------------------------
    # Demo: store a large mock payload and return ref_id
    # ------------------------------------------------------------------
    @api.post("/demo", summary="Demo: store large mock tool output")
    async def demo_store(req: DemoRequest) -> JSONResponse:
        """
        Generates a large mock tool result (syslog lines, NPM metrics, etc.),
        stores it in the cache, and returns the ref_id so you can read it back.

        This demonstrates the full P0 ToolResultCache cycle:
          1. Tool produces large payload
          2. Cache stores it → returns compact reference label
          3. Only the label enters the LLM prompt
          4. A follow-up read_result() call retrieves any page on demand

        Try it in the WebUI:
          POST /runtime/cache/demo  →  note the ref_id
          GET  /runtime/cache/{ref_id}?session_id=demo-session  →  read page 1
          GET  /runtime/cache/{ref_id}?session_id=demo-session&offset=2000  →  page 2
        """
        # Build a realistic large mock payload
        lines: list[str] = []
        if req.tool_name == "syslog_search":
            for i in range(req.size_chars // 120):
                ts = f"2025-01-01T00:{i//60:02d}:{i%60:02d}Z"
                lines.append(
                    f"{ts} payments-service ERROR [req-{1000+i}] "
                    f"connection pool exhausted after 30s timeout; "
                    f"active=50 idle=0 waiting={i%20}"
                )
        elif req.tool_name == "prometheus_query":
            for i in range(req.size_chars // 80):
                lines.append(
                    f"{{job='payments', instance='10.0.0.{i%256}', "
                    f"pod='payments-{i%10}'}} "
                    f"http_request_duration_seconds_bucket{{le='0.1'}} "
                    f"{0.85 + (i % 10) * 0.01:.4f} {1704067200000 + i * 15000}"
                )
        else:
            for i in range(req.size_chars // 60):
                lines.append(
                    f"device-{i%50:03d}: metric_value={i * 1.23:.2f} "
                    f"status={'ok' if i % 7 != 0 else 'WARN'}"
                )
        raw_payload = "\n".join(lines)

        label = cache.store(req.tool_name, raw_payload, req.session_id)
        entry = None

        if "[STORED:" in label:
            ref_id = label.split(":")[2].split(" ")[0]
            entry  = cache.get_entry(ref_id, req.session_id)

        return JSONResponse({
            "stored":        "[STORED:" in label,
            "ref_id":        entry.ref_id if entry else None,
            "tool_name":     req.tool_name,
            "session_id":    req.session_id,
            "total_chars":   len(raw_payload),
            "total_bytes":   entry.byte_size if entry else len(raw_payload),
            "prompt_label":  label[:400],
            "how_to_read":   (
                f"GET /runtime/cache/{entry.ref_id}"
                f"?session_id={req.session_id}&offset=0&length=2000"
            ) if entry else "payload was small enough to be inlined",
            "pages_available": (len(raw_payload) // 2000 + 1) if entry else 1,
        })

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