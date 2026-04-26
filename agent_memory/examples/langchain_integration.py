"""
agent_memory/examples/langchain_integration.py
===============================================
Integration patterns for agent_memory v3.

Pattern 1: AgentMemoryCallback — hooks into LangChain callback system.
Pattern 2: MemoryAugmentedPrompt — builds context-enriched system prompts.
Pattern 3: Standalone demo — runs without any LangChain dependency.

New in v3: UserModelEngine integration
  - Tracks behavioral profile per user (technical level, tool preferences, etc.)
  - Detects stated-vs-revealed preference contradictions
  - Automatically injects [USER PROFILE] block into build_context()
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from agent_memory import MemoryManager
from agent_memory.user_model import UserProfile

logger = logging.getLogger(__name__)


# ── LangChain stub types (replace with real imports in your project) ──────────
# from langchain.callbacks.base import BaseCallbackHandler
# from langchain.schema import LLMResult
class BaseCallbackHandler:
    """Replace with: from langchain.callbacks.base import BaseCallbackHandler"""

class LLMResult:
    """Replace with: from langchain.schema import LLMResult"""
    def __init__(self, generations): self.generations = generations


# ═════════════════════════════════════════════════════════════════════════════
# Pattern 1: Callback Handler
# ═════════════════════════════════════════════════════════════════════════════

class AgentMemoryCallback(BaseCallbackHandler):
    """
    Drop-in LangChain callback that writes to all memory layers automatically,
    and updates the behavioral user profile on each turn.

    Usage:
        mem = MemoryManager(data_dir="./memory_data", llm_fn=my_llm)
        callback = AgentMemoryCallback(mem, user_id="alice", session_id="s1")
        agent = initialize_agent(tools, llm, callbacks=[callback])

    To rotate sessions between conversations:
        callback.new_session("session-002")
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        user_id: str,
        session_id: str,
        distill_every_n_turns: int = 3,
        cache_threshold: int = 4_000,
    ) -> None:
        self.mem = memory_manager
        self.user_id = user_id
        self.session_id = session_id
        self.distill_every_n_turns = distill_every_n_turns
        self.cache_threshold = cache_threshold
        self._turn_count = 0
        # Buffer: list of (chunk_id, text) for pending distillation
        self._pending: List[tuple[str, str]] = []
        # Last user message (for user model update)
        self._last_user_text: str = ""
        # Last tool token (for callers to inject into next prompt)
        self.last_tool_token: Optional[str] = None

    def on_human_turn(self, text: str, **kwargs: Any) -> None:
        """Call this when the user sends a message (before LLM responds)."""
        self._last_user_text = text

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Store LLM response as a long-term chunk; update user profile."""
        for gens in response.generations:
            for gen in gens:
                text = getattr(gen, "text", str(gen)).strip()
                if not text:
                    continue
                chunk = self.mem.remember(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    text=text,
                    source="llm_response",
                )
                self._pending.append((chunk.chunk_id, text))

                # Update behavioral user profile
                if self._last_user_text:
                    self.mem.update_user_profile(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        user_text=self._last_user_text,
                        assistant_text=text,
                    )

        self._turn_count += 1
        if self._turn_count % self.distill_every_n_turns == 0:
            self._distill_pending()

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Cache large tool outputs; store small ones inline."""
        tool_name = kwargs.get("name", "unknown_tool")
        if len(output) > self.cache_threshold:
            entry = self.mem.cache_tool_result(
                user_id=self.user_id,
                session_id=self.session_id,
                tool_name=tool_name,
                content=output,
            )
            # Store in self.last_tool_token — inject into next LLM message
            self.last_tool_token = self.mem.get_cache_preview(
                self.user_id, entry.ref_id
            )
            logger.info("Large output cached: %s", self.last_tool_token[:80])

            # Update user model with tool usage
            self.mem.update_user_profile(
                user_id=self.user_id,
                session_id=self.session_id,
                user_text=self._last_user_text,
                assistant_text="",
                tool_calls=[{"tool": tool_name}],
            )
        else:
            chunk = self.mem.remember(
                user_id=self.user_id,
                session_id=self.session_id,
                text=f"[tool:{tool_name}] {output}",
                source="tool_output",
            )
            self._pending.append((chunk.chunk_id, output))
            self.last_tool_token = None

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """Flush pending distillation when agent completes a task."""
        self._distill_pending()

    def new_session(self, session_id: str) -> None:
        """
        Rotate to a new session_id between conversations.
        Flushes pending distillation from the previous session first.
        """
        self._distill_pending()
        self.session_id = session_id
        self._pending.clear()
        self._turn_count = 0
        self._last_user_text = ""
        logger.info("Session rotated → %s for user %s", session_id, self.user_id)

    def build_system_prompt(self, base_prompt: str, query: str) -> str:
        """
        Return base_prompt augmented with memory context.
        Includes: [USER PROFILE] + facts + relevant chunks.
        """
        ctx = self.mem.build_context(
            user_id=self.user_id,
            query=query,
            session_id=self.session_id,
        )
        return f"{base_prompt}\n\n{ctx}" if ctx else base_prompt

    def _distill_pending(self) -> None:
        if not self._pending:
            return
        combined = "\n\n".join(text for _, text in self._pending)
        chunk_ids = [cid for cid, _ in self._pending]
        try:
            facts = self.mem.distill(
                user_id=self.user_id,
                session_id=self.session_id,
                text=combined,
                source_chunk_ids=chunk_ids,
            )
            logger.debug("Distilled %d facts from %d pending chunks",
                         len(facts), len(self._pending))
        except Exception as e:
            logger.warning("Distillation failed: %s", e)
        finally:
            self._pending.clear()


# ═════════════════════════════════════════════════════════════════════════════
# Pattern 2: Memory-Augmented Prompt Builder
# ═════════════════════════════════════════════════════════════════════════════

class MemoryAugmentedPrompt:
    """
    Builds memory-enriched system prompts per query.

    Usage:
        augmenter = MemoryAugmentedPrompt(mem, user_id="alice", session_id="s1")
        system = augmenter.build("You are an IT ops assistant.", user_query)
        page = augmenter.read_cached(ref_id, offset=0, length=2000)
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        user_id: str,
        session_id: str,
        max_context_chars: int = 3_000,
    ) -> None:
        self.mem = memory_manager
        self.user_id = user_id
        self.session_id = session_id
        self.max_context_chars = max_context_chars

    def build(self, base_prompt: str, query: str) -> str:
        ctx = self.mem.build_context(
            user_id=self.user_id, query=query,
            session_id=self.session_id,
            max_chars=self.max_context_chars,
        )
        return f"{base_prompt}\n\n{ctx}" if ctx else base_prompt

    def read_cached(self, ref_id: str, offset: int = 0, length: int = 2_000) -> dict:
        return self.mem.read_cached(self.user_id, ref_id, offset, length)

    def get_profile(self) -> Optional[UserProfile]:
        return self.mem.get_user_profile(self.user_id)


# ═════════════════════════════════════════════════════════════════════════════
# Pattern 3: Standalone demo
# ═════════════════════════════════════════════════════════════════════════════

def demo() -> None:
    """
    Full demo of all v3 features. No LangChain required.
    Run: python -m agent_memory.examples.langchain_integration
    """
    print("=" * 62)
    print("  Agent Memory Module v3 — Integration Demo")
    print("=" * 62)

    with MemoryManager(data_dir="/tmp/agent_memory_v3_demo") as mem:
        user_id    = "alice"
        session_id = "demo-session-001"

        # ── Step 1: Store conversation turns (long-term) ──────────────────
        print("\n── Step 1: Long-term memory (conversation chunks) ──")
        mem.remember(user_id, session_id,
                     "User asked: How do I check BGP neighbor state on R1?")
        mem.remember(user_id, session_id,
                     "Agent: Use 'show ip bgp neighbors' on Cisco IOS.")
        mem.remember(user_id, session_id,
                     "User: R1 is at 10.0.0.1. I prefer CLI over REST API.")
        print("  Stored 3 conversation chunks.")

        # ── Step 2: Distill facts (mid-term) ──────────────────────────────
        print("\n── Step 2: Mid-term memory (distilled facts) ──")
        mem.distill(user_id, session_id,
                    "User prefers Cisco IOS CLI. R1 is at 10.0.0.1. BGP peer AS is 65001.")
        mem.add_fact(user_id, session_id, "R1 at 10.0.0.1, IOS 15.4", "entity", 0.95)
        mem.add_fact(user_id, session_id, "User prefers CLI over REST", "preference", 0.9)
        print("  Stored distilled facts.")

        # ── Step 3: Update user behavioral profile (NEW in v3) ────────────
        print("\n── Step 3: UserModelEngine (behavioral profiling) ── [NEW]")
        # Simulate multiple turns to build up profile
        turns = [
            ("Check ECMP iBGP MPLS redistribution paths on R1 and R2",
             "Advanced BGP config with MPLS forwarding table analysis"),
            ("I prefer httpx for REST calls, don't use requests anymore",
             "Noted, will use httpx. Here's the httpx equivalent..."),
            ("Search syslogs on ap-01 for auth failures last 24h",
             "Found 12 RADIUS auth failures. Log entries: ...",
             [{"tool": "syslog_search"}]),
            ("Brief summary please — what is BGP next-hop self?",
             "BGP next-hop self forces the router to advertise itself."),
        ]
        for turn in turns:
            user_text, asst_text = turn[0], turn[1]
            tool_calls = turn[2] if len(turn) > 2 else None
            mem.remember(user_id, session_id, user_text)
            mem.update_user_profile(user_id, session_id, user_text, asst_text, tool_calls)

        profile = mem.get_user_profile(user_id)
        print(f"  Technical level: {profile.technical_level.value}")
        print(f"  Communication:   {profile.communication_style.value}")
        print(f"  Domain focus:    {sorted(profile.domain_counts.items(), key=lambda x: -x[1])[:3]}")
        print(f"  Tool usage:      {profile.tool_usage}")
        print(f"  Total turns:     {profile.total_turns}")
        print(f"  Stated prefs:    {len(profile.stated_preferences)} entries")
        print()
        print("  Profile prompt section:")
        print("  " + profile.to_prompt_section(400).replace("\n", "\n  "))

        # ── Step 4: Cache large tool output (short-term) ──────────────────
        print("\n── Step 4: Short-term cache (P0 tool result, byte-safe) ──")
        syslog = "\n".join(
            f"Apr 10 0{i%10}:{i:02d}:00 ap-01 RADIUS: Auth {'OK' if i%3 else 'FAIL'} "
            f"user=alice@corp.com nas=10.0.1.{i%5}"
            for i in range(60)
        )
        entry = mem.cache_tool_result(user_id, session_id, "syslog_search", syslog)
        print(f"  Cached {entry.total_length} chars → ref_id: {entry.ref_id[:12]}…")
        preview = mem.get_cache_preview(user_id, entry.ref_id)
        print(f"  Token: {preview[:80]}…")

        # ── Step 5: Paged byte-offset retrieval ───────────────────────────
        print("\n── Step 5: Byte-offset paging (Unicode-safe) ──")
        page1 = mem.read_cached(user_id, entry.ref_id, offset=0, length=300)
        page2 = mem.read_cached(user_id, entry.ref_id,
                                offset=page1["next_offset"], length=300)
        print(f"  Page 1: {page1['length']} bytes, has_more={page1['has_more']}")
        print(f"  Page 2: {page2['length']} bytes, next={page2.get('next_offset')}")

        # ── Step 6: FTS5-safe query (reserved words handled) ──────────────
        print("\n── Step 6: FTS5-safe search (AND/OR/NOT handled) ──")
        r = mem.search_chunks(user_id, "AND OR NOT BGP CLI neighbor")
        print(f"  'AND OR NOT BGP CLI' → {r.total_found} results, {r.elapsed_ms:.1f}ms")

        # ── Step 7: Unified context (profile + facts + chunks) ────────────
        print("\n── Step 7: build_context (profile + facts + chunks) ──")
        ctx = mem.build_context(user_id, "BGP neighbor R1", session_id=session_id)
        print(f"  Context ({len(ctx)} chars):")
        for line in ctx.split("\n")[:12]:
            print(f"  {line}")
        if len(ctx.split("\n")) > 12:
            print(f"  ... ({len(ctx.split(chr(10)))-12} more lines)")

        # ── Step 8: Cross-session retrieval ───────────────────────────────
        print("\n── Step 8: Cross-session retrieval ──")
        mem.remember(user_id, "demo-session-002",
                     "New session: Prometheus disk alert on node01.")
        cross = mem.search_chunks(user_id, "BGP R1 Cisco RADIUS")
        print(f"  Cross-session search: {cross.total_found} results "
              f"across {len(mem.list_sessions(user_id))} sessions")

        # ── Step 9: Atomic eviction ───────────────────────────────────────
        print("\n── Step 9: Session cache lifecycle ──")
        evicted = mem.evict_expired_cache()
        print(f"  Evicted {evicted} expired entries (atomic, safe for concurrent calls)")

        # ── Step 10: Full stats ───────────────────────────────────────────
        print("\n── Step 10: Stats ──")
        s = mem.stats(user_id, session_id)
        for k, v in s.items():
            if k == "user_profile":
                v_str = f"{{ turns={v['total_turns']}, traits={len(v['traits'])}, domains={v['domain_counts']} }}" if v else "None"
                print(f"  {k}: {v_str}")
            else:
                print(f"  {k}: {v}")

    print("\n✓ Demo complete. MemoryManager closed via context manager.")
    print("  All connections released. WAL checkpointed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    demo()
