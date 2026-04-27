"""
agent_memory/memory_manager.py — v5

Unified façade integrating all memory capabilities:
  ShortTermStore      P0 tool cache, byte-offset paging
  MidTermStore        Hermes-style distilled facts (dedup, TTL, confidence decay)
  LongTermStore       Claw-style chunks (recency+importance scoring, batch)
  UserModelEngine     Dialectic behavioral profiling
  SessionState        Dual-track hot/cold memory
  ContextBudgetManager  Priority token budget
  SkillStore          Procedural memory — reusable task execution skills (v5 NEW)
  MemoryConsolidator  Context compression for long sessions (v5 NEW)
  ReflectionEngine    Reflect & write lessons after task completion (v5 NEW)
  EmbeddingIndex      Semantic vector retrieval with pluggable backends (v5 NEW)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent_memory.schemas import (
    MemoryChunk, MemoryFact, ToolResultEntry, RetrievalResult, _uid, _now,
)
from agent_memory.stores.long_term_store import LongTermStore
from agent_memory.stores.mid_term_store import MidTermStore
from agent_memory.stores.short_term_store import ShortTermStore
from agent_memory.stores.skill_store import SkillStore, Skill
from agent_memory.retrieval.fact_extractor import FactExtractor
from agent_memory.retrieval.embedding_store import EmbeddingIndex, EmbeddingBackend
from agent_memory.stores._db import close_pool
from agent_memory.user_model import UserModelEngine, UserProfile
from agent_memory.context_budget import (
    ContextBudgetManager, BudgetReport, Priority, estimate_tokens,
)
from agent_memory.session_state import (
    SessionState, SessionStateRegistry, mmr_rerank,
)
from agent_memory.consolidation import MemoryConsolidator, ReflectionEngine

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Agent Memory Module v5.

    Quick start:
        with MemoryManager(data_dir="./memory_data") as mem:
            # 基础记忆
            chunk = mem.remember(user_id, session_id, text)
            mem.distill(user_id, session_id, text)

            # 程序性记忆
            mem.save_skill(user_id, "bgp_diagnosis", description, steps, tags)
            skills = mem.find_skills(user_id, "BGP neighbor down")

            # 上下文构建（含 Skill）
            state = mem.get_session(user_id, session_id)
            ctx, report = mem.build_context_budgeted(state, query)

            # 反思写入
            mem.reflect(user_id, session_id, task="BGP排查", outcome="success",
                        summary="...", skill_id=skill.skill_id)

            # 历史压缩（长会话）
            mem.consolidate_session(user_id, session_id)
    """

    def __init__(
        self,
        data_dir: str | Path = "./agent_memory_data",
        # LLM backends
        llm_fn:              Optional[Callable[[str], str]] = None,
        user_model_llm_fn:   Optional[Callable[[str, str], str]] = None,
        # Semantic embedding (optional)
        embedding_fn:        Optional[Callable[[str], List[float]]] = None,
        embedding_dim:       int   = 0,
        embedding_backend:   Optional[EmbeddingBackend] = None,
        # Context budget
        tiktoken_fn:         Optional[Callable[[str], int]] = None,
        default_token_budget: int = 3_200,
        mmr_lambda:          float = 0.6,
        # Short-term cache
        inline_threshold:    int   = 4_000,
        session_ttl:         float = 86_400,
        # Fact quality
        min_fact_confidence: float = 0.5,
        # Index caps
        max_index_docs:      int   = 50_000,
        max_session_indexes: int   = 512,
        # User model
        contradiction_check_interval: int = 10,
        enable_user_model:   bool  = True,
        # Consolidation
        consolidate_after_n_turns: int = 30,
        consolidation_keep_recent: int = 10,
        # Skill store
        skill_deprecate_threshold: float = 0.3,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._data_dir / "memory.db"

        # Core stores
        self.long_term  = LongTermStore(db_path=self._db_path,
                                        max_index_docs=max_index_docs)
        self.mid_term   = MidTermStore(db_path=self._db_path,
                                       max_index_docs=max_index_docs // 5,
                                       max_indexes=max_session_indexes)
        self.short_term = ShortTermStore(base_dir=self._data_dir / "tool_cache",
                                         db_path=self._db_path,
                                         inline_threshold=inline_threshold,
                                         session_ttl=session_ttl)

        # v5: Skill store
        self.skill_store = SkillStore(db_path=self._db_path,
                                      deprecate_threshold=skill_deprecate_threshold)

        # Retrieval
        self.extractor = FactExtractor(llm_fn=llm_fn,
                                       min_confidence=min_fact_confidence)

        # v5: Semantic embedding index
        if embedding_backend:
            self._emb_index = EmbeddingIndex(embedding_backend)
        elif embedding_fn:
            from agent_memory.retrieval.embedding_store import CallableBackend
            self._emb_index = EmbeddingIndex(
                CallableBackend(embedding_fn, dim=embedding_dim))
        else:
            self._emb_index = EmbeddingIndex()   # TF-IDF fallback
        logger.info("Embedding backend: %s", self._emb_index.backend_name)

        # User model
        self.user_model: Optional[UserModelEngine] = (
            UserModelEngine(db_path=str(self._db_path),
                            llm_fn=user_model_llm_fn,
                            contradiction_check_interval=contradiction_check_interval)
            if enable_user_model else None
        )

        # Session management
        self._sessions = SessionStateRegistry()

        # v5: Consolidation & Reflection
        self._consolidator = MemoryConsolidator(
            llm_fn=llm_fn,
            consolidate_after_n=consolidate_after_n_turns,
            keep_recent_n=consolidation_keep_recent,
        )
        self._reflector = ReflectionEngine(llm_fn=llm_fn)

        # Budget
        self._tiktoken_fn    = tiktoken_fn
        self._default_budget = default_token_budget
        self._mmr_lambda     = mmr_lambda
        self._inline_threshold = inline_threshold

        logger.info(
            "MemoryManager v5: data_dir=%s llm=%s embedding=%s user_model=%s",
            self._data_dir, "yes" if llm_fn else "rules",
            self._emb_index.backend_name,
            "yes" if enable_user_model else "off",
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        close_pool(self._db_path)
        logger.info("MemoryManager closed: %s", self._db_path)

    def checkpoint(self) -> None:
        from agent_memory.stores._db import get_pool
        get_pool(self._db_path).checkpoint()

    # ═══════════════════════════════════════════════════════
    # WRITE API — standard memory layers
    # ═══════════════════════════════════════════════════════

    def remember(self, user_id, session_id, text, source="conversation",
                 metadata=None, chunk_id=None, importance=0.5) -> MemoryChunk:
        chunk = MemoryChunk(
            chunk_id=chunk_id or _uid(), user_id=user_id,
            session_id=session_id, text=text, source=source,
            created_at=_now(), metadata=metadata or {},
        )
        self.long_term.add_chunk(chunk, importance=importance)
        # Also index into EmbeddingIndex for semantic search
        self._emb_index.add(chunk.chunk_id, text)
        return chunk

    def remember_batch(self, user_id, session_id, texts,
                       source="conversation", importance=0.5) -> List[MemoryChunk]:
        chunks = [MemoryChunk(user_id=user_id, session_id=session_id,
                              text=t, source=source, created_at=_now())
                  for t in texts if t and t.strip()]
        if chunks:
            self.long_term.add_chunks_batch(chunks, importance=importance)
            for c in chunks:
                self._emb_index.add(c.chunk_id, c.text)
        return chunks

    def distill(self, user_id, session_id, text,
                source_chunk_ids=None) -> List[MemoryFact]:
        if not text or not text.strip():
            return []
        facts = self.extractor.extract(text=text, user_id=user_id,
                                       session_id=session_id,
                                       source_chunk_ids=source_chunk_ids or [])
        if facts:
            self.mid_term.add_facts_batch(facts)
        return facts

    def add_fact(self, user_id, session_id, fact, fact_type="general",
                 confidence=1.0, source_chunk_ids=None,
                 metadata=None, ttl_days=None) -> MemoryFact:
        mf = MemoryFact(user_id=user_id, session_id=session_id, fact=fact,
                        fact_type=fact_type, confidence=confidence,
                        source_chunk_ids=source_chunk_ids or [],
                        metadata=metadata or {})
        self.mid_term.add_fact(mf, ttl_days=ttl_days)
        return mf

    def cache_tool_result(self, user_id, session_id, tool_name, content,
                          ttl=None, metadata=None, auto_remember=False) -> ToolResultEntry:
        entry = self.short_term.store(user_id=user_id, session_id=session_id,
                                      tool_name=tool_name, content=content,
                                      ttl=ttl, metadata=metadata)
        if auto_remember:
            self.remember(user_id=user_id, session_id=session_id,
                          text=f"[tool:{tool_name}] {content[:1500]}",
                          source="tool_output")
        return entry

    # ── user model ────────────────────────────────────────────────────────────

    def update_user_profile(self, user_id, session_id, user_text,
                            assistant_text, tool_calls=None) -> Optional[UserProfile]:
        if self.user_model is None:
            return None
        return self.user_model.update_profile(user_id=user_id,
                                              session_id=session_id,
                                              user_text=user_text,
                                              assistant_text=assistant_text,
                                              tool_calls=tool_calls)

    def get_user_profile(self, user_id) -> Optional[UserProfile]:
        return self.user_model.get_profile(user_id) if self.user_model else None

    def get_user_profile_section(self, user_id, max_chars=800) -> str:
        return (self.user_model.get_prompt_section(user_id, max_chars)
                if self.user_model else "")

    # ═══════════════════════════════════════════════════════
    # v5 NEW: SKILL STORE (程序性记忆)
    # ═══════════════════════════════════════════════════════

    def save_skill(
        self,
        user_id:     str,
        skill_name:  str,
        description: str,
        steps:       List[Any],
        tags:        Optional[List[str]] = None,
        metadata:    Optional[Dict[str, Any]] = None,
    ) -> Skill:
        """
        保存或更新一个可复用技能（程序性记忆）。
        同名技能自动版本迭代，保留历史成功率。
        """
        return self.skill_store.save_skill(
            user_id=user_id, skill_name=skill_name, description=description,
            steps=steps, tags=tags, metadata=metadata,
        )

    def find_skills(
        self,
        user_id:  str,
        query:    str,
        top_k:    int  = 3,
        include_deprecated: bool = False,
    ) -> List[Skill]:
        """检索与当前任务最匹配的技能（FTS5 + TF-IDF 混合）。"""
        return self.skill_store.find_skills(
            user_id=user_id, query=query, top_k=top_k,
            include_deprecated=include_deprecated,
        )

    def record_skill_outcome(
        self,
        user_id:  str,
        skill_id: str,
        success:  bool,
    ) -> Optional[float]:
        """记录技能调用结果，更新成功率（EMA）。"""
        return self.skill_store.record_outcome(user_id, skill_id, success)

    # ═══════════════════════════════════════════════════════
    # v5 NEW: CONSOLIDATION (历史压缩)
    # ═══════════════════════════════════════════════════════

    def consolidate_session(
        self,
        user_id:    str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        压缩 session 内的历史 chunks。
        将旧 chunks 用 LLM 摘要合并为更少的 summary chunks，
        控制长会话的 token 消耗（GA 的 <30K 窗口策略）。

        建议在每 consolidate_after_n_turns 轮后调用，或手动触发。
        """
        return self._consolidator.consolidate(
            self.long_term, user_id, session_id
        )

    def should_consolidate(self, user_id: str, session_id: str) -> bool:
        """检查当前 session 是否达到压缩阈值。"""
        count = self.long_term.count_session(user_id, session_id)
        return self._consolidator.should_consolidate(count)

    # ═══════════════════════════════════════════════════════
    # v5 NEW: REFLECT (反思写入)
    # ═══════════════════════════════════════════════════════

    def reflect(
        self,
        user_id:    str,
        session_id: str,
        task:       str,
        outcome:    str,          # "success" | "failure" | "partial"
        summary:    str,
        reason:     str = "",
        skill_id:   Optional[str] = None,
        metadata:   Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        任务完成后写入反思记忆。

        同时写入：
          - 长期记忆（source="reflection", importance=0.9）
          - 中期 lesson facts（fact_type="lesson"）
          - 更新关联 Skill 的成功率（若提供 skill_id）

        Args:
            task:     任务描述
            outcome:  "success" | "failure" | "partial"
            summary:  执行过程摘要
            reason:   失败原因（outcome="failure" 时填写）
            skill_id: 关联的 Skill ID（可选，用于更新成功率）
        """
        return self._reflector.reflect(
            long_term_store=self.long_term,
            mid_term_store=self.mid_term,
            skill_store=self.skill_store,
            user_id=user_id, session_id=session_id,
            task=task, outcome=outcome, summary=summary,
            reason=reason, skill_id=skill_id, metadata=metadata,
        )

    # ═══════════════════════════════════════════════════════
    # READ API
    # ═══════════════════════════════════════════════════════

    def read_cached(self, user_id, ref_id, offset=0, length=2_000) -> dict:
        return self.short_term.read(user_id, ref_id, offset, length)

    def get_cache_preview(self, user_id, ref_id, preview_len=200) -> str:
        return self.short_term.preview(user_id, ref_id, preview_len)

    def should_cache(self, content: str) -> bool:
        return len(content) > self._inline_threshold

    # ═══════════════════════════════════════════════════════
    # SEARCH API — with optional semantic embedding
    # ═══════════════════════════════════════════════════════

    def search(self, user_id, query, session_id=None, top_k=5,
               layers=None, use_mmr=False) -> Dict[str, RetrievalResult]:
        enabled = set(layers or ["long_term", "mid_term"])
        results: Dict[str, RetrievalResult] = {}
        if "long_term" in enabled or "cross_session" in enabled:
            lt = self._search_long_term(user_id, query, top_k, use_mmr)
            if "long_term" in enabled:      results["long_term"] = lt
            if "cross_session" in enabled:  results["cross_session"] = lt
        if "mid_term" in enabled:
            mt = self.mid_term.search(user_id=user_id, query=query,
                                      session_id=session_id, top_k=top_k)
            if use_mmr and mt.items:
                mt = self._apply_mmr(mt, top_k)
            results["mid_term"] = mt
        return results

    def search_facts(self, user_id, query, session_id=None, fact_type=None,
                     min_confidence=0.0, top_k=10, use_mmr=False) -> RetrievalResult:
        r = self.mid_term.search(user_id=user_id, query=query,
                                 session_id=session_id, fact_type=fact_type,
                                 min_confidence=min_confidence, top_k=top_k)
        return self._apply_mmr(r, top_k) if use_mmr and r.items else r

    def search_chunks(self, user_id, query, session_id=None,
                      source_filter=None, top_k=5, use_mmr=False,
                      use_embedding=False) -> RetrievalResult:
        """
        检索 chunks。

        use_embedding=True 时先用 EmbeddingIndex 做语义检索，
        然后与 FTS5+TF-IDF 结果合并 (hybrid retrieval)。
        """
        if use_embedding and self._emb_index.size > 0:
            return self._hybrid_search_chunks(user_id, query, session_id,
                                              source_filter, top_k, use_mmr)
        return self._search_long_term(user_id, query, top_k, use_mmr,
                                      session_id=session_id,
                                      source_filter=source_filter)

    def _search_long_term(self, user_id, query, top_k, use_mmr,
                           session_id=None, source_filter=None) -> RetrievalResult:
        r = self.long_term.search(user_id=user_id, query=query, top_k=top_k,
                                  session_id=session_id, source_filter=source_filter)
        return self._apply_mmr(r, top_k) if use_mmr and r.items else r

    def _hybrid_search_chunks(self, user_id, query, session_id,
                               source_filter, top_k, use_mmr) -> RetrievalResult:
        """Hybrid: EmbeddingIndex + FTS5+TF-IDF, deduplicated, score-fused."""
        import time as _time
        t0 = _time.perf_counter()

        # Embedding retrieval
        if use_mmr:
            emb_hits = self._emb_index.query_mmr(query, top_k=top_k * 2,
                                                   lambda_=self._mmr_lambda)
        else:
            emb_hits = self._emb_index.query(query, top_k=top_k * 2)
        emb_scores = {cid: score for cid, score in emb_hits}

        # FTS5+TF-IDF retrieval
        fts_result = self.long_term.search(user_id=user_id, query=query,
                                           top_k=top_k * 2, session_id=session_id,
                                           source_filter=source_filter)
        fts_scores = {c.chunk_id: 1.0 / (i + 1) for i, c in enumerate(fts_result.items)}

        # Score fusion: combine reciprocal rank (FTS) + normalized cosine (embedding)
        all_ids = set(emb_scores) | set(fts_scores)
        fused: Dict[str, float] = {}
        for cid in all_ids:
            e_score = emb_scores.get(cid, 0.0)
            f_score = fts_scores.get(cid, 0.0)
            fused[cid] = 0.5 * e_score + 0.5 * f_score

        if not fused:
            return fts_result

        # Fetch chunks for fused candidates
        top_ids = sorted(fused, key=lambda x: fused[x], reverse=True)[:top_k]
        chunks_map = {c.chunk_id: c for c in fts_result.items}
        # For IDs only in emb but not fts, fetch from DB
        missing = [cid for cid in top_ids if cid not in chunks_map]
        if missing:
            ph = ",".join("?" * len(missing))
            rows = self.long_term._pool.execute_read(
                f"SELECT * FROM long_term_chunks WHERE chunk_id IN ({ph}) AND user_id=?",
                (*missing, user_id)
            )
            import json
            for r in rows:
                chunks_map[r["chunk_id"]] = MemoryChunk(
                    chunk_id=r["chunk_id"], user_id=r["user_id"],
                    session_id=r["session_id"], text=r["text"],
                    source=r["source"], created_at=r["created_at"],
                    metadata=json.loads(r["metadata"] or "{}"),
                )

        items = [chunks_map[cid] for cid in top_ids if cid in chunks_map]
        return RetrievalResult(
            items=items, layer="long_term_hybrid", query=query,
            total_found=len(items),
            elapsed_ms=(_time.perf_counter() - t0) * 1000,
        )

    def _apply_mmr(self, result: RetrievalResult, top_k: int) -> RetrievalResult:
        candidates = [(item.text if hasattr(item, "text") else item.fact,
                       float(getattr(item, "confidence", 1.0)))
                      for item in result.items]
        reranked_texts = {text for text, _ in mmr_rerank(candidates, top_k, self._mmr_lambda)}
        new_items = [item for item in result.items
                     if (item.text if hasattr(item, "text") else item.fact) in reranked_texts]
        result.items = new_items[:top_k]
        result.total_found = len(new_items)
        return result

    # ═══════════════════════════════════════════════════════
    # CONTEXT BUILDERS
    # ═══════════════════════════════════════════════════════

    def build_context_budgeted(
        self,
        session_state: SessionState,
        query:         str,
        budget:        Optional[int] = None,
        top_k:         int = 8,
        environment:   Optional[str] = None,
        tiktoken_fn:   Optional[Callable[[str], int]] = None,
        include_skills: bool = True,
    ) -> Tuple[str, BudgetReport]:
        """
        Build context with priority-based token budget.

        v5 adds:
          P2 (shared with confirmed_facts): relevant Skills from SkillStore
          Semantic embedding fallback for cold retrieval
        """
        token_budget = budget or self._default_budget
        count_fn = tiktoken_fn or self._tiktoken_fn or estimate_tokens
        mgr = ContextBudgetManager(token_budget=token_budget, tiktoken_fn=count_fn)

        user_id    = session_state.user_id
        session_id = session_state.session_id

        # P1: User profile
        if self.user_model:
            section = self.user_model.get_prompt_section(user_id, max_chars=600)
            if section:
                mgr.add(Priority.USER_PROFILE, section, "user_profile")

        # P2a: Confirmed facts (hot track)
        hot = session_state.build_hot_context()
        if "confirmed_facts" in hot:
            mgr.add(Priority.CONFIRMED_FACTS, hot["confirmed_facts"], "confirmed_facts")

        # P2b: Relevant skills (same priority as confirmed facts — procedural knowledge)
        if include_skills:
            skills_ctx = self.skill_store.build_skills_context(user_id, query, top_k=3)
            if skills_ctx:
                mgr.add(Priority.CONFIRMED_FACTS, skills_ctx, "skills")

        # P3: Working set
        if "working_set" in hot:
            mgr.add(Priority.WORKING_SET, hot["working_set"], "working_set")

        # P4: Mid-term facts (with semantic embedding if available)
        mt = self.mid_term.search(user_id=user_id, query=query,
                                  session_id=session_id, top_k=top_k * 2)
        if mt.items:
            candidates = [(f.fact, f.confidence) for f in mt.items]
            reranked   = mmr_rerank(candidates, top_k, self._mmr_lambda)
            sel_texts  = {t for t, _ in reranked}
            text = self._format_facts([f for f in mt.items if f.fact in sel_texts][:top_k])
            if text:
                mgr.add(Priority.MID_TERM_FACTS, text, "mid_term_facts")

        # P5: Long-term chunks (hybrid embedding + FTS if embedding available)
        if self._emb_index.size > 0:
            lt_result = self._hybrid_search_chunks(user_id, query, session_id,
                                                    None, top_k * 2, False)
        else:
            lt_result = self.long_term.search(user_id=user_id, query=query,
                                              top_k=top_k * 2)
        if lt_result.items:
            candidates = [(c.text, 1.0) for c in lt_result.items]
            reranked   = mmr_rerank(candidates, top_k, self._mmr_lambda)
            sel_texts  = {t for t, _ in reranked}
            text = self._format_chunks([c for c in lt_result.items
                                        if c.text in sel_texts][:top_k])
            if text:
                mgr.add(Priority.LONG_TERM_CHUNKS, text, "long_term_chunks")

        # P5 (also): Recent tool results
        if "recent_tools" in hot:
            mgr.add(Priority.LONG_TERM_CHUNKS, hot["recent_tools"], "recent_tools")

        # P6: Environment
        if environment:
            mgr.add(Priority.ENVIRONMENT, environment, "environment")

        return mgr.build()

    def build_context(self, user_id, query, session_id=None, max_chars=3_000,
                      include_facts=True, include_chunks=True,
                      include_user_profile=True, include_skills=True) -> str:
        """Backward-compatible context builder (no SessionState required)."""
        if max_chars <= 0:
            return ""
        sections: List[str] = []
        both = include_facts and include_chunks

        if include_user_profile and self.user_model:
            section = self.user_model.get_prompt_section(
                user_id, max_chars=int(max_chars * 0.15))
            if section:
                sections.append(section)

        if include_skills:
            skills_ctx = self.skill_store.build_skills_context(
                user_id, query, top_k=2)
            if skills_ctx:
                sections.append(skills_ctx[:int(max_chars * 0.20)])

        fact_budget  = int(max_chars * (0.45 if both else 0.60))
        chunk_budget = int(max_chars * (0.30 if both else 0.60))

        if include_facts:
            fr = self.mid_term.search(user_id=user_id, query=query,
                                      session_id=session_id, top_k=15)
            if fr.items:
                text = self._format_facts(fr.items, max_chars=fact_budget)
                if text:
                    sections.append(text)

        if include_chunks:
            cr = self.long_term.search(user_id=user_id, query=query, top_k=8)
            if cr.items:
                text = self._format_chunks(cr.items, max_chars=chunk_budget)
                if text:
                    sections.append(text)

        return "\n\n".join(sections)

    def _format_facts(self, facts, max_chars=9_999,
                      header="## Distilled Facts (mid-term)") -> str:
        lines = [header]
        used = len(header) + 1
        for f in facts:
            line = f"- [{f.fact_type}] {f.fact} (conf={f.confidence:.2f})"
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines) if len(lines) > 1 else ""

    def _format_chunks(self, chunks, max_chars=9_999,
                       header="## Relevant Memory (long-term)") -> str:
        lines = [header]
        used = len(header) + 1
        for c in chunks:
            tag  = f"[{c.session_id[:8]}]" if c.session_id else ""
            line = f"- {tag} {c.text[:400].replace(chr(10), ' ')}"
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines) if len(lines) > 1 else ""

    # ═══════════════════════════════════════════════════════
    # DUAL-TRACK SESSION
    # ═══════════════════════════════════════════════════════

    def get_session(self, user_id: str, session_id: str) -> SessionState:
        return self._sessions.get_or_create(user_id, session_id)

    def end_session(self, user_id: str, session_id: str) -> None:
        self._sessions.drop(user_id, session_id)

    def active_sessions(self, user_id: str) -> List[str]:
        return self._sessions.active_sessions(user_id)

    # ═══════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════

    def list_sessions(self, user_id: str) -> List[str]:
        return self.long_term.list_sessions(user_id)

    def clear_session_cache(self, user_id: str, session_id: str) -> int:
        return self.short_term.delete_session(user_id, session_id)

    def evict_expired_cache(self) -> int:
        return self.short_term.evict_expired()

    def estimate_tokens(self, text: str) -> int:
        fn = self._tiktoken_fn or estimate_tokens
        return fn(text)

    def stats(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        s: Dict[str, Any] = {
            "user_id":             user_id,
            "session_id":          session_id,
            "long_term_chunks":    self.long_term.count(user_id),
            "sessions":            len(self.list_sessions(user_id)),
            "active_hot_sessions": self._sessions.total_active,
            "skill_count":         self.skill_store.count(user_id),
            "embedding_backend":   self._emb_index.backend_name,
            "embedding_index_size":self._emb_index.size,
        }
        if session_id:
            s["mid_term_facts"]     = self.mid_term.count(user_id, session_id)
            s["short_term_entries"] = len(self.short_term.list_session(user_id, session_id))
            state = self._sessions.get(user_id, session_id)
            if state:
                s["hot_track"] = state.to_dict()
        if self.user_model:
            p = self.user_model.get_profile(user_id)
            s["user_profile"] = p.to_dict() if p else None
        return s
