"""
agent_memory/consolidation.py
==============================
Memory Consolidation（历史压缩）& Reflect（反思写入）

两个相互配合的机制：

1. MemoryConsolidator — 上下文压缩
   将 session 内过多的历史 chunks 用 LLM 摘要压缩为更少的 summary chunks，
   从而控制长会话的 token 消耗（GA 的核心竞争力：<30K 上下文窗口）。

   触发时机（任选其一）：
     a. 每 N 轮自动触发（consolidate_after_n_turns）
     b. 手动调用 mem.consolidate_session()
     c. 检测到总 token 超过阈值

   压缩策略：
     - 保留最近 keep_recent_n 条 chunks 原文（当前轮上下文）
     - 对更旧的 chunks 按时间分组，每组调用 LLM 生成摘要
     - 摘要替换原始 chunks（delete 旧的，insert summary chunk）
     - summary chunk 的 importance = 0.85（高于普通 conversation = 0.5）
     - 无 LLM 时 fallback 到拼接截断（保留前 N 字符）

2. ReflectionEngine — 反思写入
   任务完成（或失败）后，让 Agent 生成反思文本，写入记忆：
     - 成功：记录"什么做法有效"→ Skill 步骤补充，high-importance chunk
     - 失败：记录"什么没有奏效"→ lesson fact，避免下次重蹈覆辙
     - 矛盾：若结果与 mid-term fact 矛盾 → 触发 confidence decay

   反思数据同时写入：
     - LongTermStore：source="reflection"，importance=0.9
     - MidTermStore：fact_type="lesson"，confidence 根据成功/失败动态设置
     - SkillStore：更新相关 Skill 的 success_rate
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from agent_memory.schemas import MemoryChunk, MemoryFact, _uid, _now

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_SUMMARY_PROMPT = """\
请将以下对话历史压缩为一段简洁的摘要（不超过200字）。
保留所有重要的事实、决策和结论。忽略重复内容和闲聊。
用第三人称描述。不要添加分析或评价，只保留关键信息。

对话历史：
{history}

摘要（直接输出，不要前缀）："""

_REFLECT_SUCCESS_PROMPT = """\
以下任务已成功完成。请生成一段简洁的反思（不超过150字）：
  - 哪些关键步骤起到了决定性作用？
  - 有什么值得下次复用的经验？

任务：{task}
执行过程摘要：{summary}
结果：成功

反思（直接输出）："""

_REFLECT_FAILURE_PROMPT = """\
以下任务执行失败。请生成一段简洁的反思（不超过150字）：
  - 哪个环节出了问题？
  - 下次应该怎么做不同？

任务：{task}
执行过程摘要：{summary}
失败原因：{reason}

反思（直接输出）："""

_LESSON_EXTRACT_PROMPT = """\
从以下反思中提取1-3条可操作的经验教训，每条不超过60字。
格式：JSON 数组，每项有 "lesson"（教训文本）和 "confidence"（0.6-1.0）。
只返回 JSON，不要其他内容。

反思：
{reflection}

JSON："""


# ══════════════════════════════════════════════════════════════════════════════
# MemoryConsolidator
# ══════════════════════════════════════════════════════════════════════════════

class MemoryConsolidator:
    """
    历史压缩引擎。控制长会话的 token 消耗。

    与 MemoryManager 集成：
        mem = MemoryManager(data_dir="./data", llm_fn=my_llm,
                            consolidate_after_n_turns=20)

        # 自动触发（内部在 remember() 后检查）
        # 或手动触发：
        result = mem.consolidate_session(user_id, session_id)
        print(f"压缩了 {result['chunks_merged']} 条 → {result['summaries_created']} 条摘要")
    """

    def __init__(
        self,
        llm_fn:              Optional[Callable[[str], str]] = None,
        consolidate_after_n: int = 30,    # 每 N 条 chunks 触发压缩
        keep_recent_n:       int = 10,    # 最近 N 条不压缩
        group_size:          int = 8,     # 每组几条 chunks 合并为一条摘要
        summary_importance:  float = 0.85,
    ) -> None:
        self._llm_fn        = llm_fn
        self._trigger_n     = consolidate_after_n
        self._keep_recent   = keep_recent_n
        self._group_size    = group_size
        self._summary_imp   = summary_importance

    def should_consolidate(self, chunk_count: int) -> bool:
        return chunk_count >= self._trigger_n

    def consolidate(
        self,
        long_term_store,    # LongTermStore instance
        user_id:    str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        压缩 session 历史 chunks。

        Returns:
            {
              "chunks_before": int,
              "chunks_after":  int,
              "chunks_merged": int,
              "summaries_created": int,
              "skipped": bool,
            }
        """
        from agent_memory.stores._db import get_pool

        pool = long_term_store._pool

        # Fetch all chunks for this session, ordered by time
        rows = pool.execute_read(
            "SELECT chunk_id, text, created_at, source FROM long_term_chunks "
            "WHERE user_id=? AND session_id=? ORDER BY created_at ASC",
            (user_id, session_id)
        )

        if len(rows) < self._trigger_n:
            return {"skipped": True, "reason": "not enough chunks"}

        # Split: keep recent N, compress the rest
        to_keep   = rows[-self._keep_recent:]
        to_merge  = rows[:-self._keep_recent]

        if len(to_merge) < 2:
            return {"skipped": True, "reason": "too few chunks to compress"}

        chunks_before = len(rows)
        summaries_created = 0
        merged_ids: List[str] = []

        # Process in groups
        for i in range(0, len(to_merge), self._group_size):
            group = to_merge[i : i + self._group_size]
            if not group:
                continue

            history_text = "\n\n".join(
                f"[{r['source']}] {r['text'][:600]}" for r in group
            )
            summary_text = self._summarize(history_text)
            if not summary_text:
                continue

            # Write summary chunk (inherit earliest timestamp of group)
            summary_chunk = MemoryChunk(
                chunk_id=_uid(),
                user_id=user_id,
                session_id=session_id,
                text=f"[摘要] {summary_text}",
                source="summary",
                created_at=group[0]["created_at"],
                metadata={"compressed_from": [r["chunk_id"] for r in group]},
            )
            long_term_store.add_chunk(summary_chunk, importance=self._summary_imp)

            # Delete original chunks
            for r in group:
                long_term_store.delete_chunk(user_id, r["chunk_id"])
                merged_ids.append(r["chunk_id"])

            summaries_created += 1

        chunks_after = len(to_keep) + summaries_created
        logger.info(
            "Consolidated session %s: %d → %d chunks (%d summaries)",
            session_id[:12], chunks_before, chunks_after, summaries_created
        )
        return {
            "skipped":           False,
            "chunks_before":     chunks_before,
            "chunks_after":      chunks_after,
            "chunks_merged":     len(merged_ids),
            "summaries_created": summaries_created,
        }

    def _summarize(self, history: str) -> str:
        if self._llm_fn:
            try:
                prompt = _SUMMARY_PROMPT.format(history=history[:3000])
                result = self._llm_fn(prompt)
                return result.strip() if result else self._fallback_summary(history)
            except Exception as e:
                logger.warning("LLM summarization failed: %s", e)
        return self._fallback_summary(history)

    @staticmethod
    def _fallback_summary(history: str, max_chars: int = 400) -> str:
        """No-LLM fallback: truncate and label."""
        lines = [l.strip() for l in history.split("\n") if l.strip()]
        result = " | ".join(lines)
        return result[:max_chars] + ("…" if len(result) > max_chars else "")


# ══════════════════════════════════════════════════════════════════════════════
# ReflectionEngine
# ══════════════════════════════════════════════════════════════════════════════

class ReflectionEngine:
    """
    反思写入引擎。任务完成/失败后提炼经验教训。

    与 MemoryManager 集成：
        # 任务成功后
        result = mem.reflect(
            user_id, session_id,
            task="排查 AS65002 BGP 故障",
            outcome="success",
            summary="通过 ping + syslog 确认对端不可达，联系 NOC 后恢复",
            skill_id="bgp-diagnosis-skill-id",  # 可选：更新关联 Skill 的成功率
        )

        # 任务失败后
        result = mem.reflect(
            user_id, session_id,
            task="自动重启 BGP 进程",
            outcome="failure",
            reason="权限不足，缺少 write 访问",
            summary="尝试了 3 种方式均被拒绝",
        )
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._llm_fn = llm_fn

    def reflect(
        self,
        long_term_store,
        mid_term_store,
        skill_store,
        user_id:    str,
        session_id: str,
        task:       str,
        outcome:    str,          # "success" | "failure" | "partial"
        summary:    str,
        reason:     str = "",     # failure reason
        skill_id:   Optional[str] = None,
        metadata:   Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        生成反思，写入各记忆层，返回写入汇总。

        写入：
          LongTermStore  → source="reflection", importance=0.9
          MidTermStore   → fact_type="lesson"
          SkillStore     → record_outcome (if skill_id given)
        """
        success = outcome == "success"

        # 1. 生成反思文本
        reflection = self._generate_reflection(task, summary, outcome, reason)
        if not reflection:
            return {"skipped": True, "reason": "empty reflection"}

        now = _now()
        written: Dict[str, Any] = {"reflection": reflection, "outcome": outcome}

        # 2. 写入长期记忆（高 importance）
        chunk = MemoryChunk(
            chunk_id=_uid(),
            user_id=user_id,
            session_id=session_id,
            text=f"[反思/{outcome}] 任务：{task}\n{reflection}",
            source="reflection",
            created_at=now,
            metadata={"task": task, "outcome": outcome, **(metadata or {})},
        )
        long_term_store.add_chunk(chunk, importance=0.9)
        written["long_term_chunk_id"] = chunk.chunk_id

        # 3. 提炼 lesson facts 写入中期记忆
        lessons = self._extract_lessons(reflection)
        fact_ids = []
        for lesson in lessons:
            conf = lesson.get("confidence", 0.7 if success else 0.8)
            # 失败教训置信度略高（避免重蹈覆辙比复用成功经验更紧迫）
            fact = MemoryFact(
                user_id=user_id,
                session_id=session_id,
                fact=lesson.get("lesson", ""),
                fact_type="lesson",
                confidence=conf,
                source_chunk_ids=[chunk.chunk_id],
                metadata={"task": task, "outcome": outcome},
            )
            if fact.fact:
                mid_term_store.add_fact(fact)
                fact_ids.append(fact.fact_id)
        written["lesson_fact_ids"] = fact_ids

        # 4. 更新 Skill 成功率
        if skill_id and skill_store:
            new_rate = skill_store.record_outcome(user_id, skill_id, success)
            written["skill_id"]      = skill_id
            written["new_skill_rate"] = new_rate

        logger.info(
            "Reflected on task '%s' (%s): %d lessons, skill_updated=%s",
            task[:40], outcome, len(fact_ids), skill_id is not None
        )
        return written

    def _generate_reflection(self, task, summary, outcome, reason) -> str:
        if not self._llm_fn:
            return self._heuristic_reflection(task, outcome, reason, summary)
        try:
            if outcome == "success":
                prompt = _REFLECT_SUCCESS_PROMPT.format(task=task, summary=summary[:800])
            else:
                prompt = _REFLECT_FAILURE_PROMPT.format(
                    task=task, summary=summary[:600], reason=reason[:200]
                )
            result = self._llm_fn(prompt)
            return result.strip() if result else self._heuristic_reflection(task, outcome, reason, summary)
        except Exception as e:
            logger.warning("LLM reflection failed: %s", e)
            return self._heuristic_reflection(task, outcome, reason, summary)

    @staticmethod
    def _heuristic_reflection(task, outcome, reason, summary) -> str:
        if outcome == "success":
            return f"任务「{task}」成功完成。关键过程：{summary[:200]}"
        else:
            r = f"，原因：{reason}" if reason else ""
            return f"任务「{task}」{outcome}{r}。过程：{summary[:200]}"

    def _extract_lessons(self, reflection: str) -> List[Dict]:
        if not self._llm_fn:
            return self._heuristic_lessons(reflection)
        try:
            prompt = _LESSON_EXTRACT_PROMPT.format(reflection=reflection[:800])
            raw    = self._llm_fn(prompt)
            import re, json as _json
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            data = _json.loads(raw)
            return data if isinstance(data, list) else []
        except Exception:
            return self._heuristic_lessons(reflection)

    @staticmethod
    def _heuristic_lessons(reflection: str) -> List[Dict]:
        """Rule-based lesson extraction without LLM."""
        lessons = []
        # Split by sentences and take informative ones
        import re
        sentences = re.split(r'[。；\n]', reflection)
        for s in sentences:
            s = s.strip()
            if len(s) > 15 and any(kw in s for kw in
                                    ["应该", "需要", "必须", "避免", "建议", "关键",
                                     "有效", "失败", "成功", "发现", "注意", "记住"]):
                lessons.append({"lesson": s[:120], "confidence": 0.65})
        return lessons[:3]
