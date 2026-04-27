"""
agent_memory/stores/skill_store.py
===================================
程序性记忆（Procedural Memory）— Skill Store

存储 Agent 通过实际执行固化下来的可复用技能（Skill）。
不同于陈述性 facts（"R1 在 10.0.0.1"），Skill 记录的是
"如何完成一类任务"的执行路径，例如：

    Skill: "BGP 邻居故障排查"
    steps:
      1. bgp_check(router) → 查看邻居状态
      2. syslog_search(peer_ip) → 查看连接历史
      3. ping(peer_ip) → 验证 L3 可达性
      4. interface_check(router) → 排除本地接口故障
      5. 若均正常 → 联系对端 NOC
    success_rate: 0.92

核心设计：
- 技能按 (user_id, skill_name) 唯一标识，支持版本迭代
- FTS5 关键词 + TF-IDF 语义双路检索（与长期记忆相同策略）
- 每次调用后更新 success_rate（指数移动平均）
- success_rate 低于阈值时自动降级为 deprecated
- steps 以 JSON 存储，支持任意结构（文本步骤 / 工具调用序列）
- 与 ContextBudgetManager 集成：注入 P2（比 confirmed_facts 略低）

集成用法：
    # 任务成功后固化技能
    mem.save_skill("alice", "bgp_fault_diagnosis",
                   description="BGP 邻居故障排查标准流程",
                   steps=["bgp_check(R1)", "syslog_search(peer_ip)", "ping(peer_ip)"],
                   tags=["bgp", "network", "diagnosis"])

    # 新任务开始时检索匹配技能
    skills = mem.find_skills("alice", "BGP neighbor went down", top_k=3)

    # 任务完成后更新成功率
    mem.record_skill_outcome("alice", skill_id, success=True)
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_memory.schemas import _uid, _now
from agent_memory.retrieval.vector_store import TFIDFIndex
from agent_memory.stores._db import get_pool

logger = logging.getLogger(__name__)

# ── DDL ──────────────────────────────────────────────────────────────────────

_DDL = [
    """CREATE TABLE IF NOT EXISTS skills (
        skill_id       TEXT PRIMARY KEY,
        user_id        TEXT NOT NULL,
        skill_name     TEXT NOT NULL,
        description    TEXT NOT NULL,
        steps_json     TEXT NOT NULL DEFAULT '[]',
        tags_json      TEXT NOT NULL DEFAULT '[]',
        version        INTEGER NOT NULL DEFAULT 1,
        use_count      INTEGER NOT NULL DEFAULT 0,
        success_count  INTEGER NOT NULL DEFAULT 0,
        success_rate   REAL NOT NULL DEFAULT 1.0,
        status         TEXT NOT NULL DEFAULT 'active',
        created_at     REAL NOT NULL,
        updated_at     REAL NOT NULL,
        last_used_at   REAL NOT NULL DEFAULT 0,
        metadata_json  TEXT NOT NULL DEFAULT '{}'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sk_user ON skills(user_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_sk_user_name ON skills(user_id, skill_name)",
    "CREATE INDEX IF NOT EXISTS idx_sk_status ON skills(user_id, status)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
        skill_id UNINDEXED,
        user_id UNINDEXED,
        description,
        tags
    )""",
]

_FTS5_RESERVED = re.compile(r'\b(AND|OR|NOT|NEAR|COLUMN|ROW|MATCH)\b', re.IGNORECASE)
_FTS5_SPECIAL  = re.compile(r'["\'\(\)\*\+\-\:\^\.\/ ]+')

DEPRECATE_THRESHOLD = 0.3   # success_rate 低于此值自动标记为 deprecated
EMA_ALPHA           = 0.2   # 成功率指数移动平均平滑系数


def _fts_safe(q: str) -> str:
    q = _FTS5_RESERVED.sub(" ", q)
    q = _FTS5_SPECIAL.sub(" ", q).strip()
    return q if q else "skill"


# ── Skill schema ──────────────────────────────────────────────────────────────

@dataclass
class Skill:
    skill_id:     str
    user_id:      str
    skill_name:   str
    description:  str
    steps:        List[Any]          # 文本步骤 or 工具调用 dict
    tags:         List[str]
    version:      int   = 1
    use_count:    int   = 0
    success_count:int   = 0
    success_rate: float = 1.0
    status:       str   = "active"   # active | deprecated | archived
    created_at:   float = field(default_factory=_now)
    updated_at:   float = field(default_factory=_now)
    last_used_at: float = 0.0
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self, max_steps: int = 8) -> str:
        """Format skill for context injection."""
        steps_str = "\n".join(
            f"  {i+1}. {s}" if isinstance(s, str) else f"  {i+1}. {json.dumps(s, ensure_ascii=False)}"
            for i, s in enumerate(self.steps[:max_steps])
        )
        tag_str = ", ".join(self.tags) if self.tags else "—"
        return (
            f"[Skill: {self.skill_name}] (v{self.version}, "
            f"成功率={self.success_rate:.0%}, 用过 {self.use_count} 次)\n"
            f"描述: {self.description}\n"
            f"标签: {tag_str}\n"
            f"执行步骤:\n{steps_str}"
        )

    def to_dict(self) -> dict:
        return {
            "skill_id":     self.skill_id,
            "user_id":      self.user_id,
            "skill_name":   self.skill_name,
            "description":  self.description,
            "steps":        self.steps,
            "tags":         self.tags,
            "version":      self.version,
            "use_count":    self.use_count,
            "success_count":self.success_count,
            "success_rate": round(self.success_rate, 4),
            "status":       self.status,
            "created_at":   self.created_at,
            "updated_at":   self.updated_at,
            "last_used_at": self.last_used_at,
            "metadata":     self.metadata,
        }


# ── SkillStore ────────────────────────────────────────────────────────────────

class SkillStore:
    """
    程序性记忆存储。每个用户拥有独立的技能库。

    特性：
    - 技能唯一标识: (user_id, skill_name)，重名则自动版本迭代
    - 双路检索: FTS5 关键词 + TF-IDF 语义
    - 成功率跟踪: 每次调用后 EMA 更新，低于阈值自动降级
    - Context 注入: to_prompt_block() 直接供 ContextBudgetManager 使用
    """

    def __init__(
        self,
        db_path: str | Path,
        max_index_docs: int = 5_000,
        deprecate_threshold: float = DEPRECATE_THRESHOLD,
    ) -> None:
        self._pool = get_pool(db_path)
        self._deprecate_threshold = deprecate_threshold
        self._max_index_docs = max_index_docs
        # user_id → TFIDFIndex
        self._indexes: OrderedDict[str, TFIDFIndex] = OrderedDict()
        self._idx_lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._pool.execute_write_many([(sql, ()) for sql in _DDL])

    def _index(self, user_id: str) -> TFIDFIndex:
        with self._idx_lock:
            if user_id in self._indexes:
                self._indexes.move_to_end(user_id)
                return self._indexes[user_id]
            if len(self._indexes) >= 128:
                del self._indexes[next(iter(self._indexes))]
            idx = TFIDFIndex(max_docs=self._max_index_docs)
            rows = self._pool.execute_read(
                "SELECT skill_id, description, tags_json FROM skills WHERE user_id=? AND status='active'",
                (user_id,)
            )
            for r in rows:
                tags = " ".join(json.loads(r["tags_json"] or "[]"))
                idx.add(r["skill_id"], r["description"] + " " + tags)
            self._indexes[user_id] = idx
            return idx

    # ── write ─────────────────────────────────────────────────────────────────

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
        保存或更新技能。同名技能存在时自动版本 +1。
        返回保存的 Skill 对象。
        """
        tags = tags or []
        now = _now()

        # Check if skill already exists
        existing = self._pool.execute_read(
            "SELECT skill_id, version FROM skills WHERE user_id=? AND skill_name=?",
            (user_id, skill_name)
        )

        if existing:
            # Update: bump version, keep use_count and success_rate
            old_id  = existing[0]["skill_id"]
            new_ver = existing[0]["version"] + 1
            self._pool.execute_write(
                """UPDATE skills SET description=?, steps_json=?, tags_json=?,
                   version=?, status='active', updated_at=?, metadata_json=?
                   WHERE skill_id=?""",
                (description, json.dumps(steps, ensure_ascii=False),
                 json.dumps(tags, ensure_ascii=False), new_ver,
                 now, json.dumps(metadata or {}, ensure_ascii=False), old_id)
            )
            # Update FTS5
            self._pool.execute_write(
                "INSERT OR REPLACE INTO skills_fts(skill_id, user_id, description, tags) VALUES (?,?,?,?)",
                (old_id, user_id, description, " ".join(tags))
            )
            skill_id = old_id
        else:
            skill_id = _uid()
            self._pool.execute_write(
                """INSERT INTO skills
                   (skill_id, user_id, skill_name, description, steps_json, tags_json,
                    version, use_count, success_count, success_rate, status,
                    created_at, updated_at, last_used_at, metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (skill_id, user_id, skill_name, description,
                 json.dumps(steps, ensure_ascii=False),
                 json.dumps(tags, ensure_ascii=False),
                 1, 0, 0, 1.0, "active", now, now, 0.0,
                 json.dumps(metadata or {}, ensure_ascii=False))
            )
            self._pool.execute_write(
                "INSERT INTO skills_fts(skill_id, user_id, description, tags) VALUES (?,?,?,?)",
                (skill_id, user_id, description, " ".join(tags))
            )

        # Update TF-IDF index
        idx = self._index(user_id)
        idx.add(skill_id, description + " " + " ".join(tags))

        logger.info("Saved skill '%s' for user %s (id=%s)", skill_name, user_id, skill_id[:8])
        return self.get_skill(user_id, skill_id)

    def record_outcome(
        self,
        user_id:  str,
        skill_id: str,
        success:  bool,
    ) -> Optional[float]:
        """
        记录一次技能调用结果，更新成功率（EMA）。
        失败率过高时自动标记为 deprecated。
        返回更新后的 success_rate，未找到时返回 None。
        """
        rows = self._pool.execute_read(
            "SELECT use_count, success_count, success_rate FROM skills WHERE skill_id=? AND user_id=?",
            (skill_id, user_id)
        )
        if not rows:
            return None
        r       = rows[0]
        new_uc  = r["use_count"] + 1
        new_sc  = r["success_count"] + (1 if success else 0)
        # EMA: blend new outcome into existing rate
        raw_rate   = new_sc / new_uc
        new_rate   = EMA_ALPHA * raw_rate + (1 - EMA_ALPHA) * r["success_rate"]
        new_status = "deprecated" if new_rate < self._deprecate_threshold else "active"
        self._pool.execute_write(
            """UPDATE skills SET use_count=?, success_count=?, success_rate=?,
               status=?, updated_at=?, last_used_at=? WHERE skill_id=?""",
            (new_uc, new_sc, new_rate, new_status, _now(), _now(), skill_id)
        )
        if new_status == "deprecated":
            logger.warning("Skill %s deprecated (success_rate=%.2f)", skill_id[:8], new_rate)
        return new_rate

    def delete_skill(self, user_id: str, skill_id: str) -> bool:
        n = self._pool.execute_write(
            "DELETE FROM skills WHERE skill_id=? AND user_id=?", (skill_id, user_id)
        )
        # Remove from FTS5 by deleting all rows matching skill_id
        self._pool.execute_write(
            "DELETE FROM skills_fts WHERE rowid IN ("
            "  SELECT rowid FROM skills_fts WHERE skill_id=?)", (skill_id,)
        )
        self._index(user_id).remove(skill_id)
        return n > 0

    # ── retrieve ──────────────────────────────────────────────────────────────

    def find_skills(
        self,
        user_id:       str,
        query:         str,
        top_k:         int  = 3,
        include_deprecated: bool = False,
    ) -> List[Skill]:
        """
        混合检索（FTS5 + TF-IDF）找最匹配的技能。
        默认只返回 active 状态技能。
        """
        query = (query or "").strip()
        candidate_ids: set[str] = set()
        tfidf_score: dict[str, float] = {}
        fts_ids: set[str] = set()

        if query:
            safe_q = _fts_safe(query)
            fts_rows = self._pool.execute_read(
                "SELECT skill_id FROM skills_fts WHERE skills_fts MATCH ? AND user_id=? LIMIT ?",
                (safe_q, user_id, top_k * 4)
            )
            fts_ids = {r["skill_id"] for r in fts_rows}
            candidate_ids.update(fts_ids)

            for sid, sc in self._index(user_id).query(query, top_k=top_k * 4):
                candidate_ids.add(sid)
                tfidf_score[sid] = sc

        if not candidate_ids:
            # Fallback: return most recently used active skills
            status_clause = "" if include_deprecated else "AND status='active'"
            rows = self._pool.execute_read(
                f"SELECT * FROM skills WHERE user_id=? {status_clause} "
                f"ORDER BY last_used_at DESC LIMIT ?",
                (user_id, top_k)
            )
            return [self._row_to_skill(r) for r in rows]

        ph = ",".join("?" * len(candidate_ids))
        status_clause = "" if include_deprecated else "AND status='active'"
        rows = self._pool.execute_read(
            f"SELECT * FROM skills WHERE skill_id IN ({ph}) AND user_id=? {status_clause}",
            (*candidate_ids, user_id)
        )

        scored = []
        for r in rows:
            score = tfidf_score.get(r["skill_id"], 0.0) * r["success_rate"]
            score += 0.5 if r["skill_id"] in fts_ids else 0.0
            scored.append((score, r["last_used_at"], r))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        return [self._row_to_skill(r) for _, _, r in scored[:top_k]]

    def get_skill(self, user_id: str, skill_id: str) -> Optional[Skill]:
        rows = self._pool.execute_read(
            "SELECT * FROM skills WHERE skill_id=? AND user_id=?", (skill_id, user_id)
        )
        return self._row_to_skill(rows[0]) if rows else None

    def get_skill_by_name(self, user_id: str, skill_name: str) -> Optional[Skill]:
        rows = self._pool.execute_read(
            "SELECT * FROM skills WHERE user_id=? AND skill_name=?", (user_id, skill_name)
        )
        return self._row_to_skill(rows[0]) if rows else None

    def list_skills(
        self,
        user_id:  str,
        status:   Optional[str] = "active",
        limit:    int = 50,
    ) -> List[Skill]:
        if status:
            rows = self._pool.execute_read(
                "SELECT * FROM skills WHERE user_id=? AND status=? ORDER BY last_used_at DESC LIMIT ?",
                (user_id, status, limit)
            )
        else:
            rows = self._pool.execute_read(
                "SELECT * FROM skills WHERE user_id=? ORDER BY last_used_at DESC LIMIT ?",
                (user_id, limit)
            )
        return [self._row_to_skill(r) for r in rows]

    def count(self, user_id: str, status: Optional[str] = "active") -> int:
        if status:
            rows = self._pool.execute_read(
                "SELECT COUNT(*) as n FROM skills WHERE user_id=? AND status=?", (user_id, status)
            )
        else:
            rows = self._pool.execute_read(
                "SELECT COUNT(*) as n FROM skills WHERE user_id=?", (user_id,)
            )
        return rows[0]["n"] if rows else 0

    def build_skills_context(self, user_id: str, query: str, top_k: int = 3) -> str:
        """Format top matching skills as a context block for prompt injection."""
        skills = self.find_skills(user_id, query, top_k=top_k)
        if not skills:
            return ""
        lines = ["## 可复用技能 (Skill Library)"]
        for sk in skills:
            lines.append("")
            lines.append(sk.to_prompt_block())
        return "\n".join(lines)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _row_to_skill(self, r) -> Skill:
        return Skill(
            skill_id=r["skill_id"], user_id=r["user_id"],
            skill_name=r["skill_name"], description=r["description"],
            steps=json.loads(r["steps_json"] or "[]"),
            tags=json.loads(r["tags_json"] or "[]"),
            version=r["version"], use_count=r["use_count"],
            success_count=r["success_count"], success_rate=r["success_rate"],
            status=r["status"], created_at=r["created_at"],
            updated_at=r["updated_at"], last_used_at=r["last_used_at"],
            metadata=json.loads(r["metadata_json"] or "{}"),
        )
