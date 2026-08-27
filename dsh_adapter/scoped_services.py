"""Session-scoped memory recall and profile-scoped harness capability retrieval."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

from retrieval.bm25 import BM25Retriever
from retrieval.factory import skills_to_corpus, tools_to_corpus

from .backend import open_backend, resolve_backend_mode


# Curated network-operations vocabulary used only as retrieval evidence. It
# does not execute tools or infer arguments. Keeping the aliases explicit makes
# capability routing deterministic and reviewable across English/Chinese
# operator language instead of relying on an LLM to guess the intended skill.
_SKILL_ALIASES: dict[str, str] = {
    "read_stored_result": (
        "stored reference stored id durable result page pagination offset chunk "
        "truncated output load more 大结果 落盘 分页 下一页 结果片段 继续读取"
    ),
    "alert_summary": (
        "active alert warning critical event severity grouped alert table "
        "告警汇总 监控告警 严重级别 告警统计 活跃告警"
    ),
    "app_access_troubleshoot": (
        "cross domain application access troubleshoot LAN admission then DC permission "
        "RADIUS NAC VLAN 802.1X application RBAC intermittent app failure "
        "端到端应用访问 跨域诊断 LAN准入 DC权限 应用访问失败 时断时续"
    ),
    "branch_app_reachability": (
        "branch office remote office branch to DC WAN circuit tunnel SLA headquarters "
        "three domain LAN WAN DC transport fabric path "
        "分支机构 分公司 总部应用 分支到DC 电路 隧道 传输路径 三域"
    ),
    "lan_new_employee_onboarding_access": (
        "new employee new hire new starter onboarding provision access grant and verify "
        "network admission application role RBAC "
        "新员工 新入职 新人 开通权限 端到端开通 准入和应用权限"
    ),
    "lan_user_access_diagnose": (
        "user identity network access denied LAN-side admission RADIUS NAC VLAN "
        "classify LAN versus application permission "
        "用户网络权限 身份准入 访问被拒绝 NAC策略 VLAN准入 LAN问题"
    ),
    "netflow_analysis": (
        "netflow flow records traffic spike bandwidth top talkers source destination "
        "network conversations bytes east west suspicious traffic "
        "流量分析 异常流量 带宽占用 通信双方 流量排行"
    ),
    "prometheus_query": (
        "prometheus promql metrics time series rate cpu memory packet loss interface errors "
        "监控指标 时间序列 接口错误率 CPU指标 内存指标 丢包"
    ),
    "restart_service": (
        "restart service rolling restart graceful restart recycle replicas controlled restart "
        "滚动重启 逐个重启 服务重启 副本重启"
    ),
    "rollback_service": (
        "rollback service roll back revert restore previous version undo deployment "
        "回滚服务 恢复版本 撤销变更 上一版本 失败变更"
    ),
    "service_health": (
        "service health healthy degraded status latency uptime replicas database api "
        "服务健康 服务状态 延迟 副本 数据库健康 是否降级"
    ),
    "syslog_search": (
        "syslog device logs router logs switch logs interface flap link down log severity "
        "设备日志 网络日志 交换机日志 端口抖动 链路中断 日志过滤"
    ),
}


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
                item["text"] = " ".join((
                    item.get("text", ""),
                    _SKILL_ALIASES.get(str(item.get("id", "")), ""),
                )).strip()
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
