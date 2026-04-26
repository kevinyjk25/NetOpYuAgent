#!/usr/bin/env python3
"""
examples/e2e_scenario.py  — v5
================================
端到端场景演示：IT 运维 Agent 处理 BGP 故障（含 v5 新能力）

覆盖所有记忆模块能力：
  Round 0  历史注入      → remember_batch, add_fact, update_user_profile
  Round 1  初步诊断      → get_session, save_skill, build_context_budgeted (含 Skills P2)
                           cache_tool_result, read_cached (byte-offset)
                           confirm_fact, add_to_working_set
  Round 2  深入排查      → 热轨道累积, confidence_decay, 语义 embedding 检索
  Round 3  影响评估      → MMR 多样性, consolidate_session (历史压缩), BudgetReport
  Round 4  恢复确认      → reflect (反思写入), record_skill_outcome, 跨 session 召回
  After    维护操作      → evict_expired_facts, apply_retention, garbage_collect,
                           checkpoint, end_session, stats

运行：
    cd <project_root>
    python -m agent_memory.examples.e2e_scenario
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json, math, time, tempfile, hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Any

from agent_memory import (
    MemoryManager, MemoryChunk, MemoryFact,
    Priority, estimate_tokens,
)

# ── 彩色输出 ─────────────────────────────────────────────────────────────────
R="\033[0m"; B="\033[1m"; DIM="\033[2m"
RED="\033[91m"; GRN="\033[92m"; YLW="\033[93m"
BLU="\033[94m"; MAG="\033[95m"; CYN="\033[96m"; WHT="\033[97m"; GRY="\033[90m"

def banner(t, c=BLU): print(f"\n{c}{B}{'═'*68}\n  {t}\n{'═'*68}{R}")
def section(t, c=CYN): print(f"\n{c}{B}── {t} {'─'*(62-len(t))}{R}")
def ok(m): print(f"  {GRN}✓{R} {m}")
def info(m): print(f"  {CYN}→{R} {m}")
def warn(m): print(f"  {YLW}⚠{R} {m}")
def log(k, v="", c=WHT): print(f"  {GRY}{k}:{R} {c}{v}{R}")
def sep(): print(f"  {DIM}{'·'*64}{R}")

def ctx_preview(ctx, n=18):
    for ln in ctx.split("\n")[:n]:
        print(f"  {DIM}│{R} {GRY}{ln}{R}")
    lines = ctx.split("\n")
    if len(lines) > n:
        print(f"  {DIM}│{R} {GRY}... ({len(lines)-n} 更多行){R}")

def budget_summary(report):
    pct = f"{report.utilization:.0%}"
    print(f"  {CYN}Token budget:{R} {report.used_tokens}/{report.total_budget} ({pct})")
    for name, toks in report.kept_sections:
        bar = "█" * max(1, toks * 18 // max(report.total_budget, 1))
        print(f"  {GRN}  ✓ {name:<26}{R}{GRY}{toks:>5} tok  {GRN}{bar}{R}")
    for name, toks in report.dropped_sections:
        print(f"  {RED}  ✗ {name:<26}{R}{GRY}{toks:>5} tok  (dropped){R}")

# ── 模拟工具 ──────────────────────────────────────────────────────────────────
def fake_bgp_check(router):
    return f"BGP Neighbor Table — {router}\n10.0.1.2 AS65002  Idle   00:00:00  0  ← PROBLEM\n10.0.1.3 AS65003  Established 3d02h 2847\n10.0.1.4 AS65004  Established 7d14h 5921"

def fake_syslog(pattern, hours=24):
    entries = [
        "BGP: Peer 10.0.1.2 went from Established to Idle",
        "BGP: Hold timer expired for neighbor 10.0.1.2",
        "BGP: Notification sent: Hold Timer Expired (error 4)",
        "BGP: Peer 10.0.1.2 connection reset, retrying in 30s",
        "BGP: Retrying connection to 10.0.1.2 (attempt 47)",
        "SNMP: Trap sent for bgpBackwardTransition",
    ] * 14   # 大于 4KB
    base = datetime.now() - timedelta(hours=hours)
    return "\n".join(f"{(base+timedelta(minutes=i*3)).strftime('%b %d %H:%M:%S')} R1 {e}"
                     for i, e in enumerate(entries))

def fake_ping(host):
    return ("PING 10.0.1.2: 100% packet loss — host unreachable"
            if "10.0.1.2" in host else f"PING {host}: 4ms avg, 0% packet loss — OK")

def fake_interface_check(router):
    return f"GigabitEthernet0/1  UP  UP  10.0.1.1/30  (to AS65002) ← UP at L1/L2"

def fake_route_check(prefix):
    return (f"Route {prefix}: via AS65003 metric=100 (primary)\n"
            f"              via AS65004 metric=200 (backup)\n"
            f"Note: AS65002 path WITHDRAWN — peer in Idle")

# ── 无 LLM 的 Agent 推理（规则代替 LLM） ────────────────────────────────────
def agent_think(query, ctx, turn):
    responses = {
        1: ("BGP 邻居 10.0.1.2 (AS65002) 处于 Idle 状态，Hold Timer 在约3小时前到期。"
            "日志显示已重试47次均以 ETIMEDOUT 失败。Ping 确认对端不可达——建议联系 AS65002 NOC。",
            [{"tool":"bgp_check","args":{"router":"R1"}},
             {"tool":"syslog_search","args":{"pattern":"BGP 10.0.1.2","hours":24}},
             {"tool":"ping","args":{"host":"10.0.1.2"}}]),
        2: ("R1 所有接口物理层正常（UP/UP），排除本地硬件故障。AS65003 可达，其他会话正常。"
            "问题高度确认为 AS65002 侧单边故障，建议通过 NOC 工单联系。",
            [{"tool":"interface_check","args":{"router":"R1"}},
             {"tool":"ping","args":{"host":"10.0.1.3"}}]),
        3: ("192.168.100.0/24 通过 AS65003/AS65004 正常可达，AS65002 路径已撤销。"
            "流量已自动切换，服务影响最小。",
            [{"tool":"route_check","args":{"prefix":"192.168.100.0/24"}}]),
        4: ("AS65002 已恢复，BGP 会话重新建立（Established）。"
            "路由 192.168.100.0/24 通过 AS65002 重新通告，三条路径均可用。"
            "建议更新 NOC 工单并标记此类故障排查流程为标准 SOP。",
            []),
    }
    return responses.get(turn, ("…", []))

# ══════════════════════════════════════════════════════════════════════════════

def run_scenario():
    banner("Agent Memory Module v5 — 端到端场景演示", BLU)
    print(f"""
  {WHT}场景:{R} IT 运维 Agent 处理 BGP 故障排查（AS65002 断连）
  {WHT}新增:{R} 程序性记忆(Skill)、历史压缩(Consolidation)、反思写入(Reflect)、语义向量检索

  {DIM}（无 LLM，Agent 响应由规则生成，等效于 LLM 推理结果）{R}
""")

    # ── 构建可选的语义 embedding（纯 Python，无外部依赖） ─────────────────────
    def simple_embed(text: str) -> List[float]:
        """4维 hash-based embedding，仅供演示语义检索路径。"""
        h = hashlib.md5(text.encode()).digest()
        vec = [(b / 127.5) - 1.0 for b in h[:4]]
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / norm for x in vec]

    data_dir = tempfile.mkdtemp(prefix="agent_memory_v5_demo_")
    mem = MemoryManager(
        data_dir=data_dir,
        embedding_fn=simple_embed,    # 语义向量检索（v5 新）
        embedding_dim=4,
        inline_threshold=4_000,
        default_token_budget=3_200,
        mmr_lambda=0.6,
        consolidate_after_n_turns=8,  # 8 条后触发压缩
        min_fact_confidence=0.4,
        enable_user_model=True,
    )
    USER_ID, SESSION_ID = "alice", f"session-bgp-{int(time.time())}"
    log("data_dir",    data_dir,   GRN)
    log("user_id",     USER_ID,    GRN)
    log("session_id",  SESSION_ID, GRN)
    log("embedding",   mem._emb_index.backend_name, GRN)
    ok("MemoryManager v5 初始化完成")

    # ══════════════════════════════════════════════════════════════════════════
    banner("Round 0 │ 历史数据注入 + 保存历史技能", YLW)

    section("写入：长期记忆 — 批量历史 chunks", YLW)
    old_sess = "session-weekly-review"
    history = [
        "用户alice确认：R1（IP 10.0.0.1）是核心BGP路由器，运行Cisco IOS 15.4",
        "AS65001 BGP 对等关系：AS65002(10.0.1.2), AS65003(10.0.1.3), AS65004(10.0.1.4)",
        "2024年1月历史事件：AS65002断开2小时，原因是对端维护未提前通知",
        "alice偏好使用CLI工具（show命令）而非REST API",
        "AS65002 NOC联系方式：noc@as65002.net，紧急电话+1-800-555-0102",
        "BGP最佳实践：Hold Timer到期后先查L1/L2，再确认对端状态，最后联系NOC",
    ]
    mem.remember_batch(USER_ID, old_sess, history)
    ok(f"批量写入 {len(history)} 条历史 chunks（单事务 + Embedding 索引）")
    log("embedding_index_size", str(mem._emb_index.size), GRN)

    section("写入：中期记忆 — 历史提炼事实", YLW)
    facts = [
        ("R1 路由器 IP 10.0.0.1，Cisco IOS 15.4", "entity", 0.95),
        ("AS65002 BGP 邻居地址：10.0.1.2",         "entity", 0.95),
        ("alice 偏好 CLI 工具进行网络诊断",          "preference", 0.85),
        ("BGP Hold Timer 到期后先查 L1/L2",        "procedure", 0.88),
    ]
    for fact_text, fact_type, conf in facts:
        mem.add_fact(USER_ID, old_sess, fact_text, fact_type, conf)
    # 演示去重
    mem.add_fact(USER_ID, old_sess, "R1 路由器 IP 10.0.0.1，Cisco IOS 15.4", "entity", 0.9)
    ok(f"写入 {len(facts)} 条 facts（+1 重复 → INSERT OR IGNORE 去重）")

    section("写入：程序性记忆 (Skill Store) — v5 新能力", YLW)
    sk = mem.save_skill(
        USER_ID, "bgp_fault_diagnosis",
        description="BGP邻居故障排查：ping验证对端可达性、syslog分析连接历史、接口检查排除本地故障、联系NOC",
        steps=[
            "1. bgp_check(router) — 查看所有邻居状态，识别 Idle 邻居",
            "2. syslog_search(peer_ip, hours=24) — 分析连接断开历史和错误日志",
            "3. ping(peer_ip) — 验证 L3 可达性（区分 BGP 配置错误 vs 网络不通）",
            "4. interface_check(router) — 确认本地接口 L1/L2 状态正常",
            "5. 若 ping 失败且接口正常 → 问题在对端，联系 AS NOC",
        ],
        tags=["bgp", "network", "fault-diagnosis", "routing"],
        metadata={"source": "historical", "domain": "network-ops"},
    )
    ok(f"Skill 已保存：{sk.skill_name!r} v{sk.version} [{sk.skill_id[:8]}]")
    log("steps",   str(len(sk.steps)), GRN)
    log("tags",    str(sk.tags),       GRN)
    log("status",  sk.status,          GRN)

    section("写入：用户行为画像", YLW)
    mem.update_user_profile(USER_ID, old_sess,
        "R1 BGP show bgp summary 检查 CLI 命令",
        "以下是 CLI 命令...",
        tool_calls=[{"tool": "bgp_check"}])
    profile = mem.get_user_profile(USER_ID)
    log("technical_level",     profile.technical_level.value,     GRN)
    log("domain_counts",       str(profile.domain_counts),        GRN)
    ok("用户画像已构建")
    time.sleep(0.05)

    # ══════════════════════════════════════════════════════════════════════════
    banner("Round 1 │ 用户报告故障 — 初步诊断（含 Skill P2 注入）", CYN)
    Q1 = "BGP 告警：R1 与 AS65002 的 session 断了，能帮我看一下吗？"
    print(f"\n  {B}用户:{R} {Q1}\n")

    section("读取：获取会话状态 (SessionState)", CYN)
    state = mem.get_session(USER_ID, SESSION_ID)
    ok(f"热轨道已创建：confirmed_facts={len(state.confirmed_facts)}, working_set={len(state.working_set)}")

    section("读取：构建 Context（P2 包含 Skill）— v5 新行为", CYN)
    ctx1, rep1 = mem.build_context_budgeted(state, Q1)
    budget_summary(rep1)
    info("注意：'skills' 出现在 P2（与 confirmed_facts 同级）")
    print()
    ctx_preview(ctx1, n=20)

    section("执行：工具调用", CYN)
    bgp_out  = fake_bgp_check("R1")
    syslog_out = fake_syslog("BGP 10.0.1.2")
    ping_out = fake_ping("10.0.1.2")

    state.add_tool_result("bgp_check", bgp_out)
    ok(f"bgp_check: {len(bgp_out)} chars（inline）")

    syslog_entry = mem.cache_tool_result(USER_ID, SESSION_ID, "syslog_search", syslog_out)
    state.add_tool_result("syslog_search", "", ref_id=syslog_entry.ref_id)
    syslog_token = mem.get_cache_preview(USER_ID, syslog_entry.ref_id)
    ok(f"syslog_search: {syslog_entry.total_length} chars → 文件缓存")
    log("reference token", syslog_token[:72] + "…", MAG)

    # 字节 offset 分段读取
    p1 = mem.read_cached(USER_ID, syslog_entry.ref_id, 0, 300)
    p2 = mem.read_cached(USER_ID, syslog_entry.ref_id, p1["next_offset"], 300)
    ok(f"分页读取：p1={p1['length']}B, p2={p2['length']}B，总 {p1['total_bytes']}B")

    state.confirm_fact("10.0.1.2 (AS65002) 不可达 (100% packet loss)", source="tool:ping")
    state.confirm_fact("R1 GigabitEthernet0/1 二层正常 (UP/UP)",       source="tool:bgp_check")
    state.confirm_fact("AS65002 Hold Timer 到期，已重试47次均失败",     source="tool:syslog")
    state.add_to_working_set("R1",        "核心路由器 R1", "device", {"ip": "10.0.0.1"})
    state.add_to_working_set("peer-65002","AS65002 BGP 邻居", "peer", {"ip": "10.0.1.2"})
    ok(f"热轨道：confirmed_facts={len(state.confirmed_facts)}, working_set={len(state.working_set)}")

    R1, T1 = agent_think(Q1, ctx1, 1)
    chunk1 = mem.remember(USER_ID, SESSION_ID, f"User:{Q1}\nAgent:{R1}", metadata={"turn":1})
    mem.update_user_profile(USER_ID, SESSION_ID, Q1, R1, T1)
    state.increment_turn()
    print(f"\n  {B}Agent:{R} {R1[:110]}…\n")

    # ══════════════════════════════════════════════════════════════════════════
    banner("Round 2 │ 深入排查 + 语义向量检索演示", CYN)
    Q2 = "R1 自身接口有没有问题？其他 BGP 邻居正常吗？"
    print(f"\n  {B}用户:{R} {Q2}\n")

    section("读取：语义 Embedding 检索（v5 新能力）", CYN)
    info(f"Embedding 后端: {mem._emb_index.backend_name} (index size={mem._emb_index.size})")
    emb_results = mem.search_chunks(USER_ID, "BGP routing neighbor status", use_embedding=True)
    info(f"语义检索结果（layer={emb_results.layer}）：{emb_results.total_found} 条，{emb_results.elapsed_ms:.1f}ms")
    for c in emb_results.items[:2]:
        print(f"    {GRY}[{c.source}] {c.text[:60]}…{R}")

    section("读取：重建 Context（热轨道已累积）", CYN)
    ctx2, rep2 = mem.build_context_budgeted(state, Q2)
    budget_summary(rep2)

    # 接口检查 + 置信度衰减
    iface_out = fake_interface_check("R1")
    state.add_tool_result("interface_check", iface_out)
    state.confirm_fact("R1 所有接口物理层正常，排除本地硬件故障", source="tool:interface_check")
    state.confirm_fact("AS65003 ping 可达，其他 BGP 会话正常",    source="tool:ping")

    section("演示：置信度衰减 (confidence decay)", CYN)
    guess = mem.add_fact(USER_ID, SESSION_ID, "R1 接口可能存在硬件问题", "entity", 0.6)
    mem.mid_term.update_fact(guess.fact_id, USER_ID, SESSION_ID, decay=True)
    decayed = mem.mid_term.list_all(USER_ID, SESSION_ID)
    for f in decayed:
        if "接口可能" in f.fact:
            log("decay", f"'{f.fact}' → conf={f.confidence:.3f} (×0.7)", YLW)
    ok("矛盾证据 → 降低 fact 置信度，检索权重下降")

    R2, T2 = agent_think(Q2, ctx2, 2)
    chunk2 = mem.remember(USER_ID, SESSION_ID, f"User:{Q2}\nAgent:{R2}", metadata={"turn":2})
    mem.update_user_profile(USER_ID, SESSION_ID, Q2, R2, T2)
    state.increment_turn()
    print(f"\n  {B}Agent:{R} {R2[:110]}…\n")

    # ══════════════════════════════════════════════════════════════════════════
    banner("Round 3 │ 影响评估 + 历史压缩 (Consolidation) — v5 新能力", CYN)
    Q3 = "AS65002 断连后，到 192.168.100.0/24 的路由还通吗？"
    print(f"\n  {B}用户:{R} {Q3}\n")

    # 写入更多 chunks 触发压缩阈值
    for i in range(6):
        mem.remember(USER_ID, SESSION_ID,
                     f"diagnostic step {i}: 检查 BGP AS{i+65000} 邻居状态和路由通告情况")

    section("读取：历史压缩 (Memory Consolidation) — v5 新能力", CYN)
    count_before = mem.long_term.count_session(USER_ID, SESSION_ID)
    log("session chunks before", str(count_before), YLW)
    log("should_consolidate",    str(mem.should_consolidate(USER_ID, SESSION_ID)), YLW)

    if mem.should_consolidate(USER_ID, SESSION_ID):
        result = mem.consolidate_session(USER_ID, SESSION_ID)
        if not result.get("skipped"):
            ok(f"consolidate_session(): {result['chunks_merged']} 条 → "
               f"{result['summaries_created']} 条摘要（节省 token！）")
            log("chunks before", str(result["chunks_before"]), GRN)
            log("chunks after",  str(result["chunks_after"]),  GRN)
        else:
            info(f"跳过压缩：{result.get('reason', '')}")
    else:
        info("未达压缩阈值，跳过")

    section("读取：MMR 多样性检索对比", CYN)
    for t in ["BGP session AS65002 dropped Hold Timer",
              "BGP session AS65002 dropped Hold Timer retry",
              "Traffic failover 192.168.100.0/24 via AS65003"]:
        mem.remember(USER_ID, SESSION_ID, t)
    plain = mem.search_chunks(USER_ID, "BGP session dropped", top_k=3)
    mmr_r = mem.search_chunks(USER_ID, "BGP session dropped", top_k=3, use_mmr=True)
    info("普通 TF-IDF：")
    for c in plain.items[:2]: print(f"    {GRY}{c.text[:60]}…{R}")
    info("MMR 多样性过滤：")
    for c in mmr_r.items[:2]: print(f"    {GRN}{c.text[:60]}…{R}")

    route_out = fake_route_check("192.168.100.0/24")
    state.add_tool_result("route_check", route_out)
    state.confirm_fact("192.168.100.0/24 通过 AS65003/AS65004 正常可达", source="tool:route_check")
    state.confirm_fact("AS65002 路由已撤销，流量已自动切换",              source="tool:route_check")

    R3, T3 = agent_think(Q3, ctx2, 3)
    chunk3 = mem.remember(USER_ID, SESSION_ID, f"User:{Q3}\nAgent:{R3}", metadata={"turn":3})
    ctx3, rep3 = mem.build_context_budgeted(state, Q3,
                                            environment="NOC mode: active | incident: P2")
    print(); budget_summary(rep3)
    mem.update_user_profile(USER_ID, SESSION_ID, Q3, R3, T3)
    state.increment_turn()
    print(f"\n  {B}Agent:{R} {R3[:110]}…\n")

    # ══════════════════════════════════════════════════════════════════════════
    banner("Round 4 │ 故障恢复确认 + 反思写入 (Reflect) — v5 新能力", CYN)
    Q4 = "AS65002 恢复了，BGP session 已重建，请帮我记录一下这次的处理流程"
    print(f"\n  {B}用户:{R} {Q4}\n")

    section("写入：反思记忆 (Reflect) — v5 新能力", CYN)
    info("任务完成 → 写入 reflection chunk + lesson facts + 更新 Skill 成功率")

    reflect_result = mem.reflect(
        USER_ID, SESSION_ID,
        task="AS65002 BGP 故障排查（Hold Timer Expired）",
        outcome="success",
        summary=("通过 bgp_check 发现 Idle 状态，syslog 确认47次重试失败，"
                 "ping 验证 L3 不可达，interface_check 排除本地故障，"
                 "联系 AS65002 NOC 后对端侧恢复，BGP 会话在30分钟内重建。"),
        skill_id=sk.skill_id,
    )
    ok(f"反思已写入：")
    log("long_term_chunk", reflect_result["long_term_chunk_id"][:8] + "…", GRN)
    log("lesson_facts",    str(len(reflect_result.get("lesson_fact_ids", []))), GRN)
    log("skill updated",   f"成功率 → {reflect_result.get('new_skill_rate', 'N/A'):.0%}", GRN)

    section("写入：手动更新 Skill 成功率", CYN)
    rate = mem.record_skill_outcome(USER_ID, sk.skill_id, success=True)
    updated_sk = mem.skill_store.get_skill(USER_ID, sk.skill_id)
    log("skill success_rate", f"{updated_sk.success_rate:.0%}", GRN)
    log("skill use_count",    str(updated_sk.use_count),        GRN)
    ok("Skill 成功率已更新（EMA 平滑），下次检索优先级提升")

    section("读取：跨 Session 召回（历史 + 当前）", CYN)
    cross = mem.search_chunks(USER_ID, "AS65002 BGP NOC 联系 历史事件", top_k=3)
    info(f"跨 session 检索：{cross.total_found} 条（含历史 session 数据）")
    for c in cross.items[:2]:
        print(f"    {GRY}[{c.session_id[:12]}] {c.text[:55]}…{R}")

    R4, _ = agent_think(Q4, ctx3, 4)
    mem.remember(USER_ID, SESSION_ID, f"User:{Q4}\nAgent:{R4}", metadata={"turn":4})
    mem.update_user_profile(USER_ID, SESSION_ID, Q4, R4)
    state.increment_turn()
    print(f"\n  {B}Agent:{R} {R4[:130]}…\n")

    # ══════════════════════════════════════════════════════════════════════════
    banner("维护操作 & 最终统计", MAG)

    section("维护：evict_expired_facts", MAG)
    n = mem.mid_term.evict_expired_facts()
    ok(f"evict_expired_facts() → 清理 {n} 条过期 facts")

    section("维护：apply_retention (>90天数据)", MAG)
    ancient = MemoryChunk(user_id=USER_ID, session_id="old-session",
                          text="非常古老的数据需要清理", created_at=1000.0)
    mem.long_term.add_chunk(ancient)
    deleted = mem.long_term.apply_retention(USER_ID, max_age_days=90)
    ok(f"apply_retention(90天) → 删除 {deleted} 条")

    section("维护：garbage_collect", MAG)
    gc = mem.short_term.garbage_collect()
    ok(f"garbage_collect() → orphan_files={gc['orphan_files']}, missing={gc['missing_files']}")

    section("维护：end_session + checkpoint", MAG)
    mem.end_session(USER_ID, SESSION_ID)
    mem.checkpoint()
    ok("end_session() → 热轨道释放")
    ok("checkpoint()  → WAL 归零，连接释放")

    section("最终统计", GRN)
    stats = mem.stats(USER_ID, SESSION_ID)
    import os as _os
    db_size = _os.path.getsize(mem._db_path)
    log("long_term_chunks",    str(stats["long_term_chunks"]),    GRN)
    log("sessions",             str(stats["sessions"]),            GRN)
    log("skill_count",          str(stats["skill_count"]),         GRN)
    log("embedding_backend",    stats["embedding_backend"],        GRN)
    log("embedding_index_size", str(stats["embedding_index_size"]),GRN)
    log("db_size",              f"{db_size/1024:.1f} KB",          GRN)

    mem.close()

    # ══════════════════════════════════════════════════════════════════════════
    banner("用法全景图 — 所有模块调用路径汇总", WHT)
    print(f"""
  {B}写入路径（每轮对话后）:{R}
  ┌─────────────────────────────────────────────────────────────┐
  │  remember() / remember_batch() → LongTermStore + EmbIndex  │ 长期+语义
  │  distill() / add_fact()        → MidTermStore (dedup+TTL)  │ 中期
  │  cache_tool_result()           → ShortTermStore (bytes)    │ 短期
  │  save_skill()                  → SkillStore ← v5 NEW       │ 程序性
  │  reflect()                     → Long+Mid+Skill ← v5 NEW   │ 反思
  │  update_user_profile()         → UserModelEngine           │ 用户画像
  └─────────────────────────────────────────────────────────────┘

  {B}热轨道（零DB访问，会话内即时访问）:{R}
  ┌─────────────────────────────────────────────────────────────┐
  │  state = get_session(user_id, session_id)                  │
  │    confirm_fact()   P2 永不驱逐                             │
  │    add_to_working_set() P3                                  │
  │    add_tool_result()    P5 inline                           │
  └─────────────────────────────────────────────────────────────┘

  {B}读取路径（每轮 LLM 调用前）:{R}
  ┌─────────────────────────────────────────────────────────────┐
  │  build_context_budgeted(state, query, budget=3200)         │
  │    P1  user_profile        ← UserModelEngine               │
  │    P2a confirmed_facts     ← 热轨道                        │
  │    P2b skills              ← SkillStore ← v5 NEW           │
  │    P3  working_set         ← 热轨道                        │
  │    P4  mid_term_facts      ← FTS5+TF-IDF+MMR               │
  │    P5  long_term_chunks    ← FTS5+TF-IDF+Embedding+MMR     │
  │    P6  environment         ← 可选外部状态                   │
  └─────────────────────────────────────────────────────────────┘

  {B}v5 新增能力:{R}
  ┌─────────────────────────────────────────────────────────────┐
  │  save_skill() / find_skills() / record_skill_outcome()     │ 程序性记忆
  │  reflect(outcome, summary, skill_id)                       │ 反思写入
  │  consolidate_session()                                      │ 历史压缩
  │  search_chunks(use_embedding=True)                         │ 语义检索
  │  EmbeddingIndex.from_sentence_transformer(model)           │ 语义后端
  │  EmbeddingIndex.from_openai(api_key, model)                │ OpenAI 后端
  └─────────────────────────────────────────────────────────────┘

  {B}维护操作（建议每日 cron）:{R}
  ┌─────────────────────────────────────────────────────────────┐
  │  evict_expired_cache()         短期 TTL 清理                │
  │  mid_term.evict_expired_facts() 中期 TTL 清理              │
  │  long_term.apply_retention()   长期数据保留策略             │
  │  short_term.garbage_collect()  孤儿文件清理                 │
  │  consolidate_session()         历史摘要压缩 ← v5 NEW        │
  │  checkpoint()                  WAL 文件归零                 │
  │  end_session()                 热轨道内存释放               │
  └─────────────────────────────────────────────────────────────┘
""")
    print(f"  {GRN}{B}演示完成。数据在：{data_dir}{R}\n")

if __name__ == "__main__":
    run_scenario()
