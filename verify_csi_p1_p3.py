"""verify_csi_p1_p3.py — in-container verification of CSI / P1 / P3.

Run: PYTHONPATH=. python3 verify_csi_p1_p3.py
Uses REAL LAN/DC metadata. Uses the real async embedder if Ollama is reachable,
else falls back to no-embedding (tag+tool_set only) and says so.

This verifies everything that does NOT require two live agents:
  CSI  — cluster formation, similarity discrimination, skill→tool attribution
  P1   — recurring-trajectory detection + solidify trigger (fake evolve cb)
  P3   — append attribution to in-use skill + merge trigger (fake merge cb)
"""
import asyncio, importlib, sys

from skills.loader import SkillLoader
from skills.capability_index import CapabilitySemanticIndex
from skills.trajectory_miner import TrajectoryMiner, MinerConfig
from skills.append_merger import AppendMerger, MergeConfig

OK = "✅"; NO = "❌"; INFO = "•"

def load_profile(prof):
    tm = importlib.import_module(f"profiles.{prof}.tool_meta")
    meta = next((getattr(tm, a) for a in dir(tm)
                 if isinstance(getattr(tm, a), dict) and getattr(tm, a)
                 and all(isinstance(x, dict) for x in getattr(tm, a).values())), {})
    return meta, SkillLoader(mode="mock", profile=prof).skill_definitions()

async def try_embedder():
    """Return (embedder, is_real). OllamaEmbedder silently falls back to the
    deterministic sha256 StubEmbedder per call, so 'returned a vector' does
    NOT mean real embeddings — compare against the stub's known output for
    the same probe text to detect the fallback precisely."""
    try:
        from config import cfg
        from integrations.clients.embedder import build_embedder, StubEmbedder
        emb = build_embedder(cfg.embeddings)
        probe = "embedder-reality-probe"
        v = await emb.embed(probe)
        if not v:
            return None, False
        stub_v = await StubEmbedder().embed(probe)
        is_real = (v != stub_v)
        if not is_real:
            print(f"  {INFO} embedder returned STUB vectors (Ollama unreachable) — "
                  f"emb-axis scores are hash artifacts, not semantic")
        return emb, is_real
    except Exception as e:
        print(f"  {INFO} embedder unavailable ({type(e).__name__}); falling back to tag+tool_set only")
    return None, False

async def main():
    lan_t, lan_s = load_profile("lan")
    dc_t, dc_s = load_profile("dc")
    all_t, all_s = {**lan_t, **dc_t}, {**lan_s, **dc_s}

    emb, emb_real = await try_embedder()
    idx = CapabilitySemanticIndex()
    await idx.build_async(all_t, all_s, async_embed=(emb.embed if emb else None))
    space = idx.export_space()

    print("\n" + "="*64)
    print("  CSI 验证")
    print("="*64)
    clusters = space["clusters"]
    print(f"  {INFO} 形成 {len(clusters)} 个能力族: {', '.join(sorted(clusters))}")
    # check 1: DC two-level split
    c1 = "dc/fabric" in clusters and "dc/application" in clusters
    print(f"  {OK if c1 else NO} 修正1 二级聚类: DC 拆成 dc/fabric + dc/application")
    # check 2: similarity discrimination
    rel = idx.similarity("get_user_access", "check_nac_policy")
    unrel = idx.similarity("get_user_access", "dc_bgp_evpn_status")
    c2 = rel and unrel and rel.score > unrel.score
    print(f"  {OK if c2 else NO} 相似度鉴别: 准入类({rel.score:.2f}) > 跨域无关({unrel.score:.2f})")
    print(f"      理由可读: {rel.reasons}")
    # check 3: skill→tool attribution
    hit = await idx.nearest_skill_async(
        ["dc_check_user_app_access", "dc_get_app_acl"], "check user app access")
    c3 = hit and hit[0] == "dc_app_access_diagnose"
    print(f"  {OK if c3 else NO} skill→tool 归属: DC app 工具集 → {hit[0] if hit else None}"
          f" (score={hit[1].score:.2f})" if hit else f"  {NO} 归属失败")
    csi_pass = c1 and c2 and c3

    print("\n" + "="*64)
    print("  P1 验证 (重复轨迹 → 固化)")
    print("="*64)
    # Fake journal: 3 identical access-diagnosis trajectories + 1 unrelated
    entries = [
        {"session_id": f"acc{i}", "outcome": "completed",
         "tool_calls": ["get_user_access", "check_nac_policy"],
         "query": "诊断用户访问失败", "loaded_skills": []} for i in range(3)
    ] + [
        {"session_id": "fab1", "outcome": "completed",
         "tool_calls": ["dc_bgp_evpn_status"], "query": "fabric", "loaded_skills": []}
    ]
    class _J:
        def list_recent(self, limit=200): return entries
        def extract_trajectory(self, sid):
            e = next((x for x in entries if x["session_id"] == sid), None)
            return {"steps": [f"call {t}" for t in (e or {}).get("tool_calls", [])],
                    "tools": (e or {}).get("tool_calls", []), "observations": [],
                    "loaded_skills": [], "turns": 2}
    solidified = []
    async def evolve_cb(**kw):
        solidified.append(kw); return {"skill_id": "gen_access_diag"}
    miner = TrajectoryMiner(_J(), idx, evolve_cb,
                            MinerConfig(recurrence_threshold=3, similarity_threshold=0.5))
    recurring = miner.find_recurring()
    p1a = len(recurring) == 1 and recurring[0].size == 3
    print(f"  {OK if p1a else NO} 检测到重复轨迹: {len(recurring)} 个达阈值簇"
          f" (size={recurring[0].size if recurring else 0}, 阈值=3)")
    props = await miner.sweep()
    p1b = len(props) == 1 and solidified and solidified[0]["solution_steps"]
    print(f"  {OK if p1b else NO} 触发 evolver 固化 + 喂真实轨迹: "
          f"steps={solidified[0]['solution_steps'] if solidified else None}")
    p1c = recurring and recurring[0].covered_by_skill
    print(f"  {INFO} 该簇是否被存量 skill 覆盖: {recurring[0].covered_by_skill if recurring else None}"
          f" (若覆盖则 P1 跳过、交给 P3)")
    p1_pass = p1a and p1b

    print("\n" + "="*64)
    print("  P3 验证 (追加诉求 → 定向 merge)")
    print("="*64)
    merged = []
    async def merge_cb(*, skill_id, append_text, session_id, tools):
        merged.append(skill_id); return True
    # case A: session-active skill = ground truth
    m = AppendMerger(idx, merge_cb, MergeConfig(prefer_session_active=True))
    rA = await m.maybe_merge(append_text="还要顺便检查 VPN 隧道状态", session_id="s1",
                             active_skill="app_access_troubleshoot",
                             session_tools=["get_user_access"])
    p3a = rA.merged and rA.skill_id == "app_access_troubleshoot"
    print(f"  {OK if p3a else NO} 会话内活跃 skill 作 ground truth: merged→{rA.skill_id} ({rA.reason})")
    # case B: CSI attribution when no session signal
    m2 = AppendMerger(idx, merge_cb, MergeConfig(prefer_session_active=False, attribution_floor=0.1))
    rB = await m2.maybe_merge(append_text="check user app access", session_id="s2",
                              active_skill=None,
                              session_tools=["dc_check_user_app_access", "dc_get_app_acl"])
    p3b = rB.merged and rB.skill_id == "dc_app_access_diagnose"
    print(f"  {OK if p3b else NO} 无会话信号时 CSI 定向归属: merged→{rB.skill_id} (sim={rB.similarity:.2f})")
    # case C: floor gating rejects weak attribution
    m3 = AppendMerger(idx, merge_cb, MergeConfig(prefer_session_active=False, attribution_floor=0.95))
    rC = await m3.maybe_merge(append_text="something unrelated", session_id="s3",
                              active_skill=None, session_tools=["list_devices"])
    p3c = not rC.merged
    print(f"  {OK if p3c else NO} floor 拦截弱归属: merged={rC.merged} ({rC.reason})")
    p3_pass = p3a and p3b and p3c

    print("\n" + "="*64)
    print("  总结")
    print("="*64)
    print(f"  CSI: {OK if csi_pass else NO}   P1: {OK if p1_pass else NO}   P3: {OK if p3_pass else NO}")
    print(f"  embedder: {'REAL (Ollama nomic)' if emb_real else 'STUB或无 — emb轴数值不代表真实语义'}")
    return 0 if (csi_pass and p1_pass and p3_pass) else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
