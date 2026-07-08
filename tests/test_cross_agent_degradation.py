"""Cross-agent skill degradation (peer-offline is the normal case).

A cross-agent skill declares `delegates-to` (peers it hands subtasks to) and
`degraded-capability` (free-text boundary contract). When a declared peer is
offline, the loop injects the boundary contract into the prompt UP FRONT and
emits a capability_degraded event — so the agent works within its boundary
instead of improvising after a failed delegation.
"""
import asyncio
import unittest

from runtime.loop import AgentRuntimeLoop
from runtime.loop_types import RuntimeConfig


class _Sum:
    def __init__(self, sid, delegates_to, degraded):
        self.skill_id = sid; self.name = sid; self.purpose = "p"
        self.risk_level = "low"; self.requires_hitl = False; self.tags = ["x"]
        self.delegates_to = delegates_to
        self.degraded_capability = degraded


class _Sel:
    def __init__(self, sid):
        self.ambiguous = False; self.ambiguous_kind = None
        self.top_score = 0.9; self.second_score = 0.1
        self.selected = [(sid, 0.9)]
        self.summary = f"[SKILL] {sid}"


class _Cat:
    def __init__(self, summary):
        self._summary = summary
    def select_skills_for_query(self, q, top_k=5):
        return _Sel(self._summary.skill_id)
    def get_summary(self, sid):
        return self._summary if sid == self._summary.skill_id else None


def _loop(llm, catalog, peer_health_fn):
    return AgentRuntimeLoop(config=RuntimeConfig(), llm_fn=llm,
                            skill_catalog=catalog, peer_health_fn=peer_health_fn)


class TestSkillFrontmatterFields(unittest.TestCase):
    def test_parses_delegates_to_and_degraded(self):
        from skills.skill_format import parse_skill_md, to_flat_dict
        md = ('---\nname: t\ndescription: d\ndelegates-to: dc-agent, wan-agent\n'
              'degraded-capability: "DC 离线交付 LAN 层"\n'
              'metadata:\n  skill_id: t\n---\n# b')
        _, d = to_flat_dict(parse_skill_md(md))
        self.assertEqual(d["delegates_to"], ["dc-agent", "wan-agent"])
        self.assertIn("LAN", d["degraded_capability"])

    def test_real_cross_agent_skills_declare_fields(self):
        from skills.loader import SkillLoader
        defs = SkillLoader().profile_skill_definitions("lan")
        for sid, peer in (("app_access_troubleshoot", "dc-agent"),
                          ("branch_app_reachability", "wan-agent")):
            d = defs.get(sid)
            self.assertIsNotNone(d, f"{sid} should load")
            self.assertIn(peer, d.get("delegates_to", []))
            self.assertTrue(d.get("degraded_capability"),
                            f"{sid} must declare a degraded-capability contract")


class TestLoopDegradation(unittest.TestCase):
    def test_offline_peer_injects_notice_and_event(self):
        async def t():
            summ = _Sum("app_access_troubleshoot", ["dc-agent"],
                        "DC 离线:只交付 LAN 层,应用层标记待恢复")
            seen_ctx = {}
            async def llm(query, context, state):
                seen_ctx["c"] = context
                return "按边界能力交付 LAN 层,DC 离线不查应用层,答复足够长满足要求。"
            loop = _loop(llm, _Cat(summ), lambda: {"dc-agent"})
            events = []
            async for ch in loop.stream(query="诊断", session_id="deg1",
                                        tool_registry={}):
                if isinstance(ch, dict) and ch.get("type") == "capability_degraded":
                    events.append(ch)
            # event fired with the boundary contract
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["offline_peers"], ["dc-agent"])
            self.assertIn("LAN", events[0]["degraded_capability"])
            # boundary notice reached the prompt
            self.assertIn("跨域能力状态", seen_ctx["c"])
        asyncio.run(t())

    def test_healthy_peer_no_degradation(self):
        async def t():
            summ = _Sum("app_access_troubleshoot", ["dc-agent"], "x")
            async def llm(query, context, state):
                return "正常端到端诊断,DC 在线,答复足够长满足长度要求即可。"
            # dc-agent NOT in offline set
            loop = _loop(llm, _Cat(summ), lambda: set())
            events = []
            async for ch in loop.stream(query="诊断", session_id="deg2",
                                        tool_registry={}):
                if isinstance(ch, dict) and ch.get("type") == "capability_degraded":
                    events.append(ch)
            self.assertEqual(events, [])
        asyncio.run(t())

    def test_no_peer_health_fn_no_degradation(self):
        async def t():
            summ = _Sum("app_access_troubleshoot", ["dc-agent"], "x")
            async def llm(query, context, state):
                return "正常诊断,无健康探针注入,答复足够长满足长度要求即可继续。"
            loop = _loop(llm, _Cat(summ), None)   # no peer_health_fn
            events = []
            async for ch in loop.stream(query="诊断", session_id="deg3",
                                        tool_registry={}):
                if isinstance(ch, dict) and ch.get("type") == "capability_degraded":
                    events.append(ch)
            self.assertEqual(events, [])
        asyncio.run(t())

    def test_non_cross_agent_skill_never_degrades(self):
        async def t():
            # skill with NO delegates_to → offline peers are irrelevant
            summ = _Sum("local_only", [], "")
            async def llm(query, context, state):
                return "纯本地技能,不涉及委派,答复足够长满足长度要求即可继续处理。"
            loop = _loop(llm, _Cat(summ), lambda: {"dc-agent", "wan-agent"})
            events = []
            async for ch in loop.stream(query="q", session_id="deg4",
                                        tool_registry={}):
                if isinstance(ch, dict) and ch.get("type") == "capability_degraded":
                    events.append(ch)
            self.assertEqual(events, [])
        asyncio.run(t())


if __name__ == "__main__":
    unittest.main(verbosity=2)
