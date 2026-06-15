"""tests/test_p1_p3_evolution.py — P1 trajectory mining + P3 append merge."""
import asyncio
import unittest

from skills.capability_index import CapabilitySemanticIndex
from skills.trajectory_miner import TrajectoryMiner, MinerConfig, TrajectoryCluster
from skills.append_merger import AppendMerger, MergeConfig


class _FakeJournal:
    def __init__(self, entries):
        self._entries = entries
    def list_recent(self, limit=200):
        return list(self._entries)
    def extract_trajectory(self, sid):
        for e in self._entries:
            if e["session_id"] == sid:
                return {"steps": [f"call {t}" for t in e.get("tool_calls", [])],
                        "tools": e.get("tool_calls", []), "observations": [],
                        "loaded_skills": e.get("loaded_skills", []), "turns": 2}
        return {"steps": [], "tools": [], "observations": [], "loaded_skills": [], "turns": 0}


def _csi():
    idx = CapabilitySemanticIndex(embed_fn=None)
    idx.build({}, {})   # empty; cluster_trajectories doesn't need built caps
    return idx


class TestP1TrajectoryMiner(unittest.TestCase):
    def test_recurring_trajectory_triggers_solidify(self):
        async def run():
            # 3 runs with the SAME tool trajectory → should solidify
            entries = [
                {"session_id": f"s{i}", "outcome": "completed",
                 "tool_calls": ["get_user_access", "check_nac_policy"],
                 "query": "诊断用户访问失败", "loaded_skills": []}
                for i in range(3)
            ]
            calls = []
            async def evolve_cb(**kw):
                calls.append(kw); return {"skill_id": "generated_x"}
            miner = TrajectoryMiner(_FakeJournal(entries), _csi(), evolve_cb,
                                    MinerConfig(recurrence_threshold=3, similarity_threshold=0.5))
            clusters = miner.find_recurring()
            self.assertEqual(len(clusters), 1)
            self.assertEqual(clusters[0].size, 3)
            proposals = await miner.sweep()
            self.assertEqual(len(proposals), 1)
            self.assertTrue(calls)   # evolver was asked to solidify
            # real trajectory was passed (not empty)
            self.assertTrue(calls[0]["solution_steps"])
        asyncio.run(run())

    def test_below_threshold_does_not_solidify(self):
        async def run():
            entries = [
                {"session_id": f"s{i}", "outcome": "completed",
                 "tool_calls": ["get_user_access", "check_nac_policy"],
                 "query": "q", "loaded_skills": []}
                for i in range(2)   # only 2 < threshold 3
            ]
            async def evolve_cb(**kw):
                return {"skill_id": "x"}
            miner = TrajectoryMiner(_FakeJournal(entries), _csi(), evolve_cb,
                                    MinerConfig(recurrence_threshold=3))
            self.assertEqual(miner.find_recurring(), [])
            self.assertEqual(await miner.sweep(), [])
        asyncio.run(run())

    def test_covered_cluster_skipped(self):
        async def run():
            cluster = TrajectoryCluster(
                members=["s1"], rep_tools={"get_user_access"}, size=5,
                sample_query="q", covered_by_skill="lan_user_access_diagnose")
            called = []
            async def evolve_cb(**kw):
                called.append(kw); return {"skill_id": "x"}
            miner = TrajectoryMiner(_FakeJournal([]), _csi(), evolve_cb)
            res = await miner.solidify(cluster)
            self.assertIsNone(res)        # skipped — already covered (P3 territory)
            self.assertEqual(called, [])
        asyncio.run(run())


class TestP3AppendMerger(unittest.TestCase):
    def test_session_active_skill_is_ground_truth(self):
        async def run():
            merged = []
            async def merge_cb(*, skill_id, append_text, session_id, tools):
                merged.append(skill_id); return True
            m = AppendMerger(_csi(), merge_cb, MergeConfig(prefer_session_active=True))
            res = await m.maybe_merge(
                append_text="还要顺便检查 VPN 隧道状态", session_id="s1",
                active_skill="app_access_troubleshoot",
                session_tools=["get_user_access"])
            self.assertTrue(res.merged)
            self.assertEqual(res.skill_id, "app_access_troubleshoot")
            self.assertEqual(merged, ["app_access_troubleshoot"])
        asyncio.run(run())

    def test_empty_append_no_merge(self):
        async def run():
            async def merge_cb(**kw):
                return True
            m = AppendMerger(_csi(), merge_cb)
            res = await m.maybe_merge(append_text="   ", session_id="s1",
                                      active_skill="x")
            self.assertFalse(res.merged)
        asyncio.run(run())

    def test_csi_attribution_when_no_session_active(self):
        async def run():
            # build CSI with a real skill so nearest_skill can attribute
            from skills.loader import SkillLoader
            defs = SkillLoader(mode="mock", profile="dc").skill_definitions()
            idx = CapabilitySemanticIndex(embed_fn=None)
            # need tool defs for tool_set jaccard; load dc tools
            import importlib
            tm = importlib.import_module("profiles.dc.tool_meta")
            meta = next((getattr(tm, a) for a in dir(tm)
                         if isinstance(getattr(tm, a), dict) and getattr(tm, a)
                         and all(isinstance(x, dict) for x in getattr(tm, a).values())), {})
            idx.build(meta, defs)
            merged = []
            async def merge_cb(*, skill_id, append_text, session_id, tools):
                merged.append(skill_id); return True
            m = AppendMerger(idx, merge_cb,
                             MergeConfig(prefer_session_active=False, attribution_floor=0.1))
            res = await m.maybe_merge(
                append_text="check user app access", session_id="s1",
                active_skill=None,
                session_tools=["dc_check_user_app_access", "dc_get_app_acl"])
            # should attribute to the dc app access skill
            self.assertTrue(res.merged)
            self.assertEqual(res.skill_id, "dc_app_access_diagnose")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
