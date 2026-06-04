"""tests/test_skill_preference.py — preference learning core logic.

Covers the parts that don't need the loop: staging (learn/recommend/auto),
boost weighting, high-risk auto-exclusion, and record/recall round-trip
against a fake memory adapter.
"""
import asyncio
import unittest

from skills.skill_preference import (
    SkillPreferenceService, PreferenceConfig, PreferenceHit, FACT_TYPE,
)


class _FakeMem:
    """Minimal stand-in for the memory adapter."""
    def __init__(self):
        self.facts = []  # list of dicts
        self._id = 0
    async def add_fact(self, *, session_id, user_id, fact_text, fact_type,
                       confidence, ttl_days=None, metadata=None):
        self._id += 1
        fid = f"f{self._id}"
        self.facts.append({
            "fact_id": fid, "user_id": user_id, "fact_type": fact_type,
            "confidence": confidence, "metadata": metadata or {},
            "query_sample": (metadata or {}).get("query_sample", ""),
        })
        return fid
    async def find_similar_facts(self, *, user_id, query_text, fact_type=None,
                                 top_k=5, session_id=None):
        # naive: return all facts of this user+type with a fixed similarity
        return [
            {"fact_id": f["fact_id"], "confidence": f["confidence"],
             "score": 0.9, "metadata": f["metadata"]}
            for f in self.facts
            if f["user_id"] == user_id and f["fact_type"] == fact_type
        ][:top_k]
    async def update_fact_confidence(self, fact_id, new_conf, *, reason=""):
        for f in self.facts:
            if f["fact_id"] == fact_id:
                f["confidence"] = new_conf
                return True
        return False


class TestStaging(unittest.TestCase):
    def setUp(self):
        self.svc = SkillPreferenceService(_FakeMem(), PreferenceConfig(
            recommend_floor=0.5, auto_threshold=0.85))

    def test_no_hits_is_learn(self):
        stage, sid = self.svc.stage_for([])
        self.assertEqual(stage, "learn")
        self.assertIsNone(sid)

    def test_low_confidence_is_learn(self):
        hits = [PreferenceHit("skill_a", 0.4, 0.9, "f1")]
        self.assertEqual(self.svc.stage_for(hits)[0], "learn")

    def test_mid_confidence_is_recommend(self):
        hits = [PreferenceHit("skill_a", 0.7, 0.9, "f1")]
        stage, sid = self.svc.stage_for(hits)
        self.assertEqual(stage, "recommend")
        self.assertEqual(sid, "skill_a")

    def test_high_confidence_is_auto(self):
        hits = [PreferenceHit("skill_a", 0.9, 0.9, "f1")]
        stage, sid = self.svc.stage_for(hits)
        self.assertEqual(stage, "auto")
        self.assertEqual(sid, "skill_a")

    def test_high_risk_skill_never_auto(self):
        hits = [PreferenceHit("danger_skill", 0.95, 0.9, "f1")]
        stage, sid = self.svc.stage_for(
            hits, skill_requires_hitl=lambda s: s == "danger_skill")
        self.assertEqual(stage, "recommend")   # capped, not auto


class TestBoost(unittest.TestCase):
    def setUp(self):
        self.svc = SkillPreferenceService(_FakeMem(), PreferenceConfig(base_boost=0.2))

    def test_boost_reranks(self):
        selected = [("skill_a", 0.12), ("skill_b", 0.10)]
        hits = [PreferenceHit("skill_b", 1.0, 1.0, "f1")]  # full boost to b
        out = self.svc.apply_boost(selected, hits)
        self.assertEqual(out[0][0], "skill_b")   # b now on top
        self.assertAlmostEqual(out[0][1], 0.10 + 0.2, places=4)

    def test_boost_adds_absent_preferred_skill(self):
        selected = [("skill_a", 0.12)]
        hits = [PreferenceHit("skill_c", 1.0, 1.0, "f1")]  # not in candidates
        out = self.svc.apply_boost(selected, hits)
        ids = [s for s, _ in out]
        self.assertIn("skill_c", ids)

    def test_no_hits_unchanged(self):
        selected = [("skill_a", 0.12)]
        self.assertEqual(self.svc.apply_boost(selected, []), selected)


class TestRecordRecall(unittest.TestCase):
    def test_round_trip_and_demote(self):
        async def run():
            mem = _FakeMem()
            svc = SkillPreferenceService(mem, PreferenceConfig(initial_confidence=0.6))
            fid = await svc.record_choice(
                user_id="alice", session_id="s1",
                query="诊断用户访问应用失败", chosen_skill_id="app_access_troubleshoot",
                candidates=["app_access_troubleshoot", "lan_user_access_diagnose"],
            )
            self.assertIsNotNone(fid)
            hits = await svc.recall(user_id="alice", query="诊断用户访问应用失败")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0].skill_id, "app_access_troubleshoot")
            self.assertAlmostEqual(hits[0].confidence, 0.6, places=4)
            # demote (wrong auto-select feedback)
            ok = await svc.demote(fid, to=0.4)
            self.assertTrue(ok)
            hits2 = await svc.recall(user_id="alice", query="诊断用户访问应用失败")
            self.assertAlmostEqual(hits2[0].confidence, 0.4, places=4)
        asyncio.run(run())

    def test_none_choice_not_recalled_as_boost(self):
        async def run():
            mem = _FakeMem()
            svc = SkillPreferenceService(mem, PreferenceConfig())
            await svc.record_choice(
                user_id="bob", session_id="s1", query="some query",
                chosen_skill_id=None, candidates=["x"])
            hits = await svc.recall(user_id="bob", query="some query")
            self.assertEqual(hits, [])   # __none__ yields no boostable skill
        asyncio.run(run())

    def test_per_user_isolation(self):
        async def run():
            mem = _FakeMem()
            svc = SkillPreferenceService(mem, PreferenceConfig())
            await svc.record_choice(user_id="alice", session_id="s1",
                query="q", chosen_skill_id="skill_a", candidates=["skill_a"])
            # bob has no preferences
            self.assertEqual(await svc.recall(user_id="bob", query="q"), [])
            self.assertEqual(len(await svc.recall(user_id="alice", query="q")), 1)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()


class TestRealMemoryAdapterIntegration(unittest.TestCase):
    """Regression: find_similar_facts must work against the REAL memory
    backend (the bug was it read .facts/.scores off RetrievalResult which only
    has .items, so every preference recall silently returned [])."""

    def test_preference_roundtrip_real_backend(self):
        import tempfile, shutil
        from memory.adapter import MemoryAdapter
        d = tempfile.mkdtemp()
        try:
            mem = MemoryAdapter(data_dir=d)

            async def run():
                await mem.add_fact(
                    session_id="s1", user_id="alice",
                    fact_text="对于诊断 alice 访问 crm 的请求,选择 app_access_troubleshoot",
                    fact_type="skill_preference", confidence=0.6,
                    metadata={"chosen_skill_id": "app_access_troubleshoot",
                              "query_sample": "诊断 alice 访问 crm"})
                svc = SkillPreferenceService(mem)
                return await svc.recall(user_id="alice", query="再诊断一次 alice 访问 crm")

            hits = asyncio.run(run())
            self.assertTrue(hits, "preference recall returned nothing against real backend")
            self.assertEqual(hits[0].skill_id, "app_access_troubleshoot")
        finally:
            shutil.rmtree(d, ignore_errors=True)
