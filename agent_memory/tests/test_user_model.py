"""
agent_memory/tests/test_user_model.py
Tests for UserModelEngine — new in v3.
Run: python -m unittest agent_memory.tests.test_user_model -v
"""
from __future__ import annotations
import os, sys, json, tempfile, threading, time, unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_memory import MemoryManager, UserModelEngine, UserProfile
from agent_memory.user_model import (
    TechnicalLevel, CommunicationStyle, InferredTrait,
    _update_trait, _update_domains, _extract_stated_preferences,
    _update_level_heuristics,
)


def _engine(tmp):
    db = os.path.join(tmp, "m", "memory.db")
    os.makedirs(os.path.dirname(db), exist_ok=True)
    return UserModelEngine(db_path=db)

def _mem(tmp):
    return MemoryManager(data_dir=os.path.join(tmp, "m"))


# ── UserProfile data model ────────────────────────────────────────────────────
class TestUserProfile(unittest.TestCase):

    def test_to_dict_roundtrip(self):
        p = UserProfile(user_id="alice")
        p.technical_level = TechnicalLevel.EXPERT
        p.domain_counts = {"network": 5, "auth": 2}
        p.tool_usage = {"syslog_search": 3}
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        self.assertEqual(p2.user_id, "alice")
        self.assertEqual(p2.technical_level, TechnicalLevel.EXPERT)
        self.assertEqual(p2.domain_counts, {"network": 5, "auth": 2})

    def test_to_prompt_section_empty(self):
        p = UserProfile(user_id="nobody")
        section = p.to_prompt_section()
        self.assertIn("USER PROFILE", section)

    def test_to_prompt_section_with_data(self):
        p = UserProfile(user_id="alice")
        p.technical_level = TechnicalLevel.EXPERT
        p.communication_style = CommunicationStyle.TERSE
        p.domain_counts = {"network": 10, "auth": 5}
        p.tool_usage = {"bgp_check": 4, "syslog_search": 2}
        _update_trait(p, {"trait": "prefers_cli", "value": "yes",
                          "confidence": 0.9, "is_revealed": True, "evidence": "..."})
        section = p.to_prompt_section()
        self.assertIn("expert", section)
        self.assertIn("terse", section)
        self.assertIn("network", section)

    def test_prompt_section_max_chars_respected(self):
        p = UserProfile(user_id="alice")
        for i in range(50):
            _update_trait(p, {"trait": f"trait_{i}", "value": f"value_{i}",
                               "confidence": 0.9, "is_revealed": True, "evidence": ""})
        section = p.to_prompt_section(max_chars=200)
        self.assertLessEqual(len(section), 220)  # small tolerance for truncation marker

    def test_hourly_activity_serialization(self):
        p = UserProfile(user_id="alice")
        p.hourly_activity = {9: 5, 14: 3, 22: 1}
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        self.assertEqual(p2.hourly_activity[9], 5)
        self.assertEqual(p2.hourly_activity[14], 3)


# ── Pure helper functions ─────────────────────────────────────────────────────
class TestHelpers(unittest.TestCase):

    def test_update_trait_new(self):
        p = UserProfile(user_id="u")
        _update_trait(p, {"trait": "http_lib", "value": "httpx",
                          "confidence": 0.8, "is_revealed": True, "evidence": "used httpx"})
        self.assertIn("http_lib", p.traits)
        self.assertAlmostEqual(p.traits["http_lib"].confidence, 0.8)
        self.assertIn("httpx", p.revealed_preferences.values())

    def test_update_trait_smoothing(self):
        p = UserProfile(user_id="u")
        _update_trait(p, {"trait": "style", "value": "terse", "confidence": 0.9,
                          "is_revealed": True, "evidence": ""})
        _update_trait(p, {"trait": "style", "value": "terse", "confidence": 0.5,
                          "is_revealed": True, "evidence": ""})
        conf = p.traits["style"].confidence
        # 0.3*0.5 + 0.7*0.9 = 0.78
        self.assertAlmostEqual(conf, 0.78, places=2)
        self.assertEqual(p.traits["style"].evidence_count, 2)

    def test_update_trait_low_confidence_ignored(self):
        p = UserProfile(user_id="u")
        _update_trait(p, {"trait": "low", "value": "x", "confidence": 0.3,
                          "is_revealed": True, "evidence": ""})
        self.assertNotIn("low", p.traits)

    def test_update_domains_network(self):
        p = UserProfile(user_id="u")
        _update_domains(p, "BGP neighbor went down, OSPF area 0 affected")
        self.assertIn("network", p.domain_counts)
        self.assertGreater(p.domain_counts["network"], 0)

    def test_update_domains_multiple(self):
        p = UserProfile(user_id="u")
        _update_domains(p, "RADIUS auth failure during k8s pod deployment")
        self.assertIn("auth", p.domain_counts)
        self.assertIn("kubernetes", p.domain_counts)

    def test_extract_stated_preferences(self):
        p = UserProfile(user_id="u")
        _extract_stated_preferences(p, "I prefer httpx over requests for all HTTP calls")
        self.assertTrue(any("httpx" in v for v in p.stated_preferences.values()))

    def test_extract_avoidance(self):
        p = UserProfile(user_id="u")
        _extract_stated_preferences(p, "I don't use curl for production scripts")
        self.assertTrue(any("curl" in v for v in p.stated_preferences.values()))

    def test_level_heuristic_expert(self):
        p = UserProfile(user_id="u")
        _update_level_heuristics(p, "Check ECMP paths and iBGP redistribution for MPLS", "ok")
        self.assertEqual(p.technical_level, TechnicalLevel.EXPERT)

    def test_level_heuristic_novice(self):
        p = UserProfile(user_id="u")
        _update_level_heuristics(p, "What is BGP? How do I configure it? I'm not sure", "ok")
        self.assertEqual(p.technical_level, TechnicalLevel.NOVICE)

    def test_level_heuristic_no_change_when_known(self):
        p = UserProfile(user_id="u")
        p.technical_level = TechnicalLevel.EXPERT
        _update_level_heuristics(p, "What is this? How do I do it? I'm not sure what it means", "ok")
        self.assertEqual(p.technical_level, TechnicalLevel.EXPERT)  # don't downgrade

    def test_communication_style_verbose(self):
        p = UserProfile(user_id="u")
        _update_level_heuristics(p, "thank you for the detailed explanation",
                                 "x" * 1500)
        self.assertEqual(p.communication_style, CommunicationStyle.VERBOSE)

    def test_communication_style_terse(self):
        p = UserProfile(user_id="u")
        _update_level_heuristics(p, "ok got it", "short answer")
        self.assertEqual(p.communication_style, CommunicationStyle.TERSE)


# ── UserModelEngine ───────────────────────────────────────────────────────────
class TestUserModelEngine(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.engine = _engine(self.tmp)

    def test_update_creates_profile(self):
        p = self.engine.update_profile("alice", "s1",
                                       "BGP dropped on R1 router network",
                                       "Check show ip bgp on Cisco IOS")
        self.assertEqual(p.user_id, "alice")
        self.assertGreater(p.total_turns, 0)

    def test_profile_persists_across_instances(self):
        self.engine.update_profile("alice", "s1",
                                   "ECMP iBGP MPLS redistribution config",
                                   "Advanced BGP config applied")
        # New engine instance, same DB
        db = os.path.join(self.tmp, "m", "memory.db")
        engine2 = UserModelEngine(db_path=db)
        p = engine2.get_profile("alice")
        self.assertIsNotNone(p)
        self.assertEqual(p.user_id, "alice")
        self.assertGreater(p.total_turns, 0)

    def test_user_isolation(self):
        self.engine.update_profile("alice", "s1", "OSPF area 0 network config", "ok")
        self.engine.update_profile("bob", "s1", "Kubernetes pod deployment config", "ok")
        alice = self.engine.get_profile("alice")
        bob   = self.engine.get_profile("bob")
        # Alice has network domain, Bob has k8s
        self.assertIn("network", alice.domain_counts)
        self.assertIn("kubernetes", bob.domain_counts)
        # Alice profile doesn't bleed into Bob
        self.assertNotIn("network", bob.domain_counts)

    def test_hourly_activity_tracked(self):
        self.engine.update_profile("alice", "s1", "check bgp status", "ok")
        p = self.engine.get_profile("alice")
        self.assertGreater(sum(p.hourly_activity.values()), 0)

    def test_tool_usage_tracked(self):
        self.engine.update_profile("alice", "s1", "check logs", "ok",
                                   tool_calls=[{"tool": "syslog_search"},
                                               {"tool": "bgp_check"}])
        p = self.engine.get_profile("alice")
        self.assertIn("syslog_search", p.tool_usage)
        self.assertIn("bgp_check", p.tool_usage)

    def test_multiple_turns_accumulate(self):
        for i in range(5):
            self.engine.update_profile("alice", "s1",
                                       f"BGP OSPF network query {i}",
                                       f"Response {i}")
        p = self.engine.get_profile("alice")
        self.assertEqual(p.total_turns, 5)
        self.assertGreater(p.domain_counts.get("network", 0), 0)

    def test_get_prompt_section_empty_if_no_profile(self):
        section = self.engine.get_prompt_section("unknown_user")
        self.assertEqual(section, "")

    def test_get_prompt_section_non_empty_after_update(self):
        self.engine.update_profile("alice", "s1",
                                   "BGP OSPF ECMP iBGP MPLS redistribution",
                                   "Expert-level response")
        section = self.engine.get_prompt_section("alice")
        self.assertIn("USER PROFILE", section)

    def test_llm_fn_called_for_trait_inference(self):
        calls = []
        def fake_llm(system, user):
            calls.append(user)
            return '[{"trait":"test_trait","value":"v","confidence":0.8,"is_revealed":true,"evidence":"e"}]'
        db = os.path.join(self.tmp, "m", "memory.db")
        engine = UserModelEngine(db_path=db, llm_fn=fake_llm)
        engine.update_profile("alice", "s1", "I prefer httpx", "ok")
        self.assertGreater(len(calls), 0)
        p = engine.get_profile("alice")
        self.assertIn("test_trait", p.traits)

    def test_thread_safe_concurrent_updates(self):
        errors = []
        def worker(uid):
            for i in range(10):
                try:
                    self.engine.update_profile(uid, "s1",
                                               f"BGP network query {i} ospf routing",
                                               f"Response text {i}")
                except Exception as e:
                    errors.append(f"{uid}:{e}")
        threads = [threading.Thread(target=worker, args=(f"u{n}",)) for n in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [], f"Concurrent errors: {errors[:3]}")

    def test_list_users(self):
        self.engine.update_profile("alice", "s1", "BGP network config", "ok")
        self.engine.update_profile("bob", "s1", "Kubernetes pod deploy", "ok")
        users = self.engine.list_users()
        self.assertIn("alice", users)
        self.assertIn("bob", users)

    def test_stated_vs_revealed_tracking(self):
        self.engine.update_profile("alice", "s1",
                                   "I prefer httpx for all API calls",
                                   "Noted, using httpx")
        p = self.engine.get_profile("alice")
        stated_vals = list(p.stated_preferences.values())
        self.assertTrue(any("httpx" in v for v in stated_vals),
                        f"httpx not in stated: {stated_vals}")


# ── Integration with MemoryManager ────────────────────────────────────────────
class TestMemoryManagerUserModel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = _mem(self.tmp)

    def tearDown(self):
        self.mem.close()

    def test_update_user_profile_via_manager(self):
        p = self.mem.update_user_profile("alice", "s1",
                                         "BGP OSPF network routing config query",
                                         "Detailed BGP response from assistant")
        self.assertIsNotNone(p)
        self.assertEqual(p.user_id, "alice")

    def test_get_user_profile_none_before_update(self):
        p = self.mem.get_user_profile("nobody")
        self.assertIsNone(p)

    def test_get_user_profile_after_update(self):
        self.mem.update_user_profile("alice", "s1", "BGP query", "ok")
        p = self.mem.get_user_profile("alice")
        self.assertIsNotNone(p)

    def test_build_context_includes_profile(self):
        self.mem.remember("alice", "s1", "BGP session dropped on R1 router")
        self.mem.update_user_profile("alice", "s1",
                                     "BGP OSPF network routing ECMP iBGP MPLS",
                                     "Expert-level response")
        ctx = self.mem.build_context("alice", "BGP", session_id="s1")
        self.assertIn("USER PROFILE", ctx)

    def test_build_context_no_profile_section_if_disabled(self):
        mem2 = MemoryManager(data_dir=os.path.join(self.tmp, "m2"),
                             enable_user_model=False)
        mem2.remember("alice", "s1", "BGP session dropped on R1")
        ctx = mem2.build_context("alice", "BGP", session_id="s1")
        self.assertNotIn("USER PROFILE", ctx)
        mem2.close()

    def test_stats_includes_user_profile(self):
        self.mem.remember("alice", "s1", "test content chunk here")
        self.mem.update_user_profile("alice", "s1", "BGP network query", "ok")
        s = self.mem.stats("alice", "s1")
        self.assertIn("user_profile", s)
        self.assertIsNotNone(s["user_profile"])

    def test_user_model_persists_after_reopen(self):
        self.mem.update_user_profile("alice", "s1",
                                     "ECMP iBGP MPLS redistribution ospf advanced",
                                     "Expert answer provided")
        self.mem.close()
        mem2 = MemoryManager(data_dir=os.path.join(self.tmp, "m"))
        p = mem2.get_user_profile("alice")
        self.assertIsNotNone(p)
        self.assertGreater(p.total_turns, 0)
        mem2.close()

    def test_build_context_budget_respected_with_profile(self):
        for i in range(20):
            self.mem.remember("alice", "s1", f"BGP chunk {i} routing config ospf node{i}")
            self.mem.add_fact("alice", "s1", f"entity fact {i} R{i} at 10.0.{i}.1",
                              "entity", 0.9)
        self.mem.update_user_profile("alice", "s1",
                                     "BGP OSPF ECMP iBGP MPLS expert config",
                                     "Advanced response")
        ctx = self.mem.build_context("alice", "BGP", session_id="s1", max_chars=3000)
        self.assertLessEqual(len(ctx), 3200)

    def test_enable_user_model_false_returns_none(self):
        mem2 = MemoryManager(data_dir=os.path.join(self.tmp, "no_model"),
                             enable_user_model=False)
        result = mem2.update_user_profile("alice", "s1", "text", "resp")
        self.assertIsNone(result)
        self.assertIsNone(mem2.get_user_profile("alice"))
        self.assertEqual(mem2.get_user_profile_section("alice"), "")
        mem2.close()

    def test_profile_section_in_prompt_section_method(self):
        self.mem.update_user_profile("alice", "s1",
                                     "BGP network OSPF routing config query",
                                     "Network expert response")
        section = self.mem.get_user_profile_section("alice")
        self.assertIn("USER PROFILE", section)


if __name__ == "__main__":
    unittest.main(verbosity=2)
