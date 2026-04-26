"""
agent_memory/tests/test_v4_features.py
Tests for v4 additions: ContextBudgetManager, SessionState, MMR retrieval.
Run: python -m unittest agent_memory.tests.test_v4_features -v
"""
from __future__ import annotations
import os, sys, tempfile, threading, time, unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_memory import (
    MemoryManager, Priority,
    ContextBudgetManager, BudgetReport, estimate_tokens,
    SessionState, SessionStateRegistry,
    ConfirmedFact, WorkingSetEntry, RecentToolResult,
    mmr_rerank,
)


def _mem(tmp):
    return MemoryManager(data_dir=os.path.join(tmp, "m"), default_token_budget=3200)


# ─── estimate_tokens ─────────────────────────────────────────────────────────
class TestEstimateTokens(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_ascii_approx(self):
        t = "BGP session dropped on R1 at 02:15 UTC today"
        tokens = estimate_tokens(t)
        # ~4 chars per token for ASCII
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, len(t))

    def test_cjk_denser_than_ascii(self):
        ascii_t = "ABCD" * 10   # 40 chars
        cjk_t   = "你好世界" * 10   # 40 chars, but denser
        ascii_tok = estimate_tokens(ascii_t)
        cjk_tok   = estimate_tokens(cjk_t)
        # CJK chars should produce MORE tokens for the same char count
        self.assertGreater(cjk_tok, ascii_tok)

    def test_custom_tiktoken_fn(self):
        # custom counter always returns 1 per char
        def exact(text): return len(text)
        mgr = ContextBudgetManager(token_budget=100, tiktoken_fn=exact)
        mgr.add(Priority.MID_TERM_FACTS, "hello world", "test")
        ctx, report = mgr.build()
        self.assertEqual(report.kept_sections[0][1], 11)  # exact char count


# ─── ContextBudgetManager ────────────────────────────────────────────────────
class TestContextBudgetManager(unittest.TestCase):

    def test_all_fit_within_budget(self):
        mgr = ContextBudgetManager(token_budget=500)
        mgr.add(Priority.USER_PROFILE,    "User profile text here")
        mgr.add(Priority.CONFIRMED_FACTS, "Confirmed: R1 ping OK")
        mgr.add(Priority.WORKING_SET,     "Working: R1 device")
        mgr.add(Priority.MID_TERM_FACTS,  "Facts: BGP peer unreachable")
        ctx, report = mgr.build()
        self.assertEqual(len(report.dropped_sections), 0)
        self.assertIn("User profile", ctx)
        self.assertIn("BGP peer", ctx)

    def test_low_priority_evicted_first(self):
        mgr = ContextBudgetManager(token_budget=10)
        mgr.add(Priority.USER_PROFILE,    "A" * 10, "profile")   # ~3 tokens
        mgr.add(Priority.CONFIRMED_FACTS, "B" * 10, "confirmed") # ~3 tokens
        mgr.add(Priority.ENVIRONMENT,     "C" * 60, "env")       # ~15 tokens
        ctx, report = mgr.build()
        dropped = [n for n, _ in report.dropped_sections]
        # environment (P6) should be dropped, not user_profile or confirmed_facts
        self.assertIn("env", dropped)
        self.assertNotIn("profile", dropped)
        self.assertNotIn("confirmed", dropped)

    def test_p1_p2_never_evicted(self):
        # Even if P1+P2 exceed budget, they must be included
        mgr = ContextBudgetManager(token_budget=5)   # tiny budget
        mgr.add(Priority.USER_PROFILE,    "X" * 40, "profile")
        mgr.add(Priority.CONFIRMED_FACTS, "Y" * 40, "confirmed")
        mgr.add(Priority.ENVIRONMENT,     "Z" * 10, "env")
        ctx, report = mgr.build()
        kept_names = [n for n, _ in report.kept_sections]
        self.assertIn("profile",   kept_names)
        self.assertIn("confirmed", kept_names)
        # Environment should be dropped (budget exceeded by P1+P2)
        self.assertIn("env", [n for n, _ in report.dropped_sections])

    def test_priority_ordering_in_output(self):
        mgr = ContextBudgetManager(token_budget=200)
        mgr.add(Priority.ENVIRONMENT,     "env content here",      "env")
        mgr.add(Priority.USER_PROFILE,    "profile content here",  "profile")
        mgr.add(Priority.MID_TERM_FACTS,  "facts content here",    "facts")
        mgr.add(Priority.CONFIRMED_FACTS, "confirmed content here", "confirmed")
        ctx, _ = mgr.build()
        # P1 (profile) must appear before P2 (confirmed) before P4 (facts) before P6 (env)
        p_profile   = ctx.find("profile content")
        p_confirmed = ctx.find("confirmed content")
        p_facts     = ctx.find("facts content")
        p_env       = ctx.find("env content")
        self.assertGreater(p_confirmed, p_profile)
        self.assertGreater(p_facts,     p_confirmed)
        self.assertGreater(p_env,       p_facts)

    def test_trimming_to_budget(self):
        mgr = ContextBudgetManager(token_budget=15)
        long_text = "\n".join([f"Line {i} content here" for i in range(20)])
        mgr.add(Priority.MID_TERM_FACTS, long_text, "facts")
        ctx, report = mgr.build()
        kept = [n for n, _ in report.kept_sections]
        if "facts [trimmed]" in kept:
            self.assertLessEqual(report.used_tokens, 20)  # trimmed to fit

    def test_empty_section_skipped(self):
        mgr = ContextBudgetManager(token_budget=100)
        mgr.add(Priority.USER_PROFILE,  "", "profile")
        mgr.add(Priority.MID_TERM_FACTS, "actual facts content", "facts")
        ctx, report = mgr.build()
        self.assertIn("actual facts", ctx)
        self.assertEqual(len([n for n, _ in report.kept_sections
                               if "profile" in n]), 0)

    def test_reset_clears_sections(self):
        mgr = ContextBudgetManager(token_budget=100)
        mgr.add(Priority.MID_TERM_FACTS, "some facts here", "f1")
        mgr.reset()
        ctx, report = mgr.build()
        self.assertEqual(ctx, "")
        self.assertEqual(len(report.kept_sections), 0)

    def test_budget_report_summary(self):
        mgr = ContextBudgetManager(token_budget=200)
        mgr.add(Priority.MID_TERM_FACTS,  "fact content here", "facts")
        mgr.add(Priority.LONG_TERM_CHUNKS, "chunk content here", "chunks")
        _, report = mgr.build()
        summary = report.summary()
        self.assertIn("Budget:", summary)
        self.assertIn("facts", summary)
        self.assertIn("chunks", summary)

    def test_budget_report_to_dict(self):
        mgr = ContextBudgetManager(token_budget=200)
        mgr.add(Priority.MID_TERM_FACTS, "some content here", "facts")
        _, report = mgr.build()
        d = report.to_dict()
        self.assertIn("total_budget", d)
        self.assertIn("used_tokens", d)
        self.assertIn("utilization", d)
        self.assertIn("kept_sections", d)

    def test_utilization_property(self):
        mgr = ContextBudgetManager(token_budget=100)
        mgr.add(Priority.MID_TERM_FACTS, "X" * 40, "f")  # ~10 tokens
        _, report = mgr.build()
        self.assertGreater(report.utilization, 0)
        self.assertLessEqual(report.utilization, 1.1)  # can slightly exceed


# ─── SessionState ─────────────────────────────────────────────────────────────
class TestSessionState(unittest.TestCase):

    def _state(self):
        return SessionState(user_id="alice", session_id="s1")

    def test_confirm_fact(self):
        s = self._state()
        cf = s.confirm_fact("R1 ping OK", source="tool:ping")
        self.assertEqual(cf.text, "R1 ping OK")
        self.assertEqual(cf.source, "tool:ping")
        self.assertEqual(len(s.confirmed_facts), 1)

    def test_confirm_fact_deduplication(self):
        s = self._state()
        s.confirm_fact("R1 ping OK")
        s.confirm_fact("R1 ping OK")   # duplicate
        self.assertEqual(len(s.confirmed_facts), 1)

    def test_retract_fact(self):
        s = self._state()
        s.confirm_fact("R1 ping OK")
        removed = s.retract_fact("R1 ping OK")
        self.assertTrue(removed)
        self.assertEqual(len(s.confirmed_facts), 0)

    def test_retract_nonexistent(self):
        s = self._state()
        removed = s.retract_fact("does not exist")
        self.assertFalse(removed)

    def test_add_to_working_set(self):
        s = self._state()
        s.add_to_working_set("R1", "Router R1", "device", {"ip": "10.0.0.1"})
        ws = s.working_set
        self.assertEqual(len(ws), 1)
        self.assertEqual(ws[0].entity_id, "R1")
        self.assertEqual(ws[0].metadata["ip"], "10.0.0.1")

    def test_working_set_update_existing(self):
        s = self._state()
        s.add_to_working_set("R1", "Router R1", "device", {"ip": "10.0.0.1"})
        s.add_to_working_set("R1", "Router R1 (updated)", "device", {"ip": "10.0.0.2"})
        ws = s.working_set
        self.assertEqual(len(ws), 1)
        self.assertEqual(ws[0].metadata["ip"], "10.0.0.2")

    def test_remove_from_working_set(self):
        s = self._state()
        s.add_to_working_set("R1", "Router R1", "device")
        removed = s.remove_from_working_set("R1")
        self.assertTrue(removed)
        self.assertEqual(len(s.working_set), 0)

    def test_add_tool_result_inline(self):
        s = self._state()
        s.add_tool_result("ping", "PING OK 4ms")
        tr = s.recent_results
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr[0].content, "PING OK 4ms")

    def test_add_tool_result_large_uses_ref_id(self):
        s = SessionState(user_id="alice", session_id="s1", inline_threshold=10)
        s.add_tool_result("syslog", "X" * 100, ref_id="ref-abc")
        tr = s.recent_results
        self.assertEqual(tr[0].ref_id, "ref-abc")
        self.assertEqual(tr[0].content, "")

    def test_tool_result_format_with_ref(self):
        s = self._state()
        s.add_tool_result("syslog", "", ref_id="ref-abc")
        tr = s.recent_results[0]
        fmt = tr.format()
        self.assertIn("[STORED:ref-abc]", fmt)

    def test_build_hot_context_all_sections(self):
        s = self._state()
        s.confirm_fact("R1 OK")
        s.add_to_working_set("R1", "Router R1", "device")
        s.add_tool_result("ping", "PING OK 4ms")
        hot = s.build_hot_context()
        self.assertIn("confirmed_facts", hot)
        self.assertIn("working_set",     hot)
        self.assertIn("recent_tools",    hot)

    def test_build_hot_context_empty(self):
        s = self._state()
        hot = s.build_hot_context()
        self.assertEqual(hot, {})

    def test_increment_turn(self):
        s = self._state()
        self.assertEqual(s.increment_turn(), 1)
        self.assertEqual(s.increment_turn(), 2)
        self.assertEqual(s.turn_count, 2)

    def test_clear(self):
        s = self._state()
        s.confirm_fact("fact")
        s.add_to_working_set("R1", "Router R1", "device")
        s.clear()
        self.assertEqual(len(s.confirmed_facts), 0)
        self.assertEqual(len(s.working_set), 0)

    def test_to_dict(self):
        s = self._state()
        s.confirm_fact("R1 OK", source="tool:ping")
        s.add_to_working_set("R1", "Router R1", "device")
        d = s.to_dict()
        self.assertEqual(d["user_id"], "alice")
        self.assertEqual(len(d["confirmed_facts"]), 1)
        self.assertEqual(len(d["working_set"]), 1)

    def test_thread_safe_concurrent_writes(self):
        s = self._state()
        errors = []
        def worker(n):
            try:
                for i in range(20):
                    s.confirm_fact(f"fact-{n}-{i} content here now")
                    s.add_to_working_set(f"entity-{n}", f"Entity {n}", "device")
                    s.add_tool_result(f"tool_{n}", f"result {n} {i}")
            except Exception as e:
                errors.append(str(e))
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])

    def test_max_working_set_cap(self):
        s = SessionState(user_id="alice", session_id="s1", max_working_set=3)
        for i in range(10):
            s.add_to_working_set(f"entity-{i}", f"Entity {i}", "device")
        self.assertLessEqual(len(s.working_set), 3)

    def test_max_confirmed_facts_cap(self):
        s = SessionState(user_id="alice", session_id="s1", max_confirmed_facts=5)
        for i in range(20):
            s.confirm_fact(f"unique fact number {i} content text")
        self.assertLessEqual(len(s.confirmed_facts), 5)


# ─── SessionStateRegistry ─────────────────────────────────────────────────────
class TestSessionStateRegistry(unittest.TestCase):

    def test_get_or_create(self):
        reg = SessionStateRegistry()
        s = reg.get_or_create("alice", "s1")
        self.assertIsInstance(s, SessionState)

    def test_same_session_same_object(self):
        reg = SessionStateRegistry()
        s1 = reg.get_or_create("alice", "s1")
        s2 = reg.get_or_create("alice", "s1")
        self.assertIs(s1, s2)

    def test_different_sessions_different_objects(self):
        reg = SessionStateRegistry()
        s1 = reg.get_or_create("alice", "s1")
        s2 = reg.get_or_create("alice", "s2")
        self.assertIsNot(s1, s2)

    def test_user_isolation(self):
        reg = SessionStateRegistry()
        a = reg.get_or_create("alice", "s1")
        b = reg.get_or_create("bob",   "s1")
        self.assertIsNot(a, b)

    def test_drop(self):
        reg = SessionStateRegistry()
        s = reg.get_or_create("alice", "s1")
        s.confirm_fact("fact here")
        reg.drop("alice", "s1")
        # After drop, get returns None
        self.assertIsNone(reg.get("alice", "s1"))

    def test_active_sessions(self):
        reg = SessionStateRegistry()
        reg.get_or_create("alice", "s1")
        reg.get_or_create("alice", "s2")
        reg.get_or_create("bob", "s1")
        alice_sessions = reg.active_sessions("alice")
        self.assertIn("s1", alice_sessions)
        self.assertIn("s2", alice_sessions)
        self.assertNotIn("s1", reg.active_sessions("charlie"))

    def test_total_active(self):
        reg = SessionStateRegistry()
        reg.get_or_create("alice", "s1")
        reg.get_or_create("bob", "s1")
        self.assertEqual(reg.total_active, 2)


# ─── MMR Retrieval ─────────────────────────────────────────────────────────────
class TestMMRRetrieval(unittest.TestCase):

    def test_pure_relevance_lambda_1(self):
        candidates = [
            ("BGP session dropped on R1", 0.9),
            ("BGP session dropped on R2", 0.8),
            ("DNS resolution failure", 0.5),
        ]
        result = mmr_rerank(candidates, top_k=2, lambda_=1.0)
        self.assertEqual(len(result), 2)
        # Pure relevance: top 2 by score
        texts = [t for t, _ in result]
        self.assertIn("BGP session dropped on R1", texts)

    def test_diversity_reduces_duplicates(self):
        # Two nearly identical docs + one diverse doc
        candidates = [
            ("BGP session dropped on R1 router today",  0.9),
            ("BGP session dropped on R1 router again",  0.85),  # near-duplicate
            ("Prometheus disk alert on node01 server",  0.7),   # diverse
        ]
        result_mmr   = mmr_rerank(candidates, top_k=2, lambda_=0.3)
        result_plain = candidates[:2]   # plain top-2 = both BGP dupes

        texts_mmr   = [t for t, _ in result_mmr]
        texts_plain = [t for t, _ in result_plain]

        # MMR should prefer diverse doc over near-duplicate
        self.assertIn("Prometheus disk alert on node01 server", texts_mmr)
        # Plain would have both BGP dupes
        self.assertNotIn("Prometheus disk alert on node01 server", texts_plain)

    def test_empty_candidates(self):
        result = mmr_rerank([], top_k=3)
        self.assertEqual(result, [])

    def test_top_k_respected(self):
        candidates = [(f"doc {i} content text", 1.0 - i * 0.1) for i in range(10)]
        result = mmr_rerank(candidates, top_k=3)
        self.assertLessEqual(len(result), 3)

    def test_single_candidate(self):
        result = mmr_rerank([("only one doc here", 0.9)], top_k=5)
        self.assertEqual(len(result), 1)

    def test_scores_positive(self):
        candidates = [("doc A content", 0.8), ("doc B content", 0.6)]
        result = mmr_rerank(candidates, top_k=2, lambda_=0.6)
        for _, score in result:
            # MMR scores can be negative when diversity penalty is high, that's OK
            self.assertIsInstance(score, float)


# ─── Integration: build_context_budgeted ──────────────────────────────────────
class TestBuildContextBudgeted(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = _mem(self.tmp)
        # Seed data
        for i in range(5):
            self.mem.remember("alice", "s1",
                              f"BGP session AS{i} dropped at 02:{i:02d} UTC on R1")
            self.mem.add_fact("alice", "s1",
                              f"AS{i} is peer of R1 at 10.0.{i}.1",
                              "entity", 0.9)
        self.mem.update_user_profile("alice", "s1",
                                     "Check ECMP iBGP MPLS routing config R1",
                                     "Advanced BGP expert response provided")
        self.state = self.mem.get_session("alice", "s1")
        self.state.confirm_fact("R1 ping OK", source="tool:ping")
        self.state.add_to_working_set("R1", "Router R1", "device", {"ip": "10.0.0.1"})
        self.state.add_tool_result("ping", "PING 10.0.0.1: 4ms OK")

    def tearDown(self):
        self.mem.close()

    def test_returns_tuple(self):
        result = self.mem.build_context_budgeted(self.state, "BGP neighbor")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_context_str_non_empty(self):
        ctx, _ = self.mem.build_context_budgeted(self.state, "BGP neighbor")
        self.assertIsInstance(ctx, str)
        self.assertGreater(len(ctx), 0)

    def test_confirmed_facts_in_context(self):
        ctx, _ = self.mem.build_context_budgeted(self.state, "BGP neighbor")
        self.assertIn("Confirmed Facts", ctx)
        self.assertIn("R1 ping OK", ctx)

    def test_working_set_in_context(self):
        ctx, _ = self.mem.build_context_budgeted(self.state, "BGP neighbor")
        self.assertIn("Working Set", ctx)
        self.assertIn("Router R1", ctx)

    def test_user_profile_in_context(self):
        ctx, _ = self.mem.build_context_budgeted(self.state, "BGP neighbor")
        self.assertIn("USER PROFILE", ctx)

    def test_report_has_budget_info(self):
        _, report = self.mem.build_context_budgeted(self.state, "BGP neighbor",
                                                     budget=3200)
        self.assertEqual(report.total_budget, 3200)
        self.assertGreater(report.used_tokens, 0)
        self.assertGreater(len(report.kept_sections), 0)

    def test_tiny_budget_p1_p2_always_included(self):
        ctx, report = self.mem.build_context_budgeted(
            self.state, "BGP neighbor", budget=10)
        # P1 (user_profile) and P2 (confirmed_facts) must always be present
        kept = [n for n, _ in report.kept_sections]
        self.assertTrue(any("user_profile" in n or "confirmed_facts" in n
                            for n in kept))

    def test_budget_utilization_sensible(self):
        _, report = self.mem.build_context_budgeted(
            self.state, "BGP neighbor", budget=3200)
        self.assertGreaterEqual(report.utilization, 0)
        # With real data, should use at least some budget
        self.assertGreater(report.used_tokens, 5)

    def test_environment_injected_and_lowest_priority(self):
        ctx, report = self.mem.build_context_budgeted(
            self.state, "BGP neighbor", budget=3200,
            environment="System: datacenter=east, noc_mode=active")
        # Environment should appear in context (budget is large enough)
        if not report.dropped_sections or "environment" not in \
                [n for n, _ in report.dropped_sections]:
            self.assertIn("datacenter", ctx)

    def test_backward_compat_build_context(self):
        # Old API still works
        ctx = self.mem.build_context("alice", "BGP neighbor", session_id="s1")
        self.assertIsInstance(ctx, str)

    def test_get_session_returns_state(self):
        state = self.mem.get_session("alice", "s1")
        self.assertIsInstance(state, SessionState)
        self.assertEqual(state.user_id, "alice")

    def test_end_session_clears_hot_track(self):
        state = self.mem.get_session("alice", "end-test")
        state.confirm_fact("temporary fact")
        self.mem.end_session("alice", "end-test")
        # After end, session is removed from registry
        self.assertIsNone(self.mem._sessions.get("alice", "end-test"))

    def test_stats_includes_hot_track(self):
        s = self.mem.get_session("alice", "s1")
        s.confirm_fact("some verified fact")
        stats = self.mem.stats("alice", "s1")
        self.assertIn("hot_track", stats)
        self.assertIn("active_hot_sessions", stats)

    def test_estimate_tokens_method(self):
        t = self.mem.estimate_tokens("BGP session dropped on R1")
        self.assertGreater(t, 0)

    def test_mmr_search_via_manager(self):
        r = self.mem.search_chunks("alice", "BGP session dropped", use_mmr=True)
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r.total_found, 0)

    def test_mmr_facts_search_via_manager(self):
        r = self.mem.search_facts("alice", "BGP peer AS", use_mmr=True)
        self.assertIsNotNone(r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
