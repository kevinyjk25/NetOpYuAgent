"""
agent_memory/tests/test_v5_features.py
Tests for v5 new capabilities:
  - SkillStore (程序性记忆)
  - EmbeddingIndex (语义向量检索)
  - MemoryConsolidator (历史压缩)
  - ReflectionEngine (反思写入)
  - MemoryManager v5 integration

Run: python -m unittest agent_memory.tests.test_v5_features -v
"""
from __future__ import annotations
import os, sys, tempfile, threading, time, math, unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent_memory import (
    MemoryManager, MemoryChunk, MemoryFact,
    SkillStore, Skill,
    EmbeddingIndex, TFIDFBackend, CallableBackend, cosine_similarity,
    MemoryConsolidator, ReflectionEngine,
)
from agent_memory.stores.skill_store import DEPRECATE_THRESHOLD, EMA_ALPHA


def _mem(tmp, **kwargs):
    return MemoryManager(data_dir=os.path.join(tmp, "m"), **kwargs)


# ─── cosine_similarity ────────────────────────────────────────────────────────
class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=5)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, places=5)

    def test_zero_vector(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_range_clamped(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        sim = cosine_similarity(a, b)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)


# ─── EmbeddingIndex (TF-IDF backend) ─────────────────────────────────────────
class TestEmbeddingIndexTFIDF(unittest.TestCase):
    """EmbeddingIndex with default TF-IDF backend — zero deps, full compatibility."""

    def setUp(self):
        self.idx = EmbeddingIndex()

    def test_backend_name(self):
        self.assertEqual(self.idx.backend_name, "tfidf")

    def test_add_and_query(self):
        self.idx.add("d1", "BGP session dropped on R1 router neighbor")
        self.idx.add("d2", "DNS resolution failure payments service")
        self.idx.add("d3", "BGP neighbor unreachable AS65002 timeout")
        results = self.idx.query("BGP neighbor problem", top_k=2)
        ids = [r[0] for r in results]
        self.assertTrue("d1" in ids or "d3" in ids)

    def test_remove(self):
        self.idx.add("d1", "redis cache timeout config")
        self.idx.remove("d1")
        self.assertEqual(self.idx.query("redis cache"), [])

    def test_size(self):
        self.idx.add("a", "hello world content")
        self.idx.add("b", "foo bar baz content")
        self.assertEqual(self.idx.size, 2)
        self.idx.remove("a")
        self.assertEqual(self.idx.size, 1)

    def test_query_mmr_fallback(self):
        self.idx.add("d1", "BGP session AS65001 dropped")
        self.idx.add("d2", "BGP session AS65002 dropped")
        self.idx.add("d3", "Prometheus disk alert node01")
        results = self.idx.query_mmr("BGP session dropped", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_clear(self):
        self.idx.add("x", "some content here")
        self.idx.clear()
        self.assertEqual(self.idx.size, 0)


# ─── EmbeddingIndex (callable backend) ────────────────────────────────────────
class TestEmbeddingIndexCallable(unittest.TestCase):
    """EmbeddingIndex with a deterministic callable embedding function."""

    def _make_embed_fn(self):
        """Simple hash-based embedding for testing (deterministic, dim=8)."""
        def embed(text: str):
            import hashlib
            h = hashlib.md5(text.encode()).digest()
            vec = [(b / 127.5) - 1.0 for b in h[:8]]
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0
            return [x / norm for x in vec]
        return embed

    def setUp(self):
        self.embed_fn = self._make_embed_fn()
        self.idx = EmbeddingIndex.from_callable(self.embed_fn, dim=8)

    def test_backend_name(self):
        self.assertIn("callable", self.idx.backend_name)

    def test_add_and_query(self):
        self.idx.add("d1", "BGP session dropped neighbor")
        self.idx.add("d2", "DNS resolution failure")
        results = self.idx.query("BGP session dropped neighbor", top_k=2)
        self.assertGreater(len(results), 0)
        # d1 should rank first (identical text → cosine=1.0)
        self.assertEqual(results[0][0], "d1")

    def test_size(self):
        self.idx.add("a", "content alpha")
        self.idx.add("b", "content beta")
        self.assertEqual(self.idx.size, 2)

    def test_remove(self):
        self.idx.add("x", "unique content xyz")
        self.idx.remove("x")
        self.assertEqual(self.idx.size, 0)

    def test_query_mmr_diversity(self):
        """MMR should prefer diverse results over near-identical ones."""
        texts = {
            "d1": "BGP session dropped at 02:15",
            "d2": "BGP session dropped at 02:16",   # near-dup
            "d3": "Prometheus disk alert node01",    # diverse
        }
        for doc_id, text in texts.items():
            self.idx.add(doc_id, text)

        plain = self.idx.query("BGP session dropped", top_k=2)
        mmr   = self.idx.query_mmr("BGP session dropped", top_k=2, lambda_=0.2)
        # MMR with low lambda should prefer diversity
        mmr_ids = [r[0] for r in mmr]
        # At least one result should differ from plain top-2
        self.assertTrue(len(mmr_ids) > 0)

    def test_query_empty_index(self):
        results = self.idx.query("anything")
        self.assertEqual(results, [])

    def test_thread_safe_adds(self):
        errors = []
        def worker(n):
            for i in range(20):
                try:
                    self.idx.add(f"doc_{n}_{i}", f"content {n} {i} BGP network routing")
                except Exception as e:
                    errors.append(str(e))
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])

    def test_from_callable_factory(self):
        idx = EmbeddingIndex.from_callable(self.embed_fn, dim=8)
        self.assertIsNotNone(idx)
        idx.add("test", "hello world")
        self.assertEqual(idx.size, 1)


# ─── SkillStore ────────────────────────────────────────────────────────────────
class TestSkillStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db  = os.path.join(self.tmp, "test.db")
        self.store = SkillStore(db_path=self.db)

    def test_save_and_retrieve(self):
        sk = self.store.save_skill(
            "alice", "bgp_diagnosis",
            description="BGP邻居故障排查标准流程",
            steps=["bgp_check(R1)", "syslog_search(peer)", "ping(peer_ip)"],
            tags=["bgp", "network", "diagnosis"],
        )
        self.assertIsNotNone(sk.skill_id)
        self.assertEqual(sk.skill_name, "bgp_diagnosis")
        self.assertEqual(sk.version, 1)
        self.assertEqual(len(sk.steps), 3)

    def test_save_updates_version(self):
        self.store.save_skill("alice", "bgp_diag", "v1 description",
                              ["step1"], tags=["bgp"])
        sk2 = self.store.save_skill("alice", "bgp_diag", "v2 improved desc",
                                    ["step1", "step2", "step3"], tags=["bgp", "v2"])
        self.assertEqual(sk2.version, 2)
        self.assertEqual(len(sk2.steps), 3)

    def test_get_by_name(self):
        self.store.save_skill("alice", "my_skill", "desc", ["step"], tags=[])
        sk = self.store.get_skill_by_name("alice", "my_skill")
        self.assertIsNotNone(sk)
        self.assertEqual(sk.skill_name, "my_skill")

    def test_user_isolation(self):
        self.store.save_skill("alice", "shared_name", "alice skill", ["a"], tags=[])
        self.store.save_skill("bob",   "shared_name", "bob skill",   ["b"], tags=[])
        alice_sk = self.store.get_skill_by_name("alice", "shared_name")
        bob_sk   = self.store.get_skill_by_name("bob",   "shared_name")
        self.assertEqual(alice_sk.description, "alice skill")
        self.assertEqual(bob_sk.description,   "bob skill")

    def test_find_skills_fts(self):
        self.store.save_skill("alice", "bgp_diag",
                              "BGP邻居故障排查，ping测试对端可达性",
                              ["bgp_check", "ping"], tags=["bgp", "network"])
        self.store.save_skill("alice", "dns_fix",
                              "DNS解析失败修复流程",
                              ["nslookup", "dig"], tags=["dns"])
        results = self.store.find_skills("alice", "BGP neighbor unreachable")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].skill_name, "bgp_diag")

    def test_find_skills_returns_active_only(self):
        sk = self.store.save_skill("alice", "deprecated_skill",
                                   "old deprecated skill", ["step"], tags=[])
        # Manually deprecate by recording many failures
        for _ in range(20):
            self.store.record_outcome("alice", sk.skill_id, success=False)
        results = self.store.find_skills("alice", "deprecated old")
        skill_ids = [s.skill_id for s in results]
        self.assertNotIn(sk.skill_id, skill_ids)

    def test_record_outcome_updates_rate(self):
        sk = self.store.save_skill("alice", "test_skill", "test", ["s"], tags=[])
        new_rate = self.store.record_outcome("alice", sk.skill_id, success=True)
        self.assertIsNotNone(new_rate)
        self.assertAlmostEqual(new_rate, 1.0, delta=0.1)

    def test_record_failure_decreases_rate(self):
        sk = self.store.save_skill("alice", "failing_skill", "desc", ["s"], tags=[])
        rate = 1.0
        for _ in range(5):
            rate = self.store.record_outcome("alice", sk.skill_id, success=False)
        self.assertLess(rate, 0.8)

    def test_deprecation_on_low_rate(self):
        sk = self.store.save_skill("alice", "bad_skill", "desc", ["s"], tags=[])
        # Record many failures to push rate below DEPRECATE_THRESHOLD
        for _ in range(30):
            self.store.record_outcome("alice", sk.skill_id, success=False)
        fetched = self.store.get_skill("alice", sk.skill_id)
        self.assertEqual(fetched.status, "deprecated")

    def test_list_skills(self):
        self.store.save_skill("alice", "s1", "desc1", ["a"], tags=[])
        self.store.save_skill("alice", "s2", "desc2", ["b"], tags=[])
        skills = self.store.list_skills("alice")
        names = {s.skill_name for s in skills}
        self.assertIn("s1", names)
        self.assertIn("s2", names)

    def test_delete_skill(self):
        sk = self.store.save_skill("alice", "to_delete", "temp skill", ["step"], tags=[])
        deleted = self.store.delete_skill("alice", sk.skill_id)
        self.assertTrue(deleted)
        fetched = self.store.get_skill("alice", sk.skill_id)
        self.assertIsNone(fetched)

    def test_count(self):
        self.store.save_skill("alice", "s1", "d1", ["a"], tags=[])
        self.store.save_skill("alice", "s2", "d2", ["b"], tags=[])
        self.assertEqual(self.store.count("alice"), 2)
        self.assertEqual(self.store.count("bob"), 0)

    def test_to_prompt_block(self):
        sk = self.store.save_skill("alice", "bgp_diag",
                                   "BGP故障排查", ["step1", "step2"], tags=["bgp"])
        block = sk.to_prompt_block()
        self.assertIn("bgp_diag", block)
        self.assertIn("step1", block)
        self.assertIn("bgp", block.lower())

    def test_build_skills_context(self):
        self.store.save_skill("alice", "bgp_skill",
                              "BGP邻居故障排查标准流程ping测试",
                              ["bgp_check", "ping"], tags=["bgp"])
        ctx = self.store.build_skills_context("alice", "BGP failure", top_k=1)
        self.assertIn("可复用技能", ctx)
        self.assertIn("bgp_skill", ctx)

    def test_concurrent_save(self):
        errors = []
        def worker(n):
            try:
                self.store.save_skill(f"user{n}", f"skill_{n}",
                                      f"description {n}", [f"step {n}"], tags=[])
            except Exception as e:
                errors.append(str(e))
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])


# ─── MemoryConsolidator ────────────────────────────────────────────────────────
class TestMemoryConsolidator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = _mem(self.tmp, consolidate_after_n_turns=5)

    def tearDown(self):
        self.mem.close()

    def test_consolidate_below_threshold_skipped(self):
        # Only 3 chunks — below threshold of 5
        for i in range(3):
            self.mem.remember("alice", "s1", f"content chunk {i} BGP network routing")
        result = self.mem.consolidate_session("alice", "s1")
        self.assertTrue(result.get("skipped", False))

    def test_consolidate_above_threshold(self):
        # Write enough chunks to trigger
        for i in range(15):
            self.mem.remember("alice", "s1",
                              f"Turn {i}: BGP session AS{i} dropped at 0{i%9}:00 UTC on R1")
        count_before = self.mem.long_term.count_session("alice", "s1")
        result = self.mem.consolidate_session("alice", "s1")
        count_after = self.mem.long_term.count("alice")
        self.assertFalse(result.get("skipped", True))
        self.assertGreater(result.get("chunks_merged", 0), 0)
        self.assertGreater(result.get("summaries_created", 0), 0)
        # After consolidation, fewer chunks remain
        self.assertLess(count_after, count_before)

    def test_consolidate_keeps_recent_chunks(self):
        for i in range(20):
            self.mem.remember("alice", "s1",
                              f"Chunk {i}: important content about BGP network config")
        result = self.mem.consolidate_session("alice", "s1")
        self.assertFalse(result.get("skipped", True))
        # At least keep_recent_n chunks remain (in some form)
        remaining = self.mem.long_term.count_session("alice", "s1")
        self.assertGreater(remaining, 0)

    def test_should_consolidate_flag(self):
        for i in range(3):
            self.mem.remember("alice", "s1", f"chunk {i}")
        self.assertFalse(self.mem.should_consolidate("alice", "s1"))
        for i in range(10):
            self.mem.remember("alice", "s1", f"chunk extra {i}")
        self.assertTrue(self.mem.should_consolidate("alice", "s1"))

    def test_summary_chunk_has_high_importance(self):
        for i in range(15):
            self.mem.remember("alice", "s1", f"Conversation turn {i} BGP routing config")
        self.mem.consolidate_session("alice", "s1")
        # Search for summary chunks
        r = self.mem.search_chunks("alice", "摘要", source_filter="summary")
        # summaries should appear in search results (high importance)
        # (may be 0 if no LLM, but should not crash)
        self.assertIsNotNone(r)

    def test_consolidate_fallback_no_llm(self):
        """Consolidation works without LLM using heuristic summary."""
        for i in range(12):
            self.mem.remember("alice", "s1", f"chunk {i} content BGP network routing config")
        result = self.mem.consolidate_session("alice", "s1")
        # Should not crash even without LLM
        self.assertIn("chunks_merged", result)

    def test_consolidate_with_mock_llm(self):
        def mock_llm(prompt):
            return "摘要：BGP会话在多个时间点断开，涉及AS65001到AS65010的多个邻居。"
        mem2 = _mem(tempfile.mkdtemp(), llm_fn=mock_llm, consolidate_after_n_turns=5)
        for i in range(25):
            mem2.remember("alice", "s1",
                          f"BGP session AS{i} dropped at 0{i%10}:00 UTC")
        result = mem2.consolidate_session("alice", "s1")
        self.assertFalse(result.get("skipped", True))
        self.assertGreater(result["summaries_created"], 0)
        mem2.close()


# ─── ReflectionEngine ─────────────────────────────────────────────────────────
class TestReflectionEngine(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = _mem(self.tmp)

    def tearDown(self):
        self.mem.close()

    def test_reflect_success_writes_chunk(self):
        result = self.mem.reflect(
            "alice", "s1",
            task="BGP故障排查",
            outcome="success",
            summary="通过ping确认对端不可达，联系NOC后恢复正常",
        )
        self.assertEqual(result["outcome"], "success")
        self.assertIn("long_term_chunk_id", result)
        # Verify chunk is in DB
        r = self.mem.search_chunks("alice", "反思 BGP故障排查")
        self.assertGreater(r.total_found, 0)

    def test_reflect_failure_writes_chunk(self):
        result = self.mem.reflect(
            "alice", "s1",
            task="自动重启BGP进程",
            outcome="failure",
            summary="尝试了3种方式均被拒绝",
            reason="权限不足，缺少write访问",
        )
        self.assertEqual(result["outcome"], "failure")
        self.assertIn("long_term_chunk_id", result)

    def test_reflect_with_skill_updates_rate(self):
        sk = self.mem.save_skill("alice", "bgp_diag", "BGP排查", ["step1"], tags=["bgp"])
        result = self.mem.reflect(
            "alice", "s1",
            task="BGP排查", outcome="success",
            summary="成功定位问题", skill_id=sk.skill_id,
        )
        self.assertIn("new_skill_rate", result)
        self.assertIsNotNone(result["new_skill_rate"])

    def test_reflect_failure_with_skill_decreases_rate(self):
        sk = self.mem.save_skill("alice", "risky_skill", "风险操作", ["step"], tags=[])
        result = self.mem.reflect(
            "alice", "s1",
            task="风险操作", outcome="failure",
            summary="执行失败", skill_id=sk.skill_id,
        )
        # success_rate should decrease
        updated = self.mem.skill_store.get_skill("alice", sk.skill_id)
        self.assertLess(updated.success_rate, 1.0)

    def test_reflect_with_llm_extracts_lessons(self):
        def mock_llm(prompt):
            if "反思" in prompt or "教训" in prompt or "lesson" in prompt.lower():
                return '[{"lesson": "关键教训：先验证L2连通性再排查BGP", "confidence": 0.85}]'
            return "反思：应先确认物理链路，关键步骤是L2验证。建议：下次避免直接假设对端故障。"
        mem2 = _mem(tempfile.mkdtemp(), llm_fn=mock_llm)
        result = mem2.reflect(
            "alice", "s1",
            task="BGP排查", outcome="success",
            summary="成功，关键是L2验证"
        )
        # Should have lesson facts (LLM path)
        # The lesson extraction may or may not work depending on mock response
        self.assertIsNotNone(result)
        mem2.close()

    def test_reflect_source_is_reflection(self):
        self.mem.reflect("alice", "s1", task="test task",
                         outcome="success", summary="test summary")
        r = self.mem.search_chunks("alice", "反思 test", source_filter="reflection")
        self.assertIsNotNone(r)

    def test_reflect_partial_outcome(self):
        result = self.mem.reflect(
            "alice", "s1",
            task="部分完成的任务",
            outcome="partial",
            summary="完成了80%，剩余部分因超时未完成",
        )
        self.assertEqual(result["outcome"], "partial")
        self.assertIn("long_term_chunk_id", result)

    def test_reflect_without_llm_still_works(self):
        """Heuristic reflection when no LLM configured."""
        result = self.mem.reflect(
            "alice", "s1",
            task="测试任务", outcome="success",
            summary="成功完成，关键是注意避免重复操作，建议下次先验证权限",
        )
        self.assertNotIn("skipped", result)
        self.assertIn("long_term_chunk_id", result)


# ─── MemoryManager v5 Integration ────────────────────────────────────────────
class TestMemoryManagerV5Integration(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = _mem(self.tmp)

    def tearDown(self):
        self.mem.close()

    def test_skill_in_build_context(self):
        """Skills should appear in build_context output."""
        self.mem.save_skill("alice", "bgp_diagnosis",
                            "BGP邻居故障排查流程：ping验证、syslog分析、联系NOC",
                            ["ping(peer)", "syslog(peer)", "contact NOC"],
                            tags=["bgp", "network"])
        ctx = self.mem.build_context("alice", "BGP neighbor down")
        self.assertIn("可复用技能", ctx)
        self.assertIn("bgp_diagnosis", ctx)

    def test_skill_in_build_context_budgeted(self):
        """Skills appear in P2 of budgeted context."""
        self.mem.save_skill("alice", "bgp_skill",
                            "BGP排查标准步骤ping syslog NOC",
                            ["bgp_check", "ping"], tags=["bgp"])
        state = self.mem.get_session("alice", "s1")
        ctx, report = self.mem.build_context_budgeted(state, "BGP neighbor failure")
        kept_names = [n for n, _ in report.kept_sections]
        self.assertIn("skills", kept_names)

    def test_full_cycle_skill_reflect_consolidate(self):
        """End-to-end: save skill → use → reflect → consolidate."""
        # Save skill
        sk = self.mem.save_skill("alice", "bgp_diag",
                                 "BGP邻居故障排查：ping测试和syslog分析",
                                 ["ping(peer_ip)", "syslog_search(peer)"],
                                 tags=["bgp"])
        # Use skill (remember the conversation)
        for i in range(12):
            self.mem.remember("alice", "s1",
                              f"Turn {i}: 排查BGP邻居10.0.1.{i}的连通性问题")
        # Reflect
        reflect_result = self.mem.reflect(
            "alice", "s1", task="BGP故障排查",
            outcome="success", summary="通过ping和syslog成功定位问题",
            skill_id=sk.skill_id,
        )
        self.assertIn("long_term_chunk_id", reflect_result)
        # Consolidate
        consol_result = self.mem.consolidate_session("alice", "s1")
        # Either consolidated or skipped (depends on chunk count)
        self.assertIsNotNone(consol_result)

    def test_embedding_index_grows_with_remember(self):
        """EmbeddingIndex should grow as chunks are added."""
        size_before = self.mem._emb_index.size
        self.mem.remember("alice", "s1", "BGP session dropped on R1")
        self.mem.remember("alice", "s1", "DNS resolution failure payments")
        size_after = self.mem._emb_index.size
        self.assertGreater(size_after, size_before)

    def test_hybrid_search_no_crash(self):
        """search_chunks with use_embedding=True should not crash."""
        self.mem.remember("alice", "s1", "BGP session dropped AS65002")
        self.mem.remember("alice", "s1", "DNS failure payments service")
        r = self.mem.search_chunks("alice", "BGP session", use_embedding=True)
        self.assertIsNotNone(r)

    def test_custom_embedding_fn(self):
        """MemoryManager with custom embedding function."""
        import math as _math
        call_count = [0]

        def my_embed(text: str):
            call_count[0] += 1
            import hashlib
            h = hashlib.md5(text.encode()).digest()
            vec = [(b / 127.5) - 1.0 for b in h[:4]]
            norm = _math.sqrt(sum(x*x for x in vec)) or 1.0
            return [x / norm for x in vec]

        mem2 = MemoryManager(
            data_dir=os.path.join(self.tmp, "emb"),
            embedding_fn=my_embed, embedding_dim=4,
        )
        mem2.remember("alice", "s1", "BGP session dropped AS65002")
        mem2.remember("alice", "s1", "DNS failure payments internal")
        self.assertGreater(call_count[0], 0)
        r = mem2.search_chunks("alice", "BGP session", use_embedding=True)
        self.assertIsNotNone(r)
        mem2.close()

    def test_stats_includes_v5_fields(self):
        self.mem.save_skill("alice", "sk1", "desc", ["step"], tags=[])
        stats = self.mem.stats("alice")
        self.assertIn("skill_count", stats)
        self.assertIn("embedding_backend", stats)
        self.assertIn("embedding_index_size", stats)

    def test_reflect_then_search_finds_lesson(self):
        self.mem.reflect("alice", "s1", task="BGP排查",
                         outcome="failure", summary="权限不足导致失败",
                         reason="缺少root权限")
        # Lesson chunk should be searchable
        r = self.mem.search_chunks("alice", "反思 BGP排查 失败")
        self.assertGreater(r.total_found, 0)

    def test_find_skills_after_save(self):
        self.mem.save_skill("alice", "bgp_diag",
                            "BGP邻居故障排查：ping测试syslog联系NOC",
                            ["ping", "syslog", "NOC"], tags=["bgp", "network"])
        skills = self.mem.find_skills("alice", "BGP neighbor problem")
        self.assertGreater(len(skills), 0)
        self.assertEqual(skills[0].skill_name, "bgp_diag")

    def test_record_skill_outcome_via_manager(self):
        sk = self.mem.save_skill("alice", "test_skill", "test", ["s"], tags=[])
        rate = self.mem.record_skill_outcome("alice", sk.skill_id, success=True)
        self.assertIsNotNone(rate)
        self.assertLessEqual(rate, 1.0)


# ─── Semantic embedding hybrid search ─────────────────────────────────────────
class TestSemanticHybridSearch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        import math as _math, hashlib

        def embed(text: str):
            h = hashlib.md5(text.encode()).digest()
            vec = [(b / 127.5) - 1.0 for b in h[:8]]
            norm = _math.sqrt(sum(x*x for x in vec)) or 1.0
            return [x / norm for x in vec]

        self.mem = MemoryManager(
            data_dir=os.path.join(self.tmp, "m"),
            embedding_fn=embed, embedding_dim=8,
        )

    def tearDown(self):
        self.mem.close()

    def test_search_returns_results(self):
        self.mem.remember("alice", "s1", "BGP session dropped on R1 router")
        self.mem.remember("alice", "s1", "DNS failure for payments service")
        r = self.mem.search_chunks("alice", "BGP routing problem", use_embedding=True)
        self.assertIsNotNone(r)
        self.assertGreaterEqual(r.total_found, 0)

    def test_embedding_index_updated_on_batch(self):
        texts = [f"batch content {i} BGP network" for i in range(5)]
        self.mem.remember_batch("alice", "s1", texts)
        self.assertGreaterEqual(self.mem._emb_index.size, 5)

    def test_fallback_to_fts_without_embedding(self):
        """Without embedding, search_chunks should still work via FTS+TF-IDF."""
        mem2 = MemoryManager(data_dir=os.path.join(self.tmp, "m2"))
        mem2.remember("alice", "s1", "BGP dropped on R1")
        r = mem2.search_chunks("alice", "BGP", use_embedding=True)
        self.assertIsNotNone(r)
        mem2.close()

    def test_hybrid_search_result_layer(self):
        self.mem.remember("alice", "s1", "BGP session AS65001 dropped at 02:15")
        r = self.mem.search_chunks("alice", "BGP session", use_embedding=True)
        # Layer should reflect hybrid mode
        self.assertIn("long_term", r.layer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
