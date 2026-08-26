"""Retrieval layer tests — was only covered indirectly through the loop.

Covers the Retriever contract + each concrete impl + the cache wrapper:
  BM25 (incl. CJK tokenisation), Keyword, Embedding (fake async embedder),
  Hybrid late-fusion (weighted_sum + RRF), CachedRetriever hit/miss, and the
  shared filter contract (require_tags / exclude_tags / min_score / top_k).
"""
import asyncio
import unittest

from retrieval.base import RetrievalResult, Match
from retrieval.bm25 import BM25Retriever, tokenize
from retrieval.keyword import KeywordRetriever
from retrieval.embedding import EmbeddingRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.cache import CachedRetriever


CORPUS = [
    {"id": "t1", "text": "list network devices and their status",
     "tags": ["device", "read"]},
    {"id": "t2", "text": "restart a service on a host", "tags": ["service", "write"]},
    {"id": "t3", "text": "query radius authentication logs",
     "tags": ["auth", "read"]},
    {"id": "t4", "text": "查询用户网络准入状态与认证日志", "tags": ["auth", "read"]},
]


class _FakeEmbedder:
    """Deterministic toy embedder: vector = per-token presence over a tiny
    vocab, so cosine similarity tracks token overlap (good enough to assert
    ranking direction without a real model)."""
    VOCAB = ["device", "status", "service", "restart", "radius", "auth",
             "log", "network", "查询", "认证", "准入"]

    async def embed(self, text: str) -> list[float]:
        t = text.lower()
        return [1.0 if w in t else 0.0 for w in self.VOCAB]


class TestTokenizer(unittest.TestCase):
    def test_ascii_lowercased_words(self):
        self.assertEqual(tokenize("List Network Devices"),
                         ["list", "network", "devices"])

    def test_cjk_chars_and_bigrams(self):
        toks = tokenize("准入认证")
        # per-char
        for ch in "准入认证":
            self.assertIn(ch, toks)
        # adjacent bigrams
        self.assertIn("准入", toks)
        self.assertIn("认证", toks)

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.r = BM25Retriever()
        self.r.index(CORPUS)

    def test_ranks_relevant_first(self):
        res = self.r.retrieve("restart service", top_k=3)
        self.assertIsInstance(res, RetrievalResult)
        self.assertTrue(res.matches)
        self.assertEqual(res.matches[0].id, "t2")
        self.assertEqual(res.total_pool, 4)

    def test_cjk_query_matches_cjk_doc(self):
        res = self.r.retrieve("用户准入认证", top_k=3)
        ids = [m.id for m in res.matches]
        self.assertIn("t4", ids)

    def test_top_k_caps_results(self):
        res = self.r.retrieve("network device service auth log", top_k=2)
        self.assertLessEqual(len(res.matches), 2)

    def test_no_match_yields_zero_scores(self):
        # default min_score=0.0 returns the pool with 0.0 scores (filtering is
        # opt-in via min_score) — assert the scores are zero, not that the list
        # is empty.
        res = self.r.retrieve("zzzznonexistentterm", top_k=3)
        self.assertTrue(all(m.score == 0.0 for m in res.matches))

    def test_no_match_with_min_score_empties(self):
        res = self.r.retrieve("zzzznonexistentterm", top_k=3, min_score=0.01)
        self.assertEqual(res.matches, [])


class TestFilters(unittest.TestCase):
    """The shared filter contract on the base retrieve()."""
    def setUp(self):
        self.r = BM25Retriever(); self.r.index(CORPUS)

    def test_require_tags(self):
        res = self.r.retrieve("read query log device", top_k=5,
                              require_tags=["write"])
        for m in res.matches:
            self.assertIn("write", m.item.get("tags", []))

    def test_exclude_tags(self):
        res = self.r.retrieve("read query log device service", top_k=5,
                              exclude_tags=["write"])
        for m in res.matches:
            self.assertNotIn("write", m.item.get("tags", []))

    def test_min_score_floor(self):
        res = self.r.retrieve("device", top_k=5, min_score=0.5)
        for m in res.matches:
            self.assertGreaterEqual(m.score, 0.5)


class TestKeyword(unittest.TestCase):
    def test_substring_match(self):
        r = KeywordRetriever(); r.index(CORPUS)
        res = r.retrieve("radius", top_k=3)
        self.assertIn("t3", [m.id for m in res.matches])


class TestEmbedding(unittest.TestCase):
    def test_embedding_ranks_by_similarity(self):
        async def run():
            r = EmbeddingRetriever(embedder=_FakeEmbedder())
            await r.index_async(CORPUS)
            res = await r.retrieve_async("restart service", top_k=3)
            self.assertTrue(res.matches)
            self.assertEqual(res.matches[0].id, "t2")
        asyncio.run(run())


class TestHybrid(unittest.TestCase):
    def test_weighted_sum_fusion(self):
        async def run():
            r = HybridRetriever(embedder=_FakeEmbedder(),
                                bm25_weight=0.5, embed_weight=0.5,
                                fusion="weighted_sum")
            await r.index_async(CORPUS)
            res = await r.retrieve_async("restart service", top_k=3)
            self.assertEqual(res.matches[0].id, "t2")
            # breakdown carries both signals
            self.assertTrue(res.matches[0].breakdown)
        asyncio.run(run())

    def test_rrf_fusion_runs(self):
        async def run():
            r = HybridRetriever(embedder=_FakeEmbedder(), fusion="rrf")
            await r.index_async(CORPUS)
            res = await r.retrieve_async("query auth log", top_k=3)
            self.assertTrue(res.matches)
        asyncio.run(run())


class TestCacheWrapper(unittest.TestCase):
    def test_second_identical_query_is_cache_hit(self):
        async def run():
            inner = BM25Retriever(); inner.index(CORPUS)
            cached = CachedRetriever(inner, max_entries=16)
            r1 = await cached.retrieve_async("restart service", top_k=3)
            r2 = await cached.retrieve_async("restart service", top_k=3)
            self.assertFalse(r1.cache_hit)
            self.assertTrue(r2.cache_hit)
            # same ranking served from cache
            self.assertEqual([m.id for m in r1.matches],
                             [m.id for m in r2.matches])
        asyncio.run(run())

    def test_different_query_misses(self):
        async def run():
            inner = BM25Retriever(); inner.index(CORPUS)
            cached = CachedRetriever(inner, max_entries=16)
            await cached.retrieve_async("restart service", top_k=3)
            r = await cached.retrieve_async("radius logs", top_k=3)
            self.assertFalse(r.cache_hit)
        asyncio.run(run())


class TestFactory(unittest.TestCase):
    def test_tools_to_corpus_shape(self):
        from retrieval.factory import tools_to_corpus
        meta = {"list_devices": {"description": "list all network devices",
                                 "tags": ["device", "read"],
                                 "parameters": {"site": "site id"}}}
        corpus = tools_to_corpus(meta)
        self.assertEqual(len(corpus), 1)
        item = corpus[0]
        self.assertEqual(item["id"], "list_devices")
        # searchable text anchors the tool name + description
        self.assertIn("list_devices", item["text"])
        self.assertIn("network devices", item["text"])

    def test_skills_to_corpus_shape(self):
        from retrieval.factory import skills_to_corpus
        defs = {"app_access_troubleshoot":
                {"purpose": "diagnose app access", "tags": ["access"]}}
        corpus = skills_to_corpus(defs)
        self.assertEqual(corpus[0]["id"], "app_access_troubleshoot")

    def test_build_retriever_hybrid_without_embedder_degrades(self):
        from retrieval.factory import build_retriever

        class _Cfg:
            class retrieval:
                backend = "hybrid"
                hybrid = None
            class embeddings:
                dim = 768
        r = build_retriever(_Cfg, embedder=None)
        # degrades to a usable lexical retriever, doesn't crash
        r.index(CORPUS)
        res = r.retrieve("restart service", top_k=2)
        self.assertTrue(res.matches)


class TestLLMJudgeRerank(unittest.TestCase):
    def test_judge_reranks_first_stage(self):
        async def run():
            first = BM25Retriever(); first.index(CORPUS)

            # judge promotes t3 regardless of first-stage order
            async def judge(system, user):
                return '{"ranking": ["t3", "t2", "t1"]}'

            from retrieval.llm_judge import LLMJudgeRetriever
            r = LLMJudgeRetriever(first_stage=first, llm_fn=judge,
                                  first_stage_top_k=5, fusion_alpha=0.0)
            await r.index_async(CORPUS)
            res = await r.retrieve_async("query auth", top_k=3)
            self.assertTrue(res.matches)
            # judge's top pick surfaces (pure-judge fusion_alpha=0.0)
            self.assertEqual(res.matches[0].id, "t3")
        asyncio.run(run())

    def test_judge_timeout_falls_back_to_first_stage(self):
        async def run():
            first = BM25Retriever(); first.index(CORPUS)

            async def slow_judge(system, user):
                await asyncio.sleep(5)
                return "{}"

            from retrieval.llm_judge import LLMJudgeRetriever
            r = LLMJudgeRetriever(first_stage=first, llm_fn=slow_judge,
                                  timeout_seconds=0.05)
            await r.index_async(CORPUS)
            res = await r.retrieve_async("restart service", top_k=3)
            # falls back to first-stage ranking, doesn't hang or error
            self.assertTrue(res.matches)
            self.assertEqual(res.matches[0].id, "t2")
        asyncio.run(run())


class TestMetaToolRegistry(unittest.TestCase):
    def test_register_and_prompt_section(self):
        from retrieval.meta_tool import MetaTool, MetaToolRegistry

        async def _h(**kw):
            return "ok"
        reg = MetaToolRegistry()
        reg.register(MetaTool(name="list_tools", description="list tools",
                              handler=_h, parameters={"query": "search"},
                              always_inject=True))
        self.assertIsNotNone(reg.get("list_tools"))
        self.assertEqual(len(reg.list_always_injected()), 1)
        section = reg.build_prompt_section()
        self.assertIn("list_tools", section)

    def test_unregister(self):
        from retrieval.meta_tool import MetaTool, MetaToolRegistry

        async def _h(**kw):
            return "ok"
        reg = MetaToolRegistry()
        reg.register(MetaTool(name="x", description="d", handler=_h))
        self.assertTrue(reg.unregister("x"))
        self.assertIsNone(reg.get("x"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
