"""tests/test_capability_index.py — CSI v1 on real LAN/DC metadata."""
import importlib
import unittest

from skills.loader import SkillLoader
from skills.capability_index import CapabilitySemanticIndex, jaccard, cosine


def _load(prof):
    tm = importlib.import_module(f"profiles.{prof}.tool_meta")
    meta = next((getattr(tm, a) for a in dir(tm)
                 if isinstance(getattr(tm, a), dict) and getattr(tm, a)
                 and all(isinstance(x, dict) for x in getattr(tm, a).values())), {})
    skills = SkillLoader(mode="mock", profile=prof).skill_definitions()
    return meta, skills


def _build_index():
    lt, ls = _load("lan")
    dt, ds = _load("dc")
    idx = CapabilitySemanticIndex(embed_fn=None)  # no embedder in unit test
    idx.build({**lt, **dt}, {**ls, **ds})
    return idx


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.idx = _build_index()
        self.space = self.idx.export_space()

    def test_dc_splits_into_fabric_and_application(self):
        # Correction #1: DC must not be one coarse bucket.
        clusters = self.space["clusters"]
        self.assertIn("dc/fabric", clusters)
        self.assertIn("dc/application", clusters)
        # app tools land in application, fabric tools in fabric
        self.assertIn("dc_get_app_acl", clusters["dc/application"]["members"])
        self.assertIn("dc_bgp_evpn_status", clusters["dc/fabric"]["members"])

    def test_lan_access_cluster_clean(self):
        clusters = self.space["clusters"]
        self.assertIn("access", clusters)
        members = clusters["access"]["members"]
        self.assertIn("get_user_access", members)
        self.assertIn("check_nac_policy", members)


class TestSimilarity(unittest.TestCase):
    def setUp(self):
        self.idx = _build_index()

    def test_similarity_has_reasons(self):
        r = self.idx.similarity("get_user_access", "check_nac_policy")
        self.assertIsNotNone(r)
        self.assertTrue(r.reasons)         # interpretable, not black-box
        self.assertGreater(r.score, 0)

    def test_related_tools_more_similar_than_unrelated(self):
        related = self.idx.similarity("get_user_access", "check_nac_policy").score
        unrelated = self.idx.similarity("get_user_access", "dc_bgp_evpn_status").score
        self.assertGreater(related, unrelated)

    def test_cross_domain_access_skills_similar(self):
        # LAN access-diagnose vs DC app-access-diagnose — should be related
        r = self.idx.similarity("lan_user_access_diagnose", "dc_app_access_diagnose")
        self.assertIsNotNone(r)
        self.assertGreater(r.score, 0.0)


class TestRouteAndNearest(unittest.TestCase):
    def setUp(self):
        self.idx = _build_index()

    def test_nearest_skill_by_tool_set(self):
        # A trajectory that used the DC app tools should map to the DC app skill
        ts = ["dc_check_user_app_access", "dc_get_app_acl", "dc_grant_app_access"]
        res = self.idx.nearest_skill(ts, text="check user app access")
        self.assertIsNotNone(res)
        sid, sim = res
        self.assertEqual(sid, "dc_app_access_diagnose")
        self.assertTrue(sim.reasons)

    def test_route_returns_topk_with_scores(self):
        hits = self.idx.route(tool_set=["get_user_access", "check_nac_policy"],
                              kind="skill", top_k=3)
        self.assertTrue(hits)
        self.assertTrue(all(h.score >= 0 for h in hits))
        # the LAN access diagnose skill should be among top hits
        self.assertIn("lan_user_access_diagnose", [h.target for h in hits])


class TestTrajectoryClustering(unittest.TestCase):
    def setUp(self):
        self.idx = _build_index()

    def test_similar_trajectories_cluster_together(self):
        trajs = [
            {"id": "t1", "tools": ["get_user_access", "check_nac_policy"]},
            {"id": "t2", "tools": ["get_user_access", "check_nac_policy", "list_users"]},
            {"id": "t3", "tools": ["dc_bgp_evpn_status", "dc_fabric_path_trace"]},
        ]
        clusters = self.idx.cluster_trajectories(trajs, threshold=0.4)
        # t1 & t2 (access) together; t3 (fabric) separate
        sizes = sorted(c["size"] for c in clusters)
        self.assertEqual(sizes, [1, 2])

    def test_repeated_trajectory_counts(self):
        trajs = [{"id": f"t{i}", "tools": ["get_user_access", "check_nac_policy"]}
                 for i in range(3)]
        clusters = self.idx.cluster_trajectories(trajs, threshold=0.5)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["size"], 3)   # P1 will threshold on this


if __name__ == "__main__":
    unittest.main()
