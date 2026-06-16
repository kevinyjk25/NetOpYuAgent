"""Behavioral provenance test — imports & calls the real extractor.

(Was: regex-extract the function body from source + exec() it in a fresh
namespace — fragile, broke on any reformat, and didn't exercise the real
imported symbol.) Now imports `_extract_delegation_provenance` directly and
asserts its behavior. The fields originate in task/delegation.py metadata,
travel over A2A, and are placed in env_context by a2a/agent_executor.py;
peer-side HITL cards show a "Delegated from <agent>" banner from them.
"""
import unittest

from integrations.adapters.hitl_executor import _extract_delegation_provenance as extract


class TestDelegationProvenance(unittest.TestCase):
    def test_none_env_context(self):
        self.assertEqual(extract(None), (None, None, None))

    def test_empty_env_context(self):
        self.assertEqual(extract({}), (None, None, None))

    def test_user_query_not_delegated(self):
        # a normal user-initiated request carries no provenance keys
        self.assertEqual(extract({"operator_id": "op-1", "trust_mode": "cautious"}),
                         (None, None, None))

    def test_delegated_request_full_provenance(self):
        env = {"source_agent": "lan-agent",
               "source_session_id": "sess-abc",
               "source_query": "诊断 alice 访问 crm 失败"}
        self.assertEqual(extract(env),
                         ("lan-agent", "sess-abc", "诊断 alice 访问 crm 失败"))

    def test_partial_provenance(self):
        # only agent known → the other two stay None (no crash, no guessing)
        self.assertEqual(extract({"source_agent": "lan-agent"}),
                         ("lan-agent", None, None))

    def test_empty_string_fields_coerce_to_none(self):
        # the `or None` guard: blank strings are treated as absent
        self.assertEqual(extract({"source_agent": "", "source_session_id": "",
                                  "source_query": ""}),
                         (None, None, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
