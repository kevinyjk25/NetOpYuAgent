"""Local archived-source draft-builder checks, not source-label reviews.

The frozen standalone pack has separate always-on tests. Rebuilding its draft
from the original (ignored) source-acquisition archive is local integration.
"""
import copy
import json

import pytest
from jsonschema import Draft202012Validator

from evaluation.bounded_cases import NOMINATIONS, ROOT, _snmp, _sources, draft_metadata, export_draft, make_annotation_input, make_cases, make_dialogues, make_references
from evaluation.bounded_pilot import case_initial_state_digest, make_protocol, validate_protocol, validate_references
from evaluation.bounded_provider import LocalProviderPool
from network_runtime.contracts import sha256_json
from network_runtime.l0.structured_schema import checked_schema


@pytest.fixture(scope="module", autouse=True)
def require_local_source_archive():
    if not (ROOT / NOMINATIONS).is_file():
        pytest.skip("optional original source-acquisition archive absent; standalone frozen pack is tested separately")
    # Existing but changed/incomplete archives must fail, never silently skip.
    _sources()


def _protocol():
    return make_protocol("draft-material-validation", make_cases(), model_digest=sha256_json("no-model-fixture"),
                         harness_digest=sha256_json("no-harness-fixture"), support=draft_metadata()["support"])


def test_twelve_drafts_bind_sources_quotes_state_and_protocol():
    protocol = validate_protocol(_protocol())
    refs = validate_references(protocol, make_references())
    cases = protocol["development_cases"]
    assert len(cases) == len(refs) == 12
    assert sum(c["kind"] == "positive" for c in cases) == 8
    assert len({c["skill_id"] for c in cases}) == 6
    assert len({c["repository_family"] for c in cases}) == 6
    assert len({c["domain"] for c in cases}) >= 3
    assert all(c["source_kind"] == "pinned_public" for c in cases)
    assert all(r["initial_state_digest"] == case_initial_state_digest(c) for c, r in zip(cases, refs))
    assert all(r["initial_state_digest"] != sha256_json(c["provider_fixture"]) for c, r in zip(cases, refs))
    meta = draft_metadata()
    assert meta["status"] == "reference_draft_needs_two_blind_reviews"
    assert not any(meta[k] for k in ("independently_reviewed", "reference_frozen", "researchEvidenceEligible",
                                    "pilotQualified", "runtimeAuthorityGranted", "sourceScriptsExecuted", "actualModelCalls"))
    assert meta["blocking_review_items"]


@pytest.mark.parametrize("index", range(12))
def test_real_sqlite_read_evidence_isolated_and_not_prebuilt_answers(tmp_path, index):
    case = make_cases()[index]
    pool = LocalProviderPool(tmp_path / "pool")
    first = pool.create_arm(f"case-{index}-a", "control", case["provider_fixture"])
    second = pool.create_arm(f"case-{index}-b", "treatment", case["provider_fixture"])
    assert first.binding["isolation_id"] != second.binding["isolation_id"]
    assert first.binding["initial_state_digest"] == second.binding["initial_state_digest"] == case_initial_state_digest(case)
    catalog = {t["name"]: t for t in case["agent_input"]["tools"]}
    for tool in catalog.values():
        checked_schema(tool["input_schema"])
        checked_schema(tool["output_schema"])
    steps = make_dialogues()[case["case_id"]]
    assert "engineering_dialogue" not in case["agent_input"]["arguments"]
    for number, step in enumerate(steps):
        envelope = first.invoke(step["tool"], step["arguments"], request_id=f"read-{number}")
        assert envelope["result"]["ok"] is True
        Draft202012Validator(catalog[step["tool"]]["output_schema"]).validate(envelope["result"])
        receipt = envelope["receipt"]
        assert receipt["before_state_digest"] == receipt["after_state_digest"]
        assert receipt["call"]["independent"] is True
    assert second.receipts() == []
    assert first.snapshot()["state"] == second.snapshot()["state"] == case["provider_fixture"]["state"]
    assert all(t["operation"] == "read" for t in case["provider_fixture"]["tools"])
    assert all("expected_value" not in item for item in steps)
    first.close()
    second.close()


def test_fresh_copies_and_no_references_in_agent_projection():
    cases = make_cases()
    original = copy.deepcopy(cases)
    cases[0]["agent_input"]["arguments"].clear()
    assert make_cases() == original
    for case in original:
        assert not {"criteria", "duties", "reference_digest", "provider_fixture"} & set(case["agent_input"])
        # These are artifact/query-writing tasks, not canned final answers.
        assert '"final_answer"' not in json.dumps(case["provider_fixture"])


def test_old_failure_intents_and_missing_source_obligations_remain_visible():
    cases = {c["case_id"]: c for c in make_cases()}
    assert cases["irql-missing-enricher"]["kind"] == "positive"
    assert "useful unenriched query" in cases["irql-missing-enricher"]["agent_input"]["task"]
    assert "upper endpoint is excluded" in cases["irql-draft"]["agent_input"]["task"]
    assert "available-source inventory" in cases["mesh-incident"]["agent_input"]["task"]
    references = {r["case_id"]: r for _, r in _snmp(_sources()["snmp"])}
    duties = {d["id"]: d for d in references["snmp-existing-howto-single-device"]["duties"]}
    assert duties["wrapper"]["critical"] and not duties["wrapper"]["strict_eligible"]
    assert duties["howto"]["critical"]
    assert any("linear-agent-writes" in gap for gap in draft_metadata()["blocking_review_items"])


def test_full_source_pins_and_explicit_initial_reference_routing():
    sources = draft_metadata()["excluded_source_manifest"]
    snmp = next(s for s in sources if s["repository"] == "netdata/netdata")
    assert len(snmp["documents"]) == 20  # 17 old documents + 3 fixed-commit supplements.
    assert all(not d["executionAllowed"] for s in sources for d in s["documents"])
    assert any(d["path"].endswith("scripts/_lib.sh") and d["projection"] == "host_archive_inert_not_initial"
               for d in snmp["documents"])
    source_bytes = sum(d["bytes"] for d in snmp["documents"])
    initial_bytes = sum(d["bytes"] for d in snmp["documents"] if d["projection"] != "host_archive_inert_not_initial")
    assert initial_bytes < source_bytes


def test_annotation_projection_omits_labels_private_facts_and_dialogues():
    annotation = make_annotation_input()
    assert len(annotation["cases"]) == 12
    prohibited = {"criteria", "duties", "critical", "strict_eligible", "engineering_dialogue", "provider_fixture",
                  "reference_digest", "expected_value", "known_failure_sources", "known_failure_note", "blocking_review_items"}

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()

    assert not keys(annotation) & prohibited


def test_standalone_export_no_overwrite_and_complete_source_text(tmp_path):
    output = tmp_path / "draft"
    exported = export_draft(output)
    assert exported["actualModelCalls"] == 0
    assert json.loads((output / "cases.json").read_text()) == make_cases()
    assert json.loads((output / "references.draft.json").read_text()) == make_references()
    sources = json.loads((output / "sources.json").read_text())
    assert len(sources) == 6
    assert sources[0]["repository"] == "nlamirault/agentheon"
    assert len(sources[0]["documents"]) == 1
    assert "license: Apache-2.0" in next(iter(sources[0]["documents"].values()))
    before = (output / "cases.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_draft(output)
    assert (output / "cases.json").read_bytes() == before
