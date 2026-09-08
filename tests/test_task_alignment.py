"""Mechanical dossier integrity; passing these tests is not semantic accuracy."""

import copy
import json

import pytest

from evaluation.task_alignment import TaskAlignment, build_dossier, prepare
from evaluation.translation_intake import _bundle, _document
from network_runtime.contracts import sha256_json


def fixture():
    text = "Read records before writing a report.\nDo not store secrets in reports.\nExample writes the raw response.\n"
    script = "raise RuntimeError('source must stay inert')\n"
    bundle = _bundle({
        "apiVersion": "effect-runtime.io/translation-intake/v1", "candidateId": "synthetic-dossier-test",
        "repository": "local/fixture", "commitSha": "a" * 40, "snapshotDigest": "sha256:" + "b" * 64,
        "entryPath": "SKILL.md", "documents": [_document("SKILL.md", text.encode(), mode="100644", origin="test"),
            _document("scripts/inert.py", script.encode(), mode="100644", origin="test")],
        "unavailableSnapshotResources": [], "supplementAttempts": [], "parentBundleDigest": None,
    })
    def quote(start, end):
        return {"path": "SKILL.md", "start": start, "end": end, "quote": text[start:end]}
    first_end = text.index("\n") + 1
    spec = {
        "api_version": "netopyu.io/task-alignment/v1", "bundle_digest": bundle["bundleDigest"], "task_id": "fixture-task",
        "task": "Read fixture records and propose a sanitized report, without writing.",
        "task_origin": "developer_authored_evaluation_request", "task_basis": [quote(0, first_end)],
        "documents": [{"path": "SKILL.md", "ranges": [{"start": 0, "end": len(text)}], "rationale": "Read the full fixture; not independent semantic proof."},
                      {"path": "scripts/inert.py", "ranges": [], "rationale": "Retain unreviewed script text and never execute it."}],
        "host_needs": [{"id": "read", "operation": "read records", "tool_name": None,
                        "rationale": "The test declares a need, not a captured host capability."}],
        "obligations": [{"id": "read-first", "statement": "Read records before report construction.", "disposition": "candidate_read",
                         "citations": [quote(0, first_end)], "rationale": "Source statement needs a separately declared host tool.", "host_needs": ["read"]}],
        "findings": [{"id": "privacy", "category": "source_tension", "explanation": "An example may conflict with the retention instruction.",
                      "citations": [quote(first_end, text.index("\n", first_end) + 1), quote(text.rindex("Example"), len(text))],
                      "next_action": "Keep unresolved and define a reviewed output retention policy."}],
        "dependencies": [{"target": "../../outside.py", "disposition": "unresolved", "citation": quote(0, first_end),
                          "rationale": "A declared inert target string is never opened or executed."}],
    }
    return bundle, spec


def build(bundle, spec, catalog=None):
    return build_dossier(bundle, TaskAlignment.model_validate(spec), catalog)


def test_dossier_preserves_source_scripts_and_never_declares_semantic_success():
    bundle, spec = fixture()
    files = build(bundle, spec)
    report = files["report.json"]
    assert report["missingHostRequirements"] == 1
    assert report["findings"][0]["resolved"] is False
    assert report["dependencies"][0]["retainedTextPresent"] is False
    assert report["runtimeAuthorityGranted"] is report["wholeSkillTranslationProven"] is False
    assert report["taskReferenceClosureProven"] is report["semanticEntailmentProven"] is False
    assert report["providerCalls"] == report["modelCalls"] == 0
    assert files["source-bundle.json"] == bundle
    assert any("source must stay inert" in p["text"] for p in files["model-input.json"]["sourcePages"]["pages"])


def test_diagnostics_and_obligation_labels_are_not_leaked_to_future_author_input():
    bundle, spec = fixture()
    files = build(bundle, spec)
    serialized = json.dumps(files["model-input.json"])
    for hidden in ["candidate_read", "source_tension", "An example may conflict", "read-first"]:
        assert hidden not in serialized
    changed = copy.deepcopy(spec)
    changed["obligations"][0]["rationale"] = "A revised reviewer judgment, not a revised source or task."
    other = build(bundle, changed)
    assert other["model-input.json"] == files["model-input.json"]
    assert other["report.json"]["reportDigest"] != files["report.json"]["reportDigest"]


@pytest.mark.parametrize("change", ["bundle", "source_text", "offset", "quote", "unreviewed_citation", "missing_document", "duplicate_document",
                                   "overlap", "range_end", "duplicate_obligation", "unknown_host", "no_host", "duplicate_host", "one_sided_conflict"])
def test_integrity_and_declared_inventory_fail_closed(change):
    bundle, spec = fixture()
    if change == "bundle":
        spec["bundle_digest"] = "sha256:" + "c" * 64
    elif change == "source_text":
        bundle["documents"][0]["content"] += "tampered"
    elif change in {"offset", "quote"}:
        span = spec["obligations"][0]["citations"][0]
        if change == "offset":
            span["start"] += 1
        else:
            span["quote"] = "This quote is not present in the retained source."
    elif change == "unreviewed_citation":
        spec["documents"][0]["ranges"][0]["end"] = 40
    elif change == "missing_document":
        spec["documents"].pop()
    elif change == "duplicate_document":
        spec["documents"].append(copy.deepcopy(spec["documents"][0]))
    elif change == "overlap":
        spec["documents"][0]["ranges"].append({"start": 0, "end": 10})
    elif change == "range_end":
        spec["documents"][0]["ranges"][0]["end"] = 9999
    elif change == "duplicate_obligation":
        spec["obligations"].append(copy.deepcopy(spec["obligations"][0]))
    elif change == "unknown_host":
        spec["obligations"][0]["host_needs"] = ["invented"]
    elif change == "no_host":
        spec["obligations"][0]["host_needs"] = []
    elif change == "duplicate_host":
        spec["host_needs"].append(copy.deepcopy(spec["host_needs"][0]))
    elif change == "one_sided_conflict":
        spec["findings"][0]["citations"] *= 0
    with pytest.raises(ValueError):
        build(bundle, spec)


def test_one_passage_repeated_does_not_count_as_two_sides_of_conflict():
    bundle, spec = fixture()
    spec["findings"][0]["citations"] = [spec["findings"][0]["citations"][0]] * 2
    with pytest.raises(ValueError, match="distinct"):
        build(bundle, spec)


def test_even_bound_supported_schemas_do_not_prove_source_mapping_or_activation():
    bundle, spec = fixture()
    spec["host_needs"][0]["tool_name"] = "read_records"
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    catalog = {"tools": [{"name": "read_records", "inputSchema": schema, "outputSchema": schema}]}
    files = build(bundle, spec, catalog)
    report = files["report.json"]
    assert report["missingHostRequirements"] == 0
    assert report["hostRequirements"][0]["shapeReadyOnly"]
    assert report["hostRequirements"][0]["sourceApiMappingVerified"] is False
    assert report["runtimeAuthorityGranted"] is False and report["translationMetrics"] is None
    catalog["tools"][0]["name"] = "changed"
    bundle["documents"][0]["content"] = "changed"
    assert files["host-catalog.json"]["tools"][0]["name"] == "read_records"
    assert files["source-bundle.json"]["documents"][0]["content"] != "changed"


@pytest.mark.parametrize("kind", ["context", "policy"])
def test_host_context_and_policy_do_not_require_invented_tools(kind):
    bundle, spec = fixture()
    spec["host_needs"].append({"id": "separate", "kind": kind, "operation": "host trust declaration",
                               "rationale": "Host context and policy are not API calls or runtime permissions."})
    report = build(bundle, spec)["report.json"]
    row = report["hostRequirements"][-1]
    assert row["shapeReadyOnly"] is None
    assert row["diagnostics"][0]["code"] == "host_" + kind + "_not_bound"
    spec["host_needs"][-1]["tool_name"] = "invented_authentication_or_policy_api"
    with pytest.raises(ValueError, match="context/policy"):
        build(bundle, spec)


@pytest.mark.parametrize("issue", ["duplicate", "missing_output", "unsupported_pattern", "wrong_name", "scalar_root"])
def test_catalog_gaps_do_not_become_host_readiness(issue):
    bundle, spec = fixture()
    spec["host_needs"][0]["tool_name"] = "read_records"
    schema = {"type": "object", "properties": {}}
    catalog = {"tools": [{"name": "read_records", "inputSchema": schema, "outputSchema": schema}]}
    if issue == "duplicate":
        catalog["tools"] *= 2
        with pytest.raises(ValueError, match="unique"):
            build(bundle, spec, catalog)
        return
    if issue == "missing_output":
        del catalog["tools"][0]["outputSchema"]
    elif issue == "unsupported_pattern":
        catalog["tools"][0]["outputSchema"] = {"type": "string", "pattern": ".*"}
    elif issue == "scalar_root":
        catalog["tools"][0]["outputSchema"] = {"type": "string"}
    else:
        catalog["tools"][0]["name"] = "not_read_records"
    report = build(bundle, spec, catalog)["report.json"]
    assert report["missingHostRequirements"] == 1
    assert report["hostRequirements"][0]["diagnostics"]


def test_offline_cli_preparation_seals_outputs_and_rejects_overwrite_first(tmp_path):
    bundle, spec = fixture()
    b, s = tmp_path / "bundle.json", tmp_path / "alignment.json"
    b.write_text(json.dumps(bundle))
    s.write_text(json.dumps(spec))
    report = prepare(b, s, tmp_path / "out")
    stored = json.loads((tmp_path / "out/report.json").read_text())
    assert stored == report
    assert report["reportDigest"] == sha256_json({k: v for k, v in report.items() if k != "reportDigest"})
    with pytest.raises(FileExistsError):
        prepare("missing-input", "missing-spec", tmp_path / "out")
